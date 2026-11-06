"""Merge four frozen Fair300k workers, validating complete paired test results."""
from pathlib import Path
import argparse,csv,hashlib,io,json,zipfile
import numpy as np
from scipy.stats import t

VERSION='fair300k_frozen_benchmark_v1'
LABELS=('EMA','OLD')
METRICS=['throughput_mbps','qos_ues','service_coverage_ues','positive_ues','jain_ema','jain_instant_qualified',
 'jain_time_average','p05_ue_mean_rate_mbps','min_ue_mean_rate_mbps','rolling_unique_ge1_1min','rolling_unique_ge1_5min',
 'longest_zero_gap_slots_max','longest_below1_gap_slots_max','zero_service_slot_fraction',
 'energy_consumed_wh','joules_per_mbit','blocked_per_policy_move','longest_consecutive_blocked_slots',
 'effective_unavailable_uav_slot_fraction','saturated_backhaul_slot_fraction','unique_ge1_ues','unique_qos_ues',
 'arrival_below20_events','min_arrival_soc_percent','actor_ms','actor_ms_p95','energy_shortfall_total_j',
 'reward_under_ema','reward_under_instant','zero_throughput_positive_reward_slots',
 'loss_outside_access_mbps','loss_association_mbps','loss_access_radio_mbps','loss_bh_disconnected_mbps','loss_bh_schedule_capacity_mbps']

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def csv_write(path,rows):
 if not rows:return
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def validate_raw(read,prefix,row,p):
 commit=json.loads(read(prefix+'/COMMIT.json'))
 required={'slots.csv','ue_summary.csv','trace.npz','energy.npz','energy_events.json','collision.npz','collision_events.json','cycles.json','network_samples.jsonl','summary.json'}
 if set(commit['files_sha256'])!=required:raise ValueError('Incomplete scientific seed commit')
 for name,value in commit['files_sha256'].items():
  if Path(name).name!=name or hashlib.sha256(read(prefix+'/'+name)).hexdigest()!=value:raise ValueError('Corrupt seed file: '+name)
 with np.load(io.BytesIO(read(prefix+'/trace.npz')),allow_pickle=False) as z:
  raw=z['ue_rates'];pos=z['ue_positions'];ema=z['ema_rates']
  if raw.shape!=(10000,90) or not np.isfinite(raw).all() or raw.min() < -1e-9 or raw.max()>4+1e-6:raise ValueError('Invalid UE rate trace')
  r=np.maximum(raw,0.)
  checks={'throughput_mbps':r.sum(1).mean(),'qos_ues':(r>=4-1e-6).sum(1).mean(),'service_coverage_ues':(r>=1-1e-9).sum(1).mean(),'jain_time_average':float(r.mean(0).sum()**2/(90*np.dot(r.mean(0),r.mean(0)))) if r.any() else 0.}
  previous=np.vstack([np.zeros((1,90)),ema[:-1]])
  if not np.allclose(previous*(59/60)+r/60,ema,rtol=1e-11,atol=1e-11):raise ValueError('EMA recurrence mismatch')
  def jains(values):
   maxima=values.max(1);scaled=np.divide(values,maxima[:,None],out=np.zeros_like(values),where=maxima[:,None]>1e-9)
   den=90*np.square(scaled).sum(1)
   return np.divide(scaled.sum(1)**2,den,out=np.zeros(len(values)),where=den>0)
  jf=jains(ema);ji=jains(np.where(r>=1-1e-9,r,0.))
  chain='00'*32
  for values in pos:chain=hashlib.sha256(bytes.fromhex(chain)+np.asarray(values,dtype='<f8').tobytes(order='C')).hexdigest()
  if chain!=row['ue_trace_sha256']:raise ValueError('UE trace hash mismatch')
 with np.load(io.BytesIO(read(prefix+'/energy.npz')),allow_pickle=False) as z:
  con=z['consumed_j'];balance=z['energy_before_j']+z['inserted_j']-z['removed_j']-con-z['energy_after_j']
  if con.shape!=(10000,5) or not np.isfinite(con).all() or np.abs(balance).max()>1e-5:raise ValueError('Energy balance failed')
  checks['energy_consumed_wh']=con.sum()/3600
  checks['energy_shortfall_total_j']=z['shortfall_j'].sum()
  consumed=con.sum(1)
  slot_records=csv.DictReader(io.StringIO(read(prefix+'/slots.csv').decode()))
  fields=['slot','throughput_mbps','qos_ues','service_coverage_ues','jain_ema','jain_instant_qualified','reward_total','active_blocked_uavs']
  logged=np.asarray([[float(rec[k]) for k in fields] for rec in slot_records])
  if logged.shape!=(10000,len(fields)) or not np.isfinite(logged).all():raise ValueError('Invalid per-slot CSV')
  np.testing.assert_array_equal(logged[:,0],np.arange(1,10001))
  np.testing.assert_allclose(logged[:,1],r.sum(1),atol=1e-7,rtol=1e-9)
  np.testing.assert_allclose(logged[:,2],(r>=4-1e-6).sum(1),atol=0,rtol=0)
  np.testing.assert_allclose(logged[:,3],(r>=1-1e-9).sum(1),atol=0,rtol=0)
  np.testing.assert_allclose(logged[:,4],jf,atol=1e-10,rtol=1e-9)
  np.testing.assert_allclose(logged[:,5],ji,atol=1e-10,rtol=1e-9)
  tp=p['training_protocol'];ec=tp['energy_config'];penalty=tp['collision_config']['collision_weight']*logged[:,7]/5
  energy_penalty=ec['energy_weight']*consumed/(5*ec['reference_w']*5)
  shared_base=tp['reward_weights']['throughput']*r.sum(1)/360-energy_penalty-penalty
  reward_ema=shared_base+tp['reward_weights']['fairness']*jf
  reward_instant=shared_base+tp['reward_weights']['fairness']*ji
  np.testing.assert_allclose(logged[:,6],reward_ema if row['label']=='EMA' else reward_instant,atol=1e-9,rtol=1e-9)
  checks.update(jain_ema=jf.mean(),jain_instant_qualified=ji.mean(),reward_under_ema=reward_ema.mean(),reward_under_instant=reward_instant.mean())
 for k,value in checks.items():
  if not np.isclose(row[k],value,rtol=1e-9,atol=1e-7):raise ValueError('Summary/raw mismatch: '+k)


def scan_inputs(input_dir):
 results={};protocols={};provenance=[]
 def consume(p,read,prefix,origin):
  if p.get('version')!=VERSION or p.get('diagnostic'):return
  if p['label'] not in LABELS or p['part'] not in (1,2):raise ValueError('Unknown worker')
  common={k:p[k] for k in ['version','config','registry','source_sha256']}
  if digest(common)!=p['comparison_fingerprint']:raise ValueError('Invalid comparison fingerprint')
  if p['config']['test_seeds']!=list(range(930101,930111)) or p['horizon']!=10000:raise ValueError('Wrong test budget')
  expected=p['config']['test_seeds'][(p['part']-1)*5:p['part']*5]
  if p['target_seeds']!=expected:raise ValueError('Wrong worker partition')
  selection=p['registry'][p['label']]
  if p['selection']!=selection or digest(p['training_protocol'])!=selection['protocol_sha256']:raise ValueError('Training selection mismatch')
  if selection['selected_step']!={'EMA':294000,'OLD':288000}[p['label']]:raise ValueError('Test checkpoint reselected')
  identity=(p['label'],p['part'])
  if identity in protocols and protocols[identity]!=p:raise ValueError('Conflicting worker protocols')
  protocols[identity]=p
  for seed in p['target_seeds']:
   base=(prefix+'/' if prefix else '')+f'seed{seed}'
   try:row=json.loads(read(base+'/summary.json'))
   except (KeyError,FileNotFoundError):continue
   if not row['complete']:continue
   if row['label']!=p['label'] or row['seed']!=seed or row['slots']!=10000 or row['target_slots']!=10000 or row['comparison_fingerprint']!=p['comparison_fingerprint'] or row['selected_step']!=selection['selected_step']:raise ValueError('Summary/worker mismatch')
   commit=json.loads(read(base+'/COMMIT.json'))
   if commit['protocol_sha256']!=digest(p):raise ValueError('Seed commit/worker mismatch')
   validate_raw(read,base,row,p)
   key=(p['label'],seed)
   if key in results and results[key]!=row:raise ValueError('Conflicting complete seed: '+str(key))
   results[key]=row
  provenance.append(dict(origin=origin,worker=list(identity)))
 for file in sorted(Path(input_dir).rglob('*.zip')):
  with zipfile.ZipFile(file) as z:
   names=[n for n in z.namelist() if n.endswith('/protocol.json') or n=='protocol.json']
   relevant=[]
   for name in names:
    p=json.loads(z.read(name))
    if p.get('version')==VERSION:relevant.append((name,p))
   if not relevant:continue
   if z.testzip() is not None:raise ValueError('Corrupt ZIP: '+str(file))
   origin=dict(file=str(file),sha256=sha(file),kind='zip')
   for name,p in relevant:consume(p,z.read,name.rsplit('/',1)[0] if '/' in name else '',origin)
 for file in sorted(Path(input_dir).rglob('protocol.json')):
  p=json.loads(file.read_text())
  if p.get('version')!=VERSION:continue
  root=file.parent
  def read(relative):
   relative=relative.lstrip('/')
   if Path(relative).is_absolute() or '..' in Path(relative).parts:raise ValueError('Unsafe result path')
   return (root/relative).read_bytes()
  consume(p,read,'',dict(directory=str(root),kind='extracted'))
 if not protocols:raise ValueError('No Fair300k benchmark output found; attach the REVIEW ZIPs or extracted folders')
 if len({p['comparison_fingerprint'] for p in protocols.values()})!=1:raise ValueError('Different scientific benchmark protocols')
 p=next(iter(protocols.values()))
 for seed in p['config']['test_seeds']:
  if all((label,seed) in results for label in LABELS) and results['EMA',seed]['ue_trace_sha256']!=results['OLD',seed]['ue_trace_sha256']:raise ValueError('Unpaired UE traces: '+str(seed))
 return results,p,provenance


def comparison(results,seeds):
 aggregate=[];paired=[]
 for label in LABELS:
  rows=[results[label,s] for s in seeds];rec=dict(label=label,seeds=len(rows))
  for key in METRICS:
   values=[r[key] for r in rows if r.get(key) is not None];rec[key]=float(np.mean(values)) if values else None
  energy=sum(r['energy_consumed_wh']*3600 for r in rows);data=sum(r['delivered_mbit'] for r in rows);requests=sum(r['policy_move_requests_total'] for r in rows)
  rec['joules_per_mbit_pooled']=energy/data if data else None
  rec['blocked_fraction_pooled']=sum(r['active_blocked_total'] for r in rows)/requests if requests else None
  aggregate.append(rec)
 for key in METRICS:
  pairs=[(results['EMA',s][key],results['OLD',s][key]) for s in seeds if results['EMA',s].get(key) is not None and results['OLD',s].get(key) is not None]
  if not pairs:continue
  a,b=np.asarray(pairs).T;delta=b-a
  half=float(t.ppf(.975,len(delta)-1)*delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None
  paired.append(dict(metric=key,paired_seeds=len(delta),ema_mean=float(a.mean()),old_mean=float(b.mean()),old_minus_ema=float(delta.mean()),delta_ci95_low=float(delta.mean()-half) if half is not None else None,delta_ci95_high=float(delta.mean()+half) if half is not None else None,old_relative_change_percent=float((b.mean()/a.mean()-1)*100) if a.mean()!=0 else None,ci_scope='Paired t interval across scenario seeds conditional on two frozen policies; no multiple-comparison adjustment'))
 return aggregate,paired


def report(input_dir,output):
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 results,p,provenance=scan_inputs(input_dir);seeds=p['config']['test_seeds'];paired=[s for s in seeds if all((m,s) in results for m in LABELS)]
 missing={m:[s for s in seeds if (m,s) not in results] for m in LABELS}
 csv_write(output/'per_seed.csv',[results[k] for k in sorted(results)])
 meta=dict(status='complete' if len(paired)==10 else 'partial',paired_seeds=paired,missing=missing,comparison_fingerprint=p['comparison_fingerprint'],provenance=provenance,scope=p['config']['scope'])
 (output/'completeness.json').write_text(json.dumps(meta,indent=2)+'\n')
 if len(paired)!=10:
  # Remove stale aggregate outputs only; per-seed evidence remains intact.
  for name in ['aggregate.csv','paired_comparison.csv','paired_comparison.json','paired_metrics.png']:(output/name).unlink(missing_ok=True)
  text=f'# Chưa đủ benchmark Fair300k\n\nCó {len(paired)}/10 seed ghép cặp hoàn tất.\n\nThiếu: {missing}.\n\nChưa tổng hợp kết luận cuối cùng.\n'
 else:
  aggregate,stats=comparison(results,seeds);csv_write(output/'aggregate.csv',aggregate);csv_write(output/'paired_comparison.csv',stats)
  (output/'paired_comparison.json').write_text(json.dumps(stats,indent=2)+'\n')
  names={'throughput_mbps':'Thông lượng (Mbps)','qos_ues':'UE đạt4Mbps mỗi slot','service_coverage_ues':'UE đạt1Mbps mỗi slot','jain_ema':'Jain EMA','jain_time_average':'Jain tốc độ trung bình UE','p05_ue_mean_rate_mbps':'Phân vị5% tốc độ UE (Mbps)','energy_consumed_wh':'Năng lượng đội UAV (Wh)','joules_per_mbit_pooled':'Joule/Mbit (gộp)','blocked_fraction_pooled':'Tỷ lệ yêu cầu active bị chặn','longest_zero_gap_slots_max':'Gián đoạn dài nhất (slot), trung bình theo seed','reward_under_ema':'Cùng công thức reward EMA','reward_under_instant':'Cùng công thức reward tức thời'}
  lines=['# Benchmark Fair300k: 10 seed ×10.000 slot','','EMA@294k và Fair cũ@288k. Hai checkpoint đã chọn bằng validation, không chọn lại theo test.','','| Chỉ số | Fair EMA | Fair cũ |','|---|---:|---:|']
  for key,name in names.items():
   vals=[f'{r[key]:.6f}' if r.get(key) is not None else 'không xác định' for r in aggregate];lines.append('| '+name+' | '+' | '.join(vals)+' |')
  lines+=['','Khoảng tin cậy tính trên chênh lệch Fair cũ−EMA theo seed kịch bản. Đây không phải độ biến động giữa nhiều seed huấn luyện. Thông lượng là chỉ số so sánh chính; các chỉ số khác hỗ trợ giải thích, chưa điều chỉnh cho nhiều phép so sánh.','','Reward riêng của hai chế độ khác định nghĩa Fair. Chỉ hai cột reward_under_ema và reward_under_instant dùng cùng công thức để đối chiếu. Yêu cầu bị chặn không phải tai nạn vật lý. Ngưỡng SOC20% là mục tiêu dự trữ, chưa là bảo đảm cứng.']
  text='\n'.join(lines)+'\n'
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
  for ax,key,title in zip(axes,['throughput_mbps','joules_per_mbit','blocked_per_policy_move'],['Thông lượng (Mbps)','Năng lượng trên dữ liệu (J/Mbit)','Yêu cầu active bị chặn (%)']):
   scale=100 if key=='blocked_per_policy_move' else 1
   a=np.array([results['EMA',s][key]*scale if results['EMA',s].get(key) is not None else np.nan for s in seeds]);b=np.array([results['OLD',s][key]*scale if results['OLD',s].get(key) is not None else np.nan for s in seeds])
   for aa,bb in zip(a,b):ax.plot([0,1],[aa,bb],color='grey',alpha=.4,lw=1)
   ax.scatter(np.zeros(10),a,color='#cc6b28');ax.scatter(np.ones(10),b,color='#2365a0');ax.set_xticks([0,1],['EMA294k','Fair cũ288k']);ax.set_title(title);ax.grid(axis='y',alpha=.2)
  fig.savefig(output/'paired_metrics.png',dpi=180);plt.close(fig)
 (output/'REPORT_VI.md').write_text(text);print(text,flush=True);return meta

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.input,a.output)
