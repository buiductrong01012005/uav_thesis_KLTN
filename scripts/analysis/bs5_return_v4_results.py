"""Compact float64-rate/float32-input evidence; NumPy + stdlib, no env/Torch."""
from pathlib import Path
import csv,gzip,hashlib,json,statistics
import numpy as np
try:
 from bs5_continuous_common import lifecycle
 from bs5_return_audit import f32
 from bs5_return_commit_audit import CommitAuditor
 from analyze_bs5_cn_h1200_return import validate_collision
except ImportError:
 from scripts.analysis.bs5_continuous_common import lifecycle
 from scripts.analysis.bs5_return_audit import f32
 from scripts.analysis.bs5_return_commit_audit import CommitAuditor
 from scripts.analysis.analyze_bs5_cn_h1200_return import validate_collision

def metric(a,start=0,end=None):
 end=len(a['rates']) if end is None else end;rates=a['rates'][start:end];s=rates.sum(1);sq=(rates*rates).sum(1);mean=rates.mean(0)
 zero=s<=1e-12;longest=run=0
 for x in zero:run=run+1 if x else 0;longest=max(longest,run)
 status=a['status_service'][start:end];roles=a['roles_service'][start:end];assoc=a['association'][start:end];routes=a['routes'][start:end]
 relay_active=int(((status==0)&(roles==1)).sum());busy=0;relay_rate=0.
 for t in range(len(rates)):
  used=set()
  for u,aid in enumerate(assoc[t]):
   if aid>=0 and routes[t,aid]>=0 and rates[t,u]>1e-12:used.add(int(routes[t,aid]));relay_rate+=rates[t,u]
  busy+=len(used)
 return dict(utility=float(s.mean()/360),throughput_mbps=float(s.mean()),mean_qos_ue=float((rates>=4-1e-9).sum(1).mean()),
  mean_positive_service_ue=float((rates>1e-12).sum(1).mean()),mean_slot_jain=float(np.divide(s*s,90*sq,out=np.zeros_like(s),where=sq>0).mean()),
  jain_mean_rate=float(mean.sum()**2/(90*(mean*mean).sum())) if (mean*mean).sum() else 0.,
  zero_reward_slots=int(zero.sum()),longest_zero_streak=longest,terminal_zero_streak=run,
  all_active_zero_slots=int((zero&(status==0).all(1)).sum()),
  return_stranded_uav_slots=int(a['stranded'][start:end].sum()),return_wait_uav_slots=int(a['wait'][start:end].sum()),
  return_reversals=int(a['reversals'][start:end].sum()),relay_active_uav_slots=relay_active,
  relay_empty_slots=relay_active-busy,empty_relay_fraction=(relay_active-busy)/relay_active if relay_active else None,
  relay_routed_throughput_mbps=relay_rate/len(rates),per_ue_mean_rates_mbps=mean.tolist())

def life_rows(a):
 for t in range(len(a['rates'])):
  assoc={str(i):np.flatnonzero(a['association'][t]==i).tolist() for i in range(5)}
  routes={str(i):(None if a['routes'][t,i]<0 else int(a['routes'][t,i])) for i in range(5) if a['status_service'][t,i]==0 and a['roles_service'][t,i]==0}
  yield dict(slot=t+1,decision=dict(status=a['status_before'][t].tolist()),
   post=dict(status=a['status_post'][t].tolist(),positions=a['positions_post'][t].tolist()),
   service=dict(status=a['status_service'][t].tolist(),roles=a['roles_service'][t].tolist(),uav_positions=a['positions_service'][t].tolist(),
    network=dict(association=assoc,routes=routes,ue_rate_mbps=a['rates'][t].tolist())))

def summarize_game(a,index):
 d=metric(a);tag={k:index[k] for k in ['arm','scenario','group']};windows=[]
 for start in range(0,len(a['rates']),1000):windows.append(dict(**tag,first=start+1,last=min(start+1000,len(a['rates'])),**metric(a,start,min(start+1000,len(a['rates'])))))
 cycles,_=lifecycle(list(life_rows(a)),index['initial'],run=index['arm']+'_seed551',scenario=index['scenario'])
 return dict(**tag,**d),windows,[dict(arm=index['arm'],group=index['group'],**c) for c in cycles]

def audit_game(directory,index,deep=True):
 path=directory/index['data'];assert hashlib.sha256(path.read_bytes()).hexdigest()==index['data_sha256']
 with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 h=index['horizon'];assert a['rates'].shape==(h,90) and a['obs'].shape==(h+1,5,116) and a['state'].shape==(h+1,111)
 assert np.isfinite(a['rates']).all() and ((a['rates']>=0)&(a['rates']<=4+1e-9)).all()
 assert np.array_equal(a['positions_before'][1:],a['positions_post'][:-1])
 assert np.array_equal(a['status_before'][1:],a['status_post'][:-1]) and np.array_equal(a['battery_before'][1:],a['battery_post'][:-1])
 assert np.array_equal(a['roles_before'][1:],a['roles_post'][:-1])
 assert all(a['mask'][t,i,act] for t,row in enumerate(a['actions']) for i,act in enumerate(row))
 if index['arm']=='C-N':assert not a['mask'][:,:,1::2].any() and not (a['actions']%2).any() and (a['routes']==-1).all()
 auditor=CommitAuditor();count=0;prior_targets=None
 for item in index['control_chunks']:
  f=directory/item['path'];assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256']
  for control in json.load(gzip.open(f,'rt')):
   t=count;assert control['slot']==t+1
   row=dict(raw_actions=a['actions'][t].tolist(),task_condition=index['condition'],return_control=control['return_control'],collision=control['collision'],charging_station_cells=index['initial']['charging_station_cells'],
    applied_actions=[dict(autonomous=a['status_before'][t,i]!=0 or i in control['return_control']['forced_return_ids']) for i in range(5)],
    decision=dict(positions=a['positions_before'][t].tolist(),status=a['status_before'][t].tolist(),battery=a['battery_before'][t].tolist(),roles=a['roles_before'][t].tolist()),
    service=dict(uav_positions=a['positions_service'][t].tolist(),status=a['status_service'][t].tolist(),roles=a['roles_service'][t].tolist()),
    post=dict(positions=a['positions_post'][t].tolist(),status=a['status_post'][t].tolist(),battery=a['battery_post'][t].tolist(),roles=a['roles_post'][t].tolist()))
   if prior_targets is not None:assert row['return_control']['target_station_idx_before']==prior_targets
   prior_targets=row['return_control']['target_station_idx_post']
   validate_collision(row,index['initial']['charging_station_cells'])
   if deep:auditor.validate(row,index['resolved_config'])
   else:
    # No additional policy/env rollout. Deep planner is an optional offline audit.
    for i,s in enumerate(row['return_control']['effective_status']):
     b=row['decision']['battery'][i]
     if s==0:cost=index['resolved_config']['hover_drain']+index['resolved_config']['move_extra_drain']*(row['raw_actions'][i]//2!=4)+(index['resolved_config']['access_radio_drain'] if row['service']['roles'][i]==0 else index['resolved_config']['relay_radio_drain']);expected=max(0.,f32(b-f32(cost)))
     elif s==1:expected=max(0.,f32(b-f32(index['resolved_config']['returning_step_drain'])))
     elif s==2:expected=min(index['resolved_config']['max_battery'],f32(b+f32(index['resolved_config']['charge_rate_per_slot'])))
     else:expected=b
     assert expected==row['post']['battery'][i]
     if s==1 and row['decision']['positions'][i]!=row['service']['uav_positions'][i]:assert b>=index['resolved_config']['returning_step_drain']
   count+=1
 assert count==h
 return a

def write_table(out,name,rows):
 (out/(name+'.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False))
 keys=sorted({k for r in rows for k,v in r.items() if not isinstance(v,(list,dict))}) or ['empty']
 with (out/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)

def analyze_run(root,out,deep=False,require_complete=False):
 root=Path(root).resolve();out=Path(out).resolve();assert not out.is_relative_to(root);out.mkdir(parents=True,exist_ok=True)
 p=json.loads((root/'protocol.json').read_text());status=json.loads((root/'status.json').read_text());commit=json.loads((root/'commit.json').read_text());lock=json.loads((root/'input_lock.json').read_text())
 complete=status['status']=='complete'
 if require_complete:assert complete and len(commit['completed'])==len(p['jobs'])
 if (root/'SHA256SUMS.json').exists():
  for rel,h in json.loads((root/'SHA256SUMS.json').read_text()).items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
 for arm,cp in p['checkpoints'].items():assert hashlib.sha256((root/'checkpoints'/f'{arm}.pt').read_bytes()).hexdigest()==cp['sha256']
 baseline=[x for x in json.loads((root/'baseline_v3_games.json').read_text()) if x['phase']=='paired'];bykey={(x['arm'],x['scenario']):x for x in baseline}
 games=[];windows=[];cycles=[];changes=[];total=0;counts={'C-R':0,'C-N':0}
 for item in commit['completed']:
  idx=json.loads((root/item['index']).read_text());assert all(idx[k]==item[k] for k in ['arm','scenario','group'])
  assert idx['horizon']==p['horizon'] and idx['return_version']==p['return_version']
  assert idx['checkpoint_sha256']==p['checkpoints'][item['arm']]['sha256']
  assert idx['weights_sha256_before']==idx['weights_sha256_after']
  entry=lock['traces'][str(item['scenario'])];assert idx['trace_sha256']==entry['position_sha256']
  for k,v in idx['initial'].items():assert entry['initial'][k]==v
  a=audit_game((root/item['index']).parent,idx,deep);d,ww,cc=summarize_game(a,idx)
  for k,v in idx['metrics'].items():
   actual=d[k]
   if isinstance(v,float):assert abs(actual-v)<1e-9,k
   else:assert actual==v,k
  games.append(d);windows+=ww;cycles+=cc;total+=idx['horizon'];counts[item['arm']]+=idx['horizon']
  old=bykey[(item['arm'],item['scenario'])];change={k:item[k] for k in ['arm','scenario','group']}
  for k in ['utility','throughput_mbps','mean_qos_ue','mean_slot_jain','mean_positive_service_ue','zero_reward_slots','return_stranded_uav_slots','longest_zero_streak']:
   change.update({k+'_v3':old[k],k+'_v4':d[k],k+'_delta':d[k]-old[k]})
  changes.append(change)
  print(json.dumps(dict(phase='offline_audit',games=len(games),slots=total,deep_planner=deep)),flush=True)
 ledger=json.loads((root/'budget_ledger.json').read_text());assert ledger['training_steps']==0
 for arm,row in ledger['phases'].items():assert counts[arm]<=row['observed']<=row['charged']<=row['cap']
 if complete:assert total==p['planned_steps'] and all(counts[arm]==row['observed']==row['charged']==row['cap'] for arm,row in ledger['phases'].items())
 aggregate=[]
 for group in ['seen','heldout']:
  for arm in ['C-R','C-N']:
   own=[x for x in games if x['group']==group and x['arm']==arm]
   if not own:continue
   d=dict(group=group,arm=arm,games=len(own))
   for k in ['utility','throughput_mbps','mean_qos_ue','mean_positive_service_ue','mean_slot_jain','jain_mean_rate']:d[k]=statistics.fmean(x[k] for x in own)
   for k in ['zero_reward_slots','return_stranded_uav_slots','return_wait_uav_slots','return_reversals','relay_active_uav_slots','relay_empty_slots']:d[k]=sum(x[k] for x in own)
   d['empty_relay_fraction']=d['relay_empty_slots']/d['relay_active_uav_slots'] if d['relay_active_uav_slots'] else None
   d['longest_zero_streak']=max(x['longest_zero_streak'] for x in own)
   cc=[x for x in cycles if x['group']==group and x['arm']==arm];lat=[x['service_latency_slots'] for x in cc if x['service_latency_slots'] is not None]
   d.update(cycles_started=len(cc),cycles_rejoined=sum(x['rejoin_active_slot'] is not None for x in cc),cycles_useful=len(lat),median_latency=statistics.median(lat) if lat else None,
    censor_horizon=sum(x['censor_reason']=='horizon' for x in cc),next_return_without_useful=sum(x['censor_reason']=='next_return' for x in cc))
   aggregate.append(d)
 for name,data in [('games',games),('windows1000',windows),('cycles',cycles),('before_after',changes),('aggregate',aggregate)]:write_table(out,name,data)
 summary=dict(status=status,complete=complete,diagnostic=p['diagnostic'],training_steps=0,committed_steps=total,budget=ledger,deep_planner_recomputed=deep,
  scope='Frozen v3-trained CR60k/CN36k evaluated under v4; controller sensitivity, one training seed. Former heldout traces have now been inspected; not a fresh untouched test.',
  encoding='NPZ exact float32 actor inputs and float64 rates; full Return/collision telemetry chunks; UE tapes stay in immutable input Dataset and are identified by hash')
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 lines=['# Return-v4: eval lại C-R60k / C-N36k',f"\nTrạng thái: {status['status']}; diagnostic={p['diagnostic']}; slot đã commit={total}/{p['planned_steps']}; train=0.",
  '\nChỉ thay controller Return: giữ đường khi next hop còn trống, replan khi next hop bị chặn, tránh ô vừa rời nếu có đường khác. Trigger/reserve, drains, charging, reward, router, actor input và collision-v2 giữ nguyên. Không bảo đảm thoát mọi cấu hình chặn động.',
  '\n| Nhóm | Policy | Game | Utility | Mbps | UE đủ4Mbps | Jain từng slot | Slot rate0 | Stranded UAV-slot |',
  '|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for d in aggregate:lines.append(f"| {d['group']} | {d['arm']} | {d['games']} | {d['utility']:.6f} | {d['throughput_mbps']:.2f} | {d['mean_qos_ue']:.2f} | {d['mean_slot_jain']:.4f} | {d['zero_reward_slots']} | {d['return_stranded_uav_slots']} |")
 lines+=['\n`before_after.csv` so v3/v4 từng policy và trace, giữ đầy đủ cả ca190006. `cycles.json` giữ censoring horizon/next-return; useful một lần không phải service liên tục. Jain từng slot khác Jain cả game. Utility và throughput là cùng bằng chứng đổi đơn vị.',
  '\nHai checkpoint được khóa từ trước, không train hay chọn lại dựa trên kết quả mới. Dữ liệu này đo độ nhạy với controller v4. Chưa xác nhận seed552; các trace trước đây gọi heldout đã được xem và dùng để chẩn đoán nên không được gọi là test mới chưa xem.',
  '\nMặc định kiểm hash/continuity/masks/energy/collision và tính lại metric từ dữ liệu compact. Thêm --deep-planner để dựng độc lập đầy đủ route memory, BFS và lifecycle; không chạy env/policy. Muốn resume cần mọi evidence part và cùng Dataset/runtime; không reset ledger khi partial_error.']
 (out/'REPORT_VI.md').write_text('\n'.join(lines),encoding='utf-8');return summary

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--deep-planner',action='store_true');p.add_argument('--require-complete',action='store_true');a=p.parse_args();print(json.dumps(analyze_run(a.root,a.output,a.deep_planner,a.require_complete)))
