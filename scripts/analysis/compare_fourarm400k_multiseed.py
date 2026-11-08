"""Four learning curves, mean ± sample SD across training seeds; validation only."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.analysis.fourarm400k_multiseed import ARMS,SEEDS,collect_training,make_bundle

def write_csv(path,rows):
 if not rows:return
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def compare(input_dir,output,freeze_models=False):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 runs,provenance=collect_training(input_dir,include_models=freeze_models)
 expected={(a,s) for a in ARMS for s in SEEDS};missing=sorted(expected-set(runs))
 complete=not missing and all(r['data']['status']['status']=='complete' for r in runs.values())
 meta=dict(status='complete_12_training_runs' if complete else 'partial_training',missing=missing,scope='Each curve averages the two validation scenarios within a run, then averages three training seeds. Band is sample SD of training-seed means, not CI and not two validation-seed min/max.',provenance=provenance)
 bestrows=[];curve_rows=[]
 fields=[('reward_total','Reward tổng',1),('throughput_mbps','Thông lượng (Mbps)',1),('qos_ues','UE đạt 4 Mbps mỗi slot',1),('service_coverage_ues','UE đạt 1 Mbps mỗi slot',1),('jain_time_average','Jain trên tốc độ trung bình UE',1),('joules_per_mbit','J/Mbit',1)]
 fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');colors=['#777777','#3073a5','#44916f','#d77c28']
 for arm,color in zip(ARMS,colors):
  present=[s for s in SEEDS if (arm,s) in runs]
  for seed in present:
   d=runs[arm,seed]['data'];rec=dict(arm=arm,training_seed=seed,status=d['status']['status'],committed_steps=d['status']['committed_training_steps'])
   b=d['status'].get('best_by_metric',{}).get('reward_total')
   if b:
    v=[r for r in d['validation'] if r['step']==b['step']];rec.update(selected_step=b['step'],selected_reward=b['score'])
    for key,_,_ in fields:
     vals=[r.get(key) for r in v if r.get(key) is not None];rec[key]=float(np.mean(vals)) if vals else None
   bestrows.append(rec)
  if not present:continue
  step_sets=[{int(r['step']) for r in runs[arm,s]['data']['validation']} for s in present]
  steps=sorted(set.intersection(*step_sets))
  for ax,(key,title,scale) in zip(axes.flat,fields):
   matrix=[]
   for seed in present:
    vals=[]
    for st in steps:
     rows=[r[key] for r in runs[arm,seed]['data']['validation'] if r['step']==st and r.get(key) is not None]
     vals.append(float(np.mean(rows)) if len(rows)==2 else np.nan)
    matrix.append(vals)
   matrix=np.asarray(matrix);mean=matrix.mean(0);sd=matrix.std(0,ddof=1) if len(present)>1 else np.zeros(len(steps))
   ax.plot(np.asarray(steps)/1000,mean*scale,color=color,label=f'{arm} (n={len(present)}/3)')
   if len(present)>1:ax.fill_between(np.asarray(steps)/1000,(mean-sd)*scale,(mean+sd)*scale,color=color,alpha=.13)
   for j,st in enumerate(steps):curve_rows.append(dict(arm=arm,step=st,metric=key,training_seeds=len(present),mean=float(mean[j]) if np.isfinite(mean[j]) else None,sample_sd=float(sd[j]) if np.isfinite(sd[j]) else None))
 for ax,(_,title,_) in zip(axes.flat,fields):ax.set(title=title,xlabel='Bước môi trường (nghìn)');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle('Đường học: trung bình các seed train, dải ±1 SD\nMỗi điểm của một seed train là trung bình hai seed validation; chưa phải test.',fontsize=12)
 fig.savefig(output/'learning_mean_sd.png',dpi=160);plt.close(fig)
 # Separate seed curves in four panels, rather than twelve labels on the main chart.
 fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
 for ax,arm in zip(axes.flat,ARMS):
  for seed in SEEDS:
   if (arm,seed) not in runs:continue
   v=runs[arm,seed]['data']['validation'];steps=sorted({r['step'] for r in v});vals=[np.mean([r['throughput_mbps'] for r in v if r['step']==st]) for st in steps]
   ax.plot(np.asarray(steps)/1000,vals,label=f'Seed train {seed}')
  ax.set(title=arm,xlabel='Bước môi trường (nghìn)',ylabel='Mbps');ax.grid(alpha=.2)
  if ax.lines:ax.legend(fontsize=8)
 fig.suptitle('Kiểm tra riêng từng seed train, cùng tiêu chí validation');fig.savefig(output/'individual_training_seeds.png',dpi=160);plt.close(fig)
 write_csv(output/'validation_best_per_training_seed.csv',bestrows);write_csv(output/'learning_mean_sd.csv',curve_rows)
 if freeze_models and complete:
  m=make_bundle(runs,output/'FOURARM400K_FROZEN_MODELS.zip');meta['frozen_models']=len(m['models'])
 elif freeze_models:meta['frozen_models']=0
 (output/'completeness.json').write_text(json.dumps(meta,indent=2)+'\n')
 (output/'REPORT_VI.md').write_text('# Tổng hợp ba seed huấn luyện\n\n'+('Đủ 12 lượt train.' if complete else 'Chưa đủ 12 lượt train; kết quả là tạm thời.')+'\n\nĐường chính có bốn cấu hình. Dải là độ lệch chuẩn giữa seed train, không phải khoảng tin cậy. Mỗi model giữ checkpoint best theo reward validation riêng. Không chọn seed train tốt nhất để đại diện cấu hình. Chỉ đóng băng model khi đủ 12 lượt và có best.pt; benchmark test là bước riêng.\n')
 return meta
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze-models',action='store_true');a=p.parse_args();print(json.dumps(compare(a.input,a.output,a.freeze_models),indent=2))
