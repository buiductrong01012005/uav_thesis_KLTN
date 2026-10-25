"""Standalone stdlib raw audit/report for the matched Return-v3 Relay ablation.
Default checks masks, continuity, rates, exact collision choices, energy and tapes.
--deep-planner additionally reconstructs every BFS trigger/Return plan (slower).
"""
from pathlib import Path
import argparse,collections,csv,gzip,hashlib,json,math,statistics,struct
try:
    from bs5_bundle import atomic_json,canonical_hash,file_hash,read_json,validate_rates
    from bs5_continuous_common import lifecycle
    from analyze_bs5_cn_h1200_return import metrics,validate_collision
    from bs5_return_audit import validate_return,f32
except ImportError:
    from scripts.analysis.bs5_bundle import atomic_json,canonical_hash,file_hash,read_json,validate_rates
    from scripts.analysis.bs5_continuous_common import lifecycle
    from scripts.analysis.analyze_bs5_cn_h1200_return import metrics,validate_collision
    from scripts.analysis.bs5_return_audit import validate_return,f32

def safe(root,rel):
    p=(root/rel).resolve();assert p.is_relative_to(root.resolve()) and not p.is_symlink();return p
def load_game(root,rel):
    index=read_json(safe(root,rel));rows=[];updates=[]
    for c in index['chunks']:
        p=safe(root,c['path']);assert file_hash(p)==c['sha256']
        with gzip.open(p,'rt') as f:d=json.load(f)
        rows+=d['rows'];updates+=d['updates']
    assert len(rows)==index['spec']['horizon'] and [r['slot'] for r in rows]==list(range(1,len(rows)+1))
    return index,rows,updates
def validate(index,rows,arm,deep):
    cfg=index['resolved_config'];h=len(rows);assert cfg['allow_relay']==(arm=='C-R');uehash=hashlib.sha256()
    for t,r in enumerate(rows):
        assert r['arm']==arm and not r['terminated'] and r['truncated']==(t+1==h)
        assert r['env_t_before']==t and r['env_t_after']==t+1 and r['surviving_bs_ids']==[0,1,2,3,4]
        assert r['service']['ue_positions']==r['decision']['ue_positions'];validate_rates(r)
        if t:
            prev=rows[t-1];assert prev['post']==r['decision'] and prev['next_obs']==r['obs'] and prev['next_state']==r['state']
            assert prev['next_action_mask']==r['action_mask'] and prev['env_rng_after']==r['env_rng_before']
        for a,action in enumerate(r['raw_actions']):assert r['action_mask'][a][action]
        if arm=='C-N':
            assert all(not any(m[1::2]) for m in r['action_mask']+r['next_action_mask'])
            assert all(a%2==0 for a in r['raw_actions'])
            assert all(v is None for v in r['service']['network']['routes'].values())
        validate_collision(r,index['initial']['charging_station_cells'])
        hubs={tuple(v) for v in index['initial']['charging_station_cells']}
        for phase in ['decision','service','post']:
            s=r[phase];positions=s['uav_positions'] if phase=='service' else s['positions']
            cells=[tuple(p) for p,status in zip(positions,s['status']) if status!=3 and tuple(p) not in hubs]
            assert len(cells)==len(set(cells))
        d=r['return_control'];effective=d['effective_status'];frozen=not r['task_condition']['battery']
        for a,state in enumerate(effective):
            before=r['decision']['battery'][a];actual=r['post']['battery'][a]
            assert before==r['service']['battery'][a]
            if frozen:expected=before
            elif state==0:
                cost=cfg['hover_drain']+cfg['move_extra_drain']*(r['raw_actions'][a]//2!=4)+(cfg['access_radio_drain'] if r['service']['roles'][a]==0 else cfg['relay_radio_drain'])
                expected=max(0.,f32(before-f32(cost)))
            elif state==1:expected=max(0.,f32(before-f32(cfg['returning_step_drain'])))
            elif state==2:expected=min(cfg['max_battery'],f32(before+f32(cfg['charge_rate_per_slot'])))
            else:expected=before
            assert f32(expected)==actual,('energy',t+1,a)
            if state==1 and r['decision']['positions'][a]!=r['service']['uav_positions'][a]:assert before>=cfg['returning_step_drain']
        if deep:validate_return(r,cfg)
        ue=[x for pos in r['service']['ue_positions'] for x in pos];uehash.update(struct.pack('<'+'d'*len(ue),*ue))
    ue=[x for pos in rows[-1]['post']['ue_positions'] for x in pos];uehash.update(struct.pack('<'+'d'*len(ue),*ue))
    assert uehash.hexdigest()==index['trace']['position_sha256'],'UE tape digest mismatch'
    return uehash.hexdigest()
def maximum_zero(rows):
    run=longest=0;start=1;best=None
    for i,r in enumerate(rows,1):
        if sum(r['service']['network']['ue_rate_mbps'])<=1e-12:
            if not run:start=i
            run+=1
            if run>longest:longest=run;best=(start,i)
        else:run=0
    return longest,run,best
def stats(rows):
    d=metrics(rows);longest,terminal,interval=maximum_zero(rows);j=[]
    for r in rows:
        x=r['service']['network']['ue_rate_mbps'];s=sum(v*v for v in x);j.append(sum(x)**2/(len(x)*s) if s else 0.)
    d.update(mean_slot_jain=statistics.fmean(j),longest_zero_streak=longest,terminal_zero_streak=terminal,longest_zero_interval=interval,
        all_active_zero_slots=sum(all(s==0 for s in r['service']['status']) and r['reward']<=1e-12 for r in rows),
        min_return_arrival_battery=min((r['post']['battery'][a] for r in rows for a in range(5) if r['decision']['status'][a]==1 and r['post']['status'][a]==2),default=None),
        return_reversals=sum(rows[t]['return_control']['effective_status'][a]==rows[t-1]['return_control']['effective_status'][a]==1 and
            rows[t]['service']['uav_positions'][a]==rows[t-1]['decision']['positions'][a] and rows[t]['decision']['positions'][a]!=rows[t]['service']['uav_positions'][a]
            for t in range(1,len(rows)) for a in range(5)))
    return d
def table(out,name,rows):
    atomic_json(out/(name+'.json'),rows)
    keys=sorted({k for r in rows for k,v in r.items() if not isinstance(v,(list,dict))}) or ['empty']
    with (out/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
def analyze(root,out,deep=False,require_complete=False):
    root=Path(root).resolve();out=Path(out).resolve();assert not out.is_relative_to(root);out.mkdir(parents=True,exist_ok=True)
    p=read_json(root/'protocol.json');status=read_json(root/'status.json');complete=status['status']=='complete'
    if require_complete:assert complete
    if (root/'SHA256SUMS.json').exists():
        for rel,digest in read_json(root/'SHA256SUMS.json').items():assert file_hash(safe(root,rel))==digest,rel
    games=[];cycles=[];windows=[];perue=[];curve=[];slots=collections.Counter();updates=0
    croot=root/'cn'
    if (croot/'commit.json').exists():
        commit=read_json(croot/'commit.json');cp=read_json(croot/'protocol.json');ledger=read_json(croot/'budget_ledger.json')
        for eid,rel in enumerate(commit['games']):
            index,rows,us=load_game(croot,rel);assert index['spec']==cp['game_schedule'][eid];validate(index,rows,'C-N',deep)
            d=stats(rows);assert abs(d['utility']-index['metrics']['utility'])<1e-10;slots['cn_train']+=len(rows);updates+=len(us)
            games.append(dict(phase='cn_train',episode=eid,arm='C-N',**d))
        for ev in commit['evaluations']:
            values=[]
            for rel in ev['games']:
                index,rows,us=load_game(croot,rel);assert not us;validate(index,rows,'C-N',deep);d=stats(rows);values.append(d['utility']);slots['cn_validation']+=len(rows)
                games.append(dict(phase='cn_validation',step=ev['step'],scenario=index['spec']['scenario_seed'],arm='C-N',**d))
            assert abs(statistics.fmean(values)-ev['utility'])<1e-10;curve.append(dict(step=ev['step'],utility=statistics.fmean(values)))
        assert slots['cn_train']==commit['steps'] and updates==commit['updates']
        if curve:
            best=max(curve,key=lambda x:(x['utility'],-x['step']));assert best['step']==commit['best_step'] and abs(best['utility']-commit['best_utility'])<1e-10
        for phase,k in [('train','cn_train'),('eval','cn_validation')]:
            b=ledger['phases'][phase];assert slots[k]<=b['observed']<=b['charged']<=b['cap']
        if read_json(croot/'status.json')['status']=='complete':assert slots['cn_train']==cp['planned_training_steps'] and slots['cn_validation']==cp['planned_validation_steps']
    pairdir=root/'paired';paired=[];allpair=[]
    if (pairdir/'commit.json').exists():
        pair=read_json(pairdir/'protocol.json');pc=read_json(pairdir/'commit.json')
        assert len(pc['completed'])<=len(pair['jobs'])
        for arm,checkpoint in pair['weights'].items():assert file_hash(pairdir/checkpoint['path'])==checkpoint['sha256']
        for item in pc['completed']:
            arm=item['arm'];index,rows,us=load_game(pairdir/arm,item['index'].split('/',1)[1]);assert not us
            tape=validate(index,rows,arm,deep);d=stats(rows);slots['paired_'+arm]+=len(rows)
            assert index['policy_checkpoint_sha256']==pair['weights'][arm]['sha256']
            tag=dict(phase='paired',arm=arm,scenario=item['scenario'],group=item['group']);games.append(dict(**tag,**d));allpair.append(dict(**tag,trace_hash=tape,initial=index['initial'],condition=index['spec']['condition'],**d))
            cs,unused=lifecycle(rows,index['initial'],run=arm+'_seed551',scenario=item['scenario']);cycles.extend(dict(arm=arm,group=item['group'],**c) for c in cs)
            for start in range(0,len(rows),1000):windows.append(dict(**tag,first=start+1,last=min(start+1000,len(rows)),**stats(rows[start:start+1000])))
            means=d['per_ue_mean_rates_mbps']
            for ue,value in enumerate(means):perue.append(dict(**tag,ue=ue,mean_rate_mbps=value))
        for group in ['seen','heldout']:
            for scenario in sorted({x['scenario'] for x in allpair if x['group']==group}):
                rows=[x for x in allpair if x['group']==group and x['scenario']==scenario]
                if len(rows)!=2:continue
                a=next(x for x in rows if x['arm']=='C-R');b=next(x for x in rows if x['arm']=='C-N');assert a['trace_hash']==b['trace_hash']
                assert a['initial']==b['initial'] and a['condition']==b['condition'],'Paired initial conditions differ'
                paired.append(dict(group=group,scenario=scenario,cr_utility=a['utility'],cn_utility=b['utility'],utility_delta=a['utility']-b['utility'],
                    cr_qos=a['mean_qos_ue'],cn_qos=b['mean_qos_ue'],cr_zero=a['zero_reward_slots'],cn_zero=b['zero_reward_slots'],
                    cr_stranded=a['return_stranded_uav_slots'],cn_stranded=b['return_stranded_uav_slots']))
        for arm in ['C-R','C-N']:
            ledger=read_json(pairdir/arm/'budget_ledger.json')['phases']['eval'];assert slots['paired_'+arm]<=ledger['observed']<=ledger['charged']<=ledger['cap']
        if complete:assert len(pc['completed'])==len(pair['jobs']) and sum(slots['paired_'+arm] for arm in ['C-R','C-N'])==p['paired_evaluation_steps']
    aggregate=[]
    for group in ['seen','heldout']:
        scenarios={x['scenario'] for x in paired if x['group']==group}
        for arm in ['C-R','C-N']:
            own=[x for x in allpair if x['group']==group and x['arm']==arm and x['scenario'] in scenarios]
            if not own:continue
            d=dict(group=group,arm=arm,games=len(own))
            for k in ['utility','throughput_mbps','mean_qos_ue','mean_positive_service_ue','jain_mean_rate','mean_slot_jain']:d[k]=statistics.fmean(x[k] for x in own)
            for k in ['zero_reward_slots','return_stranded_uav_slots','return_wait_uav_slots','relay_empty_slots','relay_active_uav_slots']:d[k]=sum(x[k] for x in own)
            d['empty_relay_fraction']=d['relay_empty_slots']/d['relay_active_uav_slots'] if d['relay_active_uav_slots'] else None
            d['longest_zero_streak']=max(x['longest_zero_streak'] for x in own);aggregate.append(d)
            cc=[c for c in cycles if c['arm']==arm and c['group']==group and c['scenario'] in scenarios]
            latency=[c['service_latency_slots'] for c in cc if c['service_latency_slots'] is not None]
            d.update(cycles_started=len(cc),cycles_arrived=sum(c['charging_slot'] is not None for c in cc),
                cycles_rejoined=sum(c['rejoin_active_slot'] is not None for c in cc),cycles_useful=len(latency),
                cycles_horizon_censored=sum(c['censor_reason']=='horizon' for c in cc),cycles_next_return_without_useful=sum(c['censor_reason']=='next_return' for c in cc),
                rejoin_latency_median=statistics.median(latency) if latency else None,
                minimum_arrival_battery=min((x['min_return_arrival_battery'] for x in own if x['min_return_arrival_battery'] is not None),default=None))
    for name,data in [('games',games),('cycles',cycles),('windows1000',windows),('per_ue',perue),('learning_curve_cn',curve),('paired',paired),('aggregate',aggregate)]:table(out,name,data)
    budgets={str(f.relative_to(root)):read_json(f) for f in sorted(root.rglob('budget_ledger.json')) if not f.relative_to(root).as_posix().startswith('preflight/')}
    summary=dict(status=status,complete=complete,diagnostic=p['diagnostic'],committed_slots=dict(slots),budgets=budgets,updates=updates,paired_scenarios=len(paired),
        deep_planner_recomputed=deep,verification='raw hashes,UE tape digests,masks,continuity,rates,energy,exact endpoint collision,selection and paired inputs',
        scope='One training seed. Seen diagnostic and locked heldout test separately. Matched curriculum includes Relay-oriented fixtures; not an optimized C-N upper bound. No C-R training or10k checkpoint selection.')
    atomic_json(out/'summary.json',summary)
    lines=['# C-R/C-N Return-v3 — kết quả',f"\nTrạng thái: {status['status']}. Diagnostic={p['diagnostic']}.",
        '\nC-R giữ checkpoint Return-v3/seed551/60k; C-N train fresh cùng recipe60k, chọn best trên validation1200. Hai policy gặp cùng UE tape và điều kiện đầu; UAV/pin/association tự diễn biến.',
        '\n| Nhóm | Policy | Game | Utility | Mbps | UE đủ4Mbps/slot | Jain từng slot | Slot rate0 | Stranded UAV-slot |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregate:lines.append(f"| {r['group']} | {r['arm']} | {r['games']} | {r['utility']:.6f} | {r['throughput_mbps']:.2f} | {r['mean_qos_ue']:.2f} | {r['mean_slot_jain']:.4f} | {r['zero_reward_slots']} | {r['return_stranded_uav_slots']} |")
    lines+=['\n| Nhóm | Policy | Return | Đến trạm | Rejoin | Useful | Censor horizon | Next-Return trước useful | Median latency useful |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregate:lines.append(f"| {r['group']} | {r['arm']} | {r['cycles_started']} | {r['cycles_arrived']} | {r['cycles_rejoined']} | {r['cycles_useful']} | {r['cycles_horizon_censored']} | {r['cycles_next_return_without_useful']} | {r['rejoin_latency_median'] if r['rejoin_latency_median'] is not None else 'N/A'} |")
    lines+=['\nXem paired.csv để so chênh lệch từng trace; cycles.json giữ censoring horizon/next-return. Jain cả game khác Jain từng slot. Relay-empty của C-N là N/A (không có denominator). Throughput và utility là cùng bằng chứng đổi đơn vị.',
        '\nMột training seed; không kết luận hội tụ hay superiority của thuật toán/curriculum. Trace mới được khóa trước khi chạy; không dùng kết quả10k để chọn checkpoint. Review chỉ gồm game đã commit; partial_error/uncommitted compute phải được giữ, không reset ledger để chạy lại.',
        '\nKiểm đầy đủ BFS độc lập: chạy analyzer với --deep-planner; chế độ mặc định đã kiểm energy/endpoint/mask/tape và không chạy env hoặc policy.']
    (out/'REPORT_VI.md').write_text('\n'.join(lines),encoding='utf-8');return summary
def main():
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--deep-planner',action='store_true');a.add_argument('--require-complete',action='store_true');x=a.parse_args();print(json.dumps(analyze(x.root,x.output,x.deep_planner,x.require_complete)))
if __name__=='__main__':main()
