"""Offline raw-slot audit and seven-method reports; no training or new rollout."""
from pathlib import Path
import json
import csv
import gzip
import shutil
import statistics
import time

import numpy as np
from scripts.analysis.bs5_v4_fresh_report import checked_game, cycle_rows, table, figure_module, pack, full_archive
from scripts.analysis.bs5_return_v4_results import metric
from scripts.analysis.bs5_bundle import atomic_json, canonical_hash, file_hash, read_json


def write_rows(directory, name, rows):
    table(directory / (name+'.csv'), rows)
    atomic_json(directory / (name+'.json'), rows)


def outage_rows(rates, tag):
    zero = rates.sum(1) <= 1e-12
    rows = []; start = None
    for i, value in enumerate(np.r_[zero, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            rows.append(dict(**tag, first_slot=start+1, last_slot=i, duration_slots=i-start,
                             duration_seconds=(i-start)*5, reaches_horizon=i==len(zero)))
            start = None
    return rows


def compute_metrics(a,index,worker_timing=None):
    seconds=np.asarray(a['decision_s'],dtype=float)
    previews=np.asarray(a['preview_calls'])
    assert len(seconds)==index['horizon'] and np.isfinite(seconds).all() and (seconds>=0).all()
    assert (previews>=0).all() and int(previews.sum())==index['preview_calls']
    decision_total=float(seconds.sum());wall=float(index['runtime_s'])
    assert decision_total<=wall+1e-6
    d=dict(rollout_wall_s=wall,rollout_wall_minutes=wall/60,decision_total_s=decision_total,
           other_rollout_wall_s=max(0.,wall-decision_total),decision_mean_ms=float(seconds.mean()*1000),
           decision_median_ms=float(np.median(seconds)*1000),decision_p95_ms=float(np.quantile(seconds,.95)*1000),
           decision_p99_ms=float(np.quantile(seconds,.99)*1000),decision_max_ms=float(seconds.max()*1000),
           executed_slots=len(seconds),slots_per_wall_second=len(seconds)/wall,preview_calls=int(previews.sum()),
           previews_per_slot=float(previews.mean()),worker_wall_s=None,worker_cpu_s=None)
    if worker_timing:
        d.update(worker_wall_s=worker_timing['worker_wall_s'],worker_cpu_s=worker_timing['worker_cpu_s'])
    assert abs(d['decision_median_ms']-index['decision_median_ms'])<1e-6
    assert abs(d['decision_p95_ms']-index['decision_p95_ms'])<1e-6
    return d


def write_slot_timing(analysis,label,a,tag):
    folder=analysis/'decision_timing_slots';folder.mkdir(exist_ok=True)
    with gzip.open(folder/(label+'.csv.gz'),'wt',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method','evaluation_seed','slot','decision_ms','preview_calls'])
        writer.writeheader()
        for i,(seconds,previews) in enumerate(zip(a['decision_s'],a['preview_calls'])):
            writer.writerow(dict(method=tag['method'],evaluation_seed=tag['evaluation_seed'],slot=i+1,
                                 decision_ms=float(seconds)*1000,preview_calls=int(previews)))


def plot_compute(analysis,timing):
    if not timing:return
    plt=figure_module();fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    labels=[r['method'] for r in timing]
    axes[0].bar(labels,[r['rollout_wall_minutes'] for r in timing])
    axes[0].set(ylabel='Rollout wall time (minutes)',title='Includes simulation, decisions and recording')
    xx=np.arange(len(labels));width=.38
    axes[1].bar(xx-width/2,[max(r['decision_median_ms'],1e-6) for r in timing],width,label='Median')
    axes[1].bar(xx+width/2,[max(r['decision_p95_ms'],1e-6) for r in timing],width,label='P95')
    axes[1].set(xticks=xx,xticklabels=labels,yscale='log',ylabel='Decision latency (ms, logarithmic)',title='Actor or heuristic including previews')
    axes[1].legend()
    for ax in axes:ax.tick_params(axis='x',labelrotation=45);ax.grid(axis='y',alpha=.2)
    fig.savefig(analysis/'compute_cost.png',dpi=170);fig.savefig(analysis/'compute_cost.pdf');plt.close(fig)


def save_plots(analysis, summary, windows, cycles):
    if not summary:
        return
    plt = figure_module()
    order = [r['method'] for r in summary]
    fig, axes = plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    keys = [('utility','Utility'),('mean_qos_ue','UE reaching 4 Mbps'),
            ('mean_positive_service_ue','UE with positive rate'),('mean_slot_jain','Mean slot Jain'),
            ('zero_slot_fraction','Zero-service fraction'),('empty_relay_fraction','Empty relay fraction')]
    for ax,(key,label) in zip(axes.flat,keys):
        for method in order:
            own = [r for r in windows if r['method']==method and r.get(key) is not None]
            if own:
                ax.plot([r['last_slot'] for r in own],[r[key] for r in own],marker='o',markersize=3,label=method)
        ax.set(xlabel='Evaluation slots',ylabel=label)
        ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8,ncol=2)
    fig.suptitle('Scenario seed 260001; fixed best checkpoints within 120k; 1,000-slot windows')
    fig.savefig(analysis/'horizon_metrics.png',dpi=170);fig.savefig(analysis/'horizon_metrics.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,key,title in zip(axes,['utility','mean_qos_ue','mean_slot_jain'],['Utility','UE reaching 4 Mbps','Mean slot Jain']):
        ax.bar(order,[r[key] for r in summary]);ax.set(ylabel=title)
        ax.tick_params(axis='x',labelrotation=45);ax.grid(axis='y',alpha=.2)
    fig.suptitle('One evaluation scenario; no between-scenario confidence interval')
    fig.savefig(analysis/'method_comparison.png',dpi=170);fig.savefig(analysis/'method_comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for method in order:
        own=[r for r in cycles if r['method']==method]
        lat=sorted(r['service_latency_slots'] for r in own if r['service_latency_slots'] is not None)
        if lat:
            axes[0].step(lat,np.arange(1,len(lat)+1)/len(lat),where='post',label=method)
    if axes[0].get_legend_handles_labels()[0]:axes[0].legend(fontsize=8)
    axes[0].set(xlabel='Slots from rejoin to useful service',ylabel='Conditional empirical CDF',title='Observed service-return cycles only')
    no_service=[r['next_return_without_service'] for r in summary]
    censored=[r['horizon_censored_after_rejoin'] for r in summary]
    axes[1].bar(order,no_service,label='Next Return without service')
    axes[1].bar(order,censored,bottom=no_service,label='Horizon after rejoin')
    axes[1].tick_params(axis='x',labelrotation=45);axes[1].set(ylabel='Cycle count');axes[1].legend(fontsize=8)
    fig.savefig(analysis/'postcharge_service.png',dpi=170);fig.savefig(analysis/'postcharge_service.pdf');plt.close(fig)


def export_report(output, deep=False):
    from scripts.evaluation.run_bs5_benchmark7_best120k import collect, verify_protocol
    output=Path(output).resolve();p=verify_protocol(output)
    completed,progress=collect(output)
    stamp=time.strftime('%Y%m%d_%H%M%S',time.gmtime())
    out=output.parent/(output.name+'_export_'+stamp)
    suffix=1
    while out.exists():
        suffix+=1;out=output.parent/(output.name+'_export_'+stamp+'_'+str(suffix))
    analysis=out/'analysis';analysis.mkdir(parents=True)
    summary=[];windows=[];cycles=[];outages=[];per_ue=[];degradation=[];timing=[]
    traces=set();initials=set();audited_slots=0
    policies={q['label']:q for q in p['policies']}
    for item in completed:
        q=policies[item['policy']];directory=(output/item['index']).parent
        a,index=checked_game(directory,deep=deep)
        traces.add(index['trace_sha256']);initials.add(canonical_hash(index['initial']))
        assert len(traces)==len(initials)==1
        assert index['horizon']==p['horizon']==10000 and index['scenario']==260001
        assert index['policy']==q['label'] and index['privileged']==q['privileged']
        if q.get('path'):
            assert index['weights_sha256']==q['weights_sha256'] and index['checkpoint_sha256']==q['sha256']
        if q['arm'] in ['C-N','5A-Recovery']:
            assert not ((a['status_service']==0)&(a['roles_service']==1)).any()
        tag=dict(method=q['display_name'],policy=q['label'],evaluation_seed=260001,training_seed=q['training_seed'])
        worker_compute=directory.parents[2]/'worker_compute.json'
        computed=compute_metrics(a,index,read_json(worker_compute) if worker_compute.exists() else None)
        timing.append(dict(**tag,privileged=q['privileged'],**computed))
        write_slot_timing(analysis,q['label'],a,tag)
        cc=[dict(**tag,**r) for r in cycle_rows(a,index,q['label'])]
        cycles+=cc;outages+=outage_rows(a['rates'],tag)
        rejoined=[c for c in cc if c['rejoin_active_slot'] is not None]
        served=[c for c in rejoined if c['service_latency_slots'] is not None]
        lat=[c['service_latency_slots'] for c in served]
        scalar_metrics={k:v for k,v in index['metrics'].items() if not isinstance(v,(dict,list))}
        row=dict(**tag,training_budget=q.get('training_budget'),checkpoint_step=q.get('step'),horizon=10000,
                 **scalar_metrics,zero_slot_fraction=index['metrics']['zero_reward_slots']/10000,
                 runtime_s=index['runtime_s'],preview_calls=index['preview_calls'],
                 decision_median_ms=index['decision_median_ms'],decision_p95_ms=index['decision_p95_ms'],privileged=q['privileged'],
                 cycles_started=len(cc),cycles_rejoined=len(rejoined),cycles_served_again=len(served),
                 latency_median_observed=statistics.median(lat) if lat else None,
                 next_return_without_service=sum(c['censor_reason']=='next_return' for c in rejoined),
                 horizon_censored_after_rejoin=sum(c['censor_reason']=='horizon' for c in rejoined),
                 cycles_not_rejoined=sum(c['rejoin_active_slot'] is None for c in cc),
                 training_seed_replications=1 if q['training_seed'] is not None else None,
                 independent_evaluation_seeds=1,utility_ci95_low=None,utility_ci95_high=None)
        summary.append(row)
        for start in range(0,10000,1000):
            m=metric(a,start,start+1000)
            windows.append(dict(**tag,first_slot=start+1,last_slot=start+1000,
                                **{k:v for k,v in m.items() if not isinstance(v,(list,dict))},zero_slot_fraction=m['zero_reward_slots']/1000))
        first=metric(a,0,1000);last=metric(a,9000,10000);half1=metric(a,0,5000);half2=metric(a,5000,10000)
        degradation.append(dict(**tag,first1000_utility=first['utility'],last1000_utility=last['utility'],
            last_minus_first_utility=last['utility']-first['utility'],
            last_over_first_utility=last['utility']/first['utility'] if first['utility']>0 else None,
            first_half_utility=half1['utility'],second_half_utility=half2['utility'],
            first1000_zero_slots=first['zero_reward_slots'],last1000_zero_slots=last['zero_reward_slots'],
            longest_outage_slots=index['metrics']['longest_zero_streak'],terminal_outage_slots=index['metrics']['terminal_zero_streak']))
        for uid,rate in enumerate(index['metrics']['per_ue_mean_rates_mbps']):
            rr=a['rates'][:,uid]
            per_ue.append(dict(**tag,ue=uid,mean_rate_mbps=rate,positive_service_slot_fraction=float((rr>1e-12).mean()),qos_slot_fraction=float((rr>=4-1e-9).mean())))
        audited_slots+=10000
        print(json.dumps(dict(phase='offline_audit',method=q['display_name'],audited_slots=audited_slots,deep_planner=deep)),flush=True)
    complete=len(summary)==7
    status=read_json(output/'status.json')
    audit=dict(status='passed',complete=complete,completed_methods=len(summary),expected_methods=7,
               audited_committed_slots=audited_slots,planned_slots=70000,training_updates=0,
               deep_planner_recomputed=deep,identical_ue_trace_and_initial_state=bool(summary),
               independent_evaluation_seeds=1 if complete else 0,training_seed_replications=1,
               confidence_interval='Not estimated: one independent scenario seed.',progress=progress)
    write_rows(analysis,'summary',summary);write_rows(analysis,'windows1000',windows)
    write_rows(analysis,'cycles',cycles);write_rows(analysis,'outages',outages)
    write_rows(analysis,'per_ue',per_ue);write_rows(analysis,'horizon_degradation',degradation)
    write_rows(analysis,'runtime_and_previews',timing)
    write_rows(analysis,'compute_time',timing)
    paired=[]
    if complete:
        cr=next(r for r in summary if r['method']=='C-R');cn=next(r for r in summary if r['method']=='C-N')
        paired=[dict(evaluation_seed=260001,metric=k,CR=cr[k],CN=cn[k],CR_minus_CN=cr[k]-cn[k]) for k in
                ['utility','throughput_mbps','mean_qos_ue','mean_positive_service_ue','mean_slot_jain','zero_reward_slots','longest_zero_streak']]
    write_rows(analysis,'paired_CR_CN',paired)
    atomic_json(analysis/'audit_receipt.json',audit)
    for name in ['protocol.json','status.json','source_identity.json','MATCHED_EVALUATION_RECEIPT.json','COMPUTE_ENVIRONMENT.json','compute_events.jsonl']:
        if (output/name).exists():shutil.copy2(output/name,analysis/name)
    save_plots(analysis,summary,windows,cycles)
    plot_compute(analysis,timing)
    lines=['# Benchmark bảy phương pháp — best trong budget 120k','',
           f"Đã hoàn tất {len(summary)}/7 phương pháp; kiểm độc lập {audited_slots:,}/70.000 slot đã commit từ raw. Trạng thái runner: {status['status']}.",
           'C-R: checkpoint 114k; C-N: checkpoint 72k; cùng training seed 551, cùng budget 120k. Không train hoặc chọn lại checkpoint theo kết quả benchmark.',
           'Seed kịch bản đánh giá 260001, horizon 10.000 slot; các phương pháp dùng cùng UE trace và cùng trạng thái khởi tạo.',
           'Chỉ có một seed kịch bản; không ước lượng CI95%, không xem slot hoặc chu kỳ sạc là mẫu độc lập.', '',
           '| Phương pháp | Utility | Mbps | UE rate dương | UE đủ 4 Mbps | Jain từng slot | Slot mất dịch vụ | Outage dài nhất |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['method']} | {r['utility']:.6f} | {r['throughput_mbps']:.2f} | {r['mean_positive_service_ue']:.2f} | {r['mean_qos_ue']:.2f} | {r['mean_slot_jain']:.4f} | {r['zero_reward_slots']} | {r['longest_zero_streak']} |")
    if not complete:lines+=['','Kết quả chưa đủ bảy phương pháp; chưa dùng làm bảng xếp hạng chung.']
    lines+=['','## Chi phí tính toán','',
            '| Phương pháp | Wall phút | CPU giây của worker | Mean quyết định ms | P95 ms | Preview |',
            '|---|---:|---:|---:|---:|---:|']
    for t in timing:
        cpu='—' if t['worker_cpu_s'] is None else f"{t['worker_cpu_s']:.2f}"
        lines.append(f"| {t['method']} | {t['rollout_wall_minutes']:.2f} | {cpu} | {t['decision_mean_ms']:.3f} | {t['decision_p95_ms']:.3f} | {t['preview_calls']} |")
    lines+=['','compute_time.csv/JSON ghi wall/CPU time, tổng latency quyết định, mean/median/P95/P99/max và số preview. decision_timing_slots/*.csv.gz ghi latency và preview từng slot.',
            'other_rollout_wall_s = rollout wall trừ tổng decision time; phần dư gồm env, telemetry và I/O, không phải thời gian env thuần. Không có phép đo riêng thời gian từng preview.',
            'CPU time đo riêng tiến trình worker quanh core runner (gồm kiểm source, nạp model, rollout và output); không cộng thời gian chờ do tiến trình khác. Wall time chịu ảnh hưởng chạy song song. COMPUTE_ENVIRONMENT.json và compute_events.jsonl ghi máy chạy, quota CPU, số worker và các mốc thực thi.',
            '', '## Độ bền theo horizon','',
            'horizon_degradation.csv so 1.000 slot đầu/cuối và hai nửa horizon. windows1000.csv cùng horizon_metrics.png/PDF giữ cả các giai đoạn phục hồi. outages.csv liệt kê từng đoạn mất toàn bộ dịch vụ, gồm outage chạm cuối horizon.',
            'Không gán nhãn “sụp” chỉ từ một chỉ số trung bình: đối chiếu utility cuối, zero-service fraction, outage dài nhất và active/return/charging trong raw.',
            '', '## Sạc và Relay','',
            'cycles.csv/JSON giữ từng chu kỳ, rejoin, phục vụ lại, thời gian chờ và censoring. next_return_without_service là đã quay lại Return mà chưa phục vụ; horizon_censored_after_rejoin là hết thời gian quan sát sau rejoin.',
            'CDF hậu sạc chỉ có điều kiện trên các chu kỳ đã quan sát phục vụ lại; không phải tỷ lệ thành công của mọi cycle hoặc survival estimate đã điều chỉnh censoring.',
            'summary.csv giữ Relay rỗng theo active Relay UAV-slot và traffic được chuyển qua Relay. Random được phép dùng Relay; chỉ đổi tên hiển thị. 3A2R và Dynamic Relay giữ cơ chế Recovery gốc.',
            '', '## Cách đọc và thời gian chạy','',
            'Jain từng slot khác Jain trên rate trung bình cả horizon. Throughput = utility × 360, không phải bằng chứng độc lập.',
            'Heuristic dùng thông tin hiện tại chính xác và simulator preview một bước; không tương đương thông tin hoặc chi phí tính toán với policy học. runtime_and_previews.csv tách preview khỏi 70.000 slot thực thi.',
            'Thời gian từng phương pháp đo trên cùng máy, có thể có tranh chấp tài nguyên khi --workers > 1. Dùng --workers 1 cho phép đo thời gian ít chịu ảnh hưởng chạy song song.',
            f'Deep planner audit: {deep}. False nghĩa là chưa dựng lại độc lập toàn bộ Return/BFS offline; không tắt planner của env. Kiểm hash, continuity, mask, energy, collision và tính lại metric vẫn được thực hiện.',
            '', '## File bàn giao','',
            'benchmark7_review.zip: báo cáo, CSV/JSON và hình PNG/PDF. benchmark7_raw.zip: protocol, weights, UE traces, raw slots, telemetry, ledger và commits để kiểm lại; không có replay/optimizer huấn luyện.',
            'Raw có thể lớn; chỉ đưa source và hai inference weights vào repo, giữ kết quả ngoài git.']
    (analysis/'REPORT_VI.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    atomic_json(analysis/'SHA256SUMS.json',{f.relative_to(analysis).as_posix():file_hash(f) for f in sorted(analysis.rglob('*')) if f.is_file()})
    pack(out/'benchmark7_review.zip',[(f,'analysis/'+f.relative_to(analysis).as_posix()) for f in sorted(analysis.rglob('*')) if f.is_file()])
    shutil.copytree(analysis,output/'analysis',dirs_exist_ok=True)
    full_archive(output,out/'benchmark7_raw.zip')
    atomic_json(out/'EXPORT_INDEX.json',dict(complete=complete,report='analysis/REPORT_VI.md',audit_receipt=audit,
                files=[dict(path=f.name,sha256=file_hash(f),bytes=f.stat().st_size) for f in sorted(out.iterdir()) if f.is_file()]))
    return out
