"""Report the extra H heuristic, optionally paired with frozen six-arm reviews."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
LABELS=['CN','CR','CH','HO','RANDOM','DYNAMIC','HEURISTIC_HO']
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def base_identity(p):
    if p.get('label')=='HEURISTIC_HO':
        if p.get('version')!='bs5_heuristic_ho_benchmark_1':raise ValueError('Unknown H heuristic version')
        base=p['base_common']
        if digest(base)!=p['base_experiment_fingerprint']:raise ValueError('Invalid base identity')
        q={k:v for k,v in p.items() if k!='experiment_fingerprint'}
        if digest(q)!=p['experiment_fingerprint']:raise ValueError('Invalid extension fingerprint')
        for key in ['phase','diagnostic','horizon','target_seeds','runtime','reward_weights','source_sha256','checkpoint_registry','network_sample_every','counterfactual_bandwidth_probes']:
            if p[key]!=base[key]:raise ValueError('Extension modified shared contract: '+key)
        if p.get('arm')!='H-only' or not p['extension_source_sha256']:raise ValueError('Invalid all-H extension')
        return base
    if p.get('version')!='bs5_multihop_benchmark6_1' or p.get('label') not in LABELS[:6]:raise ValueError('Unknown frozen benchmark protocol')
    base={k:v for k,v in p.items() if k not in ['label','arm','experiment_fingerprint']}
    if digest(base)!=p['experiment_fingerprint']:raise ValueError('Invalid frozen benchmark identity')
    return base

def validate_protocols(protocols):
    bases=[base_identity(p) for p in protocols.values()]
    if bases and any(p!=bases[0] for p in bases[1:]):raise ValueError('Different base science/runtime/phase: do not merge')
    return bases[0] if bases else None

def generate(root):
    root=Path(root);summaries=[];protocols={}
    for f in sorted(root.glob('*/seed*/summary.json')):
        s=json.loads(f.read_text());p=json.loads((f.parent.parent/'protocol.json').read_text());label=p['label']
        if label in protocols and protocols[label]!=p:raise ValueError('Conflicting label protocol')
        if s['label']!=label or s['experiment_fingerprint']!=p['experiment_fingerprint'] or s['phase']!=p['phase'] or s['diagnostic']!=p['diagnostic'] or s['target_slots']!=p['horizon']:raise ValueError('Summary/protocol mismatch')
        if bool(s['complete'])!=(s['slots']==p['horizon']) or s['seed'] not in p['target_seeds']:raise ValueError('Invalid completeness or seed')
        protocols[label]=p;summaries.append(s)
    if not summaries:return None
    common=validate_protocols(protocols);seen=set();hashes={}
    for s in summaries:
        key=s['label'],s['seed']
        if key in seen:raise ValueError('Duplicate label/seed')
        seen.add(key);hkey=s['seed'],s['slots']
        if hkey in hashes and hashes[hkey]!=s['ue_trace_sha256']:raise ValueError('UE trace mismatch')
        hashes[hkey]=s['ue_trace_sha256']
    labels=[a for a in LABELS if a in protocols];complete=[s for s in summaries if s['complete']];partial=[s for s in summaries if not s['complete']]
    sets={a:sorted(s['seed'] for s in complete if s['label']==a) for a in labels}
    shared=sorted(set.intersection(*(set(v) for v in sets.values()))) if sets else []
    all_expected=all(sets[a]==common['target_seeds'] for a in labels)
    full7=labels==LABELS and all_expected
    means={};paired_means={}
    for a in labels:
        for target,rs in [(means,[s for s in complete if s['label']==a]),(paired_means,[s for s in complete if s['label']==a and s['seed'] in shared])]:
            if not rs:continue
            target[a]={k:float(np.mean([s[k] for s in rs])) for k in rs[0] if all(isinstance(s.get(k),(int,float)) and not isinstance(s[k],bool) for s in rs)}
            target[a]['throughput_sd_mbps']=float(np.std([s['throughput_mbps'] for s in rs],ddof=1)) if len(rs)>1 else None
            target[a]['n']=len(rs)
    out=dict(version='heuristic_ho_report_1',phase=common['phase'],diagnostic=common['diagnostic'],base_experiment_fingerprint=digest(common),all_present_labels_complete=all_expected,benchmark7_complete=full7,
        complete_seed_sets=sets,shared_complete_seeds=shared,partial_runs=[dict(label=s['label'],seed=s['seed'],slots=s['slots']) for s in partial],metrics_complete_runs=means,metrics_paired_common_seeds=paired_means)
    (root/'comparison_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    with (root/'all_seed_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for s in summaries for k in s)));w.writeheader();w.writerows(summaries)
    text='# Heuristic H-only — baseline bổ sung có kiểm tra tương thích\n\n'
    if common['diagnostic']:text+='**CHỈ KIỂM TRA CHỨC NĂNG; KHÔNG DÙNG LÀM KẾT QUẢ TEST.**\n\n'
    text+=f"Đã đủ các seed yêu cầu cho nhánh hiện có: {all_expected}. Đủ cả 7 nhánh: {full7}. Seed hoàn chỉnh chung: {shared}. Lượt dở không được gộp vào mean: {out['partial_runs']}.\n\n"
    text+='HEURISTIC_HO: mọi UAV Active luôn H; chỉ chọn một trong 5 di chuyển bằng TCF ước lượng trên observation có trễ. Cùng synthetic UE belief, khám phá và navigation zero-score như Dynamic; waypoint trong chuỗi cũng là H. Không có UE thật/trạng thái hiện tại của peer hoặc rollout tương lai trong điều khiển. Giữ full Access/BH, half-duplex, cap25 và pin H đầy đủ; không train. Đây là baseline bổ sung sau khi đã xem kết quả 6 nhánh, không retune theo từng seed test.\n\n'
    text+='| Nhánh | Seed hoàn chỉnh | Mbps | SD giữa seed | UE ≥1/slot | UE ≈4/slot | UE từng ≥1/5 phút | Pin/Gbit |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    display=paired_means if len(labels)>1 else means
    for a,m in display.items():text+=f"| {a} | {m['n']} | {m['throughput_mbps']:.3f} | {m['throughput_sd_mbps'] if m['throughput_sd_mbps'] is not None else 'n/a'} | {m['service_coverage_ues']:.3f} | {m['qos_ues']:.3f} | {m['window_5min_unique_ge1_mean']:.3f} | {m['battery_points_per_gbit']:.3f} |\n"
    text+='\nNếu hiện nhiều nhánh, bảng và hình dùng CHỈ các seed hoàn chỉnh chung. JSON còn giữ mean của từng nhánh hiện có; không so các tập seed khác nhau. Với hai phần 5+5 seed, báo cáo từng phần chưa phải kết quả 10 seed cho tới khi gộp.\n\n'
    text+='## Giải thích theo log\n\n'
    for a,m in display.items():
        text+=f"- {a}: phủ hình học {m['covered_ues']:.2f} UE/slot → association {m['associated_ues']:.2f} → có BH {m['reachable_ues']:.2f} → ≥1 Mbps {m['service_coverage_ues']:.2f}. Mất Mbps ngoài Access {m['loss_outside_access_mbps']:.2f}; association/cap {m['loss_association_mbps']:.2f}; Access radio {m['loss_access_radio_mbps']:.2f}; không đường BH {m['loss_bh_disconnected_mbps']:.2f}; lịch/dung lượng BH {m['loss_bh_schedule_capacity_mbps']:.2f}. Return+Charging {100*(1-m['active_uav_fraction']):.2f}% UAV-slot; sạc xong {m['charging_rejoins']:.1f}/seed; đã phục vụ lại {m['served_again_cycles']:.1f}/seed.\n"
    text+='\nWaterfall cộng về 360 Mbps, là kế toán tại vị trí/association/routes hiện có. Probe BH/Access×2 là chẩn đoán cùng snapshot, không phải gain rollout. Unique UE/5 phút là đạt ngưỡng ít nhất một slot, không bảo đảm liên tục. Jain dài hạn cần đi cùng rate tuyệt đối và gap. Pin là điểm chuẩn hóa, không phải Joule. Served-again được logger tính khi own+forward >1 bit/s, không đồng nghĩa đã đạt 1 Mbps cho UE. Log thật chỉ được tạo sau hành động và không đưa ngược vào controller.\n'
    text+='\n## Tệp kiểm chứng\n\nMỗi seed có slots.csv, ue_summary.csv, trace.npz, network_samples.jsonl, cycles.json, live_events.jsonl, summary.json. REVIEW không chứa resume.pkl; RESUME giữ toàn bộ trạng thái môi trường, controller, reports, RNG và dữ liệu để tiếp tục chính xác. Protocol bổ sung có fingerprint riêng và tham chiếu nguyên vẹn common protocol của 6 nhánh; không sửa fingerprint cũ để giả vờ cùng một thuật toán.\n'
    (root/'REPORT_HEURISTIC_HO_VI.md').write_text(text)
    if display:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names=list(display);fig,ax=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
        for panel,key,title in zip(ax.flat,['throughput_mbps','service_coverage_ues','qos_ues','battery_points_per_gbit'],['UE throughput (Mbps)','UEs >=1 Mbps / slot','UEs at 4 Mbps / slot','Battery points / Gbit']):
            panel.bar(names,[display[a][key] for a in names],color=['#7957ab' if a=='HEURISTIC_HO' else '#3c7ca1' for a in names]);panel.set_title(title);panel.tick_params(axis='x',rotation=25);panel.grid(axis='y',alpha=.15)
        fig.suptitle('Complete common scenarios only; diagnostic runs are not thesis test results' if common['diagnostic'] else 'Additional H-only heuristic — paired completed scenarios')
        fig.savefig(root/'heuristic_ho_comparison.png',dpi=160);plt.close(fig)
    return root/'REPORT_HEURISTIC_HO_VI.md'
