"""Generate metrics, figures and a Vietnamese evidence-based explanation."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])


def generate(root):
    root=Path(root);summaries=[];seed_dirs=[];protocols={}
    for p in sorted(root.rglob('summary.json')):
        if not p.parent.name.startswith('seed'):continue
        s=json.loads(p.read_text());summaries.append(s);seed_dirs.append(p.parent)
        protocol=json.loads((p.parent.parent/'protocol.json').read_text())
        if protocol['label'] in protocols and protocols[protocol['label']]!=protocol:
            raise ValueError('Conflicting protocols for the same algorithm')
        protocols[protocol['label']]=protocol
    if not summaries:return None
    # No mixing diagnostic/pilot/test/horizon or modified science in a report.
    identities={(s['phase'],s['target_slots'],s['diagnostic'],s['experiment_fingerprint']) for s in summaries}
    if len(identities)!=1:raise ValueError('Cannot combine different experiment phases or science')
    labels=[a for a in ['CN','CR','CH','HO','RANDOM','DYNAMIC'] if any(s['label']==a for s in summaries)]
    means={}
    for a in labels:
        rs=[s for s in summaries if s['label']==a]
        means[a]={k:float(np.mean([r[k] for r in rs])) for k in rs[0]
                  if all(isinstance(r.get(k),(int,float)) and r[k] is not None for r in rs)}
        gbit=sum(r['throughput_mbps']*r['slots']*5/1000 for r in rs)
        means[a]['battery_points_per_gbit_pooled']=sum(r['total_depleted_points'] for r in rs)/gbit if gbit>1e-9 else None
        means[a]['throughput_scenario_std_mbps']=float(np.std([r['throughput_mbps'] for r in rs],ddof=1)) if len(rs)>1 else None
    hashes={}
    for s in summaries:
        key=(s['seed'],s['slots'])
        if key in hashes and hashes[key]!=s['ue_trace_sha256']:raise ValueError('UE trace mismatch between algorithms')
        hashes[key]=s['ue_trace_sha256']
    seen=set()
    for s in summaries:
        k=(s['label'],s['seed'])
        if k in seen:raise ValueError('Duplicate label/seed result')
        seen.add(k)
    complete=all(s['slots']==s['target_slots'] for s in summaries)
    seed_sets={a:sorted(s['seed'] for s in summaries if s['label']==a) for a in labels}
    paired=all(v==seed_sets[labels[0]] for v in seed_sets.values())
    expected=next(iter(protocols.values()))['target_seeds']
    benchmark_complete=(complete and len(labels)==6 and paired and seed_sets[labels[0]]==expected)
    (root/'comparison_summary.json').write_text(json.dumps(dict(complete=complete,benchmark_complete=benchmark_complete,paired_seed_sets=paired,seed_sets=seed_sets,metrics=means),indent=2)+'\n')
    fields=list(dict.fromkeys(k for s in summaries for k in s))
    with (root/'all_seed_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summaries)
    headers=['Nhánh','Seed đã có','Mbps','UE ≥1/slot','UE ≈4/slot','UE từng nhận','Jain dữ liệu','Pin/Gbit']
    data=[]
    for a in labels:
        m=means[a];e=m['battery_points_per_gbit_pooled']
        data.append([a,len(seed_sets[a]),f"{m['throughput_mbps']:.2f}",f"{m['service_coverage_ues']:.2f}",f"{m['qos_ues']:.2f}",f"{m['unique_served_ues']:.1f}/90",f"{m['jain_total_data']:.4f}",f'{e:.2f}' if e is not None else 'n/a'])
    text='# Benchmark đa hop sáu nhánh — kết quả và giải thích\n\n'
    if summaries[0]['diagnostic']:
        text+='**CHỈ KIỂM TRA CHỨC NĂNG 96 SLOT — KHÔNG PHẢI KẾT QUẢ BENCHMARK LUẬN VĂN.**\n\n'
    text+=f"Trạng thái: {'ĐỦ 6 NHÁNH VÀ ĐỦ SEED THEO PROTOCOL' if benchmark_complete else 'CHƯA ĐỦ BỘ BENCHMARK'}; đủ horizon cho các tệp hiện có: {complete}; cùng tập seed giữa nhánh: {paired}. Chỉ so sánh chính thức khi đủ cả sáu nhánh và cùng các seed yêu cầu trong protocol.\n\n"
    text+=table(headers,data)+'\n\n'
    text+='CN/CR/CH/HO dùng checkpoint đã đóng băng; RANDOM/DYNAMIC dùng Access/Relay, không train. Dynamic chỉ nhận observation actor có trễ, không truy cập UE thật; thông tin thật chỉ phục vụ log sau bước thực thi. Pilot/test/diagnostic không được gộp. Các seed test là biến thiên kịch bản, chưa phải các training seed độc lập.\n\n'
    if any(len(seed_sets[a])>1 for a in labels):
        text+=table(['Nhánh','Độ lệch chuẩn throughput giữa scenario seeds (Mbps)'],[[a,f"{means[a]['throughput_scenario_std_mbps']:.2f}" if means[a]['throughput_scenario_std_mbps'] is not None else 'Chỉ một seed'] for a in labels])+'\n\n'
    text+='## Hiểu đúng phạm vi phục vụ\n\nUE ≥1 Mbps/slot là số nhận mức rate đó ĐỒNG THỜI, không phải số người duy nhất được phục vụ trong toàn episode. Unique UE có thể đạt 90/90 dù có gián đoạn dài. Vì vậy luôn đọc cùng dữ liệu tích lũy, rate trung bình và tỷ lệ thời gian dịch vụ.\n\n'
    cols=['Nhánh','UE đã nhận ≥1 MB','UE ≥1 lần/5 phút','UE mean ≥1 Mbps/5 phút','P05 mean rate (Mbps)','Min dữ liệu (MB)','Gap zero lớn nhất (phút)']
    data=[]
    for a in labels:
        m=means[a]
        data.append([a,f"{m['ues_received_1mb']:.1f}/90",f"{m.get('window_5min_unique_ge1_mean',float('nan')):.1f}",f"{m.get('window_5min_ues_mean_rate_ge1_mean',float('nan')):.1f}",f"{m['p05_ue_mean_rate_mbps']:.3f}",f"{m['min_ue_received_mb']:.2f}",f"{max(s['longest_zero_gap_slots_max'] for s in summaries if s['label']==a)*5/60:.2f}"])
    text+=table(cols,data)+'\n\n'
    text+='Cửa sổ 1/5 phút trượt chỉ được tính khi đủ độ dài. “UE ≥1 lần/5 phút” nghĩa là có ít nhất một slot đạt 1 Mbps trong cửa sổ đó, không phải duy trì 1 Mbps suốt 5 phút. Ngưỡng 1 MB là chẩn đoán để tránh “từng nhận” chỉ do một lượng dữ liệu rất nhỏ; không phải chuẩn QoS. Jain gần 1 mô tả sự đều nhau, không bảo đảm rate cao. Các gap là khoảng không có dịch vụ quan sát được, kể cả khoảng bị cắt ở đầu/cuối episode. Mô hình fluid theo slot không có hàng đợi packet; gap dịch vụ không phải độ trễ packet end-to-end.\n\n'
    text+='## Nút thắt throughput theo số liệu\n\nWaterfall cộng đúng về 360 Mbps nhu cầu: ngoài vùng Access + chưa association/cap + Access radio/chia sẻ + không có đường BH + dung lượng/lịch BH + dữ liệu giao cuối. Đây là phân rã tại geometry/association/route hiện tại, không phải phép đo nguyên nhân độc lập của một chính sách tối ưu.\n\n'
    losskeys=[('loss_outside_access_mbps','ngoài vùng Access'),('loss_association_mbps','association/giới hạn UE'),('loss_access_radio_mbps','radio/chia sẻ Access'),('loss_bh_disconnected_mbps','không có đường BH'),('loss_bh_schedule_capacity_mbps','dung lượng/lịch BH')]
    for a in labels:
        m=means[a];top=max(losskeys,key=lambda kv:m[kv[0]])
        text+=f"- **{a}:** phủ hình học {m['covered_ues']:.2f} UE/slot → association {m['associated_ues']:.2f} → có đường BH {m['reachable_ues']:.2f} → đạt ≥1 Mbps {m['service_coverage_ues']:.2f}. Access offer {m['access_offer_mbps']:.2f} Mbps, offer có đường về BS {m['reachable_offer_mbps']:.2f}, giao cuối {m['throughput_mbps']:.2f}. Phần mất lớn nhất trong accounting: **{top[1]}**, {m[top[0]]:.2f} Mbps. Không có đường BH làm mất {m['loss_bh_disconnected_mbps']:.2f}; dung lượng/lịch BH làm mất {m['loss_bh_schedule_capacity_mbps']:.2f} Mbps.\n"
    text+='\nCác probe tăng riêng BH/Access ×2 giữ vị trí, role, trạng thái và công suất; có thể tính lại route/association. Chúng chỉ là chẩn đoán radio thưa, không đổi cấu hình benchmark hay chứng minh tác động dài hạn của năng lượng.\n\n'
    for a in labels:
        probes=[]
        for d in seed_dirs:
            if d.parent.name!=a:continue
            p=d/'network_samples.jsonl'
            if p.is_file():probes.extend(json.loads(l) for l in p.read_text().splitlines() if l)
        probes=[p for p in probes if 'same_snapshot_bh2x_mbps' in p]
        if probes:
            actual=np.mean([p['actual_mbps'] for p in probes]);bh=np.mean([p['same_snapshot_bh2x_mbps'] for p in probes]);ac=np.mean([p['same_snapshot_access2x_mbps'] for p in probes])
            text+=f"- {a}, {len(probes)} snapshot: thực {actual:.2f} Mbps; BH×2 {bh:.2f}; Access×2 {ac:.2f}.\n"
    text+='\n## Pin, lịch sạc và khả năng phục vụ\n\n'
    text+=table(['Nhánh','Active (%)','Returning (%)','Charging (%)','Rejoin/seed','Đã phục vụ lại/seed','Pin tiêu hao/seed'],[[a,f"{means[a]['active_uav_fraction']*100:.1f}",f"{means[a]['return_uav_fraction']*100:.1f}",f"{means[a]['charging_uav_fraction']*100:.1f}",f"{means[a]['charging_rejoins']:.1f}",f"{means[a]['served_again_cycles']:.1f}",f"{means[a]['total_depleted_points']:.2f}"] for a in labels])+'\n\n'
    text+='Mô hình mỗi UAV Active: radio A0,2/R0,1/H0,3; thêm hover0,1 và move extra0,1 điểm/slot5s. H trả đủ radio ngay cả khi chỉ dùng một chức năng. Tổng pin tiêu hao khác pin/Gbit; không phải Joules. Chu kỳ dang dở cuối trace không bị tính thất bại.\n\n'
    if 'CR' in means and paired:
        cr=means['CR']
        for a in ['CH','HO']:
            if a not in means:continue
            m=means[a];extra=100*(m['total_depleted_points']/cr['total_depleted_points']-1)
            downtime=(m['return_uav_fraction']+m['charging_uav_fraction']-cr['return_uav_fraction']-cr['charging_uav_fraction'])*100
            both=100*m['hybrid_both_count']/m['active_hybrid'] if m['active_hybrid']>0 else 0
            text+=f"- {a} so với CR: tổng pin/trace {extra:+.1f}%; phần thời gian Returning+Charging {downtime:+.2f} điểm phần trăm; H vừa Access vừa forwarding {both:.1f}% H UAV-slot. "
            text+=('Có thêm thời gian không phục vụ do Return/Charging trong dữ liệu. ' if downtime>0 else 'Dữ liệu này không cho thấy Return/Charging chiếm thời gian nhiều hơn CR. ')
            text+='Đối chiếu waterfall và probe BH trước khi quy throughput thấp hoàn toàn cho pin. Đây là bằng chứng mô tả; chưa có rollout can thiệp riêng chi phí H để xác định tác động nhân quả.\n'
    text+='\n## Tệp log để kiểm tra lại\n\nMỗi seed: slots.csv (live metrics + waterfall + energy + actor runtime), ue_summary.csv, trace.npz (rate/association/path/coverage/trạng thái/pin từng slot), network_samples.jsonl (link/routes/schedules và probe), cycles.json, live_events.jsonl và summary.json. UE mobility hash phải khớp giữa các nhánh cùng seed/horizon. Điều khiển không đọc các log chẩn đoán.\n'
    (root/'REPORT_VI.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#a2a9b4','#3b72b8','#eea052','#a8cbc7','#bd7284','#4d8d76']
    fig,ax=plt.subplots(figsize=(10,4.5),constrained_layout=True);bottom=np.zeros(len(labels))
    for color,(key,label) in zip(colors,[*losskeys,('delivered_mbps','delivered')]):
        vals=np.asarray([means[a][key] for a in labels]);ax.bar(labels,vals,bottom=bottom,label=label,color=color);bottom+=vals
    ax.set(ylabel='Mbps',title='Demand accounting at executed geometry (360 Mbps total)');ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1,1))
    fig.savefig(root/'01_throughput_waterfall.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for j,(key,title) in enumerate([('mean_rate_mbps','UE mean rate (Mbps)'),('ge1_time_fraction','Fraction of slots >=1 Mbps')]):
        for a in labels:
            vals=[]
            for d in seed_dirs:
                if d.parent.name==a:
                    with (d/'ue_summary.csv').open() as f:vals.extend(float(r[key]) for r in csv.DictReader(f))
            vals=np.sort(vals);axs[j].plot(vals,np.arange(1,len(vals)+1)/len(vals),label=a)
        axs[j].set(xlabel=title,ylabel='CDF across UE x scenario seed');axs[j].grid(alpha=.2)
    axs[0].legend();fig.savefig(root/'02_ue_service_distribution.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.5),constrained_layout=True)
    keys=[('service_coverage_ues','Simultaneous >=1 Mbps / slot'),
          ('window_5min_unique_ge1_mean','Distinct >=1 Mbps at least once / 5 min'),
          ('unique_ge1_ues','Distinct >=1 Mbps at least once / episode'),
          ('ues_received_1mb','Received >=1 MB / episode')]
    x=np.arange(len(labels));width=.2
    for j,(key,label) in enumerate(keys):
        ax.bar(x+(j-1.5)*width,[means[a].get(key,float('nan')) for a in labels],width,label=label)
    ax.set(xticks=x,xticklabels=labels,ylabel='UE count out of 90',ylim=(0,95),
           title='Instantaneous service and service over time are different metrics')
    ax.axhline(90,color='grey',lw=.7,ls='--');ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1,1))
    fig.savefig(root/'03_instantaneous_vs_temporal_service.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for a in labels:
        traces=[]
        for d in seed_dirs:
            if d.parent.name==a:
                with np.load(d/'trace.npz') as t:traces.append(t['rates'])
        n=min(len(t) for t in traces);time_min=(np.arange(n)+1)*5/60
        unique=np.mean([np.maximum.accumulate(t[:n]>=1-1e-9,axis=0).sum(axis=1) for t in traces],axis=0)
        received=np.mean([((np.cumsum(t[:n],axis=0)*5/8)>=1).sum(axis=1) for t in traces],axis=0)
        axs[0].plot(time_min,unique,label=a);axs[1].plot(time_min,received,label=a)
    for ax,title in zip(axs,['Distinct UEs ever >=1 Mbps','UEs with cumulative data >=1 MB']):
        ax.set(xlabel='Simulation time (minutes)',ylabel='UE count out of 90',title=title,ylim=(0,95));ax.grid(alpha=.2)
    axs[0].legend();fig.savefig(root/'04_cumulative_user_reach.png',dpi=150);plt.close(fig)
    return root/'REPORT_VI.md'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True)
    print(generate(p.parse_args().runs))
