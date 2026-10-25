"""Standalone audit: stdlib only; never loads policies, Torch, NumPy or an env."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

try:
    from bs5_bundle import atomic_json, canonical_hash, file_hash, load_trace, read_json, validate_rates, write_csv
    from analyze_bs5_curriculum_bundle import episode_metrics
    from analyze_bs5_curriculum_extend import gateway_auditor
    from bs5_continuous_common import busy_relays, geometry_classifier, lifecycle
except ImportError:
    from scripts.analysis.bs5_bundle import atomic_json, canonical_hash, file_hash, load_trace, read_json, validate_rates, write_csv
    from scripts.analysis.analyze_bs5_curriculum_bundle import episode_metrics
    from scripts.analysis.analyze_bs5_curriculum_extend import gateway_auditor
    from scripts.analysis.bs5_continuous_common import busy_relays, geometry_classifier, lifecycle


def safe(root, rel):
    path = (root / rel).resolve()
    assert path.is_relative_to(root.resolve()) and not path.is_symlink(), f"unsafe path: {rel}"
    return path


def metrics(rows):
    m = episode_metrics(rows)
    statuses = [r["service"]["status"] for r in rows]
    full = [r for r in rows if all(s == 0 for s in r["service"]["status"])]
    m.update(mean_returning=statistics.fmean(sum(s==1 for s in ss) for ss in statuses),
        mean_charging=statistics.fmean(sum(s==2 for s in ss) for ss in statuses),
        fraction_two_unavailable=sum(sum(s!=0 for s in ss)>=2 for ss in statuses)/len(rows),
        all_active_slots=len(full), utility_all_active=statistics.fmean(r["reward"] for r in full) if full else None)
    # Count delivered UE traffic on a relay-routed Access, not physical link capacity.
    m["relay_routed_throughput_mbps"] = statistics.fmean(sum(
        r["service"]["network"]["ue_rate_mbps"][u]
        for a,rid in r["service"]["network"]["routes"].items() if rid is not None
        for u in r["service"]["network"]["association"].get(a,[])) for r in rows)
    reversals=blocked=0
    for i,r in enumerate(rows):
        blocked += sum(r["decision"]["status"][a]==0 and r["raw_actions"][a]//2!=4 and
            r["service"]["uav_positions"][a]==r["decision"]["positions"][a] for a in range(5))
        if i:
            prev=rows[i-1]
            reversals += sum(prev["decision"]["status"][a]==r["decision"]["status"][a]==0 and
                prev["decision"]["positions"][a]==r["service"]["uav_positions"][a] and
                r["decision"]["positions"][a]!=r["service"]["uav_positions"][a] for a in range(5))
    m.update(two_step_reversals=reversals, blocked_move_uav_slots=blocked)
    return m


def load_game(root, rel):
    index=read_json(safe(root,rel))
    rows,updates=[],[]
    for chunk in index["chunks"]:
        path=safe(root,chunk["path"])
        assert file_hash(path)==chunk["sha256"], f"chunk hash: {path}"
        data=load_trace(path)
        assert [r["slot"] for r in data["rows"]]==list(range(chunk["first"],chunk["last"]+1))
        assert len(data["rows"])==chunk["slots"]
        rows.extend(data["rows"]); updates.extend(data["updates"])
    return index,rows,updates


def equivalent(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        return a.keys()==b.keys() and all(equivalent(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):
        return len(a)==len(b) and all(equivalent(x,y) for x,y in zip(a,b))
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)
    return a==b


def validate_game(index,rows,p, *, evaluation):
    spec=index["spec"]; h=spec["horizon"]
    assert len(rows)==h and [r["slot"] for r in rows]==list(range(1,h+1)), "incomplete game"
    c=spec["condition"]; config=index["resolved_config"]
    assert config["horizon"]==h and config["allow_relay"] is True
    assert rows[0]["decision"]["positions"]==index["initial"]["positions"]
    assert rows[0]["decision"]["battery"]==index["initial"]["battery"]
    gateway=gateway_auditor(config)
    classify=geometry_classifier(config["grid_size"],config["cell_size_m"],
        config["access_radius_cells"],config["backhaul_radius_cells"],tuple(map(tuple,config["bs_xy_m"])))
    for i,r in enumerate(rows):
        assert r["arm"]=="C-R" and r["seed"]==p["training_seed"]
        assert r["scenario_seed"]==spec["scenario_seed"] and r["stage_id"]==c["stage"]
        assert r["task_condition"]==c and r["env_t_before"]==i and r["env_t_after"]==i+1
        assert not r["terminated"] and r["truncated"]==(i+1==h)
        terminal=bool(r["terminated"] or (r["truncated"] and c["stage"]!="S4"))
        assert r["learning_terminal"]==terminal and r["bootstrap_mask"]==int(not terminal)
        assert r["service"]["ue_positions"]==r["decision"]["ue_positions"]
        assert r["surviving_bs_ids"]==[0,1,2,3,4]
        assert tuple(r["relay_required_ue"])==classify(tuple(map(tuple,r["service"]["ue_positions"])))
        validate_rates(r)
        if c["stage"]=="S4": gateway(r)
        for key in ("action_mask","next_action_mask"):
            assert len(r[key])==5 and all(len(m)==10 and any(m) for m in r[key])
        assert len(r["obs"])==len(r["next_obs"])==5
        assert all(len(o)==116 for o in r["obs"]+r["next_obs"])
        assert len(r["state"])==len(r["next_state"])==111
        if evaluation:
            assert r["epsilon"]==0 and r["global_step"]==r["checkpoint_step"]
        else:
            assert r["global_step"]==spec["start"]+i+1
        if i:
            prev=rows[i-1]
            assert prev["post"]==r["decision"], "physical discontinuity/reset"
            assert prev["next_obs"]==r["obs"] and prev["next_state"]==r["state"], "actor input discontinuity"
            assert prev["next_action_mask"]==r["action_mask"]
            assert prev["env_rng_after"]==r["env_rng_before"], "RNG discontinuity"
    generated=metrics(rows)
    assert equivalent(index["metrics"],generated), "raw/stored metrics mismatch"
    return generated


def table(output,name,rows):
    atomic_json(output/(name+".json"),rows)
    write_csv(output/(name+".csv"),rows,sorted({k for r in rows for k in r}) if rows else ["empty"])


def analyze(root,output, *, require_complete=False, manifest_required=True):
    root,output=Path(root).resolve(),Path(output).resolve()
    assert root!=output and not output.is_relative_to(root), "analysis output must be outside evidence root"
    p=read_json(root/"protocol.json")
    assert p["arms"]==["C-R"] and p["training_seeds"]==[p["training_seed"]]
    commit=read_json(root/"commit.json")
    budget=read_json(root/"budget_ledger.json")
    assert budget["protocol_sha256"]==canonical_hash(p)
    identity=read_json(root/"source_identity.json")
    assert identity["sha256"]==canonical_hash(identity["files"])==p["source_sha256"]
    import zipfile
    with zipfile.ZipFile(root/"source_snapshot.zip") as archive:
        import hashlib
        for rel,digest in identity["files"].items():
            assert hashlib.sha256(archive.read("uav_thesis/"+rel)).hexdigest()==digest
    manifest=root/"SHA256SUMS.json"
    if manifest_required:
        assert manifest.exists(), "missing export manifest"
    if manifest.exists():
        for rel,digest in read_json(manifest).items():
            assert file_hash(safe(root,rel))==digest, f"manifest hash: {rel}"
    repro=root/"REPRO_SHA256SUMS.json"
    checked_weights=0
    if repro.exists():
        for rel,digest in read_json(repro).items():
            f=safe(root,rel)
            if f.exists():
                assert file_hash(f)==digest, f"repro hash: {rel}"
                checked_weights+=1
    complete=read_json(root/"status.json")["status"]=="complete"
    if require_complete:
        assert complete and commit["steps"]==p["total_steps"], "incomplete run"
        assert [e["step"] for e in commit["evaluations"]]==p["evaluation_steps"], "missing validation milestone"
    assert len(commit["games"])==commit["episodes"]
    game_rows,windows,cycles,uavs,per_ue,learning=[],[],[],[],[],[]
    total_train=total_eval=update_count=0
    for eid,rel in enumerate(commit["games"]):
        index,rows,updates=load_game(root,rel)
        assert index["spec"]==p["game_schedule"][eid]
        generated=validate_game(index,rows,p,evaluation=False)
        total_train+=len(rows); update_count+=len(updates)
        item=dict(kind="train",episode=eid,step=index["spec"]["start"]+len(rows),**generated)
        game_rows.append(item)
        if index["spec"]["condition"]["stage"]=="S4":
            cs,us=lifecycle(rows,index["initial"],run="C-R_seed"+str(p["training_seed"]),scenario=index["spec"]["scenario_seed"])
            cycles.extend(dict(kind="train",episode=eid,**c) for c in cs)
            uavs.extend(dict(kind="train",episode=eid,**u) for u in us)
        for j in range(0,len(rows),150):
            windows.append(dict(kind="train",episode=eid,first=j+1,last=min(j+150,len(rows)),**metrics(rows[j:j+150])))
    assert total_train==commit["steps"] and update_count==commit["updates"], "training budget/update mismatch"
    for ev in commit["evaluations"]:
        assert len(ev["games"])==len(p["validation_pool"])
        utility=[]
        for scenario,rel in zip(p["validation_pool"],ev["games"]):
            index,rows,updates=load_game(root,rel)
            assert not updates and index["spec"]["scenario_seed"]==scenario
            assert index["spec"]["horizon"]==p["full_horizon"]
            assert all(r["checkpoint_step"]==ev["step"] for r in rows)
            generated=validate_game(index,rows,p,evaluation=True)
            total_eval+=len(rows); utility.append(generated["utility"])
            game_rows.append(dict(kind="validation",step=ev["step"],scenario=scenario,**generated))
            for j in range(0,len(rows),150):
                windows.append(dict(kind="validation",step=ev["step"],scenario=scenario,
                    first=j+1,last=min(j+150,len(rows)),**metrics(rows[j:j+150])))
            cs,us=lifecycle(rows,index["initial"],run="C-R_seed"+str(p["training_seed"]),scenario=scenario)
            cycles.extend(dict(kind="validation",step=ev["step"],**c) for c in cs)
            uavs.extend(dict(kind="validation",step=ev["step"],**u) for u in us)
            for ue in range(90):
                rates=[r["service"]["network"]["ue_rate_mbps"][ue] for r in rows]
                streak=maximum=0
                for rate in rates:
                    streak=streak+1 if rate<=1e-12 else 0; maximum=max(maximum,streak)
                per_ue.append(dict(step=ev["step"],scenario=scenario,ue=ue,
                    mean_rate_mbps=statistics.fmean(rates),zero_rate_slots=sum(r<=1e-12 for r in rates),max_zero_streak=maximum))
        assert math.isclose(ev["utility"],statistics.fmean(utility),rel_tol=0,abs_tol=1e-10)
        learning.append(dict(step=ev["step"],utility=statistics.fmean(utility)))
    for phase,total in (("train",total_train),("eval",total_eval)):
        b=budget["phases"][phase]
        assert total<=b["observed"]<=b["charged"]<=b["cap"], "compute ledger inconsistency"
        if require_complete: assert total==b["observed"]==b["charged"], "uncommitted/uncertain compute"
    if require_complete:
        assert total_train==p["planned_training_steps"] and total_eval==p["planned_validation_steps"]
        assert len(commit["games"])==len(p["game_schedule"])
    best=max(learning,key=lambda e:(e["utility"],-e["step"]))
    assert best["step"]==commit["best_step"] and math.isclose(best["utility"],commit["best_utility"],abs_tol=1e-10)
    selected=read_json(root/"checkpoint_selection.json")
    assert selected["best_step"]==best["step"] and selected["curve"]==commit["evaluations"]
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in (("games",game_rows),("windows",windows),("cycles",cycles),("uavs",uavs),("per_ue",per_ue),("learning_curve",learning)):
        table(output,name,rows)
    selected_cycles=[c for c in cycles if c["kind"]=="validation" and c["step"]==best["step"]]
    lifecycle_summary={}
    for limit in (30,60,90):
        eligible=[c for c in selected_cycles if c["rejoin_active_slot"] is not None and
                  c["time_at_risk_slots"]>=limit]
        # Fixed follow-up denominator; early next-Return is a competing event, reported separately.
        lifecycle_summary[str(limit)]=dict(eligible=len(eligible),useful_within=sum(
            c["service_latency_slots"] is not None and c["service_latency_slots"]<=limit for c in eligible),
            short_followup_rejoined=sum(c["rejoin_active_slot"] is not None and c["time_at_risk_slots"]<limit for c in selected_cycles))
    summary=dict(status="complete" if complete else "partial",diagnostic=p["diagnostic"],
        training_steps=total_train,validation_steps=total_eval,updates=update_count,
        best=best,last=learning[-1],tail_three_mean=statistics.fmean(e["utility"] for e in learning[-3:]),
        peak_to_last_drop=best["utility"]-learning[-1]["utility"],
        best_cycles_started=len(selected_cycles),best_cycles_rejoined=sum(c["complete_charge_cycle"] for c in selected_cycles),
        best_cycles_useful=sum(not c["useful_service_censored"] for c in selected_cycles),
        best_cycles_censored=sum(c["useful_service_censored"] for c in selected_cycles),
        followup=lifecycle_summary,optional_repro_files_verified=checked_weights,
        source_sha256=p["source_sha256"],protocol_sha256=canonical_hash(p),
        caution="Single training seed; inspected validation; horizon/time-limit/replay protocol changed; no causal, convergence, direct or C-N claim.")
    atomic_json(output/"summary.json",summary)
    atomic_json(output/"verification.json",dict(raw_metrics_verified=True,continuity_verified=True,
        source_snapshot_verified=True,manifest_verified=manifest.exists(),require_complete=require_complete,
        training_steps=total_train,validation_steps=total_eval))
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="400"><rect width="900" height="400" fill="white"/>',
         '<path d="M70 30V340H860" stroke="black" fill="none"/>',
         '<text x="70" y="20">C-R validation utility; complete game; one training seed</text>']
    for e in learning:
        x=70+e["step"]/p["total_steps"]*790; y=340-e["utility"]*300
        svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#2563eb"/><text x="{x-15}" y="365">{e["step"]//1000}k</text>')
    points=" ".join(f"{70+e['step']/p['total_steps']*790},{340-e['utility']*300}" for e in learning)
    svg.append(f'<polyline points="{points}" fill="none" stroke="#2563eb"/><text x="10" y="40">1.0</text><text x="15" y="340">0</text></svg>')
    (output/"learning_curve.svg").write_text("".join(svg),encoding="utf-8")
    (output/"REPORT_VI.md").write_text(
        f"# Pilot C-R horizon {p['full_horizon']}\n\nTrạng thái: {summary['status']}; diagnostic={p['diagnostic']}.\n\n"
        f"Đã kiểm từ raw: {total_train} train steps, {total_eval} validation steps, {update_count} updates.\n\n"
        f"Best: bước {best['step']}, utility {best['utility']:.6f}; last {learning[-1]['utility']:.6f}; "
        f"tail ba mốc {summary['tail_three_mean']:.6f}.\n\n"
        "Xem games/windows để đối chiếu utility, QoS, fairness và relay; cycles/uavs để xem nhiều chu kỳ hậu sạc. "
        "Relay rỗng tính theo UAV-slot, denominator0 là null. Throughput và utility không phải hai bằng chứng độc lập.\n\n"
        "Một training seed, validation đã được quan sát; không kết luận hội tụ, nhân quả horizon, curriculum hơn direct hoặc ưu thế C-R/C-N mới. "
        "Horizon, semantics time-limit và clear replay tại S4 cùng thay đổi. Không tự mở rộng120k.\n",
        encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--require-complete",action="store_true")
    args=parser.parse_args()
    print(json.dumps(analyze(args.root,args.output,require_complete=args.require_complete),ensure_ascii=False))


if __name__=="__main__": main()
