"""Post-action diagnostics only. Nothing in this module enters actor inputs.

The throughput waterfall is an additive accounting identity at fixed geometry,
association and routes. It is not a causal estimate of all possible policies.
"""
from copy import copy
from dataclasses import replace
import numpy as np
from src.envs.bs5_tcf_env import rate_metrics, jain
from src.diagnostics.bs5_episode_service_metrics import episode_service_metrics, longest_true
from src.diagnostics.bs5_multihop import score_multihop

VERSION = "mh_explanation_metrics_1"


def waterfall(demand, covered, associated, raw_offer, reachable_offer, delivered):
    values = [demand-covered*4, (covered-associated)*4,
              associated*4-raw_offer, raw_offer-reachable_offer,
              reachable_offer-delivered, delivered]
    if min(values) < -1e-6 or abs(sum(values)-demand) > 1e-6:
        raise AssertionError("Invalid demand/coverage/Access/BH accounting")
    names = ["loss_outside_access_mbps", "loss_association_mbps",
             "loss_access_radio_mbps", "loss_bh_disconnected_mbps",
             "loss_bh_schedule_capacity_mbps", "delivered_mbps"]
    return dict(zip(names, map(float, values)))


def slot_sample(env, before, info, actor_stats):
    snap, net, result = env.last_service_snapshot, env.last_v2_network, env.last_probe_result
    roles, status = snap["roles"], snap["status"]
    rates = net["ue_rate_mbps"].copy()
    active = status == 0
    covered = np.zeros(90, dtype=bool)
    association_id = np.full(90, -1, dtype=np.int8)
    reachable = np.zeros(90, dtype=bool)
    hops = np.zeros(90, dtype=np.int8)
    for aid, users in net["association"].items():
        association_id[users] = aid
        path = result["paths"][aid]
        reachable[users] = bool(path)
        hops[users] = len(path)
        covered |= np.linalg.norm(snap["ue_positions"]-snap["uav_positions"][aid], axis=1) <= env.cfg.access_radius_cells+1e-9
    raw = np.asarray(net["access_raw_ue_rates"])
    total = float(rates.sum())
    reachable_offer = float(raw[reachable].sum())
    delta = np.maximum(before["battery"].astype(float)-env.battery.astype(float), 0)
    charged = np.maximum(env.battery.astype(float)-before["battery"].astype(float), 0)
    effective = np.asarray(env.last_return_control["effective_status"])
    # Classify measured depletion by the operational mode used in this slot.
    energy = {f"energy_{label}_points":float(delta[(effective==0)&(roles==r)].sum())
              for r,label in [(0,"access"),(1,"relay"),(2,"hybrid")]}
    energy["energy_return_points"] = float(delta[effective==1].sum())
    energy["energy_other_points"] = float(delta[~np.isin(effective,[0,1])].sum())
    assert abs(sum(energy.values())-delta.sum()) < 1e-8
    metric = rate_metrics(rates)
    metric["qos_ues"] = int(np.sum(rates>=4-1e-6))
    metric["qos_fraction"] = metric["qos_ues"]/90
    row = dict(slot=int(env.t), **metric,
        covered_ues=int(covered.sum()), associated_ues=int((association_id>=0).sum()),
        reachable_ues=int(reachable.sum()), access_offer_mbps=float(raw.sum()),
        reachable_offer_mbps=reachable_offer,
        **waterfall(360.,int(covered.sum()),int((association_id>=0).sum()),float(raw.sum()),reachable_offer,total),
        active_access=int(np.sum(active&(roles==0))),active_relay=int(np.sum(active&(roles==1))),
        active_hybrid=int(np.sum(active&(roles==2))), active_uavs=int(active.sum()),
        returning_uavs=int(np.sum(status==1)),charging_uavs=int(np.sum(status==2)),
        unavailable_uavs=int(np.sum(status!=0)),
        depleted_points=float(delta.sum()), charged_points=float(charged.sum()), **energy,
        extra_hybrid_debit_points=float(env.last_energy_probe["extra_hybrid_debit_points"]),
        schedule_airtime=float(net["schedule_time_sum"]),
        zero_service_slot=int(total<=1e-9),
        flow_violation_mbps=float(net["radio_flow_violation_mbps"]),
        stranded_uavs=len(env.last_return_control["stranded_ids_post"]),
        returning_wait_uavs=len(env.last_return_control["wait_ids"]),
        role_switches=int(info.get("role_switches_this_slot",0)),
        reported_map_known_cells=int(np.sum((env.get_obs()[0,25:50]>0)|(env.get_obs()[0,:25]>0))),
        report_age_slots=float(env.get_obs()[0,84]*env.service_config.report_max_age_slots),
        actor_ms=float(actor_stats.get("decision_ms",0)),
        estimated_score_calls=int(actor_stats.get("estimated_score_calls",0)),
        role_vectors_evaluated=int(actor_stats.get("roles_evaluated",0)),
        heuristic_role_cache_hit=int(actor_stats.get("role_cache_hit",0)),
        heuristic_recovery_plan=int(actor_stats.get("recovery_plan",0)),
        radio_ms=float(net["radio_runtime_ms"]))
    for k in ["own_hybrid_mbps","hybrid_forwarded_mbps",
              "mean_wireless_hops_delivered","max_wireless_hops_used",
              *[f"delivered_mbps_hops_{i}" for i in range(1,6)]]:
        row[k] = float(result["metrics"][k])
    own = np.asarray([result["flows"].get(a,0) for a in range(5)])
    forward = np.asarray([net["forwarded_mbps_by_uav"].get(a,0) for a in range(5)])
    row["hybrid_both_count"] = int(np.sum(active&(roles==2)&(own>1e-6)&(forward>1e-6)))
    row["forwarding_uavs"] = int(np.sum(forward>1e-6))
    for a in range(5):
        row[f"u{a}_own_mbps"] = float(own[a])
        row[f"u{a}_forward_mbps"] = float(forward[a])
    arrays = dict(rates=rates, covered=covered, reachable=reachable,
                  association=association_id, hops=hops,
                  positions=snap["uav_positions"].copy(), roles=roles.copy(),status=status.copy(),
                  battery=snap["battery"].copy(),post_status=env.status.copy(),
                  post_battery=env.battery.copy(),energy_depleted=delta)
    return row, arrays, (own+forward)>1e-6


def network_sample(env, counterfactual=True):
    snap, r = env.last_service_snapshot, env.last_probe_result
    # Isolated copy at service time; current post-energy status must not replace
    # the status which generated this slot's service.
    view = copy(env)
    view.uav_positions = snap["uav_positions"].copy()
    view.ue_positions = snap["ue_positions"].copy()
    view.status = snap["status"].copy()
    view.roles = np.where(snap["roles"]==2,1,snap["roles"])
    roles = snap["roles"].copy()
    edges = []
    for i,e in enumerate(r["edges"]):
        use = sum(float(r["durations"][j]) for j,s in enumerate(r["states"]) if i in s)
        capacity = float((r["capacities"]@r["durations"])[i])
        load = sum(float(r["flows"][a]) for a,path in r["paths"].items() if i in path)
        users = sum(len(r["association"][a]) for a,path in r["paths"].items() if i in path)
        edges.append(dict(source=e.source,target=e.target,solo_mbps=float(e.solo_mbps),
            load_mbps=load,scheduled_capacity_mbps=capacity,airtime=use,
            fulfillment=load/capacity if capacity>1e-9 else 0,downstream_ues=users))
    out = dict(slot=int(env.t),scope="post-action diagnostic; never fed to controller",
        positions=snap["uav_positions"].tolist(),ue_positions=snap["ue_positions"].tolist(),
        roles=roles.tolist(),status=snap["status"].tolist(),edges=edges,
        paths={str(a):path for a,path in r["paths"].items()},
        association={str(a):users for a,users in r["association"].items()},
        schedules=[dict(edges=list(s),duration=float(r["durations"][j]))
                   for j,s in enumerate(r["states"]) if r["durations"][j]>1e-9],
        actual_mbps=float(r["ue_rate_mbps"].sum()))
    if counterfactual:
        for label, field in [("bh2x","backhaul_total_mhz"),("access2x","access_channel_mhz")]:
            trial = copy(view)
            trial.service_config = replace(view.service_config,**{field:2*getattr(view.service_config,field)})
            out[f"same_snapshot_{label}_mbps"] = float(score_multihop(trial,roles)["ue_rate_mbps"].sum())
        out["counterfactual_scope"] = "Isolated fixed-position/role/status radio probes; may reroute/reassociate; fixed power; no energy/trajectory rollout; not a new benchmark configuration or causal energy attribution."
    return out


def temporal_summary(arrays, rows, cycles, initial_battery):
    rates = np.asarray(arrays["rates"])
    nslots = len(rates)
    result = episode_service_metrics(rates)
    mean = rates.mean(axis=0)
    mb = rates.sum(axis=0)*5/8
    positive = rates>1e-9
    qualified = rates>=1-1e-9
    qos = rates>=4-1e-6
    result.update(slots=nslots,simulation_seconds=nslots*5,
        jain_total_data=jain(mb),min_ue_mean_rate_mbps=float(mean.min()),
        p05_ue_mean_rate_mbps=float(np.quantile(mean,.05)),median_ue_mean_rate_mbps=float(np.median(mean)),
        min_ue_received_mb=float(mb.min()),p05_ue_received_mb=float(np.quantile(mb,.05)),
        median_ue_received_mb=float(np.median(mb)),
        ues_received_1mb=int(np.sum(mb>=1)),ues_received_5mb=int(np.sum(mb>=5)),
        unique_geometrically_covered_ues=int(np.asarray(arrays["covered"]).any(axis=0).sum()),
        unique_bh_reachable_ues=int(np.asarray(arrays["reachable"]).any(axis=0).sum()),
        longest_below1_gap_slots_max=max(longest_true(~qualified[:,u]) for u in range(90)))
    cumulative = np.cumsum(rates,axis=0)*5/8
    for fraction,label in [(.9,"90pct"),(1.,"all")]:
        reached = np.flatnonzero((cumulative>=1).sum(axis=1)>=int(np.ceil(fraction*90)))
        result[f"time_to_{label}_ues_received_1mb_seconds"] = int((reached[0]+1)*5) if len(reached) else None
    for width,label in [(12,"1min"),(60,"5min")]:
        if nslots<width:
            continue
        for boolean,name in [(positive,"unique_positive"),(qualified,"unique_ge1")]:
            c=np.vstack([np.zeros((1,90)),np.cumsum(boolean,axis=0)])
            counts=((c[width:]-c[:-width])>0).sum(axis=1)
            result[f"window_{label}_{name}_mean"]=float(counts.mean())
            result[f"window_{label}_{name}_min"]=int(counts.min())
        c=np.vstack([np.zeros((1,90)),np.cumsum(rates,axis=0)])
        counts=((c[width:]-c[:-width])/width>=1-1e-9).sum(axis=1)
        result[f"window_{label}_ues_mean_rate_ge1_mean"]=float(counts.mean())
        result[f"window_{label}_ues_mean_rate_ge1_min"]=int(counts.min())
    for key in rows[0]:
        if key!='slot': result[key]=float(np.mean([r[key] for r in rows]))
    used=sum(r["depleted_points"] for r in rows)
    charged=sum(r["charged_points"] for r in rows)
    final=float(np.asarray(arrays["post_battery"])[-1].sum())
    balance=initial_battery+charged-used-final
    if abs(balance)>1e-3:raise AssertionError("Battery balance failed")
    delivered_gbit=float(rates.sum()*5/1000)
    result.update(total_depleted_points=used,total_charged_points=charged,
        battery_balance_error_points=balance,battery_points_per_gbit=used/delivered_gbit if delivered_gbit>1e-12 else None,
        active_uav_fraction=result['active_uavs']/5,return_uav_fraction=result['returning_uavs']/5,
        charging_uav_fraction=result['charging_uavs']/5,
        bh_fulfillment=result['throughput_mbps']/result['reachable_offer_mbps'] if result['reachable_offer_mbps']>1e-9 else None,
        observed_max_meaningful_wireless_hops=max([i for i in range(1,6) if any(r[f'delivered_mbps_hops_{i}']>=1 for r in rows)],default=0),
        volume_weighted_bh_links=sum(i*result[f'delivered_mbps_hops_{i}'] for i in range(1,6))/result['throughput_mbps'] if result['throughput_mbps']>1e-9 else 0,
        charging_rejoins=cycles['rejoins'],served_again_cycles=sum(c['outcome']=='served_again' for c in cycles['completed']),
        next_return_without_service=sum(c['outcome']=='next_return_without_service' for c in cycles['completed']),pending_cycles=len(cycles['pending']))
    for mode in ['access','relay','hybrid','return','other']:
        result[f'total_energy_{mode}_points']=sum(r.get(f'energy_{mode}_points',0.) for r in rows)
    ue=[]
    for u in range(90):
        ue.append(dict(ue=u,mean_rate_mbps=float(mean[u]),received_mb=float(mb[u]),
            positive_time_fraction=float(positive[:,u].mean()),ge1_time_fraction=float(qualified[:,u].mean()),qos_time_fraction=float(qos[:,u].mean()),
            geometric_coverage_fraction=float(np.asarray(arrays['covered'])[:,u].mean()),
            bh_reachable_fraction=float(np.asarray(arrays['reachable'])[:,u].mean()),
            longest_zero_gap_slots=longest_true(~positive[:,u]),longest_below1_gap_slots=longest_true(~qualified[:,u])))
    return result,ue
