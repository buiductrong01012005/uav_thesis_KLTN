"""Post-action observers and temporal metrics; never fed to policy. Original observer is copied unchanged in semantics."""
import numpy as np
from src.diagnostics.bs5_episode_service_metrics import longest_true

PRIMARY_METRICS = ['throughput_mbps', 'throughput_norm', 'service_coverage_fraction', 'service_coverage_ues', 'positive_fraction', 'positive_ues', 'qos_fraction', 'qos_ues', 'ue_outage_fraction', 'jain_raw', 'jain_service_qualified', 'service_score', 'reward_throughput', 'reward_coverage', 'reward_fairness', 'fairness_used', 'jain_ema', 'jain_instant_qualified']

def observe_step(env, info):
    snap, net = (env.last_service_snapshot, env.last_v2_network)
    active = np.flatnonzero(snap['status'] == 0)
    access = [a for a in active if snap['roles'][a] in (0, 2)]
    relays = [a for a in active if snap['roles'][a] == 1]
    covered = np.zeros(env.cfg.num_ues, dtype=bool)
    for aid in access:
        covered |= np.linalg.norm(snap['ue_positions'] - snap['uav_positions'][aid], axis=1) <= env.cfg.access_radius_cells + 1e-09
    paths = env.last_v2_radio_detail['paths']
    forwarding = {a for a, flow in net['forwarded_mbps_by_uav'].items() if flow > 1e-09}
    pure_forwarding = {a for a in forwarding if snap['roles'][a] == 1}
    assert net['schedule_time_sum'] <= 1 + 1e-06
    assert net['radio_flow_violation_mbps'] <= 1e-06
    return {**{k: float(info[k]) for k in PRIMARY_METRICS}, 'utility': float(info['utility']), 'covered_ues': float(covered.sum()), 'associated_ues': float(sum(map(len, net['association'].values()))), 'reachable_ues': float(sum((len(net['association'][a]) for a, path in paths.items() if path))), 'active_access': float(sum((snap['roles'][a] == 0 for a in active))), 'active_relay': float(len(relays)), 'active_hybrid': float(sum((snap['roles'][a] == 2 for a in active))), 'forwarding_relay': float(len(pure_forwarding)), 'forwarding_total': float(len(forwarding)), 'forwarding_hybrid': float(sum((snap['roles'][a] == 2 for a in forwarding))), 'own_hybrid_mbps': float(env.last_probe_result['metrics']['own_hybrid_mbps']), 'hybrid_forwarded_mbps': float(env.last_probe_result['metrics']['hybrid_forwarded_mbps']), 'qos_ues_original': float(info['qos_ues_original']), 'qos_fraction_original': float(info['qos_fraction_original']), **env.last_energy_probe, 'access_offer_mbps': float(net['access_raw_ue_rates'].sum()), 'reachable_offer_mbps': float(sum((net['access_offer_mbps'][a] for a, path in paths.items() if path))), 'schedule_airtime': float(net['schedule_time_sum']), 'zero_service_slot_fraction': float(info['positive_ues'] == 0), 'returning_uavs': float(np.sum(snap['status'] == 1)), 'swapping_service_snapshot_uavs': float(np.sum(snap['status'] == 2)), 'stranded_slot_fraction': float(bool(env.last_return_control['stranded_ids_post'])), 'flow_violation_mbps': float(net['radio_flow_violation_mbps']), **{key: float(env.last_probe_result['metrics'][key]) for key in ['mean_wireless_hops_delivered', 'mean_relay_hops_delivered', 'max_wireless_hops_used', 'multihop_beyond_one_relay_mbps', 'routed_via_relay_mbps', *[f'delivered_mbps_hops_{i}' for i in range(1, 6)]]}, 'hybrid_both_count': float(sum((snap['roles'][a] == 2 and net['backhaul_caps_mbps'].get(a, 0) > 1e-06 and (net['forwarded_mbps_by_uav'].get(a, 0) > 1e-06) for a in active)))}

def rolling_unique(mask, width):
    if len(mask) < width:
        return None
    cumulative = np.vstack([np.zeros((1, mask.shape[1]), dtype=int), np.cumsum(mask, axis=0)])
    counts = ((cumulative[width:]-cumulative[:-width]) > 0).sum(axis=1)
    return float(counts.mean())


def safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else None


def extend_summary(state, p):
    rows, arrays = state['rows'], state['arrays']
    rates = np.maximum(np.asarray(arrays['ue_rates']), 0.)
    ge1 = rates >= p['training_protocol']['coverage_floor_mbps']-1e-9
    energy = {k:np.asarray([r[k] for r in state['energy']]) for k in state['energy'][0]}
    status, distance = energy['effective_status'], energy['distance_m']
    throughput = np.asarray([r['throughput_mbps'] for r in rows])
    zero = throughput <= 1e-9
    mean_rates = rates.mean(0)
    vals = {k:np.asarray([r[k] for r in rows]) for k in ['jain_ema','jain_instant_qualified','reward_throughput','reward_total','reward_energy','reward_collision']}
    extra = dict(rolling_unique_ge1_1min=rolling_unique(ge1,12), rolling_unique_ge1_5min=rolling_unique(ge1,60),
        longest_below1_gap_slots_max=max(longest_true(~ge1[:,u]) for u in range(90)),
        min_ue_mean_rate_mbps=float(mean_rates.min()),median_ue_mean_rate_mbps=float(np.median(mean_rates)),
        unique_geometrically_covered_ues=int(np.asarray(arrays['ue_covered']).any(0).sum()),
        unique_bh_reachable_ues=int(np.asarray(arrays['ue_reachable']).any(0).sum()),
        effective_active_uav_slot_fraction=float(np.mean(status==0)),
        effective_return_uav_slot_fraction=float(np.mean(status==1)),
        effective_swap_uav_slot_fraction=float(np.mean(status==2)),
        effective_unavailable_uav_slot_fraction=float(np.mean(status!=0)),
        hover_wh=float(energy['flight_j'][(status!=2)&(distance<1e-9)].sum()/3600),
        cruise_wh=float(energy['flight_j'][(status!=2)&(distance>1e-9)].sum()/3600),
        reward_under_ema=float(np.mean(vals['reward_throughput']+.6*vals['jain_ema']+vals['reward_energy']+vals['reward_collision'])),
        reward_under_instant=float(np.mean(vals['reward_throughput']+.6*vals['jain_instant_qualified']+vals['reward_energy']+vals['reward_collision'])),
        corr_throughput_reward=safe_corr(throughput,vals['reward_total']),
        corr_throughput_jain_ema=safe_corr(throughput,vals['jain_ema']),
        corr_throughput_jain_instant=safe_corr(throughput,vals['jain_instant_qualified']),
        zero_throughput_slots=int(zero.sum()),
        zero_throughput_positive_reward_slots=int(np.sum(zero&(vals['reward_total']>0))),
        reward_when_zero_throughput_mean=float(vals['reward_total'][zero].mean()) if zero.any() else None)
    access=np.asarray([r['active_access'] for r in rows]);associated=np.asarray([r['associated_ues'] for r in rows]);reachable=np.asarray([r['reachable_ues'] for r in rows])
    extra.update(zero_no_active_access_slots=int((zero&(access==0)).sum()),
        zero_no_association_slots=int((zero&(access>0)&(associated==0)).sum()),
        zero_no_route_slots=int((zero&(associated>0)&(reachable==0)).sum()),
        zero_other_slots=int((zero&(reachable>0)).sum()))
    for key in ['reward_throughput','reward_fairness','reward_energy','reward_collision']:
        extra[key+'_std']=float(np.std([r[key] for r in rows]))
    starts={};returns=[]
    for event in state['env'].energy_events:
        aid=event['agent']
        if event['event']=='return_start':starts[aid]=event
        if event['event']=='swap_start' and aid in starts:
            start=starts.pop(aid);lo,hi=start['slot']-1,event['slot']
            returns.append(dict(planned_m=start['route_m'],actual_m=float(distance[lo:hi,aid].sum()),slots=hi-lo))
    extra.update(mean_return_planned_m=float(np.mean([r['planned_m'] for r in returns])) if returns else None,
        mean_return_actual_m=float(np.mean([r['actual_m'] for r in returns])) if returns else None,
        mean_return_duration_slots=float(np.mean([r['slots'] for r in returns])) if returns else None,
        return_detour_events=sum(r['actual_m']>r['planned_m']+1e-6 for r in returns))
    return extra
