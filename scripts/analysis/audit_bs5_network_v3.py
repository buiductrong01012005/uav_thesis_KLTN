"""Matched-state radio audit, not retraining or an optimized policy benchmark."""

import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.envs.bs5_tcf_env import BS5TCFEnv, TCFWeights, rate_metrics
from src.envs.bs5_network_v3 import BS5NetworkV3Env
from src.diagnostics.bs5_radio_v3 import RadioV3Config, uma_av_components
from src.diagnostics.bs5_resource_audit import access_raw_rate
from scripts.training.run_bs5_tcf120k import write_csv, atomic_json, sha

WEIGHTS = TCFWeights(0.45, 0.45, 0.10)


def raw_links(env, distance):
    from src.diagnostics.bs5_spectrum_airtime import dbm_to_mw, spectral_rate

    cfg, radio = env.cfg, env.radio_config
    bw = cfg.backhaul_bw_hz
    noise = float(dbm_to_mw(cfg.noise_psd_dbm_hz + 10 * np.log10(bw) + cfg.noise_figure_db))
    if radio.bs_uav_channel == "uma_av_mean":
        loss = uma_av_components(distance, cfg.uav_altitude_m, cfg.bs_height_m, cfg.backhaul_fc_hz)[
            "mean_loss_db"
        ]
    else:
        loss = env._a2g_path_loss_db(
            [0, 0], cfg.bs_height_m, [distance, 0], cfg.uav_altitude_m, cfg.backhaul_fc_hz
        )
    c1 = spectral_rate(dbm_to_mw(radio.bs_tx_dbm) * 10 ** (-loss / 10), 0, noise, bw)
    loss2 = env._a2a_path_loss_db([0, 0], [distance, 0], cfg.backhaul_fc_hz)
    c2 = spectral_rate(dbm_to_mw(radio.uav_bh_tx_dbm) * 10 ** (-loss2 / 10), 0, noise, bw)
    return c1, c2, c1 * c2 / (c1 + c2)


def frame_metrics(env, net, detail):
    m = rate_metrics(net["ue_rate_mbps"], weights=WEIGHTS)
    access = [a for a in net["association"]]
    covered = np.zeros(env.cfg.num_ues, dtype=bool)
    for a in access:
        covered |= (
            np.linalg.norm(env.ue_positions - env.uav_positions[a], axis=1)
            <= env.cfg.access_radius_cells + 1e-9
        )
    paths = detail["paths"]
    m.update(
        covered_ues=float(covered.sum()),
        associated_ues=float(sum(map(len, net["association"].values()))),
        reachable_ues=float(sum(len(net["association"][a]) for a, path in paths.items() if path)),
        access_offer_mbps=float(net["access_raw_ue_rates"].sum()),
        reachable_offer_mbps=float(
            sum(net["access_offer_mbps"][a] for a, path in paths.items() if path)
        ),
        schedule_airtime=net["schedule_time_sum"],
        flow_violation_mbps=net["radio_flow_violation_mbps"],
    )
    assert net["schedule_time_sum"] <= 1 + 1e-6 and net["radio_flow_violation_mbps"] <= 1e-6
    return m


def corpus_from_reference(reference, output, stride=20):
    frames = []
    for path in sorted(reference.glob("*ue_rates.npz")):
        seed = int(path.stem.split("_seed")[1].split("_")[0])
        with np.load(path) as saved:
            actions = saved["actions"]
            rates = saved["ue_rates"]
            positions = saved["positions"]
        env = BS5TCFEnv(horizon=len(actions), reward_weights=WEIGHTS)
        env.reset(seed=seed, assignment_seed=seed)
        for slot, action in enumerate(actions):
            env.step(action.tolist())
            snap = env.last_service_snapshot
            np.testing.assert_array_equal(snap["uav_positions"], positions[slot])
            np.testing.assert_allclose(
                env.last_v2_network["ue_rate_mbps"], rates[slot], atol=1e-7, rtol=0
            )
            if slot % stride == 0:
                frames.append(
                    dict(
                        seed=seed,
                        slot=slot + 1,
                        positions=snap["uav_positions"].copy(),
                        ue_positions=snap["ue_positions"].copy(),
                        status=snap["status"].copy(),
                        roles=snap["roles"].copy(),
                        battery=snap["battery"].copy(),
                    )
                )
        print(f"verified reference seed {seed}: {len(actions)} slots", flush=True)
    if not frames:
        raise ValueError("No recorded validation UE-rate/action tapes found")
    np.savez_compressed(
        output / "matched_frames.npz", **{k: np.asarray([f[k] for f in frames]) for k in frames[0]}
    )
    return frames


def set_frame(env, frame):
    env.uav_positions = frame["positions"].copy()
    env.ue_positions = frame["ue_positions"].copy()
    env.status = frame["status"].copy()
    env.roles = frame["roles"].copy()
    env.battery = frame["battery"].copy()


def relay_fixture(radio, clients):
    env = BS5NetworkV3Env(radio_config=radio, reward_weights=WEIGHTS)
    env.status[:] = 3
    env.roles[:] = 0
    env.surviving_bs_xy = np.asarray([[125.0, 125.0]])
    env.status[0] = 0
    env.roles[0] = 1
    env.uav_positions[0] = [2, 8]
    env.status[1] = 0
    env.uav_positions[1] = [2, 14]
    env.ue_positions[:] = [18.0, 1.0]
    env.ue_positions[:25] = [2.0, 14.0]
    if clients == 2:
        env.status[2] = 0
        env.uav_positions[2] = [8, 8]
        env.ue_positions[25:50] = [8.0, 8.0]
    env._in_step = True
    net = env.compute_network()
    value = float(net["ue_rate_mbps"].sum())
    assert net["schedule_time_sum"] <= 1 + 1e-6 and net["radio_flow_violation_mbps"] <= 1e-6
    return dict(
        throughput_mbps=value,
        target_mbps=100 * clients,
        served_ues=int(np.sum(net["ue_rate_mbps"] >= 4 - 1e-9)),
        radio_solo_twohop_mbps=raw_links(env, 300)[2],
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    # Selection rules are written before the sweep is evaluated.
    policy = dict(
        channel_selection="UMa-AV for BS25m/aerial receiver by terminal geometry and primary equations, not by reward ranking",
        power_selection="minimum of 20/30/43dBm that supports an isolated 200Mbps relay load at BOTH audited heights with 300m hops; UAV fixed20dBm",
        radius_height_selection="maximize matched-frame mean TCF .45/.45/.10 among selected channel/power and raw edge Access>=4.4Mbps/UE for25UE; within0.005 score prefer smaller radius then lower height",
        edge_headroom_mbps=4.4,
        edge_headroom_is_design_assumption=True,
        scope="Reference configuration in this diagnostic; not hardware validation or globally optimal design",
    )
    atomic_json(args.output / "selection_protocol.json", policy)
    if args.corpus is None:
        if args.reference is None:
            raise ValueError("Supply --reference or --corpus")
        frames = corpus_from_reference(args.reference, args.output, args.stride)
    else:
        with np.load(args.corpus) as saved:
            arrays = {k: saved[k] for k in saved.files}
        frames = [
            {k: values[i] for k, values in arrays.items()} for i in range(len(arrays["seed"]))
        ]
        np.savez_compressed(args.output / "matched_frames.npz", **arrays)
    configs = [
        RadioV3Config(access_radius_m=r, uav_altitude_m=h, bs_tx_dbm=p, bs_uav_channel=ch)
        for ch, p, h, r in itertools.product(
            ["legacy_urban_mean", "uma_av_mean"],
            [20.0, 30.0, 43.0],
            [80.0, 100.0],
            [200.0, 250.0, 300.0],
        )
    ]
    rows, aggregates, links, loads = [], [], [], []
    for k, radio in enumerate(configs):
        env = BS5NetworkV3Env(radio_config=radio, reward_weights=WEIGHTS)
        samples = []
        for frame in frames:
            set_frame(env, frame)
            env._in_step = True
            net = env.compute_network()
            m = frame_metrics(env, net, env.last_v2_radio_detail)
            samples.append(m)
            rows.append(
                dict(config_id=k, **asdict(radio), seed=frame["seed"], slot=frame["slot"], **m)
            )
        edge = access_raw_rate(env, radio.access_radius_m, 25)
        aggregates.append(
            dict(
                config_id=k,
                **asdict(radio),
                edge_access_raw_mbps=edge,
                **{name: float(np.mean([m[name] for m in samples])) for name in samples[0]},
            )
        )
        for distance in [100.0, 200.0, 300.0]:
            c1, c2, two = raw_links(env, distance)
            links.append(
                dict(
                    config_id=k,
                    **asdict(radio),
                    distance_m=distance,
                    bs_uav_solo_mbps=c1,
                    uav_uav_solo_mbps=c2,
                    half_duplex_twohop_mbps=two,
                )
            )
        for clients in [1, 2]:
            loads.append(
                dict(config_id=k, **asdict(radio), clients=clients, **relay_fixture(radio, clients))
            )
        print(
            f"audit {k + 1}/{len(configs)} channel={radio.bs_uav_channel} Pbs={radio.bs_tx_dbm:g} h={radio.uav_altitude_m:g} R={radio.access_radius_m:g} T={aggregates[-1]['throughput_mbps']:.2f} C={aggregates[-1]['service_coverage_ues']:.2f}",
            flush=True,
        )
    chosen_power = None
    for power in [20.0, 30.0, 43.0]:
        tests = [
            r
            for r in loads
            if r["bs_uav_channel"] == "uma_av_mean"
            and r["bs_tx_dbm"] == power
            and r["clients"] == 2
        ]
        if tests and all(r["throughput_mbps"] >= 200 - 1e-6 for r in tests):
            chosen_power = power
            break
    if chosen_power is None:
        raise RuntimeError("No candidate meets the stated isolated-load design rule")
    candidates = [
        r
        for r in aggregates
        if r["bs_uav_channel"] == "uma_av_mean"
        and r["bs_tx_dbm"] == chosen_power
        and r["edge_access_raw_mbps"] >= policy["edge_headroom_mbps"]
    ]
    best = max(r["tcf_score"] for r in candidates)
    tied = [r for r in candidates if r["tcf_score"] >= best - 0.005]
    selected = min(tied, key=lambda r: (r["access_radius_m"], r["uav_altitude_m"], -r["tcf_score"]))
    radio = {k: selected[k] for k in asdict(configs[0])}
    atomic_json(
        args.output / "selected_radio.json",
        dict(
            radio=radio,
            selected_config_id=selected["config_id"],
            selection=policy,
            matched_frame_metrics=selected,
            number_frames=len(frames),
        ),
    )
    for name, data in [
        ("matched_frames.csv", rows),
        ("aggregate.csv", aggregates),
        ("link_budgets.csv", links),
        ("relay_loads.csv", loads),
    ]:
        write_csv(args.output / name, data)
    atomic_json(
        args.output / "receipt.json",
        dict(
            configs=len(configs),
            frames=len(frames),
            scored_cases=len(rows),
            load_cases=len(loads),
            link_cases=len(links),
            runtime=dict(python=sys.version.split()[0], numpy=np.__version__),
            wall_seconds=time.perf_counter() - started,
            source_sha256={
                str(p.relative_to(ROOT)): sha(p)
                for p in [
                    Path(__file__),
                    ROOT / "src/envs/bs5_network_v3.py",
                    ROOT / "src/diagnostics/bs5_radio_v3.py",
                ]
            },
        ),
    )
    print("SELECTED " + json.dumps(radio), flush=True)
    return radio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=20)
    run(parser.parse_args())
