"""Build equal-budget historical balanced curves, recounting numerical QoS."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np


def prepare(review, index_path, output):
    index = json.loads(index_path.read_text())
    digest = hashlib.sha256(review.read_bytes()).hexdigest()
    if digest != index["files"]["TCF120K_REVIEW.zip"]["sha256"]:
        raise ValueError("Historical review SHA mismatch")
    rows = []
    with zipfile.ZipFile(review) as z:
        if z.testzip() is not None:
            raise ValueError("Corrupt historical ZIP")
        for arm in ["C-N", "C-R"]:
            prefix = f"balanced/{arm}/"
            protocol = json.loads(z.read(prefix + "protocol.json"))
            assert protocol["reward_weights"] == dict(
                throughput=1 / 3, coverage=1 / 3, fairness=1 / 3
            )
            assert (
                protocol["total_steps"] == 120000 and protocol["service"]["beacon_radius_m"] == 200
            )
            old = list(csv.DictReader(io.StringIO(z.read(prefix + "validation.csv").decode())))
            for row in old:
                step, seed = int(row["step"]), int(row["seed"])
                with np.load(
                    io.BytesIO(
                        z.read(prefix + f"validation_detail/step{step}_seed{seed}_ue_rates.npz")
                    ),
                    allow_pickle=False,
                ) as arrays:
                    rates = arrays["ue_rates"]
                assert abs(rates.sum(1).mean() - float(row["throughput_mbps"])) < 1e-7
                qualified = np.where((rates > 1e-9) & (rates >= 1 - 1e-9), rates, 0)
                square = (qualified * qualified).sum(1)
                jain = np.divide(
                    qualified.sum(1) ** 2, 90 * square, out=np.zeros(len(rates)), where=square > 0
                )
                t = rates.sum(1) / 360
                c = ((rates > 1e-9) & (rates >= 1 - 1e-9)).mean(1)
                rows.append(
                    dict(
                        arm=arm,
                        step=step,
                        seed=seed,
                        slots=len(rates),
                        throughput_mbps=float(rates.sum(1).mean()),
                        service_coverage_ues=float((c * 90).mean()),
                        positive_ues=float((rates > 1e-9).sum(1).mean()),
                        qos_ues=float((rates >= 4 - 1e-6).sum(1).mean()),
                        qos_ues_original=float(row["qos_ues"]),
                        jain_raw=float(row["jain_raw"]),
                        jain_service_qualified=float(jain.mean()),
                        tcf_score=float(((t + c + jain) / 3).mean()),
                        tc_priority_posthoc=float((0.45 * t + 0.45 * c + 0.1 * jain).mean()),
                        training_reward="balanced_1/3_each",
                        runtime="historical_CPU",
                    )
                )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "historical_balanced_curve.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (output / "historical_reference.json").write_text(
        json.dumps(
            dict(
                version="historical_balanced_reference_1",
                source_review_sha256=digest,
                source_index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
                rows=len(rows),
                qos_tolerance_mbps=1e-6,
                comparison="Use same validation horizon/seeds and restrict historical steps to the new committed budget; historical CPU vs new GPU is not bitwise reproducibility",
                checkpoint_selection="Historical policies trained balanced; priority score is posthoc only, not a priority-trained baseline",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    prepare(a.review, a.index, a.output)
