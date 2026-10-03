"""Run the fault campaign of a validation study and write its dataset.

    python -m validation.run_campaign --circuit sallen_key --out data/sallen_key

It resumes if it was interrupted. With the default size (5000 healthy samples and 200
per fault) a campaign is tens of thousands of simulations: run it on an idle machine.
"""

from __future__ import annotations

import argparse

import validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--circuit", required=True, choices=sorted(validation.STUDIES))
    parser.add_argument("--out", required=True, help="folder of the dataset")
    parser.add_argument("--samples-per-fault", type=int, default=200)
    parser.add_argument("--healthy", type=int, default=5000)
    parser.add_argument("--tolerance-scale", type=float, default=1.0)
    parser.add_argument("--conditions", default="nominal", help="named set of operating conditions")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    study = validation.get(args.circuit)
    campaign = study.campaign(
        args.out,
        samples_per_fault=args.samples_per_fault,
        healthy_samples=args.healthy,
        tolerance_scale=args.tolerance_scale,
        conditions=args.conditions,
        seed=args.seed,
    )
    print(f"{study.description}: {campaign.status()['total']} simulations")
    print(campaign.validate())
    campaign.run(workers=args.workers, chunk=args.chunk)
    print(campaign.status())
    summary = campaign.summary()
    print(summary[summary["success_rate"] < 1.0].to_string() or "every simulation succeeded")


if __name__ == "__main__":
    main()
