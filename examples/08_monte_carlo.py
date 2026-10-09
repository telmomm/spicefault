# %% [markdown]
# # Monte Carlo and yield
#
# A campaign without faults: the spread of the healthy circuit, and how many meet their
# specifications. Also runnable as `python examples/08_monte_carlo.py`.
# %%

import sys
from pathlib import Path

import pandas as pd

examples_dir = next(
    (root / "examples" for root in (Path.cwd(), *Path.cwd().parents)
     if (root / "examples" / "rc_study.py").is_file()),
    None,
)
if examples_dir is not None:
    sys.path.insert(0, str(examples_dir))

# %%
from rc_study import OUTPUT, WORKERS, circuit, config, measurements, population  # noqa: E402

from spicefault import Campaign, Specification  # noqa: E402
from spicefault.statistics import samples_for_half_width, zero_failure_bound  # noqa: E402

# %% [markdown]
# ## Draw and simulate the healthy population
#
# No fault is given: every sample is a circuit drawn from the tolerances.
# %%
if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4g}".format)
    campaign = Campaign(
        circuit,
        out_dir=OUTPUT / "rc_monte_carlo",
        samples=1000,
        variations=population,
        config=config,
        measurements=measurements,
        seed=42,
    )
    campaign.run(workers=WORKERS, chunk=250, progress=False)
    dataset = campaign.dataset()
    print(dataset)
    print("problems found by verify():", dataset.verify())

# %% [markdown]
# ## The distribution of each measurement
#
# A quantile that the sample cannot support is left empty instead of being read off the
# smallest or the largest value.
# %%
if __name__ == "__main__":
    statistics = dataset.statistics(quantiles=(0.001, 0.05, 0.5, 0.95))
    print("\nmean, spread and failed simulations:")
    print(statistics[["n", "n_failed", "mean", "std", "minimum", "maximum"]].droplevel([0, 1]))
    print("\nquantiles, with their 95 % intervals:")
    print(statistics[["q0.001", "q0.05_low", "q0.05", "q0.05_high", "q0.5"]].droplevel([0, 1]))

# %% [markdown]
# ## Yield against specification limits
#
# The limits are kept apart from the simulation: changing them needs no new simulation.
# %%
if __name__ == "__main__":
    limits = [
        Specification("dc", minimum=0.4965, maximum=0.5035),
        Specification("gain_1k", minimum=0.146),
        Specification("phase_1k", minimum=-73.1),
    ]
    report = dataset.yield_report(limits)
    print("\nyield per specification and of all of them:")
    print(report[["n", "passed", "yield", "ci_low", "ci_high", "only_this", "margin_sigma"]])

    relaxed = [Specification("dc", minimum=0.495, maximum=0.505), *limits[1:]]
    print(f"\nwith the DC limits at +-0.005: yield "
          f"{dataset.yield_report(relaxed).loc['all', 'yield']:.3f}")  # fmt: skip

# %% [markdown]
# ## What the sample size allows
# %%
if __name__ == "__main__":
    wide = dataset.yield_report([Specification("dc", minimum=0.4, maximum=0.6)]).loc["all"]
    print(f"\nno failure in {wide['n']:.0f} circuits: the failure probability is below "
          f"{zero_failure_bound(int(wide['n'])):.4f}, not zero")  # fmt: skip
    print("circuits for a yield near 0.95 within +-0.005:",
          samples_for_half_width(0.005, p=0.95))  # fmt: skip
