# Validation of the statistics

The statistics of a population ([definitions](STATISTICS.md)) are checked against two
circuits whose output distribution has a closed form. Each is drawn and simulated with
ngspice as a campaign without faults, and what `Dataset.statistics` and
`Dataset.yield_report` estimate is compared with the exact value. The study is
`validation/statistical.py`; a small version of it runs with the tests.

## The circuits

**A divider with uniform tolerances.** Two equal resistors, each uniform within ±5 %.
The output of a 1 V source is the ratio $\rho = R_2 / (R_1 + R_2)$. It is below $x$
when $R_1 \ge c R_2$ with $c = (1 - x)/x$, so its distribution function is the area of
that region of the square of the two tolerances, a closed form. Its mean is exactly 0.5,
because $R_1$ and $R_2$ are exchangeable.

**An RC low-pass with log-normal values.** $R$ and $C$ are log-normal with
$\sigma = 0.03$ and $0.08$, so $RC$ is exactly log-normal with
$\sigma = \sqrt{0.03^2 + 0.08^2}$. The gain at 1 kHz, $1/\sqrt{1 + (2\pi f RC)^2}$, falls
as $RC$ grows, so each of its quantiles is the gain at a quantile of $RC$. Its mean has no
closed form and is not compared.

**A yield that is known.** The specification of each circuit is placed at its exact 5 %
and 95 % quantiles, so its yield is 0.90 by construction.

The closed forms are themselves checked, in the tests, against a direct simulation of
the two populations with 400,000 draws and no SPICE.

## One campaign

4,000 circuits of each, 95 % intervals:

| Circuit | Quantity | Estimate | Interval | Exact |
|---|---|---|---|---|
| Divider | 5 % quantile | 0.48292 | 0.48238 to 0.48348 | 0.48290 |
| Divider | Median | 0.50022 | 0.49991 to 0.50055 | 0.50000 |
| Divider | 95 % quantile | 0.51692 | 0.51641 to 0.51740 | 0.51710 |
| Divider | Mean | 0.50010 | 0.49978 to 0.50041 | 0.50000 |
| Divider | Yield | 0.9020 | 0.8924 to 0.9108 | 0.9000 |
| RC | 5 % quantile | 0.65551 | 0.65336 to 0.65784 | 0.65589 |
| RC | Median | 0.70776 | 0.70657 to 0.70908 | 0.70711 |
| RC | 95 % quantile | 0.75375 | 0.75196 to 0.75553 | 0.75486 |
| RC | Yield | 0.9033 | 0.8937 to 0.9120 | 0.9000 |

Every interval contains the exact value.

## Coverage of the intervals

An interval that claims 95 % should contain the exact value in about 95 of 100
independent campaigns. Over 20 campaigns of 500 circuits, with seeds 0 to 19:

| Quantity | Divider | RC |
|---|---|---|
| 5 % quantile | 18 of 20 | 20 of 20 |
| Median | 19 of 20 | 18 of 20 |
| 95 % quantile | 19 of 20 | 19 of 20 |
| Mean | 18 of 20 | not compared |
| Yield | 19 of 20 | 20 of 20 |

That is 170 of 180 intervals, 94 %. Twenty campaigns measure a coverage only to within
about ±10 points, so this shows that no interval is badly wrong, not that each one is
exact; the command below takes more seeds.

## Running it

```bash
python -m validation.statistical --samples 4000 --seeds 20 --workers 8
```

It takes about two minutes on eight cores and prints the two tables. The figures above
were obtained with ngspice 44.2; `--seeds 200` measures the coverage to within about
±3 points.
