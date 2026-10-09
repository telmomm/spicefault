# Related work

A search of the literature to position `spicefault`: what exists, where the framework is new, and which circuits to validate it on. It is evidence for the manuscript, not the related-work section itself.

## 1. Method

Search in OpenAlex on 3 October 2026, with `scripts/openalex_search.py`. Each query is a boolean search over title and abstract; the most relevant 200 works of each are kept, with the abstract. The queries, the number of matches and every work found are in `docs/literature/`, so the search can be repeated and the judgements below checked.

| Query | Question | Matches |
|---|---|---|
| Q1 | Frameworks or tools that automate fault injection or fault simulation of analog circuits in SPICE | 20 |
| Q2 | Open tools that drive SPICE simulators programmatically | 86 |
| Q3 | Circuits the analog fault-diagnosis literature uses as benchmarks | 172 |
| Q4 | Effect of component tolerance on detectability, testability or ambiguity | 374 (200 kept) |
| Q5 | How simulated fault datasets are generated and shared | 5 |
| Q6 | Defect-oriented analog test and analog fault coverage | 106 |
| Q7 | Circuit-level fault injection for reliability and functional safety | 60 |
| Q8 | Open-source or scripted fault-injection software for analog circuits | 42 |

655 distinct works. Titles of the first 20 to 30 of each query were read, and the abstracts of the ones cited below.

Limits of this search, which bound every conclusion in this document:

- **Abstracts, not papers.** No full text was read. What a tool does is taken from what its abstract says.
- **Software is not in a bibliographic index.** PySpice, spicelib and the commercial simulators are not found by a literature search; their capabilities must be established from their documentation, at a fixed version (SCIENTIFIC_SCOPE.md §7). That has not been done.
- **One database, title and abstract only.** Works that describe a framework without those words in the abstract are missed. No citation chaining was done from the key works.
- **Authors are not recorded** by the script. Works are cited here by title, year, venue and DOI; names must be taken from the DOI when writing.

## 2. What exists

### 2.1 Defect-oriented fault simulation of integrated circuits

This is the mature neighbour, and the one a reviewer will know. Its problem is the test coverage of analog and mixed-signal integrated circuits: which of the thousands of likely defects of a layout are detected by a production test.

- *IEEE Standard for Analog Defect Modeling and Coverage* (IEEE Std 2427, 2025, doi:10.1109/ieeestd.2025.11343929) defines a defect coverage accounting method, and states that coverage depends on detectability, process variations, defect characteristics and redundancy. It also defines the terms.
- *Analog Defect Injection and Fault Simulation Techniques: A Systematic Literature Review* (IEEE TCAD, 2023, doi:10.1109/tcad.2023.3298698) surveys the field, driven by the fault-injection requirement of ISO 26262.
- *Practical random sampling of potential defects for analog fault simulation* (ITC 2014, doi:10.1109/test.2014.7035281) and *Efficient Analog Defect Simulation* (ITC 2019, doi:10.1109/itc44170.2019.9000141) deal with the size of the defect universe: likelihood-weighted sampling with confidence intervals, and defect collapsing. The second reports results on an ITC'17 benchmark made of a bandgap reference and an LDO voltage regulator.
- *Analog Fault Simulation - a Hot Topic!* (ETS 2020, doi:10.1109/ets48528.2020.9131581) names the two open problems of that field: no accepted fault model, and simulation time.
- For functional safety: *Fault grouping for fault injection based simulation of AMS circuits in the context of functional safety* (SMACD 2016, doi:10.1109/smacd.2016.7520721), *A Layout-Based Method for Analog Fault Injection in the Context of Functional Safety* (IEEE Access 2025, doi:10.1109/access.2025.3585636), *Verilog-A Implementation of Generic Defect Templates for Analog Fault Injection* (2023, doi:10.1145/3583781.3590317). The last one says that each EDA tool has its own proprietary way of defining and injecting defects.

Consequences for `spicefault`:

- The terms *defect*, *fault*, *coverage* and *detectability* have a standard meaning there. The manuscript must use them consistently with IEEE Std 2427 or say where it differs. In particular, "fault coverage" in that community means the fraction detected; what `spicefault` calls structural coverage (the fraction of the universe that was simulated) needs its qualifier every time.
- Sampling the fault universe with likelihood weights and confidence intervals is prior art. `spicefault` simulates the whole universe it generates and does not weight by likelihood; that is a limitation to state, and weighting is an extension.
- The difference is one of object and scope: those methods work at transistor level, with layout-derived defect likelihoods, commercial simulators and production test as the goal. `spicefault` works at component level on board-level circuits, with open tools, and its goal is a reproducible experiment and its reliability metrics. The manuscript should say so in those terms and not claim novelty in fault simulation itself.

### 2.2 Earlier tools for fault simulation in SPICE

- *A virtual test-bench for analog circuit testability analysis and fault diagnosis* (AUTOTESTCON 1998, doi:10.1109/autest.1998.713467): a tool that produces testability metrics and diagnostic information from SPICE descriptions, and considers robustness to parameter variations.
- *FDSAC-SPICE: fault diagnosis software for analog circuit based on SPICE simulation* (Proc. SPIE 2009, doi:10.1117/12.855758): fault modelling, injection, simulation and a fault dictionary in one environment, with coverage statistics tied to specifications and circuits with tolerance.
- *Rapid Frequency-domain Analog Fault Simulation Under Parameter Tolerances* (DAC 1997, doi:10.1109/dac.1997.597157).
- *Statistical Test Development for Analog Circuits Under High Process Variations* (IEEE TCAD 2007, doi:10.1109/tcad.2007.891373): a fault injection and simulation technique for the probabilistic detection of faults under process variations.
- *Methods for Fault Injection Simulations and Reliability Analysis in SPICE* (MIEL 2025, doi:10.1109/miel66332.2025.11261061): by its abstract, a discussion of prerequisites and expected outcomes of fault insertion in hardware-level simulation for safety verification.

So the idea of an environment that joins fault models, injection, simulation and analysis is at least twenty-five years old, and so is the concern with tolerances. None of these abstracts mentions available code, reproducibility of the samples, or the treatment of failed simulations; whether the papers do must be read in the full texts. **The novelty claim cannot be "a framework for fault injection in SPICE".**

### 2.3 Open tools around SPICE

Q2 and Q8 found open tools that drive SPICE for other purposes: *Scripts for Easier Use of Spice (SEUS)* (JOSS 2020, doi:10.21105/joss.02183), a Perl package that creates batches of netlists for Monte Carlo simulation with ngspice; Python and LTspice frameworks for EMC filter optimisation (2023, 2024); open analog design flows. The open fault-injection frameworks found by Q8 are for digital designs and FPGAs.

No open, documented framework for fault injection in analog circuits at component level was found by these queries. This is the gap the software fills, with the limits of §1: a tool published only as a repository would not appear here.

### 2.4 Fault diagnosis with machine learning: the consumers of such data

Q3 found 172 works, growing from 2 to 7 a year before 2019 to 15 to 20 a year since 2024. They simulate a circuit with faults, extract features and train a classifier. Of the 137 with an abstract:

| Mentioned in title or abstract | Works |
|---|---|
| Sallen–Key filter | 85 |
| Four-op-amp biquad filter | 45 |
| Both of the above | 28 |
| Leapfrog filter | 14 |
| State-variable filter | 6 |
| Tolerance | 26 |
| Monte Carlo | 17 |
| PSpice or OrCAD | 6 |
| ngspice or LTspice | 0 |
| Code or data said to be available | 1 dataset (five records of it) |

The one available dataset, *DiffDA-Net benchmark analog circuit fault diagnosis datasets* (Figshare 2026, doi:10.6084/m9.figshare.32260188), describes the Sallen–Key band-pass filter and the four-op-amp biquad high-pass filter as "two widely used benchmark analog circuits", simulated with PSpice Monte Carlo analysis, with 5 % resistors, 10 % capacitors, parametric deviations of 25 % and 50 %, and a pulse as excitation.

Consequences:

- **This literature is where the framework is needed.** Its datasets are generated with a proprietary simulator, by procedures described in a paragraph, and are almost never shared. That is the reproducibility problem of SCIENTIFIC_SCOPE.md §1, and it can be stated with these counts. The counts come from abstracts: a paper may share code without saying so there.
- **The benchmark circuits are the Sallen–Key band-pass and the four-op-amp biquad high-pass.** The state-variable filter proposed earlier is rare in this literature (6 works) and is replaced by the biquad.
- **The usual set-up is known**: 5 % and 10 % tolerances, faults of 25 % and 50 %, pulse response. Validating with the same set-up makes the results comparable.

### 2.5 Tolerance and the limits of detectability

The effect of tolerance on parametric fault detection is established, and two results are close to what `spicefault` reports:

- *Test limitations of parametric faults in analog circuits* (IEEE TIM 2003, doi:10.1109/tim.2003.818541) shows that many parameter faults are undetectable whatever the test method, and that the smallest detectable fault is often two to five times the normal parameter drift.
- *Methods of Handling the Tolerance and Test-Point Selection Problem for Analog-Circuit Fault Diagnosis* (IEEE TIM 2010, doi:10.1109/tim.2010.2050356) and *Coefficient-Based Test of Parametric Faults in Analog Circuits* (IEEE TIM 2006, doi:10.1109/tim.2005.861490) handle tolerance in diagnosis and test.

So the observation that a parametric fault inside the tolerance band cannot be detected, and the idea of a minimum detectable deviation, are not new. What `spicefault` adds is a way to compute them for any circuit and fault list, as a report of the campaign (`tolerance_overlap`, `minimum_detectable`, `robustness`), with intervals. The manuscript must cite the 2003 result when it presents these.

### 2.6 Where the two benchmark filters come from

Read in the full texts.

- The Sallen–Key band-pass filter, with 5 % resistors, 10 % capacitors, faults of ±50 % and a single 5 V pulse of 10 µs as stimulus, is in *Neural-network based analog-circuit fault diagnosis using wavelet transform as preprocessor* (Aminian and Aminian, IEEE Trans. Circuits Syst. II, 2000, doi:10.1109/82.823545), which takes its circuits from *Linear circuit fault diagnosis using neuromorphic analyzers* (Spina and Upadhyaya, same journal, 1997, doi:10.1109/82.558453; not read). Its values are R1 = 5.18 kΩ, R2 = 1 kΩ, R3 = 2 kΩ, R4 = R5 = 4 kΩ, C1 = C2 = 5 nF, for a centre frequency of 25 kHz. Each fault is simulated with the other components varying within tolerance. Its second circuit is not the biquad used later: it is a two-stage low-pass filter with 26 resistors.
- The four-op-amp biquad high-pass filter, with a cut-off of 10 kHz, is in *Analog fault diagnosis of actual circuits using neural networks* (Aminian, Aminian and Collins, IEEE Trans. Instrum. Meas., 2002, doi:10.1109/TIM.2002.1017726). That paper uses another Sallen–Key filter, with R1 = 1 kΩ and R2 = 3 kΩ, which it calls a 160 kHz filter; with those values and the same topology the centre frequency computes to about 26 kHz, so the figure may be in rad/s. This has not been confirmed.
- So two Sallen–Key circuits go by the name of the benchmark. *Application of DBN and GWO-SVM in analog circuit fault diagnosis* (Sci. Rep. 2021, doi:10.1038/s41598-021-86916-6) uses the one of 2002; *Enhanced analog circuit fault diagnosis via continuous wavelet transform and dual-stream convolutional fusion* (Sci. Rep. 2025, doi:10.1038/s41598-025-02596-6) prints R4 = 2.8 kΩ, which is in neither. `validation/` has the Sallen–Key filter of 2000 and the biquad of 2002.
- How those two recent studies generate their data: the first with PSpice, of unstated version, by a Monte Carlo analysis of 240 instances per fault class; the second with Multisim 14.0.593 and 200 Monte Carlo runs. Neither states a seed or what was done with simulations that did not converge; their data are said to be in the article and available on request.
- The paper of 2002 compares simulation with hardware: features measured on the built circuits lie closer together and overlap more across fault classes than simulated ones, which it attributes to simulations not covering the tolerance of every component, and it cannot separate the fault-free class from one of the faults (R2 high) within tolerances. This is evidence for the study of tolerance and detectability.

IEEE Std 2427-2025 was published on 9 January 2026: it is a standard, no longer a draft.

## 3. Positioning

What the search supports, and what it does not:

| Claim | Supported? |
|---|---|
| Fault injection and fault simulation of analog circuits in SPICE are new | No. Decades of work, a standard, commercial tools |
| Treating tolerance together with faults is new | No. Established since the 1990s |
| An open, documented, tested framework for component-level fault experiments on analog circuits fills a gap | Yes, within the limits of the search |
| The diagnosis literature generates its data in ways that are hard to reproduce | Yes, from its abstracts: proprietary simulators, one shared dataset among 172 works |
| Reproducibility of samples under parallel execution, a record of failed simulations and per-sample provenance are contributions | Not contradicted: no abstract found mentions them. To be confirmed in the full texts of §2.1 and §2.2 |
| Reliability metrics with stated estimators and intervals, computed from the campaign | Partly. Coverage with confidence intervals exists in the defect-oriented literature; the combination with detectability against tolerance and operating condition for board-level circuits was not found |

The contribution to defend is therefore methodological and practical, not a new kind of simulation: an explicit, reproducible experiment (fault, variation, condition, seed, status, provenance) and the reliability quantities derived from it, in an open implementation, demonstrated on the circuits the diagnosis literature already uses. This matches SCIENTIFIC_SCOPE.md §5 and narrows it: the claims about the fault abstraction should be worded as an engineering choice, since fault abstractions as such are prior art.

## 4. What remains to do

1. Read the full text of the works of §2.1 and §2.2, in particular the 2023 systematic review, the IEEE standard and the two tools, to confirm what they do about reproducibility, failed simulations and provenance.
2. Establish what PySpice and spicelib provide, from their documentation and by running them, for the comparison table.
3. Follow citations from the 2023 review and from the two ITC papers.
4. Done: the schematics and component values of the two benchmark filters are those of their sources (§2.6).
5. Repeat the search in Scopus or Web of Science before submission.
