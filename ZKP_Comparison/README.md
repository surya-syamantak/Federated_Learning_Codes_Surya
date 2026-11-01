# ZKP comparison: v10 Schnorr vs v11 Pedersen vs v12 Proposed Pedersen

`compare_zkp.py` compares the zero-knowledge proofs of the three FedRF-Diff
versions that sit next to this folder:

| Key | Folder | ZKP |
|---|---|---|
| v10 | `Fed_Code_v10_Schnorr_LaplaceDP_eps5` | Schnorr proof on the update (RFC 3526 2048-bit group), non-interactive |
| v11 | `Fed_Code_v11_Pedersen_LaplaceDP_eps2-3` | Pedersen commitment + Σ-protocol + 64-bit range proof (2048/256-bit group), Fiat–Shamir |
| v12 | `Fed_Code_v12_ProposedPedersenZKP_LaplaceDP_eps2-3` | Proposed (chained) Pedersen authentication on scalars (2048/256-bit group), interactive challenge |

The three saved runs share the seed (42), data split, DP budgets
(ε_r ~ U[2, 3]), 30 rounds and 4 clients, so they do the same learning work
and differ only in the ZKP. The script checks this and warns otherwise.

## Usage

```bash
python compare_zkp.py                                   # saved runs + same-machine benchmark (30 rounds)
python compare_zkp.py --no-benchmark                    # saved runs only
python compare_zkp.py --rtt-ms 20 --uplink-mbps 50 --downlink-mbps 100   # another network
```

Requirements: numpy, pandas, matplotlib (and gmpy2 for the benchmark speed),
the same packages as the FedRF-Diff versions. Each version must have its saved
run (`outputs/run/zkp_times.csv`, `costs.csv`, `config.json`); make one with
`python -m fedrf_diff` in that folder.

## What is compared

**1. ZKP times (min, max, mean; also median and std).** Authentication time
(whole session), proof generation time (client side) and proof verification
time (server side), per client and over all clients, from each saved run's
`zkp_times.csv`.

**2. Communication.** The simulation runs all parties in one process, so
communication time is modelled from the bytes each version actually sends:

    time = round trips × RTT + 8 × bytes_up / uplink + 8 × bytes_down / downlink

(defaults: RTT 50 ms, uplink 20 Mbit/s, downlink 50 Mbit/s). One ZKP session
needs 1 round trip in v10 and v11 (non-interactive: proof up, verdict down)
and 2 in v12 (C_r and T up, challenge e down, responses up, verdict down).
The learning messages (latents up; M_glob, R_k and labels down; metrics up) take
2 round trips in every version. Reported: ZKP bytes per session, ZKP
communication time per session (min, max, mean), total communication time per
client and round, the ZKP share of time and bytes, and totals over the run.

**3. Computation.** From each saved run's `costs.csv`: per round, all clients'
computation (encoding + DP, ZKP proof generation, validation, decoding +
retraining) and the server's (learning + ZKP verification); the ZKP part
(generation + verification) with min, max, mean; and the ZKP share of the
total computation.

**4. Same-machine benchmark.** Times from three separate runs include
run-to-run noise (other programs, CPU frequency). The benchmark therefore
also loads each version's own `fedrf_diff/zkp.py` and times the three
implementations in one process, interleaved round by round (the order rotates
every round), with updates of the same sizes as in the runs. Use these numbers
when you want the fairest comparison of the ZKPs themselves.

## Outputs (`outputs/`)

| File | Content |
|---|---|
| `zkp_time_summary.csv` | min, max, mean, median, std (ms) of the 3 ZKP times per version and client (+ all clients), saved runs |
| `benchmark_zkp_times.csv`, `benchmark_zkp_time_summary.csv` | every benchmark session and its summary |
| `communication_summary.csv`, `communication_per_round.csv` | ZKP and total communication, per version and per round/client |
| `computation_summary.csv`, `computation_per_round.csv` | ZKP and total computation, per version and per round |
| `comparison_summary.csv`, `comparison_summary.tex` | one row (column in LaTeX, booktabs) per version: the key numbers |
| `figures/` | 600-dpi PNG + vector PDF of each figure |
| `../comparison_log.txt` | the terminal output of the run |

Figures (serif fonts, double-column width 7.16 in, Okabe–Ito colour-blind-safe
palette, a hatch per version so no series relies on colour alone, log scales
where the versions differ by orders of magnitude):

| Figure | Panels |
|---|---|
| `fig1_zkp_times_comparison` | (a) authentication, (b) proof generation, (c) proof verification time per client and all clients; bar = mean, whiskers = min..max (saved runs) |
| `fig2_zkp_times_per_round` | the same three times per round: mean over clients, band = min..max |
| `fig3_communication_comparison` | (a) ZKP bytes per session up/down, (b) ZKP communication time per session (latency + transmission), (c) total communication time per client and round (learning + ZKP) |
| `fig4_computation_comparison` | (a) ZKP computation per round, (b) ZKP share of all computation, (c) computation breakdown per round, partitioned into learning parts (encoding + DP, validation, decoding + retraining, server learning), ZKP parts (proof generation, proof verification) and totals; one group of version bars per part on a log axis, bar = mean, whiskers = min..max over the rounds, mean printed on each bar |
| `fig5_benchmark_zkp_times` | same-machine benchmark: box plots of the three ZKP times |
| `fig6_benchmark_zkp_times_per_client` | same-machine benchmark: as fig1 |

## Reading the results

Results with the saved runs in this folder (all 120 sessions per version
valid; benchmark: 30 rounds × 4 clients per ZKP, all valid).

ZKP times per client and round (ms), all clients, min / max / mean:

| | Schnorr (v10) | Pedersen (v11) | Proposed Pedersen (v12) |
|---|---|---|---|
| Authentication, saved runs | 6.29 / 8.56 / **6.51** | 89.26 / 152.02 / **101.56** | 1.55 / 20.85 / **5.64** |
| Proof generation, saved runs | 2.79 / 4.96 / **2.90** | 28.32 / 49.70 / **32.46** | 0.29 / 8.42 / **1.08** |
| Proof verification, saved runs | 3.50 / 5.04 / **3.61** | 60.15 / 114.46 / **69.10** | 1.26 / 12.87 / **4.55** |
| Authentication, benchmark | 4.59 / 14.04 / **6.36** | 94.26 / 237.47 / **119.23** | 1.39 / 5.50 / **1.90** |
| Proof generation, benchmark | 2.03 / 6.11 / **2.81** | 29.48 / 78.59 / **37.87** | 0.22 / 1.58 / **0.32** |
| Proof verification, benchmark | 2.55 / 8.07 / **3.55** | 64.19 / 164.01 / **81.36** | 1.18 / 4.49 / **1.59** |

Communication (RTT 50 ms, 20 / 50 Mbit/s):

| | v10 | v11 | v12 |
|---|---|---|---|
| ZKP bytes per session (up / down) | 512 / 1 | 55,872 / 1 | 576 / 33 |
| ZKP round trips | 1 | 1 | 2 |
| ZKP communication per session (ms) | 50.2 | 72.3 | 100.2 |
| Total communication per client and round (ms) | 229 | 251 | 279 |
| ZKP share of the bytes sent (%) | 0.11 | 10.5 | 0.13 |

Computation:

| | v10 | v11 | v12 |
|---|---|---|---|
| ZKP computation per round, 4 clients + server (ms, mean) | 26.1 | 406.2 | 22.5 |
| ZKP share of all computation (%) | 0.55 | 9.18 | 0.18 |

How to read this:

* **v12 has the cheapest ZKP computation.** On the same machine (benchmark)
  an authentication takes 1.9 ms against 6.4 ms for v10 and 119 ms for v11:
  v12 needs a few exponentiations with 256-bit exponents (fixed-base tables
  for g and h), v10 uses full 2048-bit exponents, and v11 proves 64 bits of
  range (an OR-proof per bit).
* **v11 is the most expensive** in computation (≈ 9 % of all computation)
  and in bytes (56 KB per proof, 10 % of all traffic), the price of its range
  proof.
* **v12 is slowest on the network when latency dominates**: its interactive
  challenge needs 2 round trips instead of 1, so with RTT = 50 ms its ZKP
  communication is ≈ 100 ms per session against 50 ms (v10) and 72 ms (v11).
  On a low-latency link this gap shrinks (try `--rtt-ms 5`).
* The saved v12 run's per-round **learning** computation (≈ 12 s against
  ≈ 4.5 s) is not caused by the ZKP: the learning code of the three versions
  is byte-identical and the ZKP is 0.18 % of v12's computation. That run was
  slowed by the machine (other load), which also widened v12's saved-run ZKP
  min–max range; the benchmark, run in one process, does not have this
  problem. Rerun the three versions back to back (or compare the benchmark
  figures 5 and 6) for a fair computation comparison.
