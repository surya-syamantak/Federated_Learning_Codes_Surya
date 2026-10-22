# DP comparison: v10 (Laplace DP) vs v13 (no DP)

`compare_dp.py` compares the communication, computation and averaged validation metrics of two FedRF-Diff
versions that sit next to this folder:

| Key | Folder | Privacy |
|---|---|---|
| v10 | `Fed_Code_v10_Schnorr_LaplaceDP_eps5` | Laplace DP on every transmitted latent vector, ε_t ~ U[2, 3] per round |
| v13 | `Fed_Code_v13_Schnorr_NoDP` | no differential privacy |

Both use the same seed (42), data split, learning settings and Schnorr ZKP
(2048-bit group); they differ only in the DP. The script checks this and
warns otherwise.

## Usage

```bash
python compare_dp.py                                    # saved runs + DP overhead benchmark
python compare_dp.py --no-benchmark
python compare_dp.py --rtt-ms 20 --uplink-mbps 50 --downlink-mbps 100   # another network
```

Each version needs its saved run (`outputs/run/costs.csv`, `history.csv`,
`zkp_times.csv`, `config.json`); make one with `python -m fedrf_diff` in that
folder.

## What is compared

**Communication.** The bytes of every client and round are split into their
parts: uplink = latent matrix, training labels, loss + TP/FP/TN/FN, ZKP proof;
downlink = global model M_G, alignment A_c, pseudo-labels, ZKP verdict. All
parties run in one process, so the communication time is modelled from these
bytes:

    time = round trips × RTT + 8 × bytes_up / uplink + 8 × bytes_down / downlink

(defaults: RTT 50 ms, uplink 20 Mbit/s, downlink 50 Mbit/s, 3 round trips per
round in both versions). Clients communicate in parallel, so a round takes as
long as its slowest client. Reported per client and round (min, max, mean),
per round, and in total until each run stopped.

**Computation.** From each run's `costs.csv`: per round, the clients' encoding
(RF kernel + diffusion map, + DP in v10), ZKP proof generation, validation
and decoding + retraining, and the server's learning and ZKP verification;
min, max and mean per round, and the total until each run stopped.

**DP overhead (benchmark).** v10's own Laplace mechanism
(`fedrf_diff/privacy.py`: L1 clipping + Laplace noise on the patient rows,
clipping of the landmark rows, exactly as in `Client.encode`) is timed on this
machine on latent matrices of the clients' sizes, 30 rounds × 4 clients. This
measures the DP cost directly instead of reading it off two separate runs.

**End to end.** Rounds until the run stopped, total computation +
communication time until then, and the Global Server's accuracy for context.

**Averaged metrics.** The seven validation metrics (accuracy, precision,
recall, specificity, F1 score, MCC, Cohen's κ) averaged over each run's rounds
(mean, min, max, std), for every client and the Global Server (all clients'
validation patients pooled), and the difference No DP − DP.

## Outputs (`outputs/`)

| File | Content |
|---|---|
| `communication_summary.csv`, `communication_per_round.csv` | bytes by message part and modelled communication time, per version and per round/client |
| `computation_summary.csv`, `computation_per_round.csv` | computation parts per round, statistics and totals |
| `dp_overhead_benchmark.csv` | every timing of v10's Laplace mechanism |
| `comparison_summary.csv`, `comparison_summary.tex` | the key numbers, one column per version (LaTeX, booktabs) |
| `metrics_summary.csv`, `metrics_summary.tex` | the seven metrics averaged over the rounds (mean, min, max, std) per version and client; LaTeX table of the means |
| `figures/` | 600-dpi PNG + vector PDF of each figure |
| `../comparison_log.txt` | the terminal output of the run |

Figures (serif fonts, double-column width 7.16 in, Okabe–Ito colour-blind-safe
colours, a hatch for v10 so the versions do not rely on colour alone):

| Figure | Panels |
|---|---|
| `fig1_communication_comparison` | (a) communication time per round, (b) cumulative communication until each run stopped, (c) bytes per client and round by message part, partitioned uplink / downlink (log axis) |
| `fig2_computation_comparison` | (a) computation per round, (b) cumulative computation until each run stopped, (c) computation breakdown per round, partitioned client side / server side / total (log axis, whiskers = min..max) |
| `fig3_dp_overhead` | (a) time of the Laplace mechanism per client and round (benchmark), (b) its share of the encoding step and of all computation |
| `fig4_end_to_end` | (a) rounds until stop, (b) total computation + communication time until stop, (c) Global Server accuracy |
| `fig5_averaged_metrics` | (a) Global Server and (b)–(e) each client: the seven metrics averaged over the rounds, both versions side by side (whiskers = min..max over the rounds); (f) heat map of the difference No DP − DP for every client and metric |

## Results (saved runs, seed 42, 4 clients)

| | Laplace DP (v10) | No DP (v13) |
|---|---|---|
| Rounds | 30 (not converged) | 7 (converged) |
| Uplink per client and round | 10.7 KB | 10.7 KB |
| Downlink per client and round | 468.3 KB | 472.5 KB |
| Communication per client and round (ms, mean) | 229.2 | 229.9 |
| Communication until stop (s, clients in parallel) | 6.96 | 1.63 |
| Data transferred until stop (MB) | 57.5 | 13.5 |
| Computation per round, all clients + server (s, mean) | 4.72 | 3.51 |
| Computation until stop (s) | 141.6 | 24.6 |
| Laplace mechanism per client and round (ms, benchmark) | 0.05 | 0 |
| Global Server accuracy (mean over the rounds) | 0.806 | 0.844 |

Validation metrics averaged over the rounds (v10: 30 rounds, v13: 7 rounds):

| Client | Version | Accuracy | Precision | Recall | Specificity | F1 | MCC | Cohen's κ |
|---|---|---|---|---|---|---|---|---|
| Global Server | Laplace DP (v10) | 0.806 | 0.854 | 0.771 | 0.845 | 0.808 | 0.618 | 0.612 |
| | No DP (v13) | 0.844 | 0.872 | 0.828 | 0.861 | 0.850 | 0.689 | 0.687 |
| Client 1(Hungary) | Laplace DP (v10) | 0.862 | 0.733 | 0.954 | 0.814 | 0.827 | 0.736 | 0.717 |
| | No DP (v13) | 0.884 | 0.748 | 0.992 | 0.829 | 0.853 | 0.781 | 0.760 |
| Client 2(Switzerland) | Laplace DP (v10) | 0.698 | 0.994 | 0.689 | 0.900 | 0.807 | 0.270 | 0.158 |
| | No DP (v13) | 0.825 | 0.968 | 0.844 | 0.429 | 0.900 | 0.213 | 0.196 |
| Client 3(VA Long Beach) | Laplace DP (v10) | 0.855 | 0.939 | 0.865 | 0.822 | 0.899 | 0.652 | 0.637 |
| | No DP (v13) | 0.909 | 0.961 | 0.915 | 0.889 | 0.938 | 0.772 | 0.768 |
| Client 4(Cleveland) | Laplace DP (v10) | 0.783 | 0.825 | 0.685 | 0.868 | 0.746 | 0.568 | 0.559 |
| | No DP (v13) | 0.804 | 0.845 | 0.711 | 0.886 | 0.772 | 0.610 | 0.603 |

How to read this:

* **Per round, DP costs (almost) nothing.** DP does not change any message
  size (the noisy latents are float64 like the clean ones; the only
  difference, ≈ 4 KB in the downlink, is the size of the pickled global
  model), so the communication time per round is the same (≈ 229 ms). The
  Laplace mechanism itself takes ≈ 0.02–0.07 ms per client and round, under
  0.01 % of the encoding step.
* **Over the whole run, DP is expensive because it prevents convergence.**
  With noise the latent representation never settles, so v10 runs to the
  30-round limit while v13 converges after 7 rounds: 4.3 × the communication
  (6.96 s vs 1.63 s, 57 MB vs 14 MB) and 5.8 × the computation
  (141.6 s vs 24.6 s), for lower accuracy (0.806 vs 0.844).
* **Metrics.** Without DP every metric of the Global Server is higher
  (accuracy +0.038, F1 +0.041, MCC +0.070, κ +0.075), and so is every metric
  of Hungary, VA Long Beach and Cleveland. Switzerland is the exception: it
  has only 22 validation patients, one of them healthy, so its specificity is
  either 0 or 1 in each round (0.900 vs 0.429 is 27/30 vs 3/7 rounds right on
  that one patient) and its precision and MCC follow; read its numbers with
  care.
* The two saved runs were made at different times, so their per-round
  computation (4.72 s vs 3.51 s) also contains machine-load differences: the
  learning code is identical and the DP step takes a fraction of a
  millisecond, as the benchmark shows. Rerun both versions back to back on an
  idle machine for a fair per-round computation comparison.
