# SMPC comparison: v14 (additive secret sharing) vs v15 (proposed pairwise masking)

`compare_smpc.py` compares the two SMPC schemes of FedRF-Diff:

| Key | Folder | SMPC |
|---|---|---|
| v14 | `Fed_Code_v14_ProposedPedersenZKP_SMPC_NoDP` | n-out-of-n additive secret sharing mod 2^127 − 1: each client sends N − 1 shares to the other clients and one partial sum to the server |
| v15 | `Fed_Code_v15_ProposedPedersenZKP_ProposedSMPC_NoDP` | Adaptive Prime Secure Aggregation with Self-Randomized Pairwise Masking: one-time Diffie–Hellman pairwise keys, prime pool from a public seed, adaptive prime p_t per round, quantization S_q = 10^6, pairwise PRG masks that cancel in the sum; each client sends one masked vector |

Both protect the same values (each client's weighted validation loss and
TP/FP/TN/FN counts; the server learns only their totals) in the same place of
the same pipeline (proposed ZKP, no DP, seed 42). The learning is therefore
identical and the comparison isolates the SMPC.

## Usage

```bash
python compare_smpc.py                                       # saved runs + benchmark (N = 2 ... 32, 50 rounds)
python compare_smpc.py --clients 2 4 8 16 32 64 --benchmark-rounds 100
python compare_smpc.py --no-benchmark
python compare_smpc.py --rtt-ms 20 --uplink-mbps 50 --downlink-mbps 100
```

Each version needs its saved run (`outputs/run/smpc_times.csv`, `costs.csv`,
`history.csv`, `final_metrics.csv`, `config.json`; `python -m fedrf_diff` in
its folder). The benchmark imports each project's own `fedrf_diff/smpc.py`.

## What is compared

1. **SMPC times in the runs**: min, max and mean per round for every client
   (v14: share generation + summation; v15: quantize + streams + self-mask +
   masked vector) and for the Global Server (v14: reconstruction; v15:
   aggregation + dequantization); v15's one-time key setup separately.
2. **SMPC communication** per client and round: bytes, messages, round trips
   and a modelled time `round trips × RTT + 8 × bytes_up / uplink + 8 ×
   bytes_down / downlink` (defaults RTT 50 ms, 20 / 50 Mbit/s). v14 needs 2
   round trips per round (shares to and from the other clients, partial sum to
   the server), v15 1 (masked vector), plus one key exchange once.
3. **Computation**: each run's total per round and the SMPC share of it.
4. **Same-machine benchmark** of both `smpc.py` modules for N = 2 … 32 clients
   (6 values per client, values of the size seen in the runs, schemes
   alternated round by round): client and server time, bytes per client, v15's
   key setup. Fair timings, and the scaling with N.
5. **Metrics**: averaged and last-round metrics of both runs and the
   per-round difference of `L_avg`.
6. **Design properties** side by side (`smpc_properties.csv/.tex`).

## Outputs (`outputs/`)

| File | Content |
|---|---|
| `smpc_time_summary.csv/.tex` | SMPC time min / max / mean per client and Global Server, one-time setup |
| `smpc_communication_summary.csv/.tex` | bytes, messages, round trips, modelled time, one-time setup, totals |
| `computation_summary.csv` | total computation per round and SMPC share |
| `benchmark_smpc.csv`, `benchmark_smpc_summary.csv/.tex` | every benchmark round and the summary per scheme and N |
| `metrics_averaged.csv`, `metrics_last_round.csv`, `metrics_global_server.tex` | metrics of both runs and their difference |
| `smpc_properties.csv/.tex` | design properties |
| `figures/` | 600-dpi PNG + vector PDF |
| `../comparison_log.txt` | the terminal output |

| Figure | Panels |
|---|---|
| `fig1_smpc_times` | (a) SMPC time per round per client and Global Server (mean, min..max, log), (b) mean client time per round, (c) one-time key setup |
| `fig2_smpc_communication` | (a) SMPC bytes up / down per client and round, (b) messages sent / received, (c) modelled SMPC communication time (latency + transmission) |
| `fig3_smpc_scaling_benchmark` | benchmark vs N: (a) client time (and v15 key setup), (b) client bytes, (c) server time |
| `fig4_metrics_same_learning` | Global Server metrics (a) averaged and (b) last round for both, (c) per-round difference of `L_avg` |

## Results

SMPC per client and round, saved runs (N = 4, 7 rounds):

| | v14 additive sharing | v15 proposed masking |
|---|---|---|
| Client time (ms, mean) | 0.020–0.046 | 0.12–0.17 |
| Server time (ms, mean) | 0.010 | 0.014 |
| One-time key setup per client | none | ≈ 1.2 ms, 256 B up, 768 B down |
| Bytes up / down | 384 / 288 | **48 / 0** |
| Messages sent / received | 4 / 3 (3 to other clients) | **1 / 0** |
| Client-to-client messages per round | 12 | **0** |
| Round trips | 2 | **1** |
| Modelled communication time | 100.2 ms | **50.0 ms** |
| SMPC bytes over the run (4 clients) | 18,816 | **5,440** |
| SMPC share of all computation | 0.002 % | 0.018 % |

Same-machine benchmark (mean ms per client and round; bytes per client and round):

| N | v14 client | v15 client | v15 key setup (once) | v14 bytes | v15 bytes |
|---|---|---|---|---|---|
| 2 | 0.013 | 0.036 | 0.59 | 288 | 48 |
| 4 | 0.020 | 0.099 | 1.20 | 672 | 48 |
| 8 | 0.035 | 0.226 | 2.75 | 1,440 | 48 |
| 16 | 0.072 | 0.521 | 6.84 | 2,976 | 48 |
| 32 | 0.346 | 3.835 | 42.1 | 6,048 | 48 |

Metrics: identical averaged and last-round metrics (Global Server accuracy
0.844 / 0.846, F1 0.850 / 0.852, MCC 0.689 / 0.692 in both);
max |L_avg(v15) − L_avg(v14)| = 1.1 × 10⁻⁹ (the two schemes' fixed-point
rounding).

How to read this:

* **Communication: v15 wins clearly.** One 48-byte masked vector per client
  and round, independent of N, no client-to-client traffic and half the round
  trips; v14 sends and receives N − 1 shares per client, so its traffic grows
  linearly with N (6 KB per client at N = 32 against 48 bytes).
* **Computation: v14 is cheaper.** v15 derives one PRG stream (SHA-256) per
  other client and needs a one-time Diffie–Hellman key setup, so its client
  time grows with N and is ≈ 3–11 × v14's; both stay far below a millisecond
  at N = 4 and are negligible next to the learning (≤ 0.02 % of all
  computation).
* **Security model:** v14's shares are information-theoretically hiding; v15
  is computationally secure (PRG, Diffie–Hellman) but tolerates a drop-out
  after masking through its recovery step, which v14 cannot.
* **No effect on learning**: both runs learn the same model.

Caveats: the run times come from two separate runs (machine-load differences,
e.g. total computation per round 5.6 s vs 7.1 s with identical learning); the
benchmark, run in one process, is the fair timing comparison. Communication
time is a model, not a measurement.
