# Comparison: FedRF-Diff v12 vs Federated-Learning

`compare_v12_fl.py` compares two federated random-forest implementations that
sit next to this folder:

| Key | Folder | Design |
|---|---|---|
| v12 | `Fed_Code_v12_ProposedPedersenZKP_LaplaceDP_eps2-3` | clients send DP-noised diffusion-map latents; server aligns them (Procrustes) and trains a global RF on the latents; clients decode and retrain |
| FL | `Federated-Learning_ProposedZKP` | clients send whole random forests as explicit trees (Laplace DP on the leaf vectors); server selects a global forest tree by tree; clients tune, refine and run their own convergence test |

Both use seed 42, 4 hospital clients, ε_t ~ U[2, 3] per round, 30 rounds and
**the same proposed Pedersen ZKP** for client authentication: the same
`zkp.py`, group and λ = 2 (chained commitment `C_t = g^{v_t} h^{r_t} (C_{t-1})^λ`,
server-sent challenge `c`, check `g^{r_v} h^{r_s} = T (C_t / C_{t-1}^λ)^c`). The
comparison therefore isolates the two learning designs. At start-up the script
reads both projects' `zkp.py` and prints whether they implement the same
protocol (it would flag the original `Federated-Learning`, whose ZKP is an
Okamoto proof for a registered key made non-interactive with Fiat–Shamir).

`Federated-Learning_ProposedZKP` is `Federated-Learning` with only its ZKP
replaced; its learning results (metrics, convergence log, DP sweep) are
identical to the original's.

## Usage

```bash
python compare_v12_fl.py
python compare_v12_fl.py --rtt-ms 20 --uplink-mbps 50 --downlink-mbps 100   # another network
```

Each project needs its saved run in `outputs/run/` (`python -m fedrf_diff` in
its folder).

## What is compared and how

The two projects log different quantities, so they are mapped onto shared
categories.

**Computation** (seconds per round):

| Category | v12 | FL |
|---|---|---|
| Client: encoding / serialization | RF kernel + diffusion map + DP | forest serialization |
| Client: ZKP proof generation | proof generation | proof generation |
| Client: local model update | decoding + RF retraining | local tuning (5-fold CV grid search) + refinement |
| Client: evaluation / convergence check | validation loss and counts | three-part convergence test |
| Server: aggregation / learning | alignment + M_G training + stopping rule | tree-level aggregation |
| Server: update validation | – | structural validation of every received tree |
| Server: ZKP proof verification | all clients' proofs | all clients' proofs |

**Communication** (per round). Bytes come from each project's own cost table.
For a client, "sent" is its uplink and "received" its downlink; for the Global
Server, "received" is everything the clients sent and "sent" everything it
sent. Communication time is modelled:

    time = round trips × RTT + 8 × bytes sent / uplink + 8 × bytes received / downlink

(defaults RTT 50 ms, uplink 20 Mbit/s, downlink 50 Mbit/s). Round trips per
round: the shared ZKP session takes 2 in both (C_t and T up / c down; r_v and
r_s up / verdict down); v12 adds 2 for its learning messages (latents and
labels up / M_G, A_c and labels down; metrics up / ack) = 4; FL sends its forest
with the responses and receives the global forest with the verdict, and adds 1
for the convergence flag / decision = 3. Both projects count the ZKP messages
(576 B up, 33 B down per session) in their bytes. The Global Server's time per
round is that of its slowest client.

**Metrics.** Accuracy, precision, recall, specificity, F1 score, MCC and
Cohen's κ of each project's *global* model (v12: M_G; FL: G^(t)) on every
client's validation patients and on all of them pooled (Global Server),
averaged over the 30 rounds and in the last round (30).

## Terminal output

1. computation per round averaged over the rounds (mean, min, max) for each
   client and the Global Server, the FL / v12 ratio, and the means by category;
2. communication per round averaged over the rounds for each client and the
   Global Server: KB sent, received and in total, modelled time (mean, min,
   max) and data over the run;
3. the seven metrics averaged over the rounds, per client and Global Server,
   v12, FL and FL − v12;
4. the seven metrics of the last round, in the same layout.

## Outputs (`outputs/`)

| File | Content |
|---|---|
| `computation_summary.csv/.tex`, `computation_per_round.csv` | computation per entity (mean by category, total mean/min/max, total over the run) |
| `communication_summary.csv/.tex`, `communication_per_round.csv` | KB sent and received, modelled time (mean/min/max), totals |
| `metrics_averaged.csv/.tex` | averaged metrics (mean, min, max per project) and FL − v12 |
| `metrics_last_round.csv/.tex` | last-round metrics and FL − v12 |
| `figures/` | 600-dpi PNG + vector PDF of each figure |
| `../comparison_log.txt` | the terminal output |

Figures (serif fonts, double-column width, Okabe–Ito colour-blind-safe
colours with a hatch per project, log scales where the projects differ by
orders of magnitude):

| Figure | Table it shows | Panels |
|---|---|---|
| `fig1_computation_comparison` | computation | (a) computation per round of each client and the Global Server; (b) client-side categories; (c) server-side categories (bars = mean, whiskers = min..max) |
| `fig2_communication_comparison` | communication | (a) KB sent, (b) KB received, (c) modelled communication time per round, for each client and the Global Server |
| `fig3_averaged_metrics` | averaged metrics | (a) Global Server and (b)–(e) each client: the seven metrics averaged over the rounds (whiskers = min..max); (f) heat map FL − v12 |
| `fig4_last_round_metrics` | last-round metrics | same layout, values of round 30 |

## Results (saved runs, 30 rounds each)

Computation per round (s, mean over the rounds):

| | Hungary | Switzerland | VA Long Beach | Cleveland | Global Server |
|---|---|---|---|---|---|
| v12 | 0.77 | 0.54 | 0.63 | 1.48 | 0.36 |
| FL | 13.57 | 12.66 | 13.27 | 13.27 | 0.12 |

ZKP part (same protocol in both): proof generation 0.3 ms (v12) and 2.8 ms (FL)
per client and round, verification of all four clients 5.7 ms (v12) and 24.5 ms
(FL) per round; the differences come from machine load during the two runs,
not from the protocol.

Communication per round (mean): v12 clients send 6.5–16.9 KB and receive
≈ 457 KB (the pickled global model), ≈ 278–282 ms; FL clients send 66–112 KB
(their forests) and receive ≈ 180 KB (the global forest), ≈ 207–226 ms. Over
the run v12 moves 54.8 MB through the server, FL 30.9 MB.

Global Server metrics:

| | Accuracy | Precision | Recall | Specificity | F1 | MCC | Cohen's κ |
|---|---|---|---|---|---|---|---|
| v12, mean over 30 rounds | 0.806 | 0.854 | 0.771 | 0.845 | 0.808 | 0.618 | 0.612 |
| FL, mean over 30 rounds | 0.845 | 0.834 | 0.896 | 0.781 | 0.864 | 0.686 | 0.683 |
| v12, round 30 | 0.846 | 0.872 | 0.833 | 0.860 | 0.852 | 0.692 | 0.691 |
| FL, round 30 | 0.838 | 0.816 | 0.912 | 0.747 | 0.861 | 0.674 | 0.668 |

How to read this:

* **Computation.** FL's clients spend ≈ 9–23 × more time per round than v12's
  (≈ 13 s vs 0.5–1.5 s), almost all in the local model update, because FL
  re-runs a 5-fold cross-validated hyper-parameter search every round. Its
  server is cheaper (0.12 s vs 0.36 s): selecting trees is lighter than v12's
  alignment and global-forest training, although FL adds tree validation. The
  ZKP is a negligible part of both (milliseconds against seconds).
* **Communication.** With the same ZKP, FL still needs one round trip fewer
  per round (3 vs 4) and downloads less (its global forest is smaller than
  v12's pickled global model with its 300 trees) while uploading more (whole
  forests), so its modelled time per round is ≈ 20–25 % lower.
* **Metrics.** Averaged over the rounds FL's Global Server is higher on
  accuracy, recall, F1, MCC and κ, v12 on precision and specificity; in round
  30 the two are within ≈ 0.01–0.02 on accuracy, F1, MCC and κ. Per client,
  FL is much better on Hungary and VA Long Beach, v12 on Cleveland.
  Switzerland's few healthy validation patients (FL's specificity is 0 in
  every round) make its MCC and κ unreliable for both.

**Caveats.** The validation sets differ: v12 keeps duplicate rows, sets 100
landmark patients aside and validates on 214 patients; FL drops duplicates and
validates on 185 (sizes per client are in the figure titles), so metric
differences are not purely due to the method. Computation times come from two
separate runs and include machine-load differences (FL's clients took ≈ 8.5 s
per round in the original `Federated-Learning` run with the same learning
work). Communication time is a model, not a measurement.
