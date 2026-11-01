# FedRF-Diff (Fed_Code_v13_Schnorr_NoDP)

Federated learning with random-forest diffusion maps, Procrustes alignment and
a Schnorr-based zero-knowledge proof on every client update, **without any
differential privacy**. It implements "Algorithm 1: Iterative Closed-Loop
FedRF-Diff Framework" on the combined UCI heart-disease data
(`data/heart_statlog_cleveland_hungary_final.csv`).

This is `Fed_Code_v10_Schnorr_LaplaceDP_eps5` with the Laplace DP removed:

* `fedrf_diff/privacy.py` is gone; clients send their latent vectors exactly
  as computed (no clipping, no noise), on patient and landmark rows alike;
* the stopping rule uses the plain ΔZ (no noise correction) with threshold ε_Z;
* the accuracy-vs-DP-budget sweep, its CSV and figure, and every ε column
  are removed;
* the Schnorr ZKP, the learning code (RF kernel, diffusion map, alignment,
  global model, decoder) and all other settings are unchanged.

## Quick start

```bash
pip install -r requirements-lock.txt   # exact versions behind the results below
python -m fedrf_diff        # seed 42, no DP, Schnorr ZKP, all CSVs and figures
python -m pytest
python make_figures.py     # (re)draw all figures from the saved run, no retraining
```

The ZKP is set in `fedrf_diff/zkp.py` and runs inside the pipeline every
round. DP options (`--epsilon`, `--sigma`, ...) are rejected: this version has
no DP. The learning settings in `fedrf_diff/config.py` are options
(`--max-rounds 40`, `--output-dir outputs/run2`, ...).

**Reproducing the numbers below.** `requirements-lock.txt` pins the package
versions used for them. With other versions (`requirements.txt`) everything
works, but scikit-learn changes can move a patient between training and
validation or change the forests, so the numbers differ slightly. The second
line of the output shows the split; the numbers below come from
`Client 3(VA Long Beach) 140/36, Client 4(Cleveland) 409/103`.

## Clients

| Client | Rows | Unique patients | Disease rate |
|---|---|---|---|
| Client 1(Hungary) | 294 | 293 | 36% |
| Client 2(Switzerland) | 123 | 123 | 93% |
| Client 3(VA Long Beach) | 198 | 197 | 74% |
| Client 4(Cleveland) (incl. Statlog) | 573 | 303 | 46% |

The CSV rows are ordered by source dataset. Statlog (rows 617–886) is a
270-patient subset of Cleveland, so it is merged into Client 4. Duplicates
are kept for training. The 2 VA rows with impossible values (resting bp = 0,
ST slope = 0) are dropped. 100 unique patients are set aside as the public
landmark cohort; each client's remaining data is split 80/20 into train and
validation, with all copies of a patient on the same side.

## Terminal output

The header shows the seed (42), the ZKP group and the stopping thresholds.
Then, for every round:

* **Schnorr ZKP (per client):** Proof (Valid / Invalid), number of vectors,
  *ZKP Authentication Time*, *ZKP Proof Generation Time* and *ZKP Proof
  Verification Time* (ms; the CSV files keep seconds);
* encoding and decoding times per client;
* **validation metrics** of that round's global model for each client and
  for the Global Server: *Accuracy, Precision, Recall, Specificity, F1 Score,
  Matthews Correlation Coefficient (MCC), Cohen's Kappa*;
* the server line: ΔZ (and its threshold), `L_avg`, ΔL, patience and round
  time.

After the last round: the metrics averaged over all rounds, the ZKP time
analysis (min, max and mean per client) and the encoding/decoding time
analysis.

"Global Server" is the global model `M_G` scored on all clients' validation
patients together (summed TP/FP/TN/FN). Clients only report their loss and
TP/FP/TN/FN counts; every metric is computed from those counts.

## Output files (`--output-dir`, default `outputs/run`)

| File | Content |
|---|---|
| `config.json` | the settings used, including `privacy` (`{"mechanism": "none"}`) and `zkp` |
| `history.csv` | per round: ΔZ, `L_avg`, ΔL, timings, and per client (and pooled, `[all]`) loss and the seven metrics |
| `final_metrics.csv` | last round's metrics per client, pooled row and macro average |
| `global_server_metrics.csv` | per round: the Global Server's TP/FP/TN/FN, log-loss, seven metrics, `L_avg`, ΔZ, ΔL |
| `zkp_times.csv` | per round and client: authenticated, vectors, the three ZKP times, ZKP bytes up/down |
| `costs.csv` | per round and client (and Global Server): computation time and its parts, bytes sent and received |

Figures (`figures/`, PNG at 600 dpi plus a vector PDF):

| Figure | Panels |
|---|---|
| `convergence.png` | (a) ΔZ with ε_Z, (b) \|ΔL\| with ε_L, (c) `L_avg`; converged round marked |
| `zkp_times.png` | (a) authentication, (b) proof generation, (c) proof verification time per client per round |
| `computation_cost.png` | (a) computation time per round per client and server, (b) mean per-client breakdown, (c) mean ZKP proof generation time per client |
| `communication_cost.png` | (a) uplink, (b) downlink KB per round per client |

## Schnorr ZKP

`fedrf_diff/zkp.py` (unchanged from v10) runs the Schnorr-based ZKP once per
client and round. The update u_c is everything the client sends: its latent
matrix (patient and landmark rows) and its training labels.

| Step | Who | What | Code |
|---|---|---|---|
| 1 | client | local training → update u_c (SHA-256 of the update, mod q) | `encode_update` |
| 2 | client | random r_c ∈ Z_q | `prove` |
| 3 | client | commitment R_c = g^{r_c} mod p | `prove` |
| 4 | client | proof s_c = r_c + e·u_c mod q; send (R_c, s_c) | `prove` |
| 1 | server | receive (R_c, s_c) and the update; recompute u_c | `authenticate` |
| 2 | server | challenge e = H(R_c ‖ u_c ‖ context) mod q | `challenge` |
| 3 | server | check g^{s_c} = R_c · (g^{u_c})^e mod p: valid → accept, invalid → reject this round | `verify` |

p is the RFC 3526 2048-bit safe prime, q = (p − 1)/2 and g = 4. H = SHA-256;
context = protocol name, client and round, so a proof cannot be replayed for
another client or round. Settings in `zkp.py`: `ENABLED = True`,
`GROUP = "modp2048"` ("test512" is fast but NOT secure).

What the proof guarantees: the server's check uses u_c computed from the
update it received, so an update changed after the proof was made fails. The
protocol has no client secret key, so it checks the update's integrity and
freshness, not the sender's identity.

## Privacy without DP

Without DP the latent vectors leave the client exactly as computed. They are
low-dimensional (d_z = 4) diffusion coordinates, not raw features, but they
are a deterministic function of the patients' data and carry no formal
privacy guarantee. Use this version as the no-DP reference (accuracy upper
bound, cost baseline) against the DP versions.

## Bounded-update enforcement

After ZKP authentication and before aggregation (pipeline Stage 1.6), the server checks every
client update against a public bound (`fedrf_diff/bounds.py`). An update outside the bound is
rejected for that round and never aggregated, just like an update that fails authentication.
The check needs no secrets: it uses only the update and the public protocol parameters.

Always checked: the shapes (`d_z` columns, the agreed number of landmark rows, one label per
patient row), all values finite, labels in {0, 1}.

Without DP the update is the diffusion map Z = sqrt(n) V Lambda^tau, with orthonormal V and
0 <= lambda <= 1. That gives two exact bounds over all n = patients + landmarks rows:
every entry |z| <= sqrt(n), and ||Z||_F^2 <= n d_z.

Outputs:
- each round prints a "Bounded-update enforcement (per client)" table with the outcome, the
  percentage of the bound used, the check time and the reason;
- the end of the run prints a summary;
- `outputs/run/bounds_log.csv` holds every check;
- the history has an `n_out_of_bound` column;
- the Global Server cost includes `bound_check_s`.

To switch the check off for comparison, set `bounds.ENABLED = False` or call
`run(..., bounds_enabled=False)`. Tests: `tests/test_bounds.py`.

## One round

| Stage | Where | What |
|---|---|---|
| 1 | client | RF kernel of `RF_c^(t-1)` over its training rows + landmarks; diffusion map `Z_c = √n V Λ^τ` (constant eigenvector dropped); sent as is (no DP) |
| 1.5 | both | Schnorr proof on each client's update; an invalid proof means the update is rejected this round |
| 2 | server | Procrustes `A_c` from landmark rows `P_c` to `P_ref`; `Z_G` = aligned rows of all clients; `M_G` = random forest on `Z_G` + out-of-bag labels `ŷ`; ΔZ |
| 3 | client | validation loss `L_c`; k-NN decode of `Z_c A_c`; retrain `RF_c` on `D_train ∪ {(x̂, ŷ)}` |
| 4 | server | `L_avg = Σ (n_c/Σn) L_c`, ΔL; stop after 3 consecutive rounds with ΔZ < ε_Z = 0.15 and ΔL < ε_L = 0.02 |

## Results (seed 42, no DP, 2048-bit group)

Split: VA Long Beach 140/36, Cleveland 409/103. The full terminal output is
in `outputs/run/run_log.txt`. **Converged at round 7** (ΔZ fell from 0.53 in
round 2 to 0.09–0.12 from round 4, below ε_Z = 0.15, with ΔL < 0.02 for 3
rounds). Schnorr ZKP: 28/28 proofs valid, no update rejected.

Metrics averaged over the 7 rounds:

| Client | Accuracy | Precision | Recall | Specificity | F1 | MCC | Kappa |
|---|---|---|---|---|---|---|---|
| Client 1(Hungary) | 0.884 | 0.748 | 0.992 | 0.829 | 0.853 | 0.781 | 0.761 |
| Client 2(Switzerland) | 0.825 | 0.968 | 0.844 | 0.429 | 0.900 | 0.213 | 0.196 |
| Client 3(VA Long Beach) | 0.909 | 0.961 | 0.915 | 0.889 | 0.938 | 0.772 | 0.768 |
| Client 4(Cleveland) | 0.804 | 0.845 | 0.711 | 0.886 | 0.772 | 0.610 | 0.603 |
| Global Server | 0.844 | 0.872 | 0.828 | 0.861 | 0.850 | 0.689 | 0.687 |

Mean ZKP times per client and round (ms): authentication 6.7–7.4, proof
generation 2.7–3.4, proof verification 3.6–4.0.

Compared with v10 (Laplace DP, ε_t ~ U[2, 3], same seed): without noise the
representation settles, so the run converges in 7 rounds instead of stopping
at 30 rounds, and the Global Server's mean accuracy is 0.844 instead of 0.806.
Switzerland (22 validation patients, one healthy) is the least stable client.
One seed: repeat over several seeds before reporting.
