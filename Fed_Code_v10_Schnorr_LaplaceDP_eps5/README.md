# FedRF-Diff (Fed_Code_v10_Schnorr_LaplaceDP_eps5)

Federated learning with random-forest diffusion maps, Procrustes alignment,
Laplace differential privacy (pure ε-DP, ε = 5 per round) and a Schnorr-based
zero-knowledge proof on every client update. It implements
"Algorithm 1: Iterative Closed-Loop FedRF-Diff Framework" on the combined UCI
heart-disease data (`data/heart_statlog_cleveland_hungary_final.csv`).

## Quick start

```bash
pip install -r requirements-lock.txt   # exact versions behind the results below
python -m fedrf_diff        # seed 42, Laplace DP (ε = 5 per round) + Schnorr ZKP, all CSVs and figures
python -m pytest
python make_figures.py     # (re)draw all figures from the saved run, no retraining
```

There are no DP or ZKP options on the command line:

* Laplace DP is set in `fedrf_diff/privacy.py` (see
  [Differential privacy](#differential-privacy-laplace));
* the ZKP is set in `fedrf_diff/zkp.py` and runs inside the pipeline every
  round (see [Schnorr ZKP](#schnorr-zkp)).

`--sigma`, `--epsilon`, `--zkp-group`, ... are rejected with a pointer to the
right file. The learning settings in `fedrf_diff/config.py` are still options
(`--max-rounds 40`, `--output-dir outputs/run2`, ...).

A round takes about 5 s; each Schnorr proof takes a few milliseconds even with
the secure 2048-bit group. The run ends with the accuracy-vs-DP-budget sweep
(about 3 minutes).

**Reproducing the numbers below.** `requirements-lock.txt` pins the package
versions used for them (Python 3.11 or newer). With other versions
(`requirements.txt`) everything works, but scikit-learn changes can move a
patient between training and validation or change the forests, so the numbers
differ slightly. The second line of the output shows the split; the README
numbers come from `Client 3(VA Long Beach) 140/36, Client 4(Cleveland) 409/103`.

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

The header shows the seed (42), the ZKP group, the DP ε, L1 clip norm, Laplace scale b,
and the stopping thresholds. Then, for every round:

* **Schnorr ZKP (per client):** Proof (Valid / Invalid), number of vectors,
  *ZKP Authentication Time*, *ZKP Proof Generation Time* and *ZKP Proof
  Verification Time* (ms; the CSV files keep seconds);
* **validation metrics** of that round's global model for each client and
  for the Global Server: *Accuracy, Precision, Recall, Specificity, F1 Score,
  Matthews Correlation Coefficient (MCC), Cohen's Kappa*;
* the server line: ΔZ (and its threshold), `L_glob`, ΔL, patience, round
  time and the DP budget spent so far.

After convergence:

* the metrics averaged over all rounds, per client and for the Global Server;
* the ZKP time analysis: min, max and mean (ms) of each of the three ZKP
  times per client, and how many proofs were valid;
* the accuracy-vs-DP-budget tables (mean over the rounds and final round).

"Global Server" is the global model `M_glob` scored on all clients' validation
patients together (summed TP/FP/TN/FN). Clients only report their loss and
TP/FP/TN/FN counts; every metric is computed from those counts.

## Output files (`--output-dir`, default `outputs/run`)

| File | Content |
|---|---|
| `config.json` | the settings used, including `privacy` and `zkp` |
| `history.csv` | per round: ΔZ, `L_glob`, ΔL, timings, DP budget spent, and per client (and pooled, `[all]`) loss and the seven metrics |
| `final_metrics.csv` | last round's metrics per client, pooled row and macro average |
| `global_server_metrics.csv` | per round: the Global Server's TP/FP/TN/FN, log-loss, seven metrics, `L_glob`, ΔZ, ΔL, ε spent |
| `zkp_times.csv` | per round and client: authenticated, vectors, the three ZKP times, ZKP bytes up/down |
| `costs.csv` | per round and client (and Global Server): computation time and its parts, bytes sent and received, ε spent |
| `dp_budget_sweep.csv` | per per-round ε (and no DP): Laplace scale b, ε spent, mean accuracy over the rounds and final-round accuracy per client and pooled |

Figures (`figures/`, PNG at 600 dpi plus a vector PDF, serif fonts,
colour-blind-safe palette with a distinct marker per client):

| Figure | Panels |
|---|---|
| `convergence.png` | (a) ΔZ with its threshold, (b) \|ΔL\| with ϵ_l, (c) `L_glob`; converged round marked |
| `zkp_times.png` | (a) authentication, (b) proof generation, (c) proof verification time per client per round (grouped bars) |
| `accuracy_vs_dp_budget.png` | (a) accuracy (mean over the rounds) vs per-round ε, incl. no DP; (b) accuracy per round vs cumulative ε spent in the main run (both grouped bars) |
| `computation_cost.png` | (a) computation time per round per client and server, (b) mean per-client breakdown (ZKP proof generation in ms under each client), (c) mean ZKP proof generation time per client (ms) |
| `communication_cost.png` | (a) uplink, (b) downlink KB per round per client |

The accuracy-vs-DP-budget sweep reruns the learning for each ε in
`privacy.EPSILON_SWEEP` with seed 42 and the same number of rounds as the
main run, in parallel processes. The ZKP does not change any learned value
(`tests/test_pipeline.py` checks this), so the sweep runs skip it. With one
seed the final-round accuracy moves a few points between rounds, so panel (a)
uses the mean over the rounds; the CSV keeps both.

## Differential privacy (Laplace)

`fedrf_diff/privacy.py` applies the Laplace mechanism to every latent vector
a client sends, every round:

1. clip: `z_i ← z_i · min(1, C / ‖z_i‖₁)`, so the L1 sensitivity is `Δ₁ = 2C`;
2. noise: `z̃_i = z_i + Lap(0, b)` on every coordinate, `b = Δ₁ / ε`.

By default (`EPSILON = "random"`) the main run (the one that uses the ZKP)
draws a new ε_r uniformly from `RANDOM_RANGE` = [2, 3] every round. The draw
comes from the run seed, so it is reproducible, and the total is ε_1 + … + ε_r.
With a fixed ε = 5 instead, the Laplace scale is b = 2C/ε = 0.4 (noise std
0.57) and each release is (5, 0)-DP (pure DP, no δ). Rounds add up, so after t
rounds each vector has spent 5t. The terminal shows the per-round ε and the
running total every round. The settings:

```python
EPSILON_RANGE = (1.0, 5.0)  # every budget used must lie here
RANDOM_RANGE = (2.0, 3.0)    # ε_r ~ U(RANDOM_RANGE) each round when EPSILON = "random"
EPSILON = "random" # per-round budget (a number, or "random": new ε_r each round)
CLIP_NORM = 1.0    # C, L1 bound (sensitivity 2C)
NOISE_ON_VALIDATION = False   # validation embeddings never leave the client
NOISE_ON_LANDMARKS = False    # landmarks embed the public cohort: clipped, not noised
EPSILON_SWEEP = (1.0, 2.0, 3.0, 4.0, 5.0, None)   # None = no-DP reference
```

Scope of the guarantee: it is per transmitted vector (local DP on what
leaves the client). It does not cover the training labels, which the server
needs to train `M_glob`, nor the influence one patient has on other patients'
vectors through the shared kernel.

**Stopping under DP.** The noise level is public, so the server subtracts its
expected energy from ΔZ, and the ΔZ threshold is `ϵ_z + std/C`
(0.15 + 0.566 = 0.72 at ε = 5). When the noise is as large as the signal the
corrected ΔZ could explode, so its denominator never drops below the signal
energy measured on the noise-free public landmark rows. ϵ_l = 0.02,
patience 3. A run with ε outside [1, 5] stops with an error
(`privacy.check_epsilon`).

## Schnorr ZKP

`fedrf_diff/zkp.py` implements the "Schnorr-based Zero-Knowledge Proof (ZKP)
in Federated Learning" protocol once per client and round. The update u_k is
everything the client sends: its DP-protected latent matrix (patient and
landmark rows) and its training labels.

| Step | Who | What | Code |
|---|---|---|---|
| 1 | client | local training → update u_k (SHA-256 of the update, mod q) | `encode_update` |
| 2 | client | random r_k ∈ Z_q | `prove` |
| 3 | client | commitment R_k = g^{r_k} mod p | `prove` |
| 4 | client | proof s_k = r_k + e·u_k mod q; send (R_k, s_k) | `prove` |
| 1 | server | receive (R_k, s_k) and the update; recompute u_k | `authenticate` |
| 2 | server | challenge e = H(R_k ‖ u_k ‖ context) mod q | `challenge` |
| 3 | server | check g^{s_k} = R_k · (g^{u_k})^e mod p: valid → accept the update for aggregation; invalid → reject it this round | `verify` |

p is the RFC 3526 2048-bit safe prime, q = (p − 1)/2 and g = 4 (generates the
order-q subgroup). H = SHA-256; context = protocol name, client and round, so a
proof cannot be replayed for another client or round. Settings in `zkp.py`:
`ENABLED = True`, `GROUP = "modp2048"` ("test512" is fast but NOT secure).

Times: *proof generation* = client steps 1–4; *proof verification* = server
steps 1–3; *authentication* = the whole session.

What the proof guarantees: the server's check uses u_k computed from the update
it received, so an update changed after the proof was made fails (tests check
this, and a replayed or forged proof). The protocol has no client secret key:
anyone who knows u_k can make a valid proof for it, so it checks the update's
integrity and freshness, not the sender's identity. A key pair registered per
client (s_k = r_k + e·x_k with public key g^{x_k}) would add identity.

## Costs

* **Computation** (s per round): per client, RF kernel + diffusion map + DP,
  ZKP proof generation, validation, decoding + RF retraining; for the server,
  alignment + aggregation + `M_glob` training + stopping rule, plus ZKP proof
  verification.
* **Communication** (bytes per round): uplink = latent matrix (float64),
  training labels (1 byte each), loss + TP/FP/TN/FN (24 bytes) and the ZKP
  messages (commitment, T, responses); downlink = challenges, the pickled
  `M_glob`, `R_k` and the out-of-bag labels.

## Bounded-update enforcement

After ZKP authentication and before aggregation (pipeline Stage 1.6), the server checks every
client update against a public bound (`fedrf_diff/bounds.py`). An update outside the bound is
rejected for that round and never aggregated, just like an update that fails authentication.
The check needs no secrets: it uses only the update and the public protocol parameters.

Always checked: the shapes (`d` columns, the agreed number of landmark rows, one label per
patient row), all values finite, labels in {0, 1}.

With Laplace DP (L1 clip C, scale b = 2C/epsilon_r):
- every landmark row is clipped but not noised, so ||p||_1 <= C exactly;
- each patient row is a clipped row plus Laplace noise, so its expected squared norm is at most
  C^2 + 2 b^2 d, with variance at most 20 b^4 d + 8 C^2 b^2.

The mean over patient rows must stay below that expectation plus 8 standard deviations of the
mean. An honest client stays far below this bound; an inflated or unclipped update does not.

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
| 1 | client | RF kernel of `RF_k^(r-1)` over its training rows + landmarks; diffusion map `Ψ_k = √N_tot U diag(μ)^γ` (constant eigenvector dropped); Laplace DP |
| 1.5 | both | Schnorr proof on each client's update; an invalid proof means the update is rejected this round |
| 2 | server | Procrustes `R_k` from landmark rows `P_k` to `P_ref`; `Z_glob` = aligned rows of all clients; `M_glob` = random forest on `Z_glob` + out-of-bag labels `ŷ`; ΔZ |
| 3 | client | validation loss `L_k`; k-NN decode of `Z_k R_k`; retrain `RF_k` on `D_train ∪ {(x̂, ŷ)}` |
| 4 | server | `L_glob = Σ (N_k/Σn) L_k`, ΔL; stop after 3 consecutive rounds with ΔZ < threshold and ΔL < ϵ_l |

## Results (seed 42, Laplace ε = 5 per round, 2048-bit group)

With `requirements-lock.txt` (split: VA Long Beach 140/36, Cleveland 409/103).
Converged at round 11; ZKP authentication successful for every client in
every round (44/44 proofs valid); ε spent per vector after 11 rounds: 55
(pure DP, 11 × 5).

Metrics averaged over the 11 rounds:

| Client | Accuracy | Precision | Recall | Specificity | F1 | MCC | Kappa |
|---|---|---|---|---|---|---|---|
| Client 1(Hungary) | 0.880 | 0.759 | 0.950 | 0.844 | 0.843 | 0.761 | 0.748 |
| Client 2(Switzerland) | 0.616 | 1.000 | 0.597 | 1.000 | 0.745 | 0.257 | 0.127 |
| Client 3(VA Long Beach) | 0.856 | 0.958 | 0.845 | 0.889 | 0.898 | 0.673 | 0.658 |
| Client 4(Cleveland) | 0.803 | 0.861 | 0.691 | 0.901 | 0.766 | 0.611 | 0.600 |
| Global Server | 0.812 | 0.878 | 0.751 | 0.881 | 0.809 | 0.634 | 0.626 |

Mean ZKP times per round (ms): authentication 6.7–7.1, proof generation
2.9–3.3, proof verification 3.7–4.0.

Global Server accuracy averaged over 11 rounds, per budget:

| ε per round | 1 | 2 | 3 | 4 | 5 | No DP |
|---|---|---|---|---|---|---|
| Laplace scale b | 2.00 | 1.00 | 0.67 | 0.50 | 0.40 | 0 |
| Total ε spent | 11 | 22 | 33 | 44 | 55 | – |
| Mean accuracy | 0.638 | 0.766 | 0.808 | 0.823 | 0.812 | 0.842 |

Accuracy levels off from ε ≈ 3 per round (ε = 4 above ε = 5 is single-seed
noise). Switzerland (22 validation patients, one healthy) is the least stable
client. One seed: repeat over several seeds before reporting.
