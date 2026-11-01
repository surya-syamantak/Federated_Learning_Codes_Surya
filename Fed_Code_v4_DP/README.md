# FedRF-Diff (Fed_Code_v4_DP)

Federated learning with random-forest diffusion maps, Procrustes alignment,
Gaussian differential privacy and ZKP vector authentication. It implements
"Algorithm 1: Iterative Closed-Loop FedRF-Diff Framework" on the combined UCI
heart-disease data (`data/heart_statlog_cleveland_hungary_final.csv`).

## Quick start

```bash
pip install -r requirements.txt
python -m fedrf_diff        # seed 42, Gaussian DP + ZKP, all CSVs and figures
python -m pytest
```

There are no DP or ZKP options on the command line:

* Gaussian DP is set in `fedrf_diff/privacy.py` (see
  [Differential privacy](#differential-privacy));
* the ZKP is set in `fedrf_diff/zkp.py` and runs inside the pipeline every
  round (see [ZKP authentication](#zkp-authentication)).

`--sigma`, `--epsilon`, `--zkp-group`, ... are rejected with a pointer to the
right file. The learning settings in `fedrf_diff/config.py` are still options
(`--max-rounds 40`, `--output-dir outputs/run2`, ...).

With the secure 2048-bit ZKP group a round takes about 50 s (the four clients
prove 1,246 vectors). Set `GROUP = "test512"` in `zkp.py` for a fast,
**insecure** test run (about 5 s per round).

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

The header shows the seed (42), the ZKP group, the DP ε, δ, clip norm and σ,
and the stopping thresholds. Then, for every round:

* **ZKP authentication (per client):** Status (Successful / Failed), number
  of vectors, *ZKP Authentication Time*, *ZKP Proof Generation Time* and
  *ZKP Proof Verification Time* (s);
* **validation metrics** of that round's global model for each client and
  for the Global Server: *Accuracy, Precision, Recall, Specificity, F1 Score,
  Matthews Correlation Coefficient (MCC), Cohen's Kappa*;
* the server line: ΔZ (and its threshold), `L_glob`, ΔL, patience, round
  time and the DP budget spent so far.

After convergence:

* the metrics averaged over all rounds, per client and for the Global Server;
* the ZKP time analysis: min, max and mean of each of the three ZKP times
  per client, and how many authentications succeeded;
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
| `dp_budget_sweep.csv` | per per-round ε (and no DP): σ, ε spent, mean accuracy over the rounds and final-round accuracy per client and pooled |

Figures (`figures/`, PNG at 600 dpi plus a vector PDF, serif fonts,
colour-blind-safe palette with a distinct marker per client):

| Figure | Panels |
|---|---|
| `convergence.png` | (a) ΔZ with its threshold, (b) \|ΔL\| with ϵ_l, (c) `L_glob`; converged round marked |
| `zkp_times.png` | (a) authentication, (b) proof generation, (c) proof verification time per client per round |
| `accuracy_vs_dp_budget.png` | (a) accuracy (mean over the rounds) vs per-round ε, incl. no DP; (b) accuracy per round vs cumulative ε spent in the main run |
| `computation_cost.png` | (a) computation time per round per client and server, (b) mean per-client breakdown |
| `communication_cost.png` | (a) uplink, (b) downlink KB per round per client |

The accuracy-vs-DP-budget sweep reruns the learning for each ε in
`privacy.EPSILON_SWEEP` with seed 42 and the same number of rounds as the
main run, in parallel processes. The ZKP does not change any learned value
(`tests/test_pipeline.py` checks this), so the sweep runs skip it. With one
seed the final-round accuracy moves a few points between rounds, so panel (a)
uses the mean over the rounds; the CSV keeps both.

## Differential privacy

`fedrf_diff/privacy.py` applies the Gaussian mechanism to every latent
vector a client sends, every round:

1. clip: `z_i ← z_i · min(1, C / ‖z_i‖₂)`, so the L2 sensitivity is `Δ = 2C`;
2. noise: `z̃_i = z_i + N(0, σ² I)`.

σ is calibrated with zero-concentrated DP (Bun & Steinke 2016): one release
is ρ-zCDP with `ρ = Δ² / (2σ²)`, and `ε = ρ + 2√(ρ ln(1/δ))`, so

    √ρ = √(ε + ln(1/δ)) − √ln(1/δ),   σ = Δ / √(2ρ).

Rounds compose additively in ρ: after t rounds each vector has spent
`(tρ + 2√(tρ ln(1/δ)), δ)`. The settings:

```python
EPSILON = 20.0     # per-round budget of every transmitted latent vector
DELTA = 1e-5
CLIP_NORM = 1.0    # C (sensitivity 2C)  ->  sigma = 0.637
NOISE_ON_VALIDATION = False   # validation embeddings never leave the client
NOISE_ON_LANDMARKS = False    # landmarks embed the public cohort: clipped, not noised
EPSILON_SWEEP = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, None)
```

Scope of the guarantee: it is per transmitted vector (local DP on what
leaves the client). It does not cover the training labels, which the server
needs to train `M_glob`, nor the influence one patient has on other patients'
vectors through the shared kernel.

**Stopping under DP.** σ is public, so the server subtracts the noise's
expected energy from ΔZ, and the ΔZ threshold is `ϵ_z + σ/C`
(0.15 + 0.637 = 0.787 at ε = 20; 0.15 without DP). ϵ_l = 0.02, patience 3.

## ZKP authentication

`fedrf_diff/zkp.py` implements the "Proposed ZKP-Based Authentication for
Vector Representation" protocol for every latent vector `z_k` of every
client, every round, before the server accepts the matrix:

1. **Commitment (client):** `C_k = Π g_j^{z_cj} · h^{r_k} · (C_{e−1})^β`,
   chained to the previous commitment.
2. **Proof (client):** random `k_j, η`; `T = Π g_j^{k_j} · h^η`.
3. **Challenge (server):** random `e`.
4. **Response (client):** `s_j = k_j + e·z_cj (mod q)`, `s_r = η + e·r_k (mod q)`.
5. **Verification (server):** `C_base = C_k / (C_{e−1})^β`; check
   `Π g_j^{s_j} · h^{s_r} = T · (C_base)^e`.
6. **Authentication Successful** if every vector verifies, otherwise
   **Authentication Failed** and the client sits out the round.

Settings in `zkp.py`:

```python
ENABLED = True
GROUP = "modp2048"     # RFC 3526 2048-bit group; "test512" is fast but NOT secure
LAMBDA = 3             # chaining exponent
SCALE_BITS = 32        # fixed-point encoding round(z * 2^32)
BIND_PLAINTEXT = False # extra proof that the commitment opens to the sent Z_k
```

Times: *proof generation* = the client's steps 1, 2 and 4; *proof
verification* = the server's steps 3, 5 and 6; *authentication* = the whole
session. Group elements are computed mod the safe prime `p` and exponents
mod `q = (p−1)/2` (the picture's "mod q").

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

With adaptive Gaussian DP (L2 clip C_r, sigma_r = noise_ratio * C_r), the server first checks the
declared release parameters: sigma_r must equal noise_ratio * C_r, and C_r <= sqrt(N_tot d).
Then:
- every landmark row satisfies ||p||_2 <= C_r;
- the mean squared norm of the patient rows must stay below C_r^2 + d sigma_r^2 plus 8
  standard deviations of that mean (per-row variance 2 d sigma^4 + 4 C^2 sigma^2).

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
| 1 | client | RF kernel of `RF_k^(r-1)` over its training rows + landmarks; diffusion map `Ψ_k = √N_tot U diag(μ)^γ` (constant eigenvector dropped); Gaussian DP |
| 1.5 | both | ZKP authentication of every vector; failed clients sit out the round |
| 2 | server | Procrustes `R_k` from landmark rows `P_k` to `P_ref`; `Z_glob` = aligned rows of all clients; `M_glob` = random forest on `Z_glob` + out-of-bag labels `ŷ`; ΔZ |
| 3 | client | validation loss `L_k`; k-NN decode of `Z_k R_k`; retrain `RF_k` on `D_train ∪ {(x̂, ŷ)}` |
| 4 | server | `L_glob = Σ (N_k/Σn) L_k`, ΔL; stop after 3 consecutive rounds with ΔZ < threshold and ΔL < ϵ_l |

## Results (seed 42, ε = 20 per round, 2048-bit ZKP)

Converged at round 8; all 32 authentications successful; ε spent per vector
after 8 rounds: 82.1 (δ = 10⁻⁵).

Global Server metrics averaged over the 8 rounds: accuracy 0.803, MCC 0.615.
Final round (round 8):

| Client | n_val | Accuracy | Precision | Recall | Specificity | F1 | MCC | Kappa |
|---|---|---|---|---|---|---|---|---|
| Client 1(Hungary) | 53 | 0.906 | 0.783 | 1.000 | 0.857 | 0.878 | 0.819 | 0.803 |
| Client 2(Switzerland) | 22 | 0.591 | 1.000 | 0.571 | 1.000 | 0.727 | 0.239 | 0.108 |
| Client 3(VA Long Beach) | 36 | 0.889 | 0.960 | 0.889 | 0.889 | 0.923 | 0.731 | 0.724 |
| Client 4(Cleveland) | 103 | 0.738 | 0.756 | 0.646 | 0.818 | 0.697 | 0.473 | 0.468 |
| Global Server | 214 | 0.790 | 0.842 | 0.746 | 0.840 | 0.791 | 0.585 | 0.581 |

Switzerland's validation split has a single healthy patient, so its
specificity, MCC and kappa rest on one case.

Mean ZKP times per round (s), 2048-bit group:

| Client | Vectors | Authentication | Proof generation | Proof verification |
|---|---|---|---|---|
| Client 1(Hungary) | 309 | 11.90 | 6.04 | 5.86 |
| Client 2(Switzerland) | 188 | 7.24 | 3.58 | 3.66 |
| Client 3(VA Long Beach) | 240 | 9.40 | 4.67 | 4.73 |
| Client 4(Cleveland) | 509 | 19.75 | 9.92 | 9.82 |

Times scale with the number of vectors (about 38 ms per vector).

### Accuracy vs DP budget (seed 42, 8 rounds each)

Global Server accuracy (the figure plots the mean over the rounds; with one
seed the final round moves a few points, e.g. ε = 20 is below ε = 10):

| ε per round | 1 | 2 | 5 | 10 | 20 | 50 | 100 | No DP |
|---|---|---|---|---|---|---|---|---|
| σ | 9.80 | 5.00 | 2.11 | 1.14 | 0.637 | 0.318 | 0.197 | 0 |
| Mean over 8 rounds | 0.452 | 0.526 | 0.672 | 0.761 | 0.803 | 0.823 | 0.839 | 0.845 |
| After round 8 | 0.341 | 0.565 | 0.752 | 0.827 | 0.790 | 0.841 | 0.836 | 0.850 |

Below ε ≈ 5 per round the noise swamps the latent geometry (σ is larger
than the clipped vectors themselves); from ε ≈ 10 the accuracy is within a
few points of no DP.
