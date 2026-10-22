# FedRF-Diff (Fed_Code_v11_Pedersen_LaplaceDP_eps2-3)

Federated learning with random-forest diffusion maps, Procrustes alignment,
Laplace differential privacy (pure ε-DP, a new ε_t ~ U[2, 3] every round) and a
Pedersen-commitment zero-knowledge proof that verifies every client every
round. It implements "Algorithm 1: Iterative Closed-Loop FedRF-Diff Framework"
on the combined UCI heart-disease data
(`data/heart_statlog_cleveland_hungary_final.csv`).

Changes from v10 (`Fed_Code_v10_Schnorr_LaplaceDP_eps5`):

* the Schnorr ZKP is replaced by the **Pedersen-based ZKP** (commitment
  `C_c = g^{u_c} h^{r_c} mod p`, Σ-protocol proof of knowledge of `(u_c, r_c)`
  and a range proof `0 ≤ u_c < B`, made non-interactive with Fiat–Shamir). It
  is used only for client verification: a client whose proof is invalid has
  its update rejected for that round;
* the DP budget of the main run is drawn at random every round:
  ε_t ~ U[2, 3] (`privacy.EPSILON = "random"`, `privacy.RANDOM_RANGE = (2, 3)`).

## Quick start

```bash
pip install -r requirements-lock.txt   # exact versions behind the results below
python -m fedrf_diff        # seed 42, Laplace DP (ε_t ~ U[2, 3] per round) + Pedersen ZKP, all CSVs and figures
python -m pytest
python make_figures.py     # (re)draw all figures from the saved run, no retraining
```

There are no DP or ZKP options on the command line:

* Laplace DP is set in `fedrf_diff/privacy.py` (see
  [Differential privacy](#differential-privacy-laplace));
* the ZKP is set in `fedrf_diff/zkp.py` and runs inside the pipeline every
  round (see [Pedersen ZKP](#pedersen-zkp-client-verification)).

`--sigma`, `--epsilon`, `--zkp-group`, ... are rejected with a pointer to the
right file. The learning settings in `fedrf_diff/config.py` are still options
(`--max-rounds 40`, `--output-dir outputs/run2`, ...).

A round takes about 5 s of learning plus about 0.5 s of ZKP per client (2048-bit
group, 64-bit range proof). The run ends with the accuracy-vs-DP-budget sweep
(a few minutes).

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

The header shows the seed (42), the ZKP group and range bound, the DP range
[2, 3], the L1 clip norm and the stopping thresholds. Then, for every round:

* **Pedersen ZKP (per client):** Proof (Valid / Invalid), number of vectors,
  *ZKP Authentication Time*, *ZKP Proof Generation Time* and *ZKP Proof
  Verification Time* (ms; the CSV files keep seconds), and the authentication
  status (update accepted / rejected);
* encoding and decoding times per client;
* **validation metrics** of that round's global model for each client and
  for the Global Server: *Accuracy, Precision, Recall, Specificity, F1 Score,
  Matthews Correlation Coefficient (MCC), Cohen's Kappa*;
* the server line: ΔZ (and its threshold), `L_avg`, ΔL, patience, round
  time, this round's ε_t and the DP budget spent so far.

After the last round:

* the metrics averaged over all rounds, per client and for the Global Server;
* the ZKP time analysis: min, max and mean (ms) of each of the three ZKP
  times per client, and how many proofs were valid;
* the accuracy-vs-DP-budget tables (mean over the rounds and final round).

"Global Server" is the global model `M_G` scored on all clients' validation
patients together (summed TP/FP/TN/FN). Clients only report their loss and
TP/FP/TN/FN counts; every metric is computed from those counts.

## Output files (`--output-dir`, default `outputs/run`)

| File | Content |
|---|---|
| `config.json` | the settings used, including `privacy` and `zkp` |
| `history.csv` | per round: ΔZ, `L_avg`, ΔL, timings, ε_t, Laplace scale, DP budget spent, and per client (and pooled, `[all]`) loss and the seven metrics |
| `final_metrics.csv` | last round's metrics per client, pooled row and macro average |
| `global_server_metrics.csv` | per round: the Global Server's TP/FP/TN/FN, log-loss, seven metrics, `L_avg`, ΔZ, ΔL, ε spent |
| `zkp_times.csv` | per round and client: authenticated, vectors, the three ZKP times, ZKP bytes up/down |
| `costs.csv` | per round and client (and Global Server): computation time and its parts, bytes sent and received, ε spent |
| `dp_budget_sweep.csv` | per per-round ε (fixed 1–5, random U[2, 3], and no DP): Laplace scale b, ε spent, mean accuracy over the rounds and final-round accuracy per client and pooled |

Figures (`figures/`, PNG at 600 dpi plus a vector PDF, serif fonts,
colour-blind-safe palette with a distinct marker per client):

| Figure | Panels |
|---|---|
| `convergence.png` | (a) ΔZ with its threshold, (b) \|ΔL\| with ε_L, (c) `L_avg`; converged round marked |
| `zkp_times.png` | (a) authentication, (b) proof generation, (c) proof verification time per client per round (grouped bars) |
| `accuracy_vs_dp_budget.png` | (a) accuracy (mean over the rounds) vs per-round ε, incl. U[2, 3] and no DP; (b) accuracy per round vs cumulative ε spent in the main run (both grouped bars) |
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
2. noise: `z̃_i = z_i + Lap(0, b_t)` on every coordinate, `b_t = Δ₁ / ε_t`.

Every round t the main run draws a new budget **ε_t uniformly from [2, 3]**.
The draw comes from the run seed (its own random stream), so it is
reproducible, and all clients use the same ε_t in a round. Each release is
(ε_t, 0)-DP (pure DP, no δ); rounds add up, so after t rounds each vector has
spent ε_1 + … + ε_t (between 2t and 3t). With ε_t ∈ [2, 3] the Laplace scale
is b_t = 2C/ε_t ∈ [0.67, 1.0] (noise std √2·b_t ∈ [0.94, 1.41]). The terminal
shows ε_t and the running total every round. The settings:

```python
EPSILON_RANGE = (1.0, 5.0)  # every budget used must lie here
RANDOM_RANGE = (2.0, 3.0)    # ε_t ~ U(RANDOM_RANGE) each round
EPSILON = "random" # per-round budget: "random" (new ε_t each round) or a fixed number
CLIP_NORM = 1.0    # C, L1 bound (sensitivity 2C)
NOISE_ON_VALIDATION = False   # validation embeddings never leave the client
NOISE_ON_LANDMARKS = False    # landmarks embed the public cohort: clipped, not noised
EPSILON_SWEEP = (1.0, 2.0, 3.0, 4.0, 5.0, None)   # None = no-DP reference
```

Scope of the guarantee: it is per transmitted vector (local DP on what
leaves the client). It does not cover the training labels, which the server
needs to train `M_G`, nor the influence one patient has on other patients'
vectors through the shared kernel.

**Stopping under DP.** The noise level is public, so the server subtracts its
expected energy from ΔZ (using that round's and the previous round's noise
std), and the ΔZ threshold is `ε_Z + std_t/C`, recomputed each round from ε_t.
When the noise is as large as the signal the corrected ΔZ could explode, so
its denominator never drops below the signal energy measured on the
noise-free public landmark rows. ε_L = 0.02, patience 3. A run with ε outside
[1, 5] stops with an error (`privacy.check_epsilon`).

## Pedersen ZKP (client verification)

`fedrf_diff/zkp.py` implements the "Pedersen-based Zero-Knowledge Proof (ZKP)
in Federated Learning" protocol once per client and round. The ZKP is used
only to verify clients before the server accepts their updates. The update is
everything the client sends: its DP-protected latent matrix (patient and
landmark rows) and its training labels.

| Step | Who | What | Code |
|---|---|---|---|
| 1 | client | local training → update; u_c = its SHA-256 fingerprint (first 64 bits), 0 ≤ u_c < B = 2^64 | `encode_update` |
| 2 | client | random r_c ∈ Z_q | `authenticate` |
| 3 | client | Pedersen commitment C_c = g^{u_c} h^{r_c} mod p; send C_c | `PublicParams.commit` |
| 4 | client | proof π_c: knowledge of (u_c, r_c) with C_c = g^{u_c} h^{r_c} and 0 ≤ u_c < B (Σ-protocol / Fiat–Shamir); send π_c | `prove` |
| 1 | server | receive (C_c, π_c) and the update | `authenticate` |
| 2 | server | verify the Σ-protocol equations: C_c = g^{u_c} h^{r_c}, u_c in range, nothing revealed; valid → accept the update for aggregation, invalid → reject it this round | `verify` |
| 3 | server | federated aggregation of the verified clients only → global model `M_G` | `pipeline.run` |

How the proof π_c is built:

* **Opening (Okamoto Σ-protocol):** T = g^a h^s; z_u = a + e·u_c,
  z_r = s + e·r_c (mod q); the server checks g^{z_u} h^{z_r} = T · C_c^e.
* **Range 0 ≤ u_c < 2^64:** u_c = Σ 2^i b_i, one commitment C_i = g^{b_i} h^{r_i}
  per bit with r_c = Σ 2^i r_i. Each C_i carries an OR-proof (Cramer–Damgård–
  Schoenmakers) that it commits to 0 or 1, and the server checks
  ∏ C_i^{2^i} = C_c (the homomorphism), so u_c fits in 64 bits.
* **Fiat–Shamir:** one challenge e = H(context ‖ SHA-256(update) ‖ C_c ‖ T ‖
  all C_i, A_i0, A_i1) mod q, H = SHA-256. The context is the protocol name,
  client and round, so a proof cannot be replayed for another client or round.
  The update's hash is computed by the server from the update it actually
  received, so an update changed in transit fails.

**Group.** A 2048-bit prime p = k·q + 1 with a 256-bit prime q (a Schnorr
group, like RFC 5114's 2048/256 groups); g and h generate the order-q
subgroup. p, q, g and h are derived from the public string `zkp.SEED` with
SHA-256 ("nothing up my sleeve"), so nobody knows log_g(h) — needed for the
commitment to be binding — and anyone can regenerate them (`generate_group`,
`setup`; the tests check primality, q | p − 1 and the generators' order).
Exponentiations by g and h use precomputed fixed-base tables.

Settings in `zkp.py`: `ENABLED = True`, `GROUP = "pedersen2048"`
("test512" is fast but NOT secure, used by the tests), `RANGE_BITS = 64`
(the bound B = 2^64; any value below 256 works, cost grows linearly).

Times: *proof generation* = client steps 1–4; *proof verification* = server
steps 1–2; *authentication* = the whole session. Communication per proof:
C_c, T, z_u, z_r and per bit (C_i, A_i0, A_i1, e_i0, z_i0, z_i1) ≈ 55 KB up,
1 byte (valid / invalid) down.

What the proof guarantees: the commitment hides u_c perfectly
(information-theoretic; a fresh r_c every round), the server learns nothing
about u_c or r_c beyond "0 ≤ u_c < B", and the proof is bound to the exact
update, client and round (tests check altered updates, replays to another
round or client, forged responses, broken bit proofs and out-of-range
values). As in the diagram, the protocol has no long-term client key, so it
verifies a well-formed, unaltered and fresh update, not who made it; a key
registered per client would add identity.

## Costs

* **Computation** (s per round): per client, RF kernel + diffusion map + DP,
  ZKP proof generation, validation, decoding + RF retraining; for the server,
  alignment + aggregation + `M_G` training + stopping rule, plus ZKP proof
  verification.
* **Communication** (bytes per round): uplink = latent matrix (float64),
  training labels (1 byte each), loss + TP/FP/TN/FN (24 bytes) and the ZKP
  messages (C_c and π_c); downlink = the ZKP result, the pickled `M_G`, `A_c`
  and the out-of-bag labels.

## One round

| Stage | Where | What |
|---|---|---|
| 1 | client | RF kernel of `RF_c^(t-1)` over its training rows + landmarks; diffusion map `Z_c = √n V Λ^τ` (constant eigenvector dropped); Laplace DP with this round's ε_t ~ U[2, 3] |
| 1.5 | both | Pedersen ZKP verifies each client's update; an invalid proof means the update is rejected this round |
| 2 | server | Procrustes `A_c` from landmark rows `P_c` to `P_ref`; `Z_G` = aligned rows of the verified clients; `M_G` = random forest on `Z_G` + out-of-bag labels `ŷ`; ΔZ |
| 3 | client | validation loss `L_c`; k-NN decode of `Z̃_c A_c`; retrain `RF_c` on `D_train ∪ {(x̂, ŷ)}` |
| 4 | server | `L_avg = Σ (n_c/Σn) L_c`, ΔL; stop after 3 consecutive rounds with ΔZ < threshold and ΔL < ε_L |

## Results (seed 42, ε_t ~ U[2, 3] per round, 2048-bit Pedersen group)

Split: VA Long Beach 140/36, Cleveland 409/103. The full terminal output is
in `outputs/run/run_log.txt`. The run did **not** meet the stopping rule and
ended at `max_rounds` = 30 (ΔL stayed above ε_L = 0.02 too often). Pedersen
ZKP: every client was verified in every round (120/120 proofs valid, no update
rejected). ε_t ranged over [2.01, 2.99] (mean 2.46); ε spent per vector after
30 rounds: 73.9 (pure DP).

Metrics averaged over the 30 rounds:

| Client | Accuracy | Precision | Recall | Specificity | F1 | MCC | Kappa |
|---|---|---|---|---|---|---|---|
| Client 1(Hungary) | 0.862 | 0.733 | 0.954 | 0.814 | 0.827 | 0.736 | 0.717 |
| Client 2(Switzerland) | 0.699 | 0.994 | 0.689 | 0.900 | 0.807 | 0.271 | 0.158 |
| Client 3(VA Long Beach) | 0.855 | 0.939 | 0.865 | 0.822 | 0.899 | 0.652 | 0.637 |
| Client 4(Cleveland) | 0.783 | 0.825 | 0.685 | 0.868 | 0.746 | 0.568 | 0.559 |
| Global Server | 0.806 | 0.855 | 0.771 | 0.845 | 0.808 | 0.618 | 0.612 |

Mean Pedersen ZKP times per client and round (ms; 2048-bit group, 64-bit
range proof): authentication 327–332, proof generation 101–106, proof
verification 223–231. Each proof adds about 55 KB to the uplink.

Global Server accuracy averaged over 30 rounds, per budget:

| ε per round | 1 | 2 | 3 | 4 | 5 | U[2, 3] | No DP |
|---|---|---|---|---|---|---|---|
| Mean Laplace scale b | 2.00 | 1.00 | 0.67 | 0.50 | 0.40 | 0.83 | 0 |
| Total ε spent | 30 | 60 | 90 | 120 | 150 | 73.9 | – |
| Mean accuracy | 0.641 | 0.789 | 0.809 | 0.822 | 0.810 | 0.806 | 0.842 |

The random budget U[2, 3] performs between fixed ε = 2 and ε = 3, as
expected, at a total cost of 73.9. The learning values are identical to v10's
U[2, 3] run because the ZKP changes no learned value. Switzerland (22
validation patients, one healthy) is the least stable client. One seed:
repeat over several seeds before reporting.
