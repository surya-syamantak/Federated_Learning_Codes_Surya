# FedRF-Diff (Fed_Code_v12_ProposedPedersenZKP_LaplaceDP_eps2-3)

Federated learning with random-forest diffusion maps, Procrustes alignment,
Laplace differential privacy (pure ε-DP, a new ε_t ~ U[2, 3] every round) and
a proposed Pedersen zero-knowledge proof that authenticates every client every
round. It implements "Algorithm 1: Iterative Closed-Loop FedRF-Diff Framework"
on the combined UCI heart-disease data
(`data/heart_statlog_cleveland_hungary_final.csv`).

Changes from v11 (`Fed_Code_v11_Pedersen_LaplaceDP_eps2-3`): only the ZKP.
`fedrf_diff/zkp.py` now implements the **Proposed ZKP-Based Authentication**
protocol: a chained Pedersen commitment `C_t = g^{v_t} h^{r_t} (C_{t-1})^λ`,
`T = g^k h^τ`, a fresh random challenge `c` from the server, responses
`r_v`, `r_s`, and the check `g^{r_v} h^{r_s} = T (C_t / (C_{t-1})^λ)^c`. It
works on scalars and is used only for client authentication. The learning
code (RF kernel, diffusion map, DP, alignment, global model, decoder,
stopping rule) is byte-identical to v10/v11; only the ZKP call in
`pipeline.py`, the ZKP labels in `report.py` and the ZKP tests changed.

## Quick start

```bash
pip install -r requirements-lock.txt   # exact versions behind the results below
python -m fedrf_diff        # seed 42, Laplace DP (ε_t ~ U[2, 3] per round) + ZKP client authentication, all CSVs and figures
python -m pytest
python make_figures.py     # (re)draw all figures from the saved run, no retraining
```

There are no DP or ZKP options on the command line:

* Laplace DP is set in `fedrf_diff/privacy.py` (see
  [Differential privacy](#differential-privacy-laplace));
* the ZKP is set in `fedrf_diff/zkp.py` and runs inside the pipeline every
  round (see [ZKP client authentication](#zkp-client-authentication-proposed-pedersen)).

`--sigma`, `--epsilon`, `--zkp-group`, ... are rejected with a pointer to the
right file. The learning settings in `fedrf_diff/config.py` are still options
(`--max-rounds 40`, `--output-dir outputs/run2`, ...).

A round takes about 5 s of learning; each client's ZKP authentication takes a
few milliseconds (2048-bit group). The run ends with the accuracy-vs-DP-budget sweep
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

* **ZKP client authentication (per client):** Proof (Valid / Invalid), number of vectors,
  *ZKP Authentication Time*, *ZKP Proof Generation Time* and *ZKP Proof
  Verification Time* (ms; the CSV files keep seconds), and the status
  (Authentication Successful / Authentication Failed);
* encoding and decoding times per client;
* **validation metrics** of that round's global model for each client and
  for the Global Server: *Accuracy, Precision, Recall, Specificity, F1 Score,
  Matthews Correlation Coefficient (MCC), Cohen's Kappa*;
* the server line: ΔZ (and its threshold), `L_avg`, ΔL, patience, round
  time, this round's ε_t and the DP budget spent so far.

After the last round:

* the metrics averaged over all rounds, per client and for the Global Server;
* the ZKP time analysis: min, max and mean (ms) of each of the three ZKP
  times per client, and how many authentications succeeded;
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

## ZKP client authentication (proposed Pedersen)

`fedrf_diff/zkp.py` implements the "Proposed Zero-Knowledge Proof
(ZKP)-Based Authentication" protocol once per client and round. The ZKP is
used **only for client authentication** and works on **scalars**: the client's
secret v_t, the blinding factors r_t, k, τ, the server's challenge c and the
responses r_v, r_s. It does not touch the transmitted update and changes no
learned value; a client that fails authentication has its update rejected for
that round.

Public parameters known to both: prime q, generators g and h. The client holds
a secret v_t (committed across rounds); in round 1 the previous commitment
C_{t-1} is initialised to 1.

| Step | Direction | What | Code |
|---|---|---|---|
| 1 | client → server | Pedersen commitment C_t = g^{v_t} h^{r_t} (C_{t-1})^λ | `ClientProver.commit` |
| 2 | client → server | chooses two random integers k, τ (fresh every session); sends T = g^k h^τ | `ClientProver.announce` |
| 3 | server → client | random challenge c (fresh every time) | `ServerVerifier.challenge` |
| 4 | client → server | responses r_v = k + c·v_t (mod q), r_s = τ + c·r_t (mod q) | `ClientProver.respond` |
| – | server | verifies g^{r_v} h^{r_s} =? T (C_base)^c with C_base = C_t / (C_{t-1})^λ; decision: do both sides match? | `ServerVerifier.verify` |
| 5 | server → client | values match → **Authentication Successful** (update accepted for aggregation) | `ServerVerifier.decide` |
| 6 | server → client | values do not match → **Authentication Failed** (update rejected this round) | `ServerVerifier.decide` |

Why it works: C_base = C_t / (C_{t-1})^λ = g^{v_t} h^{r_t}, so
g^{r_v} h^{r_s} = g^{k + c·v_t} h^{τ + c·r_t} = T · (C_base)^c exactly when the
client knows the opening (v_t, r_t). Nothing about v_t or r_t is revealed:
C_t is a Pedersen commitment (perfectly hiding) and r_v, r_s are masked by the
fresh k and τ.

Implementation choices the diagram leaves open:

* **Group.** The diagram's "prime q" is realised as a 2048-bit prime
  p = k·q + 1 with a 256-bit prime q: group elements (g, h, C_t, T) are
  reduced mod p and every scalar (v_t, r_t, k, τ, c, r_v, r_s, λ) mod q, the
  order of the subgroup generated by g and h. p, q, g and h are derived from
  the public string `zkp.SEED` with SHA-256 ("nothing up my sleeve"), so
  nobody knows log_g(h) and anyone can regenerate them (`generate_group`,
  `setup`; the tests check primality, q | p − 1 and the generators' order).
  Exponentiations by g and h use precomputed fixed-base tables.
* **Secret.** At enrolment each client draws a long-term 256-bit key that
  never leaves the client; its round secret is the scalar
  v_t = SHA-256(key ‖ client ‖ t) mod q.
* **Chain.** C_{t-1} is the client's last *successfully authenticated*
  commitment; client and server both keep it. After "Authentication
  Successful" both move on to C_t; after "Authentication Failed" both keep
  C_{t-1}. λ = `zkp.LAMBDA` = 2 (a public scalar).
* **Challenge.** c is drawn uniformly from Z_q by the server in every session
  (interactive, as in the diagram), so replaying an old session fails.

Settings in `zkp.py`: `ENABLED = True`, `GROUP = "pedersen2048"`
("test512" is fast but NOT secure, used by the tests), `LAMBDA = 2`.

Times: *proof generation* = client steps 1, 2 and 4; *proof verification* =
server step 3 plus the check and the decision; *authentication* = the whole
session (steps 1–6). Communication per session: up C_t, T (256 bytes each) and
r_v, r_s (32 bytes each) = 576 bytes; down c (32 bytes) + the outcome (1 byte).

Tests check the diagram's equations step by step over several rounds, the
chain update, and that a replayed session, forged responses, an altered
commitment, a commitment that does not continue the client's chain and a
group element outside the subgroup all fail. `pipeline.run(tamper={t: client})`
simulates an impostor who replays the client's previous session in round t:
it is rejected and its update is not aggregated.

What it guarantees, and what not: the server learns only that the client
knows an opening of C_base; v_t and r_t stay hidden. As drawn in the diagram,
no commitment is registered with the server in advance (C_0 = 1), so
continuing a client's chain proves knowledge of *some* opening rather than of
a secret the server enrolled. Registering each client's first commitment
before training would tie authentication to the enrolled client.

## Costs

* **Computation** (s per round): per client, RF kernel + diffusion map + DP,
  ZKP proof generation, validation, decoding + RF retraining; for the server,
  alignment + aggregation + `M_G` training + stopping rule, plus ZKP proof
  verification.
* **Communication** (bytes per round): uplink = latent matrix (float64),
  training labels (1 byte each), loss + TP/FP/TN/FN (24 bytes) and the ZKP
  messages (C_t, T, r_v, r_s); downlink = the challenge c and the outcome, the pickled `M_G`, `A_c`
  and the out-of-bag labels.

## One round

| Stage | Where | What |
|---|---|---|
| 1 | client | RF kernel of `RF_c^(t-1)` over its training rows + landmarks; diffusion map `Z_c = √n V Λ^τ` (constant eigenvector dropped); Laplace DP with this round's ε_t ~ U[2, 3] |
| 1.5 | both | ZKP client authentication (steps 1–6); a failed authentication means the update is rejected this round |
| 2 | server | Procrustes `A_c` from landmark rows `P_c` to `P_ref`; `Z_G` = aligned rows of the authenticated clients; `M_G` = random forest on `Z_G` + out-of-bag labels `ŷ`; ΔZ |
| 3 | client | validation loss `L_c`; k-NN decode of `Z̃_c A_c`; retrain `RF_c` on `D_train ∪ {(x̂, ŷ)}` |
| 4 | server | `L_avg = Σ (n_c/Σn) L_c`, ΔL; stop after 3 consecutive rounds with ΔZ < threshold and ΔL < ε_L |

## Results (seed 42, ε_t ~ U[2, 3] per round, 2048-bit group)

Split: VA Long Beach 140/36, Cleveland 409/103. The full terminal output is
in `outputs/run/run_log.txt`. The run did **not** meet the stopping rule and
ended at `max_rounds` = 30 (ΔL stayed above ε_L = 0.02 too often). ZKP client
authentication: every client was authenticated in every round (120/120
successful, no update rejected). ε_t ranged over [2.01, 2.99] (mean 2.46); ε
spent per vector after 30 rounds: 73.9 (pure DP).

Metrics averaged over the 30 rounds:

| Client | Accuracy | Precision | Recall | Specificity | F1 | MCC | Kappa |
|---|---|---|---|---|---|---|---|
| Client 1(Hungary) | 0.862 | 0.733 | 0.954 | 0.814 | 0.827 | 0.736 | 0.717 |
| Client 2(Switzerland) | 0.699 | 0.994 | 0.689 | 0.900 | 0.807 | 0.271 | 0.158 |
| Client 3(VA Long Beach) | 0.855 | 0.939 | 0.865 | 0.822 | 0.899 | 0.652 | 0.637 |
| Client 4(Cleveland) | 0.783 | 0.825 | 0.685 | 0.868 | 0.746 | 0.568 | 0.559 |
| Global Server | 0.806 | 0.855 | 0.771 | 0.845 | 0.808 | 0.618 | 0.612 |

Mean ZKP times per client and round (ms; 2048-bit group): authentication
3.8–4.1, proof generation 0.56–0.73, proof verification 3.2–3.4. Each session
sends 576 bytes up and 33 bytes down.

Global Server accuracy averaged over 30 rounds, per budget:

| ε per round | 1 | 2 | 3 | 4 | 5 | U[2, 3] | No DP |
|---|---|---|---|---|---|---|---|
| Mean Laplace scale b | 2.00 | 1.00 | 0.67 | 0.50 | 0.40 | 0.83 | 0 |
| Total ε spent | 30 | 60 | 90 | 120 | 150 | 73.9 | – |
| Mean accuracy | 0.641 | 0.789 | 0.809 | 0.822 | 0.810 | 0.806 | 0.842 |

The learning values are identical to v11 because the ZKP changes no learned
value; only the ZKP times and message sizes differ. Switzerland (22
validation patients, one healthy) is the least stable client. One seed:
repeat over several seeds before reporting.
