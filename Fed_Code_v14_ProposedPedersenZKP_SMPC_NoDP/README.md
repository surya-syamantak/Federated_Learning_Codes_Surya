# FedRF-Diff (Fed_Code_v14_ProposedPedersenZKP_SMPC_NoDP)

Federated learning with random-forest diffusion maps, Procrustes alignment, the
proposed Pedersen zero-knowledge proof that authenticates every client every
round, and **simple SMPC** (additive secret sharing) for secure aggregation,
**without any differential privacy**. It implements "Algorithm 1: Iterative
Closed-Loop FedRF-Diff Framework" on the combined UCI heart-disease data
(`data/heart_statlog_cleveland_hungary_final.csv`).

Changes from v12 (`Fed_Code_v12_ProposedPedersenZKP_LaplaceDP_eps2-3`):

* **Laplace DP removed completely**, as in v13: `fedrf_diff/privacy.py` is
  gone, clients send their latent vectors exactly as computed (no clipping,
  no noise), the stopping rule uses the plain ΔZ with threshold ε_Z, and the
  accuracy-vs-DP-budget sweep and every ε column are gone.
* **Simple SMPC added** (`fedrf_diff/smpc.py`): every client's weighted
  validation loss and TP/FP/TN/FN counts reach the server only through
  n-out-of-n additive secret sharing, so the server learns only their totals
  (`L_avg` and the Global Server metrics). See [SMPC](#smpc-secure-aggregation).
* The proposed ZKP (`zkp.py`) and the learning code (RF kernel, diffusion map,
  alignment, global model, decoder) are unchanged.

## Quick start

```bash
pip install -r requirements-lock.txt   # exact versions behind the results below
python -m fedrf_diff        # seed 42, no DP, ZKP client authentication + SMPC, all CSVs and figures
python -m pytest
python make_figures.py     # (re)draw all figures from the saved run, no retraining
```

The ZKP is set in `fedrf_diff/zkp.py` and the SMPC in `fedrf_diff/smpc.py`;
neither takes a command-line option (`--zkp-...`, `--smpc-...` and DP options
such as `--epsilon` are rejected). The learning settings in
`fedrf_diff/config.py` are options (`--max-rounds 40`, `--output-dir
outputs/run2`, ...).

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

The header shows the seed (42), the ZKP group and λ, that no DP is used, the
SMPC scheme and the stopping thresholds. Then, for every round:

* **ZKP client authentication (per client):** Proof (Valid / Invalid), number
  of vectors, *ZKP Authentication Time*, *ZKP Proof Generation Time* and *ZKP
  Proof Verification Time* (ms; the CSV files keep seconds), and the status
  (Authentication Successful / Authentication Failed);
* encoding and decoding times per client;
* **SMPC secure aggregation:** per client the shares sent, *SMPC Share
  Generation Time* and *SMPC Share Summation Time* (ms) and the bytes sent and
  received; then the server's reconstruction time and the totals it learned;
* **validation metrics** of that round's global model for each client and for
  the Global Server: *Accuracy, Precision, Recall, Specificity, F1 Score,
  Matthews Correlation Coefficient (MCC), Cohen's Kappa*;
* the server line: ΔZ (and its threshold), `L_avg`, ΔL, patience and round
  time.

After the last round: the metrics averaged over all rounds, the ZKP time
analysis, the SMPC time analysis (min, max and mean per client and for the
server's reconstruction) and the encoding/decoding time analysis.

**Who knows which metric.** The Global Server row is computed from the SMPC
totals only. The per-client rows are the *evaluation harness* for the paper:
each client computes its own metrics locally, and they are never sent to the
server.

## Output files (`--output-dir`, default `outputs/run`)

| File | Content |
|---|---|
| `config.json` | the settings used, including `privacy` (`{"mechanism": "none"}`), `smpc` and `zkp` |
| `history.csv` | per round: ΔZ, `L_avg`, ΔL, timings (incl. ZKP and SMPC seconds), and per client (harness) and pooled (`[all]`, SMPC) loss and the seven metrics |
| `final_metrics.csv` | last round's metrics per client (harness), pooled row (SMPC totals) and macro average |
| `global_server_metrics.csv` | per round: the Global Server's TP/FP/TN/FN, log-loss and seven metrics (all from the SMPC totals), `L_avg`, ΔZ, ΔL |
| `zkp_times.csv` | per round and client: authenticated, vectors, the three ZKP times, ZKP bytes up/down |
| `smpc_times.csv` | per round and client: values aggregated, shares sent, share generation / summation / total time, SMPC bytes up/down; Global Server: reconstruction time |
| `costs.csv` | per round and client (and Global Server): computation time and its parts (incl. `smpc_s`), bytes sent and received |
| `run_log.txt` | the terminal output of the run |

Figures (`figures/`, PNG at 600 dpi plus a vector PDF):

| Figure | Panels |
|---|---|
| `convergence.png` | (a) ΔZ with ε_Z, (b) \|ΔL\| with ε_L, (c) `L_avg`; converged round marked |
| `zkp_times.png` | (a) authentication, (b) proof generation, (c) proof verification time per client per round |
| `smpc_times.png` | (a) share generation and (b) share summation time per client per round, (c) the Global Server's reconstruction time per round |
| `computation_cost.png` | (a) computation time per round per client and server, (b) mean per-client breakdown (ZKP and SMPC times in ms under each client), (c) mean ZKP proof generation time per client |
| `communication_cost.png` | (a) uplink, (b) downlink KB per round per client |

## SMPC (secure aggregation)

The server needs only totals of what the clients report after the local
evaluation: the weighted validation loss for `L_avg = Σ_c (n_c / Σ n) L_c` and
the summed TP/FP/TN/FN for the Global Server metrics. `fedrf_diff/smpc.py`
gives it exactly those totals by **n-out-of-n additive secret sharing** among
the clients that passed the ZKP that round.

Every authenticated client c holds the vector
`x_c = (n_c·L_c, n_val,c·L_c, TP_c, FP_c, TN_c, FN_c)` (n_c = training size,
the weight of L_c in `L_avg`; n_val,c = validation size, for the pooled
log-loss). One aggregation with N clients:

| Step | Who | What |
|---|---|---|
| 1 | client c | encode: reals in fixed point (`round(v · 2^32)`), counts as they are, every value in Z_P, P = 2^127 − 1 (a Mersenne prime) |
| 2 | client c | share: draw N − 1 uniformly random vectors `s_c,1 … s_c,N−1` in Z_P^6 and set `s_c,N = x_c − Σ_j s_c,j (mod P)`; keep one share, send one to each other client (pairwise secure channels) |
| 3 | client j | add the shares it holds: `y_j = Σ_c s_c,j (mod P)`; send `y_j` to the server |
| 4 | server | reconstruct `Σ_j y_j = Σ_c x_c (mod P)` and decode: `L_avg`, pooled TP/FP/TN/FN and log-loss |

What it guarantees: every share and every partial sum `y_j` is a uniformly
random element of Z_P^6, so the server learns only `Σ_c x_c`, never one
client's loss or counts; any N − 1 shares of a client's vector are independent
of it. A coalition of the server and up to N − 2 clients learns nothing about
the other clients beyond the sum of their vectors. Assumptions:
honest-but-curious parties and no drop-out during one aggregation (only the
clients that passed the ZKP take part; the set is fixed before the shares are
drawn; a rejected client takes no part, which the tests check).

What it does **not** cover: the latent rows each client sends for the global
model are not secret-shared (the server trains its random forest on the
individual aligned rows) and, with no DP, they are sent exactly as computed.
The SMPC protects the evaluation results, not the latent representation.

Settings in `smpc.py`: `ENABLED = True`, `PRIME = 2^127 − 1`,
`FRACTION_BITS = 32`. Costs per client and round: 3 shares of 6 values × 16
bytes to the other clients plus one 96-byte partial sum to the server
(384 bytes up), 3 shares received (288 bytes down). The fixed-point encoding
changes `L_avg` by less than 10⁻¹² (tests check that every learned value and
stopping decision equals the run with plain sums).

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

* **Computation** (s per round): per client, RF kernel + diffusion map, ZKP
  proof generation, validation, decoding + RF retraining, SMPC share
  generation and summation; for the server, alignment + aggregation + `M_G`
  training + stopping rule, ZKP proof verification and SMPC reconstruction.
* **Communication** (bytes per round): uplink = latent matrix (float64),
  training labels (1 byte each), the ZKP messages (C_t, T, r_v, r_s) and the
  SMPC shares and partial sum; downlink = the ZKP challenge and outcome, the
  SMPC shares from the other clients, the pickled `M_G`, `A_c` and the
  out-of-bag labels. (The plain loss + TP/FP/TN/FN message of v12 is gone.)

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
| 1.5 | both | ZKP client authentication (steps 1–6); a failed authentication means the update is rejected this round |
| 2 | server | Procrustes `A_c` from landmark rows `P_c` to `P_ref`; `Z_G` = aligned rows of the authenticated clients; `M_G` = random forest on `Z_G` + out-of-bag labels `ŷ`; ΔZ |
| 3 | client | validation loss `L_c` and counts (kept on the client); k-NN decode of `Z_c A_c`; retrain `RF_c` on `D_train ∪ {(x̂, ŷ)}` |
| 4 | both | SMPC secure aggregation of `(n_c L_c, n_val,c L_c, TP_c, FP_c, TN_c, FN_c)`; server: `L_avg`, ΔL, Global Server metrics from the totals; stop after 3 consecutive rounds with ΔZ < ε_Z = 0.15 and ΔL < ε_L = 0.02 |

## Results (seed 42, no DP, 2048-bit ZKP group, SMPC)

Split: VA Long Beach 140/36, Cleveland 409/103. The full terminal output is
in `outputs/run/run_log.txt`. **Converged at round 7** (ΔZ fell from 0.53 in
round 2 to 0.09–0.12 from round 4, below ε_Z = 0.15, with ΔL < 0.02 in
rounds 5–7). ZKP: 28/28 authentications successful. SMPC: 7 secure
aggregations; the server never saw a client's own loss or counts.

Metrics averaged over the 7 rounds (clients: evaluation harness; Global
Server: SMPC totals):

| Client | Accuracy | Precision | Recall | Specificity | F1 | MCC | Kappa |
|---|---|---|---|---|---|---|---|
| Client 1(Hungary) | 0.884 | 0.748 | 0.992 | 0.829 | 0.853 | 0.781 | 0.761 |
| Client 2(Switzerland) | 0.825 | 0.968 | 0.844 | 0.429 | 0.900 | 0.213 | 0.196 |
| Client 3(VA Long Beach) | 0.909 | 0.961 | 0.915 | 0.889 | 0.938 | 0.772 | 0.768 |
| Client 4(Cleveland) | 0.804 | 0.845 | 0.711 | 0.886 | 0.772 | 0.610 | 0.603 |
| Global Server | 0.844 | 0.872 | 0.828 | 0.861 | 0.850 | 0.689 | 0.687 |

Times per client and round (ms, mean): ZKP authentication 3.0–3.5 (proof
generation 0.42–0.69, verification 2.5–2.8); SMPC 0.023–0.049 (share
generation 0.020–0.044, summation 0.003–0.005); the server's SMPC
reconstruction 0.009. Both are negligible next to ≈ 1–2.7 s of local work per
client per round. SMPC adds 384 bytes up and 288 bytes down per client and
round.

The learned values are identical to v13 (`Fed_Code_v13_Schnorr_NoDP`, the same
learning code without DP) to within 4 × 10⁻¹³: neither the ZKP nor the SMPC
changes what is learned. Switzerland (22 validation patients, one healthy) is
the least stable client. One seed: repeat over several seeds before reporting.
