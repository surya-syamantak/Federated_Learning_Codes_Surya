# FedRF-Diff (Fed_Code_v15_ProposedPedersenZKP_ProposedSMPC_NoDP)

Federated learning with random-forest diffusion maps, Procrustes alignment, the
proposed Pedersen zero-knowledge proof that authenticates every client every
round, and the **proposed SMPC: Adaptive Prime Secure Aggregation with
Self-Randomized Pairwise Masking**, **without any differential privacy**. It
implements "Algorithm 1: Iterative Closed-Loop FedRF-Diff Framework" on the
combined UCI heart-disease data (`data/heart_statlog_cleveland_hungary_final.csv`).

Changes from v14 (`Fed_Code_v14_ProposedPedersenZKP_SMPC_NoDP`): only the SMPC.
`fedrf_diff/smpc.py` now implements the proposed scheme (Phases 1-8 of the
diagram) instead of n-out-of-n additive secret sharing, in the same place and
for the same values: every client's weighted validation loss and TP/FP/TN/FN
counts, of which the server learns only the totals (`L_avg` and the Global
Server metrics). No client-to-client messages are needed any more: each client
sends the server one masked vector. The proposed ZKP, the learning code and
"no DP" are unchanged; the learned values equal v14's (and v13's) to within
the quantization (10⁻⁹). See [Proposed SMPC](#proposed-smpc-adaptive-prime-secure-aggregation).

## Quick start

```bash
pip install -r requirements-lock.txt   # exact versions behind the results below
python -m fedrf_diff        # seed 42, no DP, proposed ZKP + proposed SMPC, all CSVs and figures
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
* **SMPC (adaptive prime secure aggregation):** per client the *SMPC Masking
  Time* (ms), the number of masked values and the bytes sent and received;
  then the round's prime `p_t = Pool[idx_t]`, the server's aggregation +
  dequantization time and the totals it learned;
* **validation metrics** of that round's global model for each client and for
  the Global Server: *Accuracy, Precision, Recall, Specificity, F1 Score,
  Matthews Correlation Coefficient (MCC), Cohen's Kappa*;
* the server line: ΔZ (and its threshold), `L_avg`, ΔL, patience and round
  time.

After the last round: the metrics averaged over all rounds, the ZKP time
analysis, the SMPC time analysis (min, max and mean masking and total time per
client, the server's aggregation time, the one-time key setup per client, the
prime used in every round and the bytes per client and round) and the
encoding/decoding time analysis.

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
| `smpc_times.csv` | per round and client: prime index and p_t, values, key setup time (first round), masking time, recovery time, total, SMPC bytes up/down; Global Server: aggregation + dequantization time |
| `costs.csv` | per round and client (and Global Server): computation time and its parts (incl. `smpc_s`), bytes sent and received |
| `run_log.txt` | the terminal output of the run |

Figures (`figures/`, PNG at 600 dpi plus a vector PDF):

| Figure | Panels |
|---|---|
| `convergence.png` | (a) ΔZ with ε_Z, (b) \|ΔL\| with ε_L, (c) `L_avg`; converged round marked |
| `zkp_times.png` | (a) authentication, (b) proof generation, (c) proof verification time per client per round |
| `smpc_times.png` | (a) masking time per client per round, (b) one-time key setup per client, (c) the Global Server's aggregation + dequantization time per round, labelled with the prime p_t used |
| `computation_cost.png` | (a) computation time per round per client and server, (b) mean per-client breakdown (ZKP and SMPC times in ms under each client), (c) mean ZKP proof generation time per client |
| `communication_cost.png` | (a) uplink, (b) downlink KB per round per client |

## Proposed SMPC (adaptive prime secure aggregation)

`fedrf_diff/smpc.py` implements **Adaptive Prime Secure Aggregation with
Self-Randomized Pairwise Masking** for the values the server needs only as
totals. Every client c that passed the ZKP this round (the *confirmed
participants*) holds `x_c = (n_c·L_c, n_val,c·L_c, TP_c, FP_c, TN_c, FN_c)`
(n_c = training size, the weight of L_c in `L_avg`; n_val,c = validation
size, for the pooled log-loss).

| Phase | When / who | What | Code |
|---|---|---|---|
| 1 | one-time, clients | pairwise keys `K_ij` (Diffie-Hellman, below) | `SecureAggregator.__init__` |
| 2 | one-time, public | prime pool `Pool = {p_0, p_1, p_2}` from the public seed; quantization factor `S_q` fixed | `prime_pool` |
| 3 | round t, everyone | adaptive prime: `idx_t = H(seed ‖ t) mod 3`, `p_t = Pool[idx_t]` | `select_prime` |
| 4 | client i | quantize: `Q(x_i) = round(S_q · x_i)` | `quantize` |
| 5 | client i | stream outputs `r_ij = PRG(K_ij, t, p_t)` for every other participant j; self-mask `mask_i = Σ_j sign(i, j) r_ij`, `sign = +1` if i < j, `−1` if i > j | `stream`, `sign`, `SecureAggregator.mask` |
| 6 | client i | masked model `c_i = Q(x_i) + mask_i (mod p_t)`; send `c_i` to the server | `SecureAggregator.aggregate` |
| 7 | server | secure aggregation `Agg = Σ_i c_i (mod p_t) = Σ_i Q(x_i)`: every `r_ij` enters once with + and once with − | `SecureAggregator.aggregate` |
| 8 | server | dequantization `x̂ = Agg / S_q`: `L_avg`, pooled TP/FP/TN/FN and log-loss | `dequantize` |

The tests reproduce the diagram's worked example number for number (round 5,
p = 997: masks [28, 19], [982, 992], [984, 983]; c = [70, 37], [16, 21],
[14, 1]; Agg = [100, 59]; θ₅ = [1.00, 0.59]).

**Choices made so that it is correct and secure on this data** (the diagram's
toy numbers would fail here):

| Diagram | Here | Why |
|---|---|---|
| Pool {997, 1009, 1013} | 3 primes of 61 bits from the public seed `POOL_SEED` (`2118740186616301883`, `1355351681047425827`, `1211642731495667789`) | the aggregate must stay below p_t, or it wraps around mod p_t: Σ n_c L_c ≈ 330 is 3.3·10⁸ after quantization. Every client checks `abs(Q(x_i)) < p_t / (2N)`, so the sum can neither wrap nor be mistaken for a negative number (`OverflowError` otherwise) |
| S_q = 100 | S_q = 10⁶ | 2 decimals would blur `L_avg` for the stopping test ΔL < 0.02; with 10⁶ the totals differ from the plain sums by < 10⁻⁹ (tests) |
| keys "generated once from seed" | each client draws a secret scalar a_i and publishes `A_i = g^{a_i}` in the ZKP's 2048-bit group (relayed by the server); `K_ij = SHA-256(A_j^{a_i} ‖ i ‖ j)` | if the server knew the seed of the keys it could recompute every mask and unmask every client |
| stream outputs r_ij | `PRG(K_ij, t, p_t)`: SHA-256 in counter mode, 512 bits per element reduced mod p_t | fresh masks every round: subtracting two rounds' messages reveals nothing |
| note on drop-outs | masks are built over the confirmed participants (clients that passed the ZKP, fixed before masking); if a participant still drops after masking, the survivors send their pairwise terms with it and the server removes them (dropout-recovery step) | the masks cancel only over the clients whose `c_i` arrive (the diagram's note); the server must then discard any late `c_i` of the dropped client |

What it guarantees: each `c_i` is `Q(x_i)` plus pseudorandom terms keyed with
secrets the server does not have, so the server learns only `Σ_i Q(x_i)`,
never one client's loss or counts (honest-but-curious server). The masks of
client i are known to its partners, so only a coalition of the server with
all the other clients could isolate `x_i`, as with any sum. A client rejected
by the ZKP takes no part (tests). Not covered: the latent rows each client
sends for the global model are not masked (the server trains its random
forest on them) and, with no DP, they are sent exactly as computed.

Settings in `smpc.py`: `ENABLED = True`, `POOL_SEED`, `POOL_SIZE = 3`,
`PRIME_BITS = 61`, `SCALE = 10**6`. Costs: one-time key setup ≈ 2 ms per
client (publish `A_i`: 256 bytes up; receive the other three: 768 bytes down,
charged to round 1); per round one masked vector of 6 × 8 bytes = **48 bytes
up and nothing down** per client, against 384 up and 288 down for v14's
additive sharing, and no client-to-client messages.

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
  proof generation, validation, decoding + RF retraining, SMPC masking (and
  the one-time key setup in round 1); for the server, alignment + aggregation
  + `M_G` training + stopping rule, ZKP proof verification and SMPC
  aggregation + dequantization.
* **Communication** (bytes per round): uplink = latent matrix (float64),
  training labels (1 byte each), the ZKP messages (C_t, T, r_v, r_s) and the
  SMPC masked vector `c_i` (plus `A_i` once); downlink = the ZKP challenge and
  outcome, the other clients' `A_j` (once), the pickled `M_G`, `A_c` and the
  out-of-bag labels.

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
| 4 | both | proposed SMPC (Phases 3-8) of `(n_c L_c, n_val,c L_c, TP_c, FP_c, TN_c, FN_c)` with this round's prime p_t; server: `L_avg`, ΔL, Global Server metrics from the totals; stop after 3 consecutive rounds with ΔZ < ε_Z = 0.15 and ΔL < ε_L = 0.02 |

## Results (seed 42, no DP, 2048-bit ZKP group, proposed SMPC)

Split: VA Long Beach 140/36, Cleveland 409/103. The full terminal output is
in `outputs/run/run_log.txt`. **Converged at round 7** (ΔZ fell from 0.53 in
round 2 to 0.09–0.12 from round 4, below ε_Z = 0.15, with ΔL < 0.02 in
rounds 5–7). ZKP: 28/28 authentications successful. SMPC: 7 secure
aggregations; adaptive prime `Pool[0]` in rounds 1 and 5, `Pool[1]` in rounds
2 and 7, `Pool[2]` in rounds 3, 4 and 6; the server never saw a client's own
loss or counts.

Metrics averaged over the 7 rounds (clients: evaluation harness; Global
Server: SMPC totals):

| Client | Accuracy | Precision | Recall | Specificity | F1 | MCC | Kappa |
|---|---|---|---|---|---|---|---|
| Client 1(Hungary) | 0.884 | 0.748 | 0.992 | 0.829 | 0.853 | 0.781 | 0.761 |
| Client 2(Switzerland) | 0.825 | 0.968 | 0.844 | 0.429 | 0.900 | 0.213 | 0.196 |
| Client 3(VA Long Beach) | 0.909 | 0.961 | 0.915 | 0.889 | 0.938 | 0.772 | 0.768 |
| Client 4(Cleveland) | 0.804 | 0.845 | 0.711 | 0.886 | 0.772 | 0.610 | 0.603 |
| Global Server | 0.844 | 0.872 | 0.828 | 0.861 | 0.850 | 0.689 | 0.687 |

Times per client and round (ms, mean): ZKP authentication 3.1–3.8 (proof
generation 0.50–0.68, verification 2.6–3.2); SMPC masking 0.13–0.18; the
server's SMPC aggregation + dequantization 0.02; one-time key setup 1.7–2.2
per client. All negligible next to ≈ 1–2.7 s of local work per client per
round. SMPC adds 48 bytes up per client and round.

The learned values are identical to v14 (additive sharing) to within 1.2 × 10⁻⁹
and to v13 (no SMPC): neither the ZKP nor the SMPC changes what is learned.
Switzerland (22 validation patients, one healthy) is the least stable client.
One seed: repeat over several seeds before reporting.
