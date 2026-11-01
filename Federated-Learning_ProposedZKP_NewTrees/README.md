# FedRF-Diff: Federated Random Forest with Tree-Level Aggregation (proposed ZKP, new trees only)

Folder `Federated-Learning_ProposedZKP_NewTrees`: the `Federated-Learning_ProposedZKP` project changed so that
**each client uploads only its newly generated trees**, never a tree it has already sent. The ZKP, local training,
refinement, convergence test, DP and bounded-update enforcement are the same; see
[New trees only](#new-trees-only-what-changed-from-federated-learning_proposedzkp).

A horizontal federated Random Forest for heart disease prediction across 4 hospitals. Clients send
**newly grown trees as explicit decision trees** (feature, threshold, children and leaf class
distribution for every node). The server **never averages parameters** and **never computes the stopping
condition**: every client runs a strict three-part convergence test on its own validation patients and sends
only a boolean flag.

```bash
pip install -r requirements.txt
python -m fedrf_diff        # full run, CSVs, forests and journal figures in outputs/run/
python -m pytest            # unit and pipeline tests
```

## Algorithm

1. **Initialisation (t = 0)**: each client trains `F_k^(0)` (25 trees) on its private data. `G^(0)` is empty and `q_k = 0`.
2. **Client → server**: at the start of round r each client authenticates with the proposed Pedersen ZKP
   (one interactive session, below) and uploads only its **newly generated trees** `N_k^(r-1)` with the schema
   metadata: `F_k^(0)` in round 1 and the supplementary forest `H_k^(r-1)` afterwards (25 trees). It also sends the
   ids of the trees in its forest `F_k^(r-1)` (ids only, for the consensus vote). It sends no raw data and no latent
   encodings. The server drops the upload of a client that fails authentication; that client re-sends the same new
   trees in its next accepted upload.
3. **Tree-level aggregation**: the server validates every tree (schema, classes, node references, splits, leaves,
   size and depth, provenance: only its creator may upload a tree, and a tree id is uploaded exactly once).
   Every accepted tree is stored in the server's repository. It pools
   `U^(r) = G^(r-1) ∪ N^(r-1) ∪ {repository trees named in a kept-id list}` (the same trees as the union of the
   client forests) and selects `G^(r) = SelectTrees(U^(r), A, B = 100)`. Policy `A = consensus`
   is a round-robin over creating clients that prefers the trees kept by the most client forests, newest first.
   The `client-balanced` and `newest` policies are also available.
4. **Broadcast**: `G^(r)`, round r and the verified schema go to every client.
5. **Local refinement**: each client first selects its Random Forest hyperparameters with a stratified 5-fold
   grid search on its own training patients (`max_features` × `min_samples_leaf`, 12 settings, balanced accuracy).
   The cross-validation forests are never shared, so they cost no DP budget. This local model selection is the bulk of
   the round time, about 30 s per round on a 4-core machine; adjust `local_tuning` in `config.py` to change it.
   The client then grows a supplementary forest `H_k^(r)` (25 trees) and builds
   `F_k^(r) = CapOrPrune(TreePool(G^(r), H_k^(r)), B_k = 50)`. This keeps the trees with the highest balanced
   accuracy on the client's training patients that each tree did not train on: out-of-bag patients for its own
   trees, all training patients for foreign trees. Validation data is never used for pruning.
6. **Three-part convergence test** (each client, on its fixed validation split):
   relative log-loss change `≤ τ_loss = 0.05`, share of changed predictions `≤ τ_pred = 0.05`, and share of
   predictions where `F_k^(r)` and `G^(r)` disagree `≤ τ_agree = 0.15`. All pass → `q_k += 1`, otherwise `q_k = 0`.
   The client sends only `b_k = [q_k ≥ P = 3]`.
7. **Stopping**: the server tallies the flags. If every client reports True it stops with "all-client convergence";
   at `t = R_max = 50` it stops with "maximum rounds reached". It returns `G^(r)` and every `F_k^(r)`.

Each newly grown tree gets Laplace DP on its leaf vectors (`b = 2C/ε`, `C = 1`, `ε ~ U[2, 3]` per round).
All parameters are in `fedrf_diff/config.py`.

The per-round validation tables in the console are written by the **evaluation harness** for the paper.
In the protocol itself, no validation metric reaches the server.

## Proposed ZKP (client authentication)

`fedrf_diff/zkp.py` is FedRF-Diff v12's proposed ZKP: the same protocol, group (2048-bit p, 256-bit q, from the
public seed "FedRF-Diff Pedersen ZKP group") and code; only β is read from `config.py` (`zkp_lambda = 2`).
One session per client and round:

| Step | Direction | What |
|---|---|---|
| 1 | client → server | Pedersen commitment `C_r = g^{v_r} h^{ρ_r} (C_{r-1})^β`, `C_0 = 1` (no registration) |
| 2 | client → server | fresh random `w, η`; `T = g^w h^η` |
| 3 | server → client | fresh random challenge `e` |
| 4 | client → server | `y_v = w + e·v_r`, `y_s = η + e·ρ_r` (mod q) |
| – | server | checks `g^{y_v} h^{y_s} = T · (C_base)^e` with `C_base = C_r / (C_{r-1})^β` |
| 5 / 6 | server → client | Authentication Successful (forest accepted) / Authentication Failed (forest dropped) |

`v_r = SHA-256(key ‖ client ‖ r) mod q` from a secret key that never leaves the client. `C_{r-1}` is the last
successfully authenticated commitment; client and server both move on to `C_r` only after a success.
Each session sends `C_r`, `T` (256 B each) and `y_v`, `y_s` (32 B each) up and `e` (32 B) plus the verdict (1 B)
down; both are counted in `communication_computation_cost.csv`. The forest travels with the responses and the
global forest with the verdict, so the ZKP adds one round trip per round (the convergence flag is the other).

What changed from `Federated-Learning`:

| | `Federated-Learning` (before) | this version |
|---|---|---|
| ZKP | Okamoto proof for a registered key `Y_k = g_1^{x_1} g_2^{x_2}`, Fiat–Shamir | proposed chained commitment, server-sent challenge |
| Registration | `Y_k` registered once | none (`C_0 = 1`) |
| Messages per session | 1 up (T, s_1, s_2) | C_r and T up, e down, y_v and y_s up, verdict down |
| Bound to the update | yes (hash of the forest in the challenge) | no: authenticates the client only, as in the diagram; the forest is checked by the server's structural validation |
| β | number of generators | weight of `C_{r-1}` |

`run_federation(..., impersonate={t: "client_k"})` makes an impostor replay client w's previous session in
round r; the fresh challenge makes it fail and its forest is not aggregated (tested). The ZKP changes no learned
value (tested: same metrics with and without the ZKP), so the learning results equal `Federated-Learning`'s.

## Data

`data/raw/heart_statlog_cleveland_hungary_final.csv` (1190 rows) stacks five UCI sources in a fixed order:
Hungary 0–293, Switzerland 294–416, VA Long Beach 417–616, Statlog 617–886, Cleveland 887–1189.
Statlog is Cleveland Clinic data, so it joins the Cleveland client, and its copies drop out with the 272 exact duplicates.

| Client | Patients | Train / validation | Heart disease | Cholesterol missing |
|---|---|---|---|---|
| Client 1(Hungary) | 293 | 234 / 59 | 36.2% | 0% |
| Client 2(Switzerland) | 123 | 98 / 25 | 93.5% | 100% |
| Client 3(VA Long Beach) | 199 | 159 / 40 | 74.4% | 24.6% |
| Client 4(Cleveland) | 303 | 242 / 61 | 45.9% | 0% |

## Outputs (`outputs/run/`)

`validation_metrics.csv` (G^(r) per client and Global Server), `local_tuning.csv` (selected hyperparameters, CV score and tuning time), `local_convergence.csv` (the three tests, q_k, b_k
and F_k^(r) metrics), `rounds.csv` (|U|, |G|, flags, decision, DP budget), `zkp_times.csv`, `tree_transfer_times.csv`,
`communication_computation_cost.csv`, `dp_budget_accuracy.csv`, `console_output.txt`, and `forests/`
(the returned G and every F_k as explicit trees). The figures are `convergence`, `zkp_times`, `computation_cost`,
`communication_cost` and `accuracy_vs_dp_budget`.

## Bounded-update enforcement

After ZKP authentication and the per-tree validation, `FederatedServer.receive` checks each
upload against public bounds derived from the configuration (`server.Limits`):

- **trees per update** and **new tree ids** <= max(`initial_trees`, `supplementary_trees`) x (r - r_last),
  because a client grows one H_k per round and every uploaded tree must be new. Here r_last
  is the client's last accepted round;
- **kept ids** <= max(`initial_trees`, `local_budget`), because CapOrPrune keeps F_k within B_k.
  Every kept id must name a tree the server already holds or one in the same upload.

An upload outside a bound is rejected as a whole: none of its trees enter the repository or the
candidate pool U^(r), and none of its new tree ids are registered.

Each round prints a "Bounded-update enforcement (per client)" table, and the run summary adds a
"Bounded-update enforcement summary". The checks are also saved as extra columns in
`tree_transfer_times.csv`.

Set `"bounded_update": False` in `config.py` to switch the check off. Tests: `tests/test_bounds.py`.

## New trees only (what changed from `Federated-Learning_ProposedZKP`)

| | `Federated-Learning_ProposedZKP` | this version |
|---|---|---|
| Upload in round r | the complete forest `F_k^(r-1)` (up to B_k = 50 trees, global and own trees) | only the new trees `N_k^(r-1)`: `F_k^(0)` (round 1), then `H_k^(r-1)` (25 trees) + kept ids |
| Re-sent trees | every kept tree travels again each round | never: a known tree id is rejected |
| Server state | only the current upload | repository of every accepted tree |
| Candidate pool U^(r) | union of the uploaded forests | G^(r-1) ∪ new trees ∪ repository trees named in kept ids |
| Consensus vote | number of uploaded forests holding the tree | number of kept-id lists naming the tree |
| Bounded update | ≤ max(initial, B_k) trees, ≤ max(initial, H) x Δr new ids | ≤ max(initial, H) x Δr trees (all new), ≤ max(initial, B_k) kept ids |

Code: `client.py` (`pending`, `serialize`, `acknowledge`), `server.py` (`check_update`, `receive`,
`update_bound`, `aggregate`, `repository`), `pipeline.py` (Steps 2-3). The transfer table adds a
"Trees in F_k^(r-1)" column next to "Trees Sent", so the saving is visible per client and round.

## Privacy scope

Each released leaf vector is ε-DP (pure DP). Leaves partition a tree's training sample, so a patient costs ε per
tree, `m·ε` for the m trees grown in a round, and these add up over rounds (no tree is sent twice) (the per-patient bound is printed as "per patient").
Split features and thresholds are learned on the raw records and are **not** privatised, so whole-tree sharing
is not an end-to-end DP release. The noise comes from a seeded numpy generator for reproducibility, not from a
hardened DP sampler. The proposed ZKP authenticates the clients (it reveals nothing about their secrets); it
does not hide the trees from the server and does not sign their bytes.
