"""Stage 4: simple SMPC - secure aggregation by n-out-of-n additive secret sharing.

The server needs only *totals* of what the clients report after the local
evaluation (Stage 3): the weighted validation loss for L_glob and the summed
TP/FP/TN/FN counts for the Global Server metrics. With this module it learns
those totals and nothing else; no client's own loss or counts reach it.

Every authenticated client k holds a vector v_k of m values

    v_k = (N_k * L_k, N_val,k * L_k, TP_k, FP_k, TN_k, FN_k)

(N_k = training-set size, the weight of L_k in L_glob; N_val,k = validation
size). One aggregation, for the N authenticated clients:

    1. encode   reals in fixed point (round(v * 2^32)), integers as they are,
                every value as an element of Z_P, P = 2^127 - 1 (a Mersenne prime)
    2. share    draw N - 1 uniformly random vectors s_{k,1..N-1} in Z_P^m and set
                s_{k,N} = v_k - sum_j s_{k,j} (mod P); keep one share and send
                one share to each other client (pairwise secure channels)
    3. sum      each client j adds the shares it holds: y_j = sum_k s_{k,j} (mod P),
                and sends y_j to the server
    4. reconstruct  the server computes sum_j y_j = sum_k v_k (mod P) and decodes it

Each single share, and each partial sum y_j, is a uniformly random element of
Z_P^m, so the server learns only sum_k v_k; any N - 1 shares of a client's
vector are independent of it. A coalition of the server and up to N - 2
clients learns nothing about the remaining honest clients beyond the sum of
their vectors. The scheme assumes honest-but-curious parties and no drop-outs
during one aggregation: only the clients that passed the ZKP authentication
take part (the set is fixed before the shares are drawn).

Not covered: the latent rows each client sends for the global model are not
secret-shared (the server trains its random forest on them) and, with no DP,
they are sent as computed.

All SMPC settings live below; there is no command-line option for them.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

# --- SMPC settings -----------------------------------------------------------
ENABLED = True  # secure aggregation of the losses and confusion counts every round
PRIME = 2**127 - 1  # P, a Mersenne prime: share arithmetic is mod P
FRACTION_BITS = 32  # fixed point for real values: v -> round(v * 2^FRACTION_BITS)
ELEMENT_BYTES = (PRIME.bit_length() + 7) // 8  # 16 bytes per element of Z_P
# The aggregated vector: (name, is_real).
FIELDS = (("loss_x_n_train", True), ("loss_x_n_val", True),
          ("tp", False), ("fp", False), ("tn", False), ("fn", False))


def settings() -> dict:
    return {"enabled": ENABLED, "scheme": "n-out-of-n additive secret sharing (secure aggregation)",
            "prime": "2^127 - 1", "fraction_bits": FRACTION_BITS, "fields": [f for f, _ in FIELDS]}


# --- encoding ------------------------------------------------------------------
def encode(value: float, real: bool) -> int:
    """A value as an element of Z_P (fixed point for reals; negatives wrap around)."""
    if not real and int(value) != value:
        raise ValueError(f"expected an integer, got {value!r}")
    v = round(value * (1 << FRACTION_BITS)) if real else int(value)
    return v % PRIME


def decode(element: int, real: bool) -> float | int:
    """Inverse of encode (elements above P/2 are negative)."""
    v = element - PRIME if element > PRIME // 2 else element
    return v / (1 << FRACTION_BITS) if real else v


# --- shares --------------------------------------------------------------------
def share(vector: list[int], n: int) -> list[list[int]]:
    """n additive shares of ``vector`` (each a list of elements of Z_P) that sum to it mod P."""
    if n < 2:
        raise ValueError("secret sharing needs at least 2 parties")
    shares = [[secrets.randbelow(PRIME) for _ in vector] for _ in range(n - 1)]
    last = [(x - sum(s[i] for s in shares)) % PRIME for i, x in enumerate(vector)]
    return shares + [last]


def add(vectors: list[list[int]]) -> list[int]:
    """Element-wise sum mod P."""
    return [sum(col) % PRIME for col in zip(*vectors)]


# --- one aggregation -------------------------------------------------------------
@dataclass
class ClientCost:
    """One client's part of one aggregation: times (s) and message sizes (bytes)."""

    share_s: float  # encode + draw the shares
    sum_s: float  # add the shares it holds
    shares_sent: int  # to the other clients
    bytes_up: int  # shares to the other clients + its partial sum to the server
    bytes_down: int  # shares from the other clients

    @property
    def total_s(self) -> float:
        return self.share_s + self.sum_s


@dataclass
class Aggregation:
    totals: dict[str, float | int]  # what the server learns: the sums of every field
    clients: dict[str, ClientCost] = field(default_factory=dict)
    reconstruct_s: float = 0.0  # server: add the partial sums and decode
    partial_sums: dict[str, list[int]] = field(default_factory=dict)  # what the server receives


def secure_aggregate(vectors: dict[str, list[float]]) -> Aggregation:
    """Sum the clients' vectors (one value per FIELDS entry) by additive secret sharing.

    ``vectors`` maps a client to its plain vector; it never leaves the client in
    the protocol: only shares and partial sums are exchanged.
    """
    names = list(vectors)
    n, m = len(names), len(FIELDS)
    if any(len(v) != m for v in vectors.values()):
        raise ValueError(f"every vector needs {m} values: {[f for f, _ in FIELDS]}")
    held: dict[str, list[list[int]]] = {name: [] for name in names}
    share_s = {}
    for name in names:  # 1-2. encode and share; share j goes to client j
        t0 = time.perf_counter()
        x = [encode(v, real) for v, (_, real) in zip(vectors[name], FIELDS)]
        for holder, s in zip(names, share(x, n)):
            held[holder].append(s)
        share_s[name] = time.perf_counter() - t0
    partial, sum_s = {}, {}
    for name in names:  # 3. each client adds the shares it holds
        t0 = time.perf_counter()
        partial[name] = add(held[name])
        sum_s[name] = time.perf_counter() - t0
    t0 = time.perf_counter()  # 4. the server adds the partial sums and decodes
    total = add(list(partial.values()))
    totals = {f: decode(e, real) for e, (f, real) in zip(total, FIELDS)}
    reconstruct_s = time.perf_counter() - t0
    vec_bytes = m * ELEMENT_BYTES
    costs = {name: ClientCost(share_s=share_s[name], sum_s=sum_s[name], shares_sent=n - 1,
                              bytes_up=(n - 1) * vec_bytes + vec_bytes, bytes_down=(n - 1) * vec_bytes)
             for name in names}
    return Aggregation(totals=totals, clients=costs, reconstruct_s=reconstruct_s, partial_sums=partial)
