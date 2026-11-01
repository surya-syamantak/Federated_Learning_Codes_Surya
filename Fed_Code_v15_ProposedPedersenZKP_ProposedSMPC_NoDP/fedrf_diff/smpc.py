"""Stage 4: proposed SMPC - Adaptive Prime Secure Aggregation with Self-Randomized Pairwise Masking.

The server needs only *totals* of what the clients report after the local
evaluation (Stage 3): the weighted validation loss for L_glob and the summed
TP/FP/TN/FN counts for the Global Server metrics. Every authenticated client k
holds the vector

    v_k = (N_k * L_k, N_val,k * L_k, TP_k, FP_k, TN_k, FN_k)

(N_k = training-set size, the weight of L_k in L_glob; N_val,k = validation
size) and sends the server only a masked copy of it; the masks cancel in the sum.

  Phase 1 (one-time)  pairwise key setup: S_ij for every pair of clients
  Phase 2 (one-time)  prime pool Pool = {p_0, ..., p_{|Pool|-1}} from a public seed;
                      quantization scaling factor S_quant fixed
  Phase 3 (round r)   adaptive prime selection: idx_r = H(seed || r) mod |Pool|,
                      p_r = Pool[idx_r]
  Phase 4 (client k)  quantize: Q(v_k) = round(S_quant * v_k)
  Phase 5 (client k)  stream outputs: m_kj = PRG(S_ij, r, p_r) in Z_{p_r}^m for every
                      other participant j (m_kj = m_jk: both ends compute the same stream)
           self-mask  M_k = sum_j sign(k, j) * m_kj,  sign(k, j) = +1 if k < j, -1 if k > j
  Phase 6 (client k)  masked model: y_k = Q(v_k) + M_k (mod p_r); send y_k to the server
  Phase 7 (server)    secure aggregation: Σ_r = sum_k y_k (mod p_r) = sum_k Q(v_k) (mod p_r),
                      because every m_kj enters once with + (at k < j) and once with -
  Phase 8 (server)    dequantization: x_hat = Σ_r / S_quant  (the totals; no client's own vector)

Implementation choices the diagram leaves open (made so that it is correct
and secure on this data, not only on the toy numbers of the diagram):

* Prime size. The aggregate must stay below p_r, or it wraps around mod p_r.
  The pool holds POOL_SIZE primes of PRIME_BITS = 61 bits (the diagram's 997,
  1009, 1013 cannot hold a sum of N_k * L_k ~ 330 at S_quant = 10^6). Every client
  checks |Q(v_k)| < p_r / (2 N) for each value, so the signed sum of N clients
  can neither wrap nor be confused with a negative number.
* Quantization. S_quant = 10^6 (fixed, as in the diagram, but finer than its 100):
  L_glob keeps 6 decimals, ample for the stopping test Delta L < 0.02; counts
  are integers and decode exactly.
* Pairwise keys. Each client draws a secret scalar d_k and publishes
  D_k = g^{d_k} (Diffie-Hellman in the ZKP's 2048-bit group, relayed by the
  server); S_ij = SHA-256(D_k^{d_j} || i || j) for k < j. The server sees only
  the public D_k, so it cannot recompute any mask (a key derived from a seed
  the server knows would let it unmask every client).
* Fresh streams. m_kj is derived from S_ij, the round r and p_r (SHA-256 in
  counter mode, 512 bits per element reduced mod p_r), so masks never repeat:
  subtracting two rounds' messages reveals nothing.
* Participants and drop-outs (the diagram's note). The masks are built over
  the confirmed participant set: the clients that passed the ZKP this round,
  fixed before any mask is drawn. If a participant still drops out after its
  mask was formed, the survivors send the server their pairwise terms with the
  dropped clients (dropout-recovery step) and the server removes them, so the
  aggregate is the exact sum over the survivors. The server must then discard
  any late message of a dropped client (a revealed pairwise term would unmask it).

Security: each y_k is Q(v_k) plus pseudorandom terms keyed with secrets the
server does not have, so it reveals nothing beyond the final sum (honest-but-
curious server; the masks of client k are known to its partners j, so a
coalition of the server with all other clients learns v_k, as with any sum).
Not covered: the latent rows each client sends for the global model are not
masked (the server trains its random forest on them) and, with no DP, they are
sent as computed.

All SMPC settings live below; there is no command-line option for them.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field

from . import zkp

# --- SMPC settings -----------------------------------------------------------
ENABLED = True  # secure aggregation of the losses and confusion counts every round
POOL_SEED = "FedRF-Diff adaptive prime pool"  # public seed of the prime pool and of idx_r
POOL_SIZE = 3  # primes in the pool (the diagram's |Pool| = 3)
PRIME_BITS = 61  # bit length of every pool prime
SCALE = 10**6  # S_quant, the quantization scaling factor
ELEMENT_BYTES = (PRIME_BITS + 7) // 8  # 8 bytes per masked value
# The aggregated vector (every value is quantized with S_quant; counts decode exactly).
FIELDS = ("loss_x_n_train", "loss_x_n_val", "tp", "fp", "tn", "fn")


def settings() -> dict:
    return {"enabled": ENABLED,
            "scheme": "adaptive prime secure aggregation with self-randomized pairwise masking",
            "pool_seed": POOL_SEED, "pool": [str(p) for p in prime_pool()], "prime_bits": PRIME_BITS,
            "scale_S_q": SCALE, "key_agreement": "Diffie-Hellman in the ZKP group, S_ij = SHA-256",
            "fields": list(FIELDS)}


# --- Phase 2 and 3: prime pool, adaptive prime -------------------------------------
def _hash_int(*parts, bits: int = 256) -> int:
    out, counter = b"", 0
    while len(out) * 8 < bits:
        h = hashlib.sha256()
        for part in (*parts, counter):
            h.update(str(part).encode() if not isinstance(part, bytes) else part)
            h.update(b"|")
        out += h.digest()
        counter += 1
    return int.from_bytes(out, "big") >> (len(out) * 8 - bits)


def prime_pool(seed: str = POOL_SEED, size: int = POOL_SIZE, bits: int = PRIME_BITS) -> tuple[int, ...]:
    """Phase 2: ``size`` distinct ``bits``-bit primes derived from the public seed."""
    pool: list[int] = []
    i = 0
    while len(pool) < size:
        p = int(zkp._next_prime(_hash_int(seed, "pool", i, bits=bits) | (1 << (bits - 1))))
        if p.bit_length() == bits and p not in pool:
            pool.append(p)
        i += 1
    return tuple(pool)


def select_prime(round_: int, pool: tuple[int, ...] | None = None, seed: str = POOL_SEED) -> tuple[int, int]:
    """Phase 3: idx_r = H(seed || r) mod |Pool| and p_r = Pool[idx_r]."""
    pool = prime_pool() if pool is None else pool
    idx = _hash_int(seed, "round", round_) % len(pool)
    return idx, pool[idx]


# --- Phase 4 and 8: quantization ------------------------------------------------------
def quantize(value: float) -> int:
    """Q(x) = round(S_quant * x)."""
    return round(value * SCALE)


def dequantize(element: int, p: int) -> float:
    """Inverse of quantize for a value in Z_p (elements above p/2 are negative)."""
    v = element - p if element > p // 2 else element
    return v / SCALE


# --- Phase 5: pairwise streams and self-mask ---------------------------------------
def stream(key: bytes, round_: int, p: int, m: int) -> list[int]:
    """m_kj: m pseudorandom elements of Z_p from S_ij, the round and the prime."""
    return [_hash_int(key, "stream", round_, p, k, bits=512) % p for k in range(m)]


def sign(i: int, j: int) -> int:
    """+1 if k < j, -1 if k > j (client indices)."""
    return 1 if i < j else -1


# --- the parties ------------------------------------------------------------------
@dataclass
class ClientCost:
    """One client's part of one aggregation: times (s) and message sizes (bytes)."""

    mask_s: float  # quantize + streams + self-mask + masked model
    recovery_s: float = 0.0  # dropout recovery (only if a participant dropped)
    bytes_up: int = 0  # y_k (and recovery terms)
    bytes_down: int = 0

    @property
    def total_s(self) -> float:
        return self.mask_s + self.recovery_s


@dataclass
class Aggregation:
    round: int
    prime: int  # p_r
    prime_index: int  # idx_r
    totals: dict[str, float]  # what the server learns: the sums of every field
    participants: list[str]
    dropped: list[str] = field(default_factory=list)
    clients: dict[str, ClientCost] = field(default_factory=dict)
    masked: dict[str, list[int]] = field(default_factory=dict)  # y_k as the server receives them
    aggregate_s: float = 0.0  # server: sum, recovery, dequantization


class SecureAggregator:
    """All clients' SMPC state (keys, indices) and the server's aggregation, for one federation.

    Phase 1 and 2 run once in the constructor; ``aggregate`` runs Phases 3-8 for one round.
    In a deployment each client holds only its own secret and keys; here one object
    simulates all parties, and only y_k (and recovery terms) cross to the server side.
    """

    def __init__(self, clients: list[str], pp: zkp.PublicParams | None = None):
        self.pp = zkp.setup() if pp is None else pp
        self.index = {name: i + 1 for i, name in enumerate(clients)}  # C_1, C_2, ... (fixed order)
        self.pool = prime_pool()  # Phase 2
        # Phase 1: Diffie-Hellman key agreement for every pair (one-time).
        self.setup_s: dict[str, float] = {}
        secret, public = {}, {}
        for name in clients:
            t0 = time.perf_counter()
            secret[name] = secrets.randbelow(int(self.pp.q) - 1) + 1
            public[name] = int(self.pp.g_pow(secret[name]))
            self.setup_s[name] = time.perf_counter() - t0
        self._keys: dict[tuple[str, str], bytes] = {}
        for a in clients:
            t0 = time.perf_counter()
            for b in clients:
                if a == b:
                    continue
                lo, hi = sorted((a, b), key=self.index.get)
                shared = int(zkp._powmod(zkp._mpz(public[b]), secret[a], self.pp.p))
                key = hashlib.sha256(shared.to_bytes(self.pp.p_bytes, "big")
                                     + f"|{self.index[lo]}|{self.index[hi]}".encode()).digest()
                self._keys[(a, b)] = key
            self.setup_s[a] += time.perf_counter() - t0
        n = len(clients)
        # Publish D_k (up), receive the other N - 1 public keys (down, relayed by the server).
        self.setup_bytes_up = self.pp.p_bytes
        self.setup_bytes_down = (n - 1) * self.pp.p_bytes

    def key(self, a: str, b: str) -> bytes:
        return self._keys[(a, b)]

    def mask(self, name: str, round_: int, p: int, participants: list[str], m: int) -> list[int]:
        """Phase 5 self-mask of one client over the confirmed participant set."""
        i = self.index[name]
        total = [0] * m
        for other in participants:
            if other == name:
                continue
            s = sign(i, self.index[other])
            for k, r in enumerate(stream(self.key(name, other), round_, p, m)):
                total[k] = (total[k] + s * r) % p
        return total

    def aggregate(self, round_: int, vectors: dict[str, list[float]],
                  drop_after_masking: tuple[str, ...] = ()) -> Aggregation:
        """Phases 3-8 for one round over the confirmed participants ``vectors``.

        ``drop_after_masking`` simulates participants whose masked model never arrives
        (tests); the survivors then run the dropout-recovery step.
        """
        participants = sorted(vectors, key=self.index.get)
        m = len(FIELDS)
        if any(len(v) != m for v in vectors.values()):
            raise ValueError(f"every vector needs {m} values: {list(FIELDS)}")
        idx, p = select_prime(round_, self.pool)  # Phase 3
        limit = p // (2 * len(participants))
        costs, masked = {}, {}
        for name in participants:  # Phases 4-6 on every client
            t0 = time.perf_counter()
            q = [quantize(v) for v in vectors[name]]
            if any(abs(v) >= limit for v in q):
                raise OverflowError(f"{name}: a quantized value exceeds p_r / (2N); use larger primes")
            mask = self.mask(name, round_, p, participants, m)
            masked[name] = [(v + r) % p for v, r in zip(q, mask)]
            costs[name] = ClientCost(mask_s=time.perf_counter() - t0, bytes_up=m * ELEMENT_BYTES)
        dropped = [n for n in participants if n in drop_after_masking]
        survivors = [n for n in participants if n not in dropped]
        received = {n: masked[n] for n in survivors}
        # Dropout recovery: every survivor reveals its pairwise terms with the dropped clients.
        recovery = {}
        for name in survivors if dropped else []:
            t0 = time.perf_counter()
            i = self.index[name]
            terms = [0] * m
            for other in dropped:
                s = sign(i, self.index[other])
                for k, r in enumerate(stream(self.key(name, other), round_, p, m)):
                    terms[k] = (terms[k] + s * r) % p
            recovery[name] = terms
            costs[name].recovery_s = time.perf_counter() - t0
            costs[name].bytes_up += m * ELEMENT_BYTES
        for name in dropped:
            costs[name].bytes_up = 0  # its masked model never arrived
        t0 = time.perf_counter()  # Phases 7-8 on the server
        agg = [sum(col) % p for col in zip(*received.values())]
        if recovery:
            agg = [(a - sum(col)) % p for a, col in zip(agg, zip(*recovery.values()))]
        totals = {f: dequantize(e, p) for e, f in zip(agg, FIELDS)}
        aggregate_s = time.perf_counter() - t0
        return Aggregation(round=round_, prime=p, prime_index=idx, totals=totals, participants=participants,
                           dropped=dropped, clients=costs, masked=received, aggregate_s=aggregate_s)
