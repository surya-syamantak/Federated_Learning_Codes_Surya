"""Stage 1.5: Pedersen-based zero-knowledge proof (ZKP) for client verification.

Implements the "Pedersen-based Zero-Knowledge Proof (ZKP) in Federated
Learning" protocol once per client and round. The client's update is
everything it sends the server that round: its DP-protected latent matrix
Z_k (patient and landmark rows) and its training labels. The ZKP is used only
to verify clients: a client whose proof is invalid has its update rejected for
that round; the proof changes no learned value.

  Public: primes p and q with q | p - 1, generators g and h of the order-q
  subgroup, log_g(h) unknown to everyone.

  Client side - proof generation
    1. local training -> update u_k (an integer in [0, B), B = 2^RANGE_BITS)
    2. choose random r_k in Z_q
    3. Pedersen commitment  C_k = g^{u_k} h^{r_k} mod p        (send C_k)
    4. proof pi_k: knowledge of (u_k, r_k) with C_k = g^{u_k} h^{r_k}, and
       0 <= u_k < B (Sigma-protocols made non-interactive with Fiat-Shamir)
                                                              (send pi_k)

  Server side - proof verification
    1. receive (C_k, pi_k) and the update
    2. verify the Sigma-protocol equations:
         * C_k = g^{u_k} h^{r_k} for some (u_k, r_k) the client knows,
         * u_k is within the valid range 0 <= u_k < B,
         * no sensitive data is revealed (u_k and r_k stay hidden)
       valid   -> accept the client's update, use it in the global aggregation
       invalid -> reject the client's update for this round

Implementation choices the diagram leaves open:

* Group: a 2048-bit prime p = κ q + 1 with a 256-bit prime q (a Schnorr
  group, like RFC 5114's 2048/256 groups). p, q, g and h are derived from the
  public string SEED with SHA-256 ("nothing up my sleeve"), so nobody knows
  log_g(h) and anyone can regenerate them (generate_group, setup).
* u_k = the first RANGE_BITS bits of SHA-256 of the update's bytes (its
  fingerprint), so 0 <= u_k < B = 2^RANGE_BITS.
* Proof of knowledge of the opening (Okamoto's Sigma-protocol):
  T = g^a h^b; z_u = a + e u_k, z_r = b + e r_k; check g^{z_u} h^{z_r} = T C_k^e.
* Range proof (bit decomposition): u_k = sum_i 2^i b_i with a commitment
  C_i = g^{b_i} h^{r_i} per bit and r_k = sum_i 2^i r_i, so by the
  homomorphism prod_i C_i^{2^i} = C_k. Each C_i gets an OR-proof
  (Cramer-Damgaard-Schoenmakers) that it commits to 0 or 1, so u_k < 2^RANGE_BITS.
* Fiat-Shamir: one challenge e = H(context || SHA-256(update) || C_k || T ||
  all C_i, A_i0, A_i1) mod q, H = SHA-256. The context is the protocol name,
  client and round, so a proof cannot be replayed for another client or round;
  the update's hash binds the proof to the exact update the server received,
  so an update changed in transit fails.

Timing per client and round: proof generation = client steps 1-4;
proof verification = server steps 1-2; authentication = the whole session.

All ZKP settings live below; there is no command-line option for them.
"""

from __future__ import annotations

import functools
import hashlib
import secrets
import time
from dataclasses import dataclass, field

import numpy as np

try:  # gmpy2 makes 2048-bit modular arithmetic several times faster.
    import gmpy2

    _mpz = gmpy2.mpz
    _powmod = gmpy2.powmod
    _next_prime = gmpy2.next_prime

    def _is_prime(n: int) -> bool:
        return bool(gmpy2.is_prime(n, 50))
except ImportError:  # pragma: no cover - exercised only without gmpy2
    _mpz = int
    _powmod = pow

    def _is_prime(n: int) -> bool:  # Miller-Rabin with 50 random bases
        if n < 4:
            return n in (2, 3)
        d, s = n - 1, 0
        while d % 2 == 0:
            d, s = d // 2, s + 1
        for _ in range(50):
            x = pow(secrets.randbelow(n - 3) + 2, d, n)
            if x in (1, n - 1):
                continue
            for _ in range(s - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    def _next_prime(n: int) -> int:
        n += 1
        while not _is_prime(n):
            n += 1
        return n


# (p bits, q bits) of each group; the primes are derived from SEED.
GROUPS = {"pedersen2048": (2048, 256), "test512": (512, 256)}

# --- ZKP settings ----------------------------------------------------------
ENABLED = True  # prove and verify every client's update every round
GROUP = "pedersen2048"  # 2048-bit p, 256-bit q (secure); "test512" is fast but NOT secure
RANGE_BITS = 64  # valid range of u_k: 0 <= u_k < B = 2^RANGE_BITS
PROTOCOL = "FedRF-Diff Pedersen ZKP"  # first part of the context in H
SEED = "FedRF-Diff Pedersen ZKP group"  # public seed of p, q, g, h


def settings() -> dict:
    p_bits, q_bits = GROUPS[GROUP]
    return {"enabled": ENABLED, "protocol": "pedersen commitment + sigma-protocol (fiat-shamir)",
            "group": GROUP, "p_bits": p_bits, "q_bits": q_bits, "range_bits": RANGE_BITS,
            "range_bound": f"2^{RANGE_BITS}", "hash": "sha256", "seed": SEED}


# --- group ----------------------------------------------------------------
def _expand(label: str, bits: int) -> int:
    """``bits`` pseudo-random bits from SHA-256(SEED | label | counter)."""
    out, counter = b"", 0
    while len(out) * 8 < bits:
        out += hashlib.sha256(f"{SEED}|{label}|{counter}".encode()).digest()
        counter += 1
    return int.from_bytes(out, "big") >> (len(out) * 8 - bits)


def generate_group(p_bits: int, q_bits: int) -> tuple[int, int]:
    """Primes (p, q) with q | p - 1, p of p_bits and q of q_bits bits, from SEED."""
    q = int(_next_prime(_expand(f"q{q_bits}", q_bits) | (1 << (q_bits - 1))))
    if q.bit_length() != q_bits:
        raise ValueError("q overflowed its bit length; change SEED")
    k = ((1 << (p_bits - 1)) + _expand(f"k{p_bits}", p_bits - 2)) // q + 1
    k += k % 2  # κ even, so p = κ q + 1 is odd
    while True:
        p = k * q + 1
        if p.bit_length() != p_bits:
            raise ValueError("no prime found in range; change SEED")
        if _is_prime(p):
            return p, q
        k += 2


def _subgroup_element(p: int, q: int, label: str) -> int:
    """Hash to the order-q subgroup: x^((p-1)/q) mod p for x = H(SEED | label)."""
    counter = 0
    while True:
        x = _expand(f"{label}{counter}", p.bit_length() + 64) % p
        y = int(_powmod(_mpz(x), _mpz((p - 1) // q), _mpz(p)))
        if y > 1:
            return y
        counter += 1


class FixedBase:
    """Fast powers of a fixed base: precomputed base^(d * 256^j) for every byte d.

    An exponent below q (q_bits bits) then costs at most q_bits / 8
    multiplications instead of a full square-and-multiply.
    """

    WINDOW = 8

    def __init__(self, base: int, p: int, bits: int):
        self.p = _mpz(p)
        self.tables = []
        b = _mpz(base)
        for _ in range((bits + self.WINDOW - 1) // self.WINDOW):
            row = [_mpz(1)]
            for _ in range((1 << self.WINDOW) - 1):
                row.append(row[-1] * b % self.p)
            self.tables.append(row)
            b = row[-1] * b % self.p  # base^(256^(j+1))

    def pow(self, e: int) -> int:
        acc, e = _mpz(1), int(e)
        for row in self.tables:
            d = e & 0xFF
            if d:
                acc = acc * row[d] % self.p
            e >>= self.WINDOW
            if not e:
                break
        return acc


@dataclass(frozen=True)
class PublicParams:
    p: int  # prime, p = κ q + 1
    q: int  # prime order of the subgroup generated by g and h
    g: int
    h: int  # second generator, log_g(h) unknown
    group: str = ""
    g_fixed: FixedBase | None = field(default=None, repr=False, compare=False)
    h_fixed: FixedBase | None = field(default=None, repr=False, compare=False)

    @property
    def p_bytes(self) -> int:
        return (int(self.p).bit_length() + 7) // 8

    @property
    def q_bytes(self) -> int:
        return (int(self.q).bit_length() + 7) // 8

    def g_pow(self, e: int) -> int:
        return self.g_fixed.pow(e % self.q)

    def h_pow(self, e: int) -> int:
        return self.h_fixed.pow(e % self.q)

    def commit(self, u: int, r: int) -> int:
        """Pedersen commitment g^u h^r mod p."""
        return self.g_pow(u) * self.h_pow(r) % self.p

    def in_subgroup(self, x: int) -> bool:
        return 0 < x < self.p and _powmod(_mpz(x), self.q, self.p) == 1


@functools.lru_cache(maxsize=None)
def _setup(group: str) -> PublicParams:
    p, q = generate_group(*GROUPS[group])
    g, h = _subgroup_element(p, q, "g"), _subgroup_element(p, q, "h")
    bits = q.bit_length()
    return PublicParams(p=_mpz(p), q=_mpz(q), g=_mpz(g), h=_mpz(h), group=group,
                        g_fixed=FixedBase(g, p, bits), h_fixed=FixedBase(h, p, bits))


def setup(group: str | None = None) -> PublicParams:
    """Public parameters p, q, g, h; the group defaults to this module's GROUP."""
    group = GROUP if group is None else group
    if group not in GROUPS:
        raise ValueError(f"unknown ZKP group {group!r}; choose from {sorted(GROUPS)}")
    pp = _setup(group)
    if not 0 < RANGE_BITS < int(pp.q).bit_length():
        raise ValueError(f"RANGE_BITS must lie in [1, {int(pp.q).bit_length() - 1}] so that B < q")
    return pp


# --- update encoding and challenge -------------------------------------------
def update_digest(*arrays: np.ndarray) -> bytes:
    """SHA-256 of the update's arrays (dtype, shape and bytes)."""
    digest = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        digest.update(f"{a.dtype.str}{a.shape}".encode())
        digest.update(a.tobytes())
    return digest.digest()


def encode_update(*arrays: np.ndarray) -> int:
    """u_k: the first RANGE_BITS bits of the update's SHA-256, so 0 <= u_k < 2^RANGE_BITS."""
    return int.from_bytes(update_digest(*arrays), "big") >> (256 - RANGE_BITS)


def context_for(client_id: str, round_: int) -> str:
    return f"{PROTOCOL}|{client_id}|round {round_}"


def challenge(pp: PublicParams, context: str, digest: bytes, *elements: int) -> int:
    """e = H(context || SHA-256(update) || C_k || T || C_i, A_i0, A_i1 ...) mod q."""
    data = hashlib.sha256()
    data.update(context.encode("utf-8") + b"|" + digest)
    for x in elements:
        data.update(int(x).to_bytes(pp.p_bytes, "big"))
    return _mpz(int.from_bytes(data.digest(), "big") % int(pp.q))


# --- proof -----------------------------------------------------------------
@dataclass(frozen=True)
class BitProof:
    """C_i = g^{b_i} h^{r_i} and the OR-proof that b_i is 0 or 1."""

    C: int
    A0: int  # commitment of the "b_i = 0" branch
    A1: int  # commitment of the "b_i = 1" branch
    e0: int  # challenge of branch 0 (branch 1 gets e - e0)
    z0: int
    z1: int


@dataclass(frozen=True)
class Proof:
    """pi_k: knowledge of the opening of C_k, plus the range proof 0 <= u_k < B."""

    T: int
    z_u: int
    z_r: int
    bits: tuple[BitProof, ...]

    def size_bytes(self, pp: PublicParams) -> int:
        return pp.p_bytes + 2 * pp.q_bytes + len(self.bits) * (3 * pp.p_bytes + 3 * pp.q_bytes)


def _rand(pp: PublicParams) -> int:
    return _mpz(secrets.randbelow(int(pp.q)))


def _invert(x: int, p: int) -> int:
    return _mpz(pow(int(x), -1, int(p)))


def prove(pp: PublicParams, u: int, r: int, C: int, digest: bytes, context: str) -> Proof:
    """Client step 4: pi_k for C_k = g^u h^r with 0 <= u < 2^RANGE_BITS."""
    q, p = pp.q, pp.p
    if not 0 <= u < (1 << RANGE_BITS):
        raise ValueError(f"u_k is outside the valid range [0, 2^{RANGE_BITS})")
    # Bit blinding factors that add up (with weights 2^i) to r_k.
    r_bits = [_rand(pp) for _ in range(RANGE_BITS)]
    r_bits[0] = (r - sum((1 << i) * r_bits[i] for i in range(1, RANGE_BITS))) % q
    g_inv = _invert(pp.g, p)
    pending = []
    for i, r_i in enumerate(r_bits):
        b = (u >> i) & 1
        C_i = pp.commit(b, r_i)
        # Real branch b: A_b = h^w. Simulated branch 1 - b: random (e, z), A = h^z X^-e,
        # where X would equal h^{r_i} if b_i were 1 - b.
        w, e_sim, z_sim = _rand(pp), _rand(pp), _rand(pp)
        X_sim = C_i * g_inv % p if b == 0 else C_i
        A_sim = pp.h_pow(z_sim) * _powmod(X_sim, q - e_sim, p) % p
        A_real = pp.h_pow(w)
        A0, A1 = (A_real, A_sim) if b == 0 else (A_sim, A_real)
        pending.append((b, r_i, C_i, A0, A1, w, e_sim, z_sim))
    a, s = _rand(pp), _rand(pp)
    T = pp.commit(a, s)
    e = challenge(pp, context, digest, C, T, *(x for _, _, C_i, A0, A1, *_ in pending for x in (C_i, A0, A1)))
    bit_proofs = []
    for b, r_i, C_i, A0, A1, w, e_sim, z_sim in pending:
        e_real = (e - e_sim) % q
        z_real = (w + e_real * r_i) % q
        if b == 0:
            bit_proofs.append(BitProof(C_i, A0, A1, e0=e_real, z0=z_real, z1=z_sim))
        else:
            bit_proofs.append(BitProof(C_i, A0, A1, e0=e_sim, z0=z_sim, z1=z_real))
    return Proof(T=T, z_u=(a + e * u) % q, z_r=(s + e * r) % q, bits=tuple(bit_proofs))


def verify(pp: PublicParams, C: int, proof: Proof, digest: bytes, context: str) -> bool:
    """Server step 2: the Sigma-protocol equations and the range of u_k."""
    q, p = pp.q, pp.p
    if len(proof.bits) != RANGE_BITS:
        return False
    if not (0 <= proof.z_u < q and 0 <= proof.z_r < q):
        return False
    if not all(0 <= x < q for bp in proof.bits for x in (bp.e0, bp.z0, bp.z1)):
        return False
    if not all(0 < x < p for bp in proof.bits for x in (bp.A0, bp.A1)):
        return False
    if not all(pp.in_subgroup(x) for x in (C, proof.T, *(bp.C for bp in proof.bits))):
        return False
    e = challenge(pp, context, digest, C, proof.T,
                  *(x for bp in proof.bits for x in (bp.C, bp.A0, bp.A1)))
    # Knowledge of (u_k, r_k): g^{z_u} h^{z_r} == T C_k^e.
    if pp.commit(proof.z_u, proof.z_r) != proof.T * _powmod(C, e, p) % p:
        return False
    # Range: every C_i commits to 0 or 1 ...
    g_inv = _invert(pp.g, p)
    for bp in proof.bits:
        e1 = (e - bp.e0) % q
        if pp.h_pow(bp.z0) != bp.A0 * _powmod(bp.C, bp.e0, p) % p:
            return False
        if pp.h_pow(bp.z1) != bp.A1 * _powmod(bp.C * g_inv % p, e1, p) % p:
            return False
    # ... and prod_i C_i^{2^i} == C_k (homomorphism), so u_k = sum_i 2^i b_i < 2^RANGE_BITS.
    acc = _mpz(1)
    for bp in reversed(proof.bits):
        acc = acc * acc % p * bp.C % p
    return acc == C


# --- one session -------------------------------------------------------------
@dataclass
class Session:
    """Outcome, timing (seconds) and message sizes (bytes) of one client's proof."""

    authenticated: bool
    vectors: int  # latent vectors in the update
    proof_generation_s: float  # client steps 1-4
    proof_verification_s: float  # server steps 1-2
    authentication_s: float  # whole session
    bytes_up: int  # client -> server: C_k and pi_k
    bytes_down: int  # server -> client: valid / invalid

    @property
    def outcome(self) -> str:
        return "Valid" if self.authenticated else "Invalid"


def authenticate(pp: PublicParams, client_id: str, round_: int, update: tuple,
                 received: tuple | None = None) -> Session:
    """One proof session for a client's update (a tuple of arrays).

    ``received`` is the update as it reaches the server (defaults to
    ``update``); tests use it to simulate an update altered in transit.
    """
    received = update if received is None else received
    context = context_for(client_id, round_)
    t0 = time.perf_counter()
    # Client side.
    digest = update_digest(*update)
    u = encode_update(*update)  # 1. update u_k
    r = _rand(pp)  # 2. random r_k in Z_q
    C = pp.commit(u, r)  # 3. C_k = g^{u_k} h^{r_k} mod p
    proof = prove(pp, u, r, C, digest, context)  # 4. pi_k
    t1 = time.perf_counter()
    # Server side: 1. receive (C_k, pi_k) and the update; 2. verify.
    ok = verify(pp, C, proof, update_digest(*received), context)
    t2 = time.perf_counter()
    return Session(authenticated=bool(ok), vectors=len(update[0]),
                   proof_generation_s=t1 - t0, proof_verification_s=t2 - t1,
                   authentication_s=t2 - t0,
                   bytes_up=pp.p_bytes + proof.size_bytes(pp), bytes_down=1)
