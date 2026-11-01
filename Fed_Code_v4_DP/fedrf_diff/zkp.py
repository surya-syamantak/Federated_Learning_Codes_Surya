"""Stage 1.5: proposed ZKP-based authentication of the latent vectors.

Implements the protocol of the "Proposed Zero-Knowledge Proof (ZKP)-Based
Authentication for Vector Representation" diagram, once per latent vector
z_k = (z_k1, ..., z_kd) a client sends (d = 4):

  Public: prime q, generators g_1..g_d and h (known to client and server).
  Round 1: previous commitment C_(r-1) = 1.
  (1) client -> server  vector Pedersen commitment
          C_r = (prod_j g_j^{z_kj}) * h^{r_k} * (C_(r-1))^beta
  (2) client -> server  temporary commitment, fresh w_1..w_d, eta
          T = (prod_j g_j^{w_j}) * h^eta
  (3) server -> client  fresh random challenge e
  (4) client -> server  responses s_j = w_j + e z_kj (mod q), s_r = eta + e r_k (mod q)
  (5)/(6) server checks
          (prod_j g_j^{s_j}) * h^{s_r} == T * (C_base)^e,
          C_base = C_r / (C_(r-1))^beta,
      and answers "Authentication Successful" (values match) or
      "Authentication Failed" (Z_k is discarded for this round).

beta is the chaining exponent: a fixed non-zero element of Z_q that links
each commitment to the previous round's.

Two details the diagram leaves implicit, without which the check cannot hold:

* Group elements are computed mod a safe prime p = 2q + 1 and exponents mod
  the prime q, the order of the subgroup of squares that g_j and h lie in
  (with "mod q" for both, s_j would be reduced modulo the wrong number).
* z is real-valued, so it is encoded as round(z * 2^SCALE_BITS) mod q.

Timing per client and round: proof generation = the client's steps 1, 2 and
4; proof verification = the server's steps 3 and 5/6; authentication = the
whole session end to end.

All ZKP settings live below; there is no command-line option for them.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass

import numpy as np

try:  # gmpy2 makes 2048-bit modular exponentiation ~7x faster.
    import gmpy2

    _mpz = gmpy2.mpz
    _powmod = gmpy2.powmod
    _invert = gmpy2.invert
except ImportError:  # pragma: no cover - exercised only without gmpy2
    _mpz = int
    _powmod = pow

    def _invert(a, m):
        return pow(a, -1, m)


# RFC 3526, group 14: 2048-bit safe prime.
MODP_2048 = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74"
    "020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F1437"
    "4FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF05"
    "98DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB"
    "9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF695581718"
    "3995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF",
    16,
)
# 512-bit safe prime for fast tests only - NOT secure.
TEST_512 = int(
    "eaeb56159b708402670d63e96c18279650037b0bdeb2f48dacf72b16ac3ebf0e"
    "867e1af57bc946898dfd194ba38f68ae09ff9026859a3b4adf750e2d5d8b43c7",
    16,
)
GROUPS = {"modp2048": MODP_2048, "test512": TEST_512}

# --- ZKP settings ----------------------------------------------------------
ENABLED = True  # authenticate every client's latent vectors every round
GROUP = "modp2048"  # RFC 3526 2048-bit group (secure); "test512" is fast but NOT secure
LAMBDA = 3  # chaining exponent beta (fixed, non-zero, in Z_q)
SCALE_BITS = 32  # fixed-point encoding of the latent values
# Optional extra proof (not in the diagram) that the commitment opens to the
# exact vector the server receives; off = the diagram's protocol.
BIND_PLAINTEXT = False


def settings() -> dict:
    return {"enabled": ENABLED, "group": GROUP, "lambda": LAMBDA,
            "scale_bits": SCALE_BITS, "bind_plaintext": BIND_PLAINTEXT}


@dataclass(frozen=True)
class PublicParams:
    p: int
    q: int
    g: tuple  # g_1..g_dz
    h: int
    lam: int  # chaining exponent beta
    scale_bits: int

    @property
    def d_z(self) -> int:
        return len(self.g)


def _hash_to_group(p: int, seed: bytes, label: str) -> int:
    """Nothing-up-my-sleeve generator: square of a hash, so it has order q."""
    n_bytes = (p.bit_length() + 7) // 8 + 16
    counter = 0
    while True:
        stream = b"".join(
            hashlib.sha256(seed + label.encode() + counter.to_bytes(4, "big") + i.to_bytes(4, "big")).digest()
            for i in range(n_bytes // 32 + 1)
        )
        x = int.from_bytes(stream[:n_bytes], "big") % p
        y = x * x % p
        if y not in (0, 1):
            return y
        counter += 1


def setup(d_z: int, group: str | None = None, lam: int | None = None,
          scale_bits: int | None = None, seed: bytes = b"FedRF-Diff/ZKP") -> PublicParams:
    """Public parameters; unset arguments take this module's settings."""
    group = GROUP if group is None else group
    lam = LAMBDA if lam is None else lam
    scale_bits = SCALE_BITS if scale_bits is None else scale_bits
    if group not in GROUPS:
        raise ValueError(f"unknown ZKP group {group!r}; choose from {sorted(GROUPS)}")
    p = GROUPS[group]
    g = tuple(_mpz(_hash_to_group(p, seed, f"g{j}")) for j in range(1, d_z + 1))
    h = _mpz(_hash_to_group(p, seed, "h"))
    return PublicParams(p=_mpz(p), q=_mpz((p - 1) // 2), g=g, h=h, lam=lam, scale_bits=scale_bits)


def encode(Z: np.ndarray, scale_bits: int) -> list[list[int]]:
    """Fixed-point integers (signed) for every latent coordinate."""
    scaled = np.rint(np.asarray(Z, dtype=float) * float(2**scale_bits))
    return [[int(v) for v in row] for row in scaled]


def _pow_signed(base, exp: int, p):
    """base^exp mod p for a small signed exponent (cheap for negative values)."""
    if exp >= 0:
        return _powmod(base, exp, p)
    return _invert(_powmod(base, -exp, p), p)


def _vector_part(pp: PublicParams, z_row: list[int]):
    acc = _mpz(1)
    for g_j, z_j in zip(pp.g, z_row):
        acc = acc * _pow_signed(g_j, z_j, pp.p) % pp.p
    return acc


@dataclass
class Response:
    s: list  # s_ij, j = 1..d
    s_r: int
    s_bind: int | None = None


class Prover:
    """Client side. Keeps C^(r-1) per row for the chain."""

    def __init__(self, pp: PublicParams):
        self.pp = pp
        self.prev: list | None = None
        self._state = None

    def _rand(self):
        return _mpz(secrets.randbelow(int(self.pp.q)))

    def commit(self, Z_tilde: np.ndarray) -> list:
        pp = self.pp
        z = encode(Z_tilde, pp.scale_bits)
        prev = self.prev if self.prev is not None else [_mpz(1)] * len(z)
        if len(prev) != len(z):
            raise ValueError("row count changed between rounds; the chain needs fixed rows")
        r = [self._rand() for _ in z]
        C = [
            _vector_part(pp, z_i) * _powmod(pp.h, r_i, pp.p) % pp.p * _powmod(c_prev, pp.lam, pp.p) % pp.p
            for z_i, r_i, c_prev in zip(z, r, prev)
        ]
        self._state = {"z": z, "r": r, "C": C}
        return C

    def announce(self, bind: bool = False) -> tuple[list, list | None]:
        pp, st = self.pp, self._state
        k = [[self._rand() for _ in range(pp.d_z)] for _ in st["z"]]
        tau = [self._rand() for _ in st["z"]]
        T = []
        for k_i, tau_i in zip(k, tau):
            acc = _powmod(pp.h, tau_i, pp.p)
            for g_j, k_ij in zip(pp.g, k_i):
                acc = acc * _powmod(g_j, k_ij, pp.p) % pp.p
            T.append(acc)
        st.update(k=k, tau=tau)
        T_bind = None
        if bind:
            rho = [self._rand() for _ in st["z"]]
            T_bind = [_powmod(pp.h, rho_i, pp.p) for rho_i in rho]
            st["rho"] = rho
        return T, T_bind

    def respond(self, e: list) -> list[Response]:
        pp, st = self.pp, self._state
        out = []
        for i, e_i in enumerate(e):
            s = [(k_ij + e_i * z_ij) % pp.q for k_ij, z_ij in zip(st["k"][i], st["z"][i])]
            s_r = (st["tau"][i] + e_i * st["r"][i]) % pp.q
            s_bind = (st["rho"][i] + e_i * st["r"][i]) % pp.q if "rho" in st else None
            out.append(Response(s=s, s_r=s_r, s_bind=s_bind))
        return out

    def finalize(self, accepted: bool) -> None:
        """Advance the chain only if the server accepted this round."""
        if accepted:
            self.prev = self._state["C"]
        self._state = None


class Verifier:
    """Server side. Stores each client's last accepted commitments."""

    def __init__(self, pp: PublicParams):
        self.pp = pp
        self.prev: dict[str, list] = {}

    def challenge(self, n: int) -> list:
        return [_mpz(secrets.randbelow(int(self.pp.q))) for _ in range(n)]

    def verify(self, client_id: str, C: list, T: list, e: list, responses: list[Response],
               Z_plain: np.ndarray | None = None, T_bind: list | None = None) -> bool:
        pp = self.pp
        prev = self.prev.get(client_id, [_mpz(1)] * len(C))
        if not (len(C) == len(T) == len(e) == len(responses) == len(prev)):
            return False
        z_plain = encode(Z_plain, pp.scale_bits) if Z_plain is not None else None
        if z_plain is not None and (T_bind is None or len(z_plain) != len(C)):
            return False
        for i, (C_i, T_i, e_i, resp) in enumerate(zip(C, T, e, responses)):
            C_base = C_i * _invert(_powmod(prev[i], pp.lam, pp.p), pp.p) % pp.p
            lhs = _powmod(pp.h, resp.s_r, pp.p)
            for g_j, s_ij in zip(pp.g, resp.s):
                lhs = lhs * _powmod(g_j, s_ij, pp.p) % pp.p
            if lhs != T_i * _powmod(C_base, e_i, pp.p) % pp.p:
                return False
            if z_plain is not None:
                # h^{r_i} must equal C_base / prod_j g_j^{enc(z~_ij)}.
                if resp.s_bind is None:
                    return False
                Y = C_base * _invert(_vector_part(pp, z_plain[i]), pp.p) % pp.p
                if _powmod(pp.h, resp.s_bind, pp.p) != T_bind[i] * _powmod(Y, e_i, pp.p) % pp.p:
                    return False
        return True

    def accept(self, client_id: str, C: list) -> None:
        self.prev[client_id] = C


@dataclass
class Session:
    """Outcome, timing (seconds) and message sizes (bytes) of one session."""

    authenticated: bool
    vectors: int
    proof_generation_s: float  # client: steps 1, 2, 4
    proof_verification_s: float  # server: steps 3, 5/6
    authentication_s: float  # whole session
    bytes_up: int  # client -> server: C, T, responses
    bytes_down: int  # server -> client: challenges, outcome


def authenticate(prover: Prover, verifier: Verifier, client_id: str, Z_tilde: np.ndarray,
                 bind_plaintext: bool | None = None, Z_sent: np.ndarray | None = None) -> Session:
    """Run one session of the diagram's protocol for every row of ``Z_tilde``.

    ``Z_sent`` is what the client actually transmits (defaults to the
    committed ``Z_tilde``); tests use it to simulate a tampering client.
    """
    Z_sent = Z_tilde if Z_sent is None else Z_sent
    bind_plaintext = BIND_PLAINTEXT if bind_plaintext is None else bind_plaintext
    pp = prover.pp
    t0 = time.perf_counter()
    C = prover.commit(Z_tilde)  # (1) vector Pedersen commitments
    T, T_bind = prover.announce(bind=bind_plaintext)  # (2) temporary commitments
    t1 = time.perf_counter()
    c = verifier.challenge(len(C))  # (3) fresh random challenges
    t2 = time.perf_counter()
    responses = prover.respond(c)  # (4) vector responses
    t3 = time.perf_counter()
    ok = verifier.verify(client_id, C, T, c, responses,  # (5)/(6) check and decide
                         Z_plain=Z_sent if bind_plaintext else None, T_bind=T_bind)
    if ok:
        verifier.accept(client_id, C)
    t4 = time.perf_counter()
    prover.finalize(ok)  # client keeps C as C_(r-1) only if authenticated
    t5 = time.perf_counter()

    p_bytes, q_bytes = (int(pp.p).bit_length() + 7) // 8, (int(pp.q).bit_length() + 7) // 8
    n = len(C)
    up = n * (2 * p_bytes + (pp.d_z + 1) * q_bytes)
    if bind_plaintext:
        up += n * (p_bytes + q_bytes)
    return Session(authenticated=ok, vectors=n,
                   proof_generation_s=(t1 - t0) + (t3 - t2) + (t5 - t4),
                   proof_verification_s=(t2 - t1) + (t4 - t3),
                   authentication_s=t5 - t0,
                   bytes_up=up, bytes_down=n * q_bytes + 1)
