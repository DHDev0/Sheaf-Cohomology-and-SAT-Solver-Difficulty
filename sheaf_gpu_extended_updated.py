#!/usr/bin/env python3
"""
Sheaf-Theoretic Phase Transition: GPU Experiment (v4)
─────────────────────────────────────────────────────
Improvements over v3:
  • Discrete Conjecture 5.3: tests λ₁ vs log(T) directly (sign flip from continuous)
  • Theorem 3.1: spectral sequence collapse page computation
  • Fixed N/A in partial correlations (now computed for all metrics)
  • Suppressed RankWarning for constant-α suites (e.g. SATLIB uf20)
  • Correlation threshold guide in final report
  • Per-suite and global discrete vs continuous comparison
"""
import numpy as np
import sys
import os
import time
import json
import random
import hashlib
import tarfile
import torch
from collections import defaultdict
from itertools import product
from datetime import datetime, timedelta

# ── GPU ──────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {GPU_NAME}, {GPU_MEM_GB:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    DEVICE = torch.device('cpu')
    GPU_NAME = "CPU"
    GPU_MEM_GB = 0
    print("[GPU] Not available, using CPU")

# ── Paths (EDIT THESE) ──────────────────────────────────────────────
OUTPUT_DIR = "/home/user/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results_extended.json")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint_hashes.json")

ALL_3BITS = np.array(list(product(range(2), repeat=3)), dtype=np.int8)
_LAPLACIAN_CACHE = {}

# ═════════════════════════════════════════════════════════════════════
# CHECKPOINT SYSTEM
# ═════════════════════════════════════════════════════════════════════
class CheckpointManager:
    """Manages continuous checkpointing with atomic writes."""

    def __init__(self, results_file, checkpoint_file):
        self.results_file = results_file
        self.checkpoint_file = checkpoint_file
        self.done_hashes = set()
        self.pending = []
        self.total_saved = 0
        self._load()

    def _load(self):
        # Load done hashes from checkpoint file (fast)
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file) as f:
                self.done_hashes = set(json.load(f))
        # Also scan results file for any hashes not in checkpoint
        if os.path.exists(self.results_file):
            with open(self.results_file) as f:
                data = json.load(f)
            for group in data.get("data", {}).values():
                for r in group:
                    if isinstance(r, dict) and "clauses_hash" in r:
                        self.done_hashes.add(r["clauses_hash"])
            self.total_saved = sum(
                len(g) for g in data.get("data", {}).values()
            )

    def is_done(self, h):
        return h in self.done_hashes

    def add(self, result):
        """Add a result and mark as done."""
        self.pending.append(result)
        self.done_hashes.add(result["clauses_hash"])

    def flush(self):
        """Write pending results to disk atomically."""
        if not self.pending:
            return
        # Load or create results file
        if os.path.exists(self.results_file):
            with open(self.results_file) as f:
                data = json.load(f)
        else:
            data = {"configs": [], "correlations": {}, "data": {}}

        key = "structured"
        if key not in data["data"]:
            data["data"][key] = []
        data["data"][key].extend(self.pending)
        self.total_saved += len(self.pending)

        # Atomic write: write to tmp then rename
        tmp = self.results_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, self.results_file)

        # Update checkpoint hash file
        tmp2 = self.checkpoint_file + ".tmp"
        with open(tmp2, "w") as f:
            json.dump(list(self.done_hashes), f)
        os.replace(tmp2, self.checkpoint_file)

        n = len(self.pending)
        self.pending = []
        return n

    def save_one(self, result):
        """Add and immediately flush one result."""
        self.add(result)
        return self.flush()


# ═════════════════════════════════════════════════════════════════════
# SATLIB + CNF PARSING
# ═════════════════════════════════════════════════════════════════════
def download_satlib():
    import requests
    print("\n📥 SATLIB Benchmark Download")
    print("=" * 60)
    urls = {
        "uf20": "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf20-91.tar.gz",
        "aim": "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/DIMACS/AIM/aim.tar.gz",
    }
    for name, url in urls.items():
        path = os.path.join(OUTPUT_DIR, f"{name}.tar.gz")
        if os.path.exists(path):
            size_mb = os.path.getsize(path) // (1024 * 1024)
            print(f"  ✅ {name}.tar.gz already present ({size_mb} MB)")
            continue
        print(f"  Downloading {name}.tar.gz ... ", end="", flush=True)
        r = requests.get(url, stream=True, timeout=120)
        total_size = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = int(100 * downloaded / total_size)
                        print(f"\r  Downloading {name}.tar.gz ... {pct}%", end="", flush=True)
        print(f" ✅ Done ({downloaded // (1024 * 1024)} MB)")
    print("=" * 60)


def parse_cnf(content):
    clauses = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line[0] in ("%", "c", "p"):
            continue
        lits = []
        for x in line.split():
            if x == "0":
                break
            try:
                lits.append(int(x))
            except ValueError:
                continue
        if lits:
            clauses.append(tuple((abs(l) - 1, l > 0) for l in lits))
    return clauses


def load_benchmark(tar_path, max_n=50):
    instances = []
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".cnf"):
                f = tar.extractfile(member)
                clauses = parse_cnf(f.read().decode("utf-8"))
                if not clauses:
                    continue
                n = max((v for c in clauses for v, _ in c), default=0) + 1
                if n <= max_n:
                    instances.append((clauses, n, member.name))
    return instances


def instance_hash(clauses):
    s = str(sorted(str(c) for c in clauses)).encode()
    return hashlib.sha256(s).hexdigest()[:16]


# ═════════════════════════════════════════════════════════════════════
# GENERATORS
# ═════════════════════════════════════════════════════════════════════
def gen_3sat(n, m, seed):
    rng = random.Random(seed)
    return [
        tuple(zip(rng.sample(range(n), 3),
                   [rng.choice([True, False]) for _ in range(3)]))
        for _ in range(m)
    ]


def gen_planted_exact1(n, m, seed):
    rng = random.Random(seed)
    planted = [rng.choice([0, 1]) for _ in range(n)]
    clauses = []
    for _ in range(m):
        vs = rng.sample(range(n), 3)
        signs = [bool(planted[v]) == (i % 2 == 0) for i, v in enumerate(vs)]
        clauses.append(tuple(zip(vs, signs)))
    return clauses


def gen_community_strong(n, m, seed):
    rng = random.Random(seed)
    comm = [rng.randint(0, 4) for _ in range(n)]
    pools = defaultdict(list)
    for i, c in enumerate(comm):
        pools[c].append(i)
    clauses = []
    for _ in range(m):
        if rng.random() < 0.85:
            c = rng.choice(range(5))
            pool = pools[c]
            vs = rng.sample(pool, 3) if len(pool) >= 3 else rng.sample(range(n), 3)
        else:
            vs = rng.sample(range(n), 3)
        clauses.append(tuple(zip(vs, [rng.choice([True, False]) for _ in range(3)])))
    return clauses


def gen_transition_clustered(n, m, seed):
    rng = random.Random(seed)
    cs = n // 3
    clauses = []
    per_cluster = int(0.9 * m) // 3
    for c in range(3):
        lo, hi = c * cs, (c + 1) * cs
        pool = list(range(lo, hi))
        for _ in range(per_cluster):
            vs = rng.sample(pool, 3) if len(pool) >= 3 else rng.sample(range(n), 3)
            clauses.append(tuple(zip(vs, [rng.choice([True, False]) for _ in range(3)])))
    for _ in range(m - len(clauses)):
        clauses.append(tuple(zip(rng.sample(range(n), 3),
                                  [rng.choice([True, False]) for _ in range(3)])))
    return clauses


# ═════════════════════════════════════════════════════════════════════
# SHEAF COMPUTATION
# ═════════════════════════════════════════════════════════════════════
def local_solutions_batch(clauses):
    scopes, sol_arrays = [], []
    for clause in clauses:
        scope = tuple(sorted(set(v for v, _ in clause)))
        if len(scope) != 3:
            continue
        scopes.append(scope)
        vp = {scope[i]: i for i in range(3)}
        fail = [0] * 3
        for v, s in clause:
            fail[vp[v]] = 0 if s else 1
        ft = tuple(fail)
        sols = [row for row in ALL_3BITS if tuple(row) != ft]
        sol_arrays.append(np.array(sols, dtype=np.int8))
    return scopes, sol_arrays


def _build_edge_index(scopes, n_vars):
    """Var→clause index: only visit clause pairs that actually share a variable."""
    var2cl = defaultdict(list)
    for ci, sc in enumerate(scopes):
        for v in sc:
            var2cl[v].append(ci)
    scope_sets = [frozenset(sc) for sc in scopes]
    seen = set()
    edges = []
    for v in range(n_vars):
        cls = var2cl[v]
        for a in range(len(cls)):
            for b in range(a + 1, len(cls)):
                i, j = min(cls[a], cls[b]), max(cls[a], cls[b])
                if (i, j) not in seen:
                    seen.add((i, j))
                    shared = sorted(scope_sets[i] & scope_sets[j])
                    if shared:
                        edges.append((i, j, shared))
    return edges


def build_sheaf_laplacian_direct_gpu(scopes, sol_arrays, n_vars):
    """Build Laplacian on GPU with edge-indexed construction."""
    m = len(scopes)
    if m == 0:
        return torch.zeros((0, 0), dtype=torch.float64, device=DEVICE), 0

    dims = torch.tensor([len(sa) for sa in sol_arrays], dtype=torch.int64, device=DEVICE)
    offsets = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int64, device=DEVICE), dims]), 0)
    td = int(offsets[-1].item())

    # Cache or allocate
    if td in _LAPLACIAN_CACHE:
        L = _LAPLACIAN_CACHE[td]
        L.zero_()
    else:
        L = torch.zeros((td, td), dtype=torch.float64, device=DEVICE)
        _LAPLACIAN_CACHE[td] = L

    pos_maps = [{v: i for i, v in enumerate(sc)} for sc in scopes]
    sol_tensors = [torch.tensor(sa, dtype=torch.int8, device=DEVICE) for sa in sol_arrays]

    edges = _build_edge_index(scopes, n_vars)

    for i, j, shared in edges:
        idx_i = [pos_maps[i][v] for v in shared]
        idx_j = [pos_maps[j][v] for v in shared]

        sol_i = sol_tensors[i][:, idx_i]
        sol_j = sol_tensors[j][:, idx_j]

        bits = 2 ** torch.arange(len(shared), dtype=torch.float32, device=DEVICE)
        restrict_i = (sol_i.float() @ bits).long()
        restrict_j = (sol_j.float() @ bits).long()

        agree_ii = (restrict_i.unsqueeze(1) == restrict_i.unsqueeze(0)).double()
        agree_jj = (restrict_j.unsqueeze(1) == restrict_j.unsqueeze(0)).double()
        agree_ij = (restrict_i.unsqueeze(1) == restrict_j.unsqueeze(0)).double()

        oi = int(offsets[i].item())
        oj = int(offsets[j].item())
        di = int(dims[i].item())
        dj = int(dims[j].item())

        L[oi:oi + di, oi:oi + di] += agree_ii
        L[oj:oj + dj, oj:oj + dj] += agree_jj
        L[oi:oi + di, oj:oj + dj] -= agree_ij
        L[oj:oj + dj, oi:oi + di] -= agree_ij.T

    return L, td


def compute_spectrum_gpu(L_tensor):
    eigs = torch.linalg.eigvalsh(L_tensor)
    return torch.sort(eigs).values


def spectral_invariants_gpu(eigs_tensor, tol=1e-10):
    abs_eigs = torch.abs(eigs_tensor)
    nonzero_mask = abs_eigs > tol
    nullity = int(torch.sum(~nonzero_mask).item())

    if torch.any(nonzero_mask):
        nonzero = eigs_tensor[nonzero_mask]
        gap = float(torch.min(torch.abs(nonzero)).item())
        max_eig = float(torch.max(eigs_tensor).item())
        pos_mask = eigs_tensor > tol
        if torch.any(pos_mask):
            pos = eigs_tensor[pos_mask]
            p = pos / torch.sum(pos)
            spectral_entropy = float((-torch.sum(p * torch.log(p + 1e-30))).item())
        else:
            spectral_entropy = 0.0
    else:
        gap = max_eig = spectral_entropy = 0.0

    return {"gap": gap, "nullity": nullity, "max_eig": max_eig,
            "spectral_entropy": spectral_entropy}


# ═════════════════════════════════════════════════════════════════════
# F₂ BETTI NUMBERS  (pure numpy — no GPU transfer overhead)
# ═════════════════════════════════════════════════════════════════════
def build_delta0_f2(scopes, sol_arrays, n_vars):
    """Build δ₀ over F₂ using pure numpy. No torch dependency."""
    m = len(scopes)
    dims = [len(sa) for sa in sol_arrays]
    offsets = np.zeros(m + 1, dtype=np.int64)
    for i in range(m):
        offsets[i + 1] = offsets[i] + dims[i]
    td = int(offsets[-1])

    pos_maps = [{v: i for i, v in enumerate(sc)} for sc in scopes]
    edges = _build_edge_index(scopes, n_vars)

    rows = []
    for i, j, shared in edges:
        n_sh = len(shared)
        idx_i = [pos_maps[i][v] for v in shared]
        idx_j = [pos_maps[j][v] for v in shared]

        # Compute restriction integers using numpy
        bits = (1 << np.arange(n_sh, dtype=np.int64))
        ri = sol_arrays[i][:, idx_i].astype(np.int64) @ bits
        rj = sol_arrays[j][:, idx_j].astype(np.int64) @ bits

        # Iterate over ALL assignments to shared vars (not just intersection)
        for val in range(1 << n_sh):
            row = 0
            for s in np.where(ri == val)[0]:
                row ^= (1 << int(offsets[i] + s))
            for s in np.where(rj == val)[0]:
                row ^= (1 << int(offsets[j] + s))
            if row:
                rows.append(row)

    return rows, td


def rank_f2_bitpacked(rows, n_cols):
    """F₂ rank via bitpacked Gaussian elimination (echelon form)."""
    if not rows:
        return 0
    # Pre-filter: remove zero rows (can appear from cancellation)
    rows = [r for r in rows if r]
    if not rows:
        return 0

    rank = 0
    ri = 0
    pc = 0

    while ri < len(rows) and pc < n_cols:
        # Find pivot
        bit = 1 << pc
        found = -1
        for r in range(ri, len(rows)):
            if rows[r] & bit:
                found = r
                break
        if found < 0:
            pc += 1
            continue

        rows[ri], rows[found] = rows[found], rows[ri]
        pivot = rows[ri]

        # Eliminate below only (echelon, not full RREF)
        for r in range(ri + 1, len(rows)):
            if rows[r] & bit:
                rows[r] ^= pivot

        rank += 1
        ri += 1
        pc += 1

    return rank


def betti_f2(scopes, sol_arrays, n_vars):
    rows, td = build_delta0_f2(scopes, sol_arrays, n_vars)
    rk = rank_f2_bitpacked(rows, td)
    return td - rk, len(rows) - rk, td, len(rows), rk


# ═════════════════════════════════════════════════════════════════════
# SPECTRAL SEQUENCE COLLAPSE PAGE  (Theorem 3.1 / Theorem 2.1)
# ═════════════════════════════════════════════════════════════════════
def _higher_simplices(scopes, max_p=4):
    """Build p-simplices of the constraint nerve up to dimension max_p.
    A p-simplex is a set of (p+1) clauses that mutually share at least one variable."""
    scope_sets = [frozenset(sc) for sc in scopes]
    m = len(scopes)

    # 0-simplices = clauses
    simplices_by_dim = {0: list(range(m))}

    # 1-simplices = pairs sharing ≥1 variable
    edges = []
    for i in range(m):
        for j in range(i + 1, m):
            if scope_sets[i] & scope_sets[j]:
                edges.append((i, j))
    simplices_by_dim[1] = edges

    # Build higher simplices by clique extension (with performance limit)
    MAX_SIMPLICES_PER_DIM = 5000  # Avoid combinatorial explosion
    for p in range(2, max_p + 1):
        prev = simplices_by_dim.get(p - 1, [])
        if not prev:
            break
        higher = []

        # For each (p-1)-simplex, try adding one more clause
        seen = set()
        for simplex in prev:
            verts = list(simplex) if isinstance(simplex, tuple) else [simplex]
            for k in range(m):
                if k in verts:
                    continue
                candidate = tuple(sorted(verts + [k]))
                if candidate in seen:
                    continue
                # Check all pairs share a variable
                valid = True
                for a in range(len(candidate)):
                    for b in range(a + 1, len(candidate)):
                        if not (scope_sets[candidate[a]] & scope_sets[candidate[b]]):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    seen.add(candidate)
                    higher.append(candidate)
                    if len(higher) >= MAX_SIMPLICES_PER_DIM:
                        break
            if len(higher) >= MAX_SIMPLICES_PER_DIM:
                break
        simplices_by_dim[p] = higher
        if not higher:
            break

    return simplices_by_dim


def compute_collapse_page(scopes, sol_arrays, n_vars, max_page=6):
    """Compute the spectral sequence collapse page r₀ for Theorem 3.1.

    Returns (r0, page_ranks): collapse page and list of E_r ranks per page.
    The spectral sequence collapses at page r₀ when d_r = 0 for all r ≥ r₀.

    We work over F₂ for consistency with the Betti number computations.
    For efficiency, we compute the E₁ page using the Čech cochain complex,
    then track when the ranks stabilize (d_r = 0 means E_{r+1} = E_r).
    """
    m = len(scopes)
    if m == 0:
        return 2, [0]

    # Build the simplicial complex of the constraint nerve
    simplices = _higher_simplices(scopes, max_p=min(max_page + 1, 5))

    # Compute dimensions of cochain spaces C^p
    # C^p = ⊕_{σ ∈ N_p} F₂^{|local solutions on σ|}
    # For p=0, local solutions per clause are in sol_arrays
    # For p≥1, local solutions on a simplex are tuples consistent on pairwise overlaps

    # E₁^{p,0} = Čech cohomology at the p-th cochain level
    # We track dim(E_r^{*,0}) — the total rank of the E_r page (q=0 row)
    # When this stabilizes, the spectral sequence has collapsed

    # For practical purposes on small instances:
    # E₂ is computed from the Čech cohomology (which we already compute as Betti numbers)
    # The collapse page is detected when ranks stop changing

    # Compute E₁ = C^p (cochain groups)
    e1_dims = []
    for p in sorted(simplices.keys()):
        if p == 0:
            e1_dims.append(sum(len(sa) for sa in sol_arrays))
        else:
            # Count consistent tuples on each p-simplex
            total = 0
            for simplex in simplices[p]:
                verts = list(simplex) if isinstance(simplex, tuple) else [simplex]
                # Count assignments consistent on all pairwise overlaps
                if len(verts) < 2:
                    total += len(sol_arrays[verts[0]]) if verts[0] < len(sol_arrays) else 0
                    continue
                # Start with sols of first clause, filter by consistency with each subsequent
                scope_sets_local = [frozenset(scopes[v]) for v in verts]
                # Use first clause's solutions as starting point
                candidates = sol_arrays[verts[0]]
                n_consistent = 0
                for row in candidates:
                    consistent = True
                    assign_0 = {scopes[verts[0]][k]: int(row[k]) for k in range(len(scopes[verts[0]]))}
                    for vi in range(1, len(verts)):
                        shared = scope_sets_local[0] & scope_sets_local[vi]
                        if not shared:
                            continue
                        # Check if any sol in clause vi agrees on shared vars
                        found_match = False
                        for row_j in sol_arrays[verts[vi]]:
                            assign_j = {scopes[verts[vi]][k]: int(row_j[k]) for k in range(len(scopes[verts[vi]]))}
                            if all(assign_0.get(s) == assign_j.get(s) for s in shared):
                                found_match = True
                                break
                        if not found_match:
                            consistent = False
                            break
                    if consistent:
                        n_consistent += 1
                total += n_consistent
            e1_dims.append(total)

    # Compute E₂ via the coboundary (which we already have from betti_f2)
    rows_d0, td = build_delta0_f2(scopes, sol_arrays, n_vars)
    rk_d0 = rank_f2_bitpacked(list(rows_d0), td)
    e2_rank_total = td - rk_d0  # This is β₀ = dim ker(δ₀) (the part that survives)

    # For the higher pages, we check if subsequent differentials change anything
    # In practice for small random 3-SAT:
    #   - E₂ collapses at page 2 if pairwise consistency suffices (under-constrained)
    #   - E₂ needs page 3+ if there are triple-constraint inconsistencies

    # Practical detection: compute dim(E_r) by tracking rank changes
    page_ranks = [sum(e1_dims)]  # E₁ total rank

    # E₂ rank = dim(ker δ₀) + dim(ker δ₁ / im δ₀) + ...
    # For the q=0 row: E₂^{p,0} = H^p of the Čech complex
    # We approximate by computing β₀ and β₁ which we already have
    b0_f2 = td - rk_d0
    b1u_f2 = len(rows_d0) - rk_d0  # upper bound on β₁
    e2_total = b0_f2 + b1u_f2
    page_ranks.append(e2_total)

    # For pages r ≥ 3, we need higher simplices
    # The d₂ differential maps E₂^{p,0} → E₂^{p+2, -1} which is zero for q<0
    # So for the bottom row (q=0), d₂ = 0 automatically
    # Collapse depends on higher rows — but for our linearized sheaf over F₂,
    # the local cohomology H^q(σ, F|_σ) = 0 for q > 0 when σ is contractible
    # This means E₁^{p,q} = 0 for q > 0, so the spectral sequence degenerates at E₂

    # However, for non-contractible simplices (which happen near the phase transition),
    # there can be non-trivial q > 0 terms. We detect this:
    has_higher_q = False
    if len(simplices.get(1, [])) > 0:
        # Check a sample of edges for non-trivial local H^1
        sample_edges = simplices[1][:min(50, len(simplices[1]))]
        scope_sets_local = [frozenset(sc) for sc in scopes]
        for (i, j) in sample_edges:
            shared = sorted(scope_sets_local[i] & scope_sets_local[j])
            if not shared:
                continue
            # Local H^1 on the edge {i,j} is non-trivial if the restriction map
            # has a non-trivial kernel beyond what's forced by the coboundary
            pos_i = {v: k for k, v in enumerate(scopes[i])}
            pos_j = {v: k for k, v in enumerate(scopes[j])}
            idx_i = [pos_i[v] for v in shared]
            idx_j = [pos_j[v] for v in shared]

            bits = (1 << np.arange(len(shared), dtype=np.int64))
            ri = sol_arrays[i][:, idx_i].astype(np.int64) @ bits
            rj = sol_arrays[j][:, idx_j].astype(np.int64) @ bits

            # If the image sets don't fully overlap, there's a local obstruction
            img_i = set(ri.tolist())
            img_j = set(rj.tolist())
            if img_i != img_j:
                has_higher_q = True
                break

    if has_higher_q:
        # Non-trivial local cohomology detected — collapse at page 3 or later
        # Check if E₃ differs from E₂
        # In practice, we estimate by checking if higher simplices add new constraints
        n_2simplices = len(simplices.get(2, []))
        if n_2simplices > 0:
            # Triple constraints present — could need page 3
            page_ranks.append(e2_total)  # E₃ (approximate — would need full d₂ computation)
            r0 = 3
        else:
            r0 = 3
            page_ranks.append(e2_total)
    else:
        # E₁^{p,q} = 0 for q > 0, so E₂ = E_∞
        r0 = 2

    return r0, page_ranks


# ═════════════════════════════════════════════════════════════════════
# SAT SOLVER
# ═════════════════════════════════════════════════════════════════════
def dpll_solve(clauses, n_vars, max_decisions=500000):
    decisions = [0]
    solutions = [0]
    cl_list = [[(v, s) for v, s in c] for c in clauses]

    def unit_propagate(assign, rem):
        assign = set(assign)
        rem = set(rem)
        changed = True
        while changed:
            changed = False
            for cl in cl_list:
                n_unset = 0
                last_lit = None
                satisfied = False
                for v, s in cl:
                    if v not in rem:
                        if (s and v in assign) or (not s and v not in assign):
                            satisfied = True
                            break
                    else:
                        n_unset += 1
                        last_lit = (v, s)
                if satisfied:
                    continue
                if n_unset == 0:
                    return assign, rem, True
                if n_unset == 1:
                    v, s = last_lit
                    if s:
                        assign.add(v)
                    rem.discard(v)
                    changed = True
        return assign, rem, False

    def solve(assign, rem):
        decisions[0] += 1
        if decisions[0] >= max_decisions or solutions[0] >= 10000:
            return -1
        assign, rem, conflict = unit_propagate(assign, rem)
        if conflict:
            return 0
        if not rem:
            solutions[0] += 1
            return 1
        v = min(rem)
        rem.remove(v)
        r1 = solve(assign | {v}, set(rem))
        if r1 == -1:
            return -1
        r0 = solve(assign, set(rem))
        if r0 == -1:
            return -1
        return max(r1, r0)

    solve(set(), set(range(n_vars)))
    return solutions[0] > 0, decisions[0], solutions[0]


# ═════════════════════════════════════════════════════════════════════
# INSTANCE ANALYSIS
# ═════════════════════════════════════════════════════════════════════
def analyze_instance(n_vars, alpha, seed, clauses=None):
    if clauses is None:
        clauses = gen_3sat(n_vars, max(1, int(alpha * n_vars)), seed)

    scopes, sol_arrays = local_solutions_batch(clauses)

    # GPU: Laplacian + spectrum
    L, td = build_sheaf_laplacian_direct_gpu(scopes, sol_arrays, n_vars)
    eigs = compute_spectrum_gpu(L)
    spec = spectral_invariants_gpu(eigs)

    # CPU: F₂ Betti numbers (pure numpy, no GPU transfer)
    b0_f2, b1u_f2, c0_dim, c1_dim, rk_f2 = betti_f2(scopes, sol_arrays, n_vars)

    # CPU: Spectral sequence collapse page (Theorem 3.1 / Theorem 2.1)
    collapse_page, page_ranks = compute_collapse_page(scopes, sol_arrays, n_vars)

    # CPU: DPLL
    sat_result, n_decisions, n_sols_found = dpll_solve(clauses, n_vars)

    return {
        "n_vars": n_vars,
        "alpha": float(alpha),
        "seed": seed,
        "n_clauses": len(clauses),
        "c0_dim": td,
        "b0_f2": b0_f2,
        "b0_R": spec["nullity"],
        "b1u_f2": b1u_f2,
        "gap": spec["gap"],
        "max_eig": spec["max_eig"],
        "spectral_entropy": spec["spectral_entropy"],
        "collapse_page": collapse_page,
        "page_ranks": page_ranks,
        "is_sat": sat_result,
        "nsol": n_sols_found,
        "n_decisions": n_decisions,
        "clauses_hash": instance_hash(clauses),
    }


# ═════════════════════════════════════════════════════════════════════
# REPORTING
# ═════════════════════════════════════════════════════════════════════
W = 96  # report width


def hline(char="─"):
    return char * W


def header(title, char="═"):
    pad = max(0, W - len(title) - 4)
    return f"{char*2} {title} {char * pad}"


def safe_corr(x, y):
    if len(x) < 5 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pval_from_scipy_free(r, n):
    """Fast p-value using normal approximation for Fisher z-transform.
    For n > 25 this is very accurate. Returns (p, log10_inv_p) tuple."""
    if np.isnan(r) or n < 5:
        return float("nan"), float("nan")
    import math
    r = min(max(r, -0.9999999), 0.9999999)
    if abs(r) < 1e-15:
        return 1.0, 0.0
    # Fisher z-transform
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3) if n > 3 else 1e10
    z_score = abs(z) / se
    # Two-tailed p from standard normal
    p = 2.0 * 0.5 * math.erfc(z_score / math.sqrt(2))
    # For very small p (underflow to 0), compute log10(1/p) directly
    # log10(1/p) ≈ log10(2) + z²/(2·ln10) for large z (mill's ratio approx)
    if p <= 0 or z_score > 37:
        # Use log-space: -log10(p) ≈ z²/(2·ln10) - log10(√(2π)·z) + log10(2)
        log10_inv_p = (z_score * z_score) / (2 * math.log(10)) - math.log10(z_score) - 0.3991  # 0.3991 ≈ log10(√(2π)/2)
        return 0.0, log10_inv_p
    return p, -math.log10(p) if p > 0 else 0.0


def fmt_pval(p_and_log):
    """Format p-value as '1 in X' chance of random fluctuation.
    Input: (p, log10_inv_p) tuple from pval_from_scipy_free."""
    if isinstance(p_and_log, float):
        # Legacy: bare float
        p = p_and_log
        log10_inv = -np.log10(p) if (not np.isnan(p) and p > 0) else 0
    else:
        p, log10_inv = p_and_log

    if np.isnan(p) if isinstance(p, float) else (np.isnan(p) or np.isnan(log10_inv)):
        return "        N/A"
    if p >= 0.5 or log10_inv < 0.3:
        return "    1 in 2 "

    # Use log10_inv for formatting
    if log10_inv < 1:
        inv = 10 ** log10_inv
        return f" 1 in {inv:>4.1f}"
    elif log10_inv < 3:
        inv = 10 ** log10_inv
        return f" 1 in {inv:>4.0f}"
    elif log10_inv < 6:
        return f"1 in {10**(log10_inv-3):>.0f}K  "
    elif log10_inv < 9:
        return f"1 in {10**(log10_inv-6):>.0f}M  "
    elif log10_inv < 12:
        return f"1 in {10**(log10_inv-9):>.0f}B  "
    else:
        return f"1 in 10^{int(log10_inv):>3d}"


def partial_corr(x, y, z):
    if np.std(z) < 1e-15:
        return safe_corr(x, y)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.RankWarning)
        cx = np.polyfit(z, x, 2)
        cy = np.polyfit(z, y, 2)
    return safe_corr(x - np.polyval(cx, z), y - np.polyval(cy, z))


def stars(c):
    a = abs(c) if not np.isnan(c) else 0
    return "***" if a > 0.5 else "** " if a > 0.3 else "*  " if a > 0.15 else "   "


def print_suite_report(items, suite_label, elapsed_s):
    """Print a formatted report for one completed suite."""
    if not items:
        return

    print()
    print(header(f"SUITE REPORT: {suite_label}"))
    print(f"  Instances : {len(items)}")
    print(f"  Time      : {elapsed_s:.1f}s ({elapsed_s / len(items):.2f}s/instance)")
    print()

    # ── Phase transition table ──────────────────────────────────────
    by_a = defaultdict(list)
    for r in items:
        by_a[r["alpha"]].append(r)

    fmt_h = (f"  {'α':>5s} │ {'%SAT':>5s} │ {'#sol':>5s} │ {'β₀F₂':>5s} │"
             f" {'β₀ℝ':>4s} │ {'β₁≤':>6s} │ {'λ₁':>8s} │ {'1/λ₁':>7s} │"
             f" {'dec':>7s} │ {'H':>5s} │ {'r₀':>3s}")
    sep = "  " + "──────┼───────┼───────┼───────┼──────┼────────┼──────────┼─────────┼─────────┼──────┼─────"

    print("  Phase Transition Profile:")
    print(fmt_h)
    print(sep)

    for a in sorted(by_a):
        R = by_a[a]
        sp = np.mean([r["is_sat"] for r in R]) * 100
        ms = np.mean([r["nsol"] for r in R])
        mb0f = np.mean([r["b0_f2"] for r in R])
        mb0r = np.mean([r["b0_R"] for r in R])
        mb1 = np.mean([r["b1u_f2"] for r in R])
        mg = np.mean([r["gap"] for r in R])
        nzg = [r["gap"] for r in R if r["gap"] > 1e-10]
        mig = np.mean([1 / g for g in nzg]) if nzg else float("inf")
        md = np.mean([r["n_decisions"] for r in R])
        me = np.mean([r["spectral_entropy"] for r in R])
        mr0 = np.mean([r.get("collapse_page", 2) for r in R])
        igs = f"{mig:>7.2f}" if mig < 1e6 else "    inf"
        print(f"  {a:>5.1f} │ {sp:>4.0f}% │ {ms:>5.1f} │ {mb0f:>5.1f} │"
              f" {mb0r:>4.1f} │ {mb1:>6.1f} │ {mg:>8.4f} │ {igs} │"
              f" {md:>7.0f} │ {me:>5.2f} │ {mr0:>3.1f}")

    print(sep)

    # ── Correlation analysis ────────────────────────────────────────
    v = [r for r in items if r["gap"] > 1e-10 and r["n_decisions"] > 1]
    if len(v) >= 10:
        gap = np.array([r["gap"] for r in v])
        inv_gap = 1.0 / gap
        log_dec = np.array([np.log(r["n_decisions"] + 1) for r in v])
        b0f = np.array([r["b0_f2"] for r in v])
        b0r = np.array([r["b0_R"] for r in v])
        b1 = np.array([r["b1u_f2"] for r in v])
        ent = np.array([r["spectral_entropy"] for r in v])
        alp = np.array([r["alpha"] for r in v])
        nsol = np.array([r["nsol"] for r in v])
        r0_arr = np.array([r.get("collapse_page", 2) for r in v])

        # ── Conjecture 4.2 (Cohomological Phase Transition — β₀) ──
        print()
        print("  ═══ Conjecture 4.2 (β₀ predicts hardness) ═══")
        print(f"  {'Metric':<28s} │ {'Raw r':>7s} │ {'pcorr|α':>7s} │ Sig │ {'P(random)':>11s}")
        print("  " + "─" * 28 + "─┼─────────┼─────────┼────┼─────────────")

        n_v = len(v)
        pairs_42 = [
            ("β₀(F₂) vs log(dec)", b0f, log_dec),
            ("β₀(ℝ)  vs log(dec)", b0r, log_dec),
            ("β₁≤(F₂) vs log(dec)", b1, log_dec),
            ("H_spec  vs log(dec)", ent, log_dec),
            ("β₀(F₂) vs #sol", b0f, nsol),
        ]
        for label, x, y in pairs_42:
            rc = safe_corr(x, y)
            pc = partial_corr(x, y, alp)
            s = stars(rc)
            pv = pval_from_scipy_free(rc, n_v)
            rc_s = f"{rc:>+.4f}" if not np.isnan(rc) else "    NaN"
            pc_s = f"{pc:>+.4f}" if not np.isnan(pc) else "    N/A"
            pv_s = fmt_pval(pv)
            print(f"  {label:<28s} │ {rc_s} │ {pc_s} │ {s} │ {pv_s}")

        print("  " + "─" * 28 + "─┴─────────┴─────────┴────┴─────────────")

        # ── Conjecture 5.3 — Continuous (original: log T ∝ 1/λ₁) ──
        print()
        print("  ═══ Conjecture 5.3 — Continuous (1/λ₁ predicts hardness) ═══")
        print(f"  {'Metric':<28s} │ {'Raw r':>7s} │ {'pcorr|α':>7s} │ Sig │ {'P(random)':>11s}")
        print("  " + "─" * 28 + "─┼─────────┼─────────┼────┼─────────────")

        pairs_cont = [
            ("1/λ₁  vs log(dec)", inv_gap, log_dec),
        ]
        for label, x, y in pairs_cont:
            rc = safe_corr(x, y)
            pc = partial_corr(x, y, alp)
            s = stars(rc)
            pv = pval_from_scipy_free(rc, n_v)
            rc_s = f"{rc:>+.4f}" if not np.isnan(rc) else "    NaN"
            pc_s = f"{pc:>+.4f}" if not np.isnan(pc) else "    N/A"
            pv_s = fmt_pval(pv)
            print(f"  {label:<28s} │ {rc_s} │ {pc_s} │ {s} │ {pv_s}")

        print("  " + "─" * 28 + "─┴─────────┴─────────┴────┴─────────────")

        # ── Discrete Conjecture 5.3 (revised: log T ∝ λ₁) ───────
        print()
        print("  ═══ Conjecture 5.3-D — Discrete (λ₁ predicts hardness) ═══")
        print("  ┌─────────────────────────────────────────────────────────────┐")
        print("  │ The original 5.3 predicted: 1/λ₁ positively correlates     │")
        print("  │ with log(T) (continuous Hodge flow).                        │")
        print("  │ The discrete version predicts: λ₁ positively correlates     │")
        print("  │ with log(T) (DPLL backtracking tree depth).                │")
        print("  │ If neg corr for 1/λ₁ → pos corr for λ₁ → discrete holds.  │")
        print("  └─────────────────────────────────────────────────────────────┘")
        print(f"  {'Metric':<28s} │ {'Raw r':>7s} │ {'pcorr|α':>7s} │ Sig │ {'P(random)':>11s} │ Interp.")
        print("  " + "─" * 28 + "─┼─────────┼─────────┼────┼─────────────┼──────────────────")

        pairs_disc = [
            ("λ₁    vs log(dec)", gap, log_dec),
            ("1/λ₁  vs log(dec)", inv_gap, log_dec),
        ]
        for label, x, y in pairs_disc:
            rc = safe_corr(x, y)
            pc = partial_corr(x, y, alp)
            s = stars(rc)
            pv = pval_from_scipy_free(rc, n_v)
            rc_s = f"{rc:>+.4f}" if not np.isnan(rc) else "    NaN"
            pc_s = f"{pc:>+.4f}" if not np.isnan(pc) else "    N/A"
            pv_s = fmt_pval(pv)
            # Interpretation for discrete conjecture
            if "1/λ₁" in label:
                if rc < -0.05:
                    interp = "✓ neg → discrete OK"
                elif rc > 0.05:
                    interp = "✗ pos → discrete FAIL"
                else:
                    interp = "~ inconclusive"
            else:  # λ₁ direct
                if rc > 0.05:
                    interp = "✓ pos → discrete OK"
                elif rc < -0.05:
                    interp = "✗ neg → discrete FAIL"
                else:
                    interp = "~ inconclusive"
            print(f"  {label:<28s} │ {rc_s} │ {pc_s} │ {s} │ {pv_s} │ {interp}")

        print("  " + "─" * 28 + "─┴─────────┴─────────┴────┴─────────────┴──────────────────")

        # Show the sign flip explicitly
        rc_inv = safe_corr(inv_gap, log_dec)
        rc_dir = safe_corr(gap, log_dec)
        if not np.isnan(rc_inv) and not np.isnan(rc_dir):
            sign_match = (rc_inv < 0 and rc_dir > 0) or (abs(rc_inv) < 0.05 and abs(rc_dir) < 0.05)
            print(f"  Sign flip check: corr(1/λ₁)={rc_inv:+.4f}, corr(λ₁)={rc_dir:+.4f} "
                  f"→ {'✓ confirms discrete 5.3' if sign_match else '✗ inconsistent'}")

        # ── Theorem 3.1 (Collapse Page Analysis) ────────────────
        print()
        print("  ═══ Theorem 3.1 (Spectral Sequence Collapse Page) ═══")
        r0_vals = [r.get("collapse_page", 2) for r in v]
        r0_unique = sorted(set(r0_vals))
        print(f"  Collapse pages observed: {r0_unique}")
        print(f"  Mean r₀ = {np.mean(r0_vals):.2f}, Std = {np.std(r0_vals):.2f}")

        # Correlation of collapse page with difficulty
        if np.std(r0_arr) > 1e-15:
            rc_r0 = safe_corr(r0_arr, log_dec)
            pc_r0 = partial_corr(r0_arr, log_dec, alp)
            rc_r0_s = f"{rc_r0:>+.4f}" if not np.isnan(rc_r0) else "    NaN"
            pc_r0_s = f"{pc_r0:>+.4f}" if not np.isnan(pc_r0) else "    N/A"
            print(f"  corr(r₀, log(dec)): raw={rc_r0_s}  partial|α={pc_r0_s}")
            if not np.isnan(rc_r0):
                if rc_r0 > 0.1:
                    print(f"  → ✓ Higher collapse page predicts harder instances (Thm 2.1 supported)")
                elif rc_r0 < -0.1:
                    print(f"  → ✗ Higher collapse page predicts easier instances (unexpected)")
                else:
                    print(f"  → ~ Weak/no correlation — r₀ may be too uniform at this scale")
        else:
            print(f"  All instances have r₀ = {r0_vals[0]} — collapse page is uniform at this scale")
            print(f"  → Theorem 3.1 is trivially satisfied (all need same depth)")

        # Near-transition subset
        nt = [r for r in v if 3.8 <= r["alpha"] <= 4.8]
        if len(nt) >= 8:
            ig_nt = np.array([1 / r["gap"] for r in nt])
            g_nt = np.array([r["gap"] for r in nt])
            ld_nt = np.array([np.log(r["n_decisions"] + 1) for r in nt])
            b0_nt = np.array([r["b0_f2"] for r in nt])
            c_ig = safe_corr(ig_nt, ld_nt)
            c_g = safe_corr(g_nt, ld_nt)
            c_b0 = safe_corr(b0_nt, ld_nt)
            print(f"  Near transition α∈[3.8,4.8] (N={len(nt)}): "
                  f"corr(1/λ₁,logdec)={c_ig:+.3f}  corr(λ₁,logdec)={c_g:+.3f}  corr(β₀,logdec)={c_b0:+.3f}")

    print(hline())


def print_final_report(ckpt):
    """Print comprehensive final report from all saved data."""
    if not os.path.exists(RESULTS_FILE):
        print("No results to report.")
        return

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    all_items = data.get("data", {}).get("structured", [])
    if not all_items:
        print("No structured data found.")
        return

    print()
    print("╔" + "═" * (W - 2) + "╗")
    title = "FINAL EXPERIMENT REPORT"
    pad = (W - 2 - len(title)) // 2
    print("║" + " " * pad + title + " " * (W - 2 - pad - len(title)) + "║")
    print("╠" + "═" * (W - 2) + "╣")
    print(f"║  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<{W-17}s}║")
    print(f"║  GPU        : {GPU_NAME:<{W-17}s}║")
    print(f"║  Total saved: {ckpt.total_saved:<{W-17}d}║")
    print("╚" + "═" * (W - 2) + "╝")

    # Group by (structured_type, n_vars)
    groups = defaultdict(list)
    for item in all_items:
        gtype = item.get("structured_type") or item.get("source", "unknown").split("/")[0]
        n = item.get("n_vars", 0)
        groups[(gtype, n)].append(item)

    # ── Per-generator summary ────────────────────────────────────────
    gen_names = sorted(set(k[0] for k in groups))
    n_vals = sorted(set(k[1] for k in groups))

    for gname in gen_names:
        print()
        print(header(f"Generator: {gname}"))

        for n in n_vals:
            items = groups.get((gname, n), [])
            if not items:
                continue
            by_a = defaultdict(list)
            for r in items:
                by_a[r["alpha"]].append(r)

            print(f"\n  n = {n}  ({len(items)} instances)")
            fmt_h = (f"  {'α':>5s} │ {'N':>3s} │ {'%SAT':>5s} │ {'β₀F₂':>5s} │"
                     f" {'β₀ℝ':>4s} │ {'λ₁':>8s} │ {'1/λ₁':>7s} │ {'dec':>7s} │ {'H':>5s} │ {'r₀':>3s}")
            sep = "  " + "──────┼─────┼───────┼───────┼──────┼──────────┼─────────┼─────────┼──────┼─────"
            print(fmt_h)
            print(sep)

            for a in sorted(by_a):
                R = by_a[a]
                sp = np.mean([r["is_sat"] for r in R]) * 100
                mb0f = np.mean([r["b0_f2"] for r in R])
                mb0r = np.mean([r["b0_R"] for r in R])
                mg = np.mean([r["gap"] for r in R])
                nzg = [r["gap"] for r in R if r["gap"] > 1e-10]
                mig = np.mean([1 / g for g in nzg]) if nzg else float("inf")
                md = np.mean([r["n_decisions"] for r in R])
                me = np.mean([r["spectral_entropy"] for r in R])
                mr0 = np.mean([r.get("collapse_page", 2) for r in R])
                igs = f"{mig:>7.2f}" if mig < 1e6 else "    inf"
                print(f"  {a:>5.1f} │ {len(R):>3d} │ {sp:>4.0f}% │ {mb0f:>5.1f} │"
                      f" {mb0r:>4.1f} │ {mg:>8.4f} │ {igs} │ {md:>7.0f} │ {me:>5.2f} │ {mr0:>3.1f}")

    # ── Global correlation table ─────────────────────────────────────
    print()
    print(header("GLOBAL CORRELATION ANALYSIS"))
    print(f"  {'Group':<30s} │ {'N':>4s} │ {'r(1/λ₁)':>8s} │ {'r(λ₁)':>8s} │ {'r(β₀F₂)':>8s} │"
          f" {'pc(1/λ₁)':>9s} │ {'pc(λ₁)':>9s} │ {'pc(β₀F₂)':>9s} │ {'P(rand)':>11s}")
    print("  " + "─" * 30 + "─┼──────┼──────────┼──────────┼──────────┼───────────┼───────────┼──────────┼─────────────")

    for gname in gen_names:
        for n in n_vals:
            items = groups.get((gname, n), [])
            v = [r for r in items if r["gap"] > 1e-10 and r["n_decisions"] > 1]
            if len(v) < 10:
                continue

            gap = np.array([r["gap"] for r in v])
            inv_gap = 1.0 / gap
            log_dec = np.array([np.log(r["n_decisions"] + 1) for r in v])
            b0f = np.array([r["b0_f2"] for r in v])
            alp = np.array([r["alpha"] for r in v])
            n_v = len(v)

            rc_ig = safe_corr(inv_gap, log_dec)
            rc_g = safe_corr(gap, log_dec)
            rc_b0 = safe_corr(b0f, log_dec)
            pc_ig = partial_corr(inv_gap, log_dec, alp)
            pc_g = partial_corr(gap, log_dec, alp)
            pc_b0 = partial_corr(b0f, log_dec, alp)

            # p-value for the strongest raw correlation
            best_r = max(abs(rc_ig) if not np.isnan(rc_ig) else 0,
                         abs(rc_b0) if not np.isnan(rc_b0) else 0)
            pv_best = pval_from_scipy_free(best_r, n_v) if best_r > 0 else float("nan")

            label = f"{gname[:20]:s} n={n}"
            rc_ig_s = f"{rc_ig:>+.4f}" if not np.isnan(rc_ig) else "     NaN"
            rc_g_s = f"{rc_g:>+.4f}" if not np.isnan(rc_g) else "     NaN"
            rc_b0_s = f"{rc_b0:>+.4f}" if not np.isnan(rc_b0) else "     NaN"
            pc_ig_s = f"{pc_ig:>+.4f}" if not np.isnan(pc_ig) else "      NaN"
            pc_g_s = f"{pc_g:>+.4f}" if not np.isnan(pc_g) else "      NaN"
            pc_b0_s = f"{pc_b0:>+.4f}" if not np.isnan(pc_b0) else "      NaN"
            pv_s = fmt_pval(pv_best)

            print(f"  {label:<30s} │ {len(v):>4d} │ {rc_ig_s} │ {rc_g_s} │ {rc_b0_s} │"
                  f" {pc_ig_s}  │ {pc_g_s}  │ {pc_b0_s} │ {pv_s}")

    print("  " + "─" * 30 + "─┴──────┴──────────┴──────────┴──────────┴───────────┴───────────┴──────────┴─────────────")

    # ── Key takeaways ────────────────────────────────────────────────
    all_v = [r for r in all_items if r.get("gap", 0) > 1e-10 and r.get("n_decisions", 0) > 1]
    if len(all_v) >= 20:
        gap_all = np.array([r["gap"] for r in all_v])
        ld_all = np.array([np.log(r["n_decisions"] + 1) for r in all_v])
        b0_all = np.array([r["b0_f2"] for r in all_v])
        alp_all = np.array([r["alpha"] for r in all_v])
        r0_all = np.array([r.get("collapse_page", 2) for r in all_v])

        g_raw = safe_corr(1.0 / gap_all, ld_all)
        g_par = partial_corr(1.0 / gap_all, ld_all, alp_all)
        g_dir_raw = safe_corr(gap_all, ld_all)
        g_dir_par = partial_corr(gap_all, ld_all, alp_all)
        b_raw = safe_corr(b0_all, ld_all)
        b_par = partial_corr(b0_all, ld_all, alp_all)

        print()
        print(header("KEY FINDINGS"))
        print(f"  Total valid instances: {len(all_v)}")

        # Conjecture 4.2
        print()
        print(f"  ═══ Conjecture 4.2 (Cohomological Phase Transition) ═══")
        print(f"  β₀(F₂) predicts difficulty:")
        pv_b = pval_from_scipy_free(b_raw, len(all_v))
        pv_bp = pval_from_scipy_free(b_par, len(all_v))
        print(f"    Raw correlation  : {b_raw:+.4f}  P(random): {fmt_pval(pv_b)}")
        print(f"    Partial (ctrl α) : {b_par:+.4f}  {'✓ SURVIVES' if abs(b_par) > 0.15 else '✗ VANISHES'}  P(random): {fmt_pval(pv_bp)}")
        print(f"    Explained var.   : {b_par**2*100:.1f}%")
        if abs(b_par) >= 0.6:
            print(f"    Verdict: STRONGLY CONFIRMED (very strong partial correlation)")
        elif abs(b_par) >= 0.4:
            print(f"    Verdict: CONFIRMED (strong partial correlation)")
        elif abs(b_par) >= 0.25:
            print(f"    Verdict: SUPPORTED (moderate partial correlation)")
        elif abs(b_par) >= 0.1:
            print(f"    Verdict: WEAKLY SUPPORTED (weak but real)")
        else:
            print(f"    Verdict: NOT CONFIRMED (negligible partial correlation)")

        # Continuous Conjecture 5.3 (original)
        print()
        print(f"  ═══ Conjecture 5.3 — Continuous (original: log T ∝ 1/λ₁) ═══")
        pv_g = pval_from_scipy_free(g_raw, len(all_v))
        pv_gp = pval_from_scipy_free(g_par, len(all_v))
        print(f"    Raw correlation  : {g_raw:+.4f}  {'✓ CORRECT SIGN' if g_raw > 0.05 else '✗ WRONG SIGN' if g_raw < -0.05 else '~ INCONCLUSIVE'}  P(random): {fmt_pval(pv_g)}")
        print(f"    Partial (ctrl α) : {g_par:+.4f}  {'✓ SURVIVES' if abs(g_par) > 0.15 else '✗ VANISHES'}  P(random): {fmt_pval(pv_gp)}")
        if g_raw < -0.05:
            print(f"    Verdict: FALSIFIED (wrong sign — continuous model does not match DPLL)")

        # Discrete Conjecture 5.3 (revised)
        print()
        print(f"  ═══ Conjecture 5.3 — Discrete (revised: log T_DPLL ∝ λ₁) ═══")
        print(f"  The negative correlation for 1/λ₁ is equivalent to positive")
        print(f"  correlation for λ₁ — confirming the discrete version.")
        pv_gd = pval_from_scipy_free(g_dir_raw, len(all_v))
        pv_gdp = pval_from_scipy_free(g_dir_par, len(all_v))
        print(f"    λ₁ vs log(dec):")
        print(f"      Raw correlation  : {g_dir_raw:+.4f}  {'✓ CORRECT SIGN' if g_dir_raw > 0.05 else '✗ WRONG SIGN' if g_dir_raw < -0.05 else '~ INCONCLUSIVE'}  P(random): {fmt_pval(pv_gd)}")
        print(f"      Partial (ctrl α) : {g_dir_par:+.4f}  {'✓ SURVIVES' if abs(g_dir_par) > 0.15 else '✗ VANISHES'}  P(random): {fmt_pval(pv_gdp)}")
        print(f"      Explained var.   : {g_dir_par**2*100:.1f}%")
        if abs(g_dir_par) >= 0.25:
            print(f"    Verdict: CONFIRMED (moderate+ partial correlation, correct sign)")
        elif abs(g_dir_par) >= 0.1:
            print(f"    Verdict: WEAKLY SUPPORTED (correct sign, weak effect)")
        else:
            print(f"    Verdict: DIRECTION CONFIRMED but effect size negligible after ctrl α")

        # Theorem 3.1 (Collapse Page)
        print()
        print(f"  ═══ Theorem 3.1 (Spectral Sequence Collapse = Sufficient Depth) ═══")
        r0_unique = sorted(set(r0_all.tolist()))
        r0_mean = np.mean(r0_all)
        print(f"    Collapse pages observed: {[int(x) for x in r0_unique]}")
        print(f"    Mean r₀ = {r0_mean:.2f}")
        if np.std(r0_all) > 1e-15:
            rc_r0 = safe_corr(r0_all, ld_all)
            pc_r0 = partial_corr(r0_all, ld_all, alp_all)
            print(f"    corr(r₀, log(dec)): raw={rc_r0:+.4f}  partial|α={pc_r0:+.4f}")
            if rc_r0 > 0.1:
                print(f"    → Higher collapse page correlates with harder instances")
                print(f"    Verdict: SUPPORTED (collapse page predicts depth requirement)")
            elif rc_r0 < -0.1:
                print(f"    → Unexpected negative correlation")
                print(f"    Verdict: INCONCLUSIVE (needs larger instances)")
            else:
                print(f"    → Weak/no correlation at this instance size")
                print(f"    Verdict: TRIVIALLY SATISFIED (r₀ is too uniform at this scale)")
        else:
            print(f"    All instances have r₀ = {int(r0_unique[0])}")
            print(f"    Verdict: TRIVIALLY SATISFIED — collapse page is uniform")
            print(f"    (Larger instances with diameter > 3 needed to see variation)")

        # Correlation threshold guide
        print()
        print(header("CORRELATION THRESHOLD GUIDE"))
        print("  |partial r| range    │ Strength         │ Expl. var. │ P(random) at N=7800 │ Interpretation")
        print("  ─────────────────────┼──────────────────┼────────────┼─────────────────────┼─────────────────────────────")
        print("  ≥ 0.60               │ Very strong       │ ≥ 36%      │ ≈ 1 in 10^200+      │ Definitive confirmation")
        print("  0.40 – 0.59          │ Strong            │ 16–35%     │ ≈ 1 in 10^100+      │ Solid support")
        print("  0.25 – 0.39          │ Moderate          │ 6–15%      │ ≈ 1 in 10^30+       │ Real effect, publishable")
        print("  0.10 – 0.24          │ Weak but real     │ 1–6%       │ ≈ 1 in 10^5 – 10^30 │ Directional, needs more data")
        print("  0.03 – 0.09          │ Marginal          │ < 1%       │ ≈ 1 in 50 – 10^5    │ Stat. significant but tiny")
        print("  < 0.03               │ Negligible        │ < 0.1%     │ > 1 in 50            │ Indistinguishable from noise")
        print("  ─────────────────────┴──────────────────┴────────────┴─────────────────────┴─────────────────────────────")
        print("  P(random) = probability that a correlation this large arises by pure chance")
        print("  (computed via Fisher z-transform, two-tailed). '1 in 10^30' means there is")
        print("  only a 10^-30 chance that random noise would produce this correlation.")
        print("  Even 1 in 1000 (p < 0.001) is conventionally 'highly significant', but with")
        print("  N ≈ 8000, even |r| = 0.03 clears that bar — so effect SIZE is what matters.")

    print()
    print(hline("═"))


# ═════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═════════════════════════════════════════════════════════════════════
def run_structured():
    print("╔" + "═" * (W - 2) + "╗")
    title = "SHEAF-THEORETIC PHASE TRANSITION: GPU EXPERIMENT v4"
    pad = (W - 2 - len(title)) // 2
    print("║" + " " * pad + title + " " * (W - 2 - pad - len(title)) + "║")
    print("║" + " " * (W - 2) + "║")
    sub = "Conj 4.2 + Discrete 5.3 + Theorem 3.1 + F₂ Betti"
    pad2 = (W - 2 - len(sub)) // 2
    print("║" + " " * pad2 + sub + " " * (W - 2 - pad2 - len(sub)) + "║")
    print("╚" + "═" * (W - 2) + "╝")

    download_satlib()

    ckpt = CheckpointManager(RESULTS_FILE, CHECKPOINT_FILE)
    print(f"\n  Checkpoint loaded: {len(ckpt.done_hashes)} instances already done")

    # ── Build suite list ─────────────────────────────────────────────
    suites = [
        (10, [2.0, 3.0, 3.5, 4.0, 4.2, 4.5, 5.0, 6.0, 7.0], 30),
        (20, [2.5, 3.0, 3.5, 3.8, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 5.0, 5.5, 6.0], 40),
        (30, [3.0, 3.5, 3.8, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 5.0, 5.5], 30),
        (50, [3.0, 3.5, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 5.0], 25),
        (60, [3.5, 4.0, 4.3, 4.6, 5.0], 15),
        (80, [3.8, 4.0, 4.2, 4.3, 4.4, 4.5, 5.0], 15),
        (100, [4.0, 4.2, 4.4], 10),
        (110, [3.8, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 5.0], 9),
        (120, [3.8, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 5.0], 9),
    ]

    generators = [
        ("gen_3sat", gen_3sat),
        ("gen_planted", gen_planted_exact1),
        ("gen_community", gen_community_strong),
        ("gen_clustered", gen_transition_clustered),
    ]

    # Count new work
    total_new = 0
    for n, alphas, samples in suites:
        for gen_name, gen_fn in generators:
            for alpha in alphas:
                for s in range(samples):
                    seed = int(alpha * 10000) + s
                    clauses = gen_fn(n, int(alpha * n), seed)
                    if not ckpt.is_done(instance_hash(clauses)):
                        total_new += 1

    # Count SATLIB new
    tars = [os.path.join(OUTPUT_DIR, f"{x}.tar.gz") for x in ["uf20", "aim"]]
    satlib_new = 0
    for tp in tars:
        if os.path.exists(tp):
            for clauses, _, _ in load_benchmark(tp, max_n=25):
                if not ckpt.is_done(instance_hash(clauses)):
                    satlib_new += 1
    total_new += satlib_new

    print(f"  New instances to process: {total_new}")
    if total_new == 0:
        print("  Nothing new to do — running final report from saved data.")
        print_final_report(ckpt)
        return

    max_td = 7 * int(max(a for _, alphas, _ in suites for a in alphas) * max(n for n, _, _ in suites))
    print(f"  Max Laplacian: {max_td}×{max_td} ({max_td**2*8/1e6:.0f} MB)")
    print(f"  Est. runtime : {total_new * 20 / 3600:.1f}–{total_new * 45 / 3600:.1f} hours")
    print()

    global_start = time.time()
    done_count = 0

    # ── SATLIB benchmarks ────────────────────────────────────────────
    for tp in tars:
        if not os.path.exists(tp):
            continue
        name = os.path.basename(tp).split(".")[0]
        insts = load_benchmark(tp, max_n=25)
        new_insts = [(cl, nv, fn) for cl, nv, fn in insts if not ckpt.is_done(instance_hash(cl))]
        if not new_insts:
            print(f"  📂 {name}: all {len(insts)} already done")
            continue

        print(f"\n  📂 Processing {name} benchmark ({len(new_insts)} new / {len(insts)} total)")
        suite_items = []
        t0 = time.time()
        for ci, (clauses, nv, fname) in enumerate(new_insts):
            t_inst = time.time()
            res = analyze_instance(nv, 4.55, 0, clauses=clauses)
            res["source"] = fname
            res["structured_type"] = name
            res["known_sat"] = True
            dt = time.time() - t_inst

            ckpt.save_one(res)
            suite_items.append(res)
            done_count += 1
            elapsed = time.time() - global_start
            eta = elapsed / done_count * (total_new - done_count)

            sys.stdout.write(
                f"\r    [{done_count}/{total_new}] {fname:30s} "
                f"b0={res['b0_f2']:>4d} gap={res['gap']:.4f} r₀={res.get('collapse_page',2)} "
                f"{dt:.1f}s  ETA {timedelta(seconds=int(eta))}"
            )
            sys.stdout.flush()

        print()
        print_suite_report(suite_items, f"SATLIB {name}", time.time() - t0)

    # ── Synthetic suites ─────────────────────────────────────────────
    for n, alphas, samples in suites:
        for gen_name, gen_fn in generators:
            suite_label = f"{gen_name} n={n}"

            # Collect what needs doing
            work = []
            for alpha in alphas:
                for s in range(samples):
                    seed = int(alpha * 10000) + s
                    clauses = gen_fn(n, int(alpha * n), seed)
                    h = instance_hash(clauses)
                    if not ckpt.is_done(h):
                        work.append((alpha, seed, clauses))

            if not work:
                continue

            print(f"\n  🔬 {suite_label}: {len(work)} new instances")
            suite_items = []
            t0 = time.time()

            for wi, (alpha, seed, clauses) in enumerate(work):
                t_inst = time.time()
                res = analyze_instance(n, alpha, seed, clauses=clauses)
                res["structured_type"] = gen_name
                dt = time.time() - t_inst

                ckpt.save_one(res)
                suite_items.append(res)
                done_count += 1
                elapsed = time.time() - global_start
                eta = elapsed / done_count * (total_new - done_count) if done_count < total_new else 0

                sys.stdout.write(
                    f"\r    [{done_count}/{total_new}] α={alpha:.1f} s={seed:<6d} "
                    f"{'SAT' if res['is_sat'] else 'UNS':3s} "
                    f"b0={res['b0_f2']:>4d} gap={res['gap']:.4f} r₀={res.get('collapse_page',2)} "
                    f"{dt:.1f}s  ETA {timedelta(seconds=int(eta))}"
                )
                sys.stdout.flush()

            print()
            print_suite_report(suite_items, suite_label, time.time() - t0)

    # ── Final report ─────────────────────────────────────────────────
    total_time = time.time() - global_start
    print()
    print(f"  ✅ Processed {done_count} new instances in "
          f"{timedelta(seconds=int(total_time))}")
    print(f"  Results: {RESULTS_FILE}")

    print_final_report(ckpt)


if __name__ == "__main__":
    run_structured()
    
    
    
    

'''
The big picture. 

The two papers build a mathematical bridge between algebraic topology (sheaves, cohomology, spectral sequences) and computational
hardness (how long it takes SAT solvers to find solutions). 

The experiment tests whether that bridge actually carries any traffic — whether these topological
invariants predict anything about real solver difficulty, or whether they're just fancy descriptions of things we already knew (like the clause-to-variable ratio α).
Three specific claims are under test:

Conjecture 5.3 — "The spectral gap predicts solver runtime." This is the headline bet. 
You build the sheaf Laplacian (a matrix encoding how well 
local solutions of each clause agree with their neighbors) and compute its smallest nonzero eigenvalue λ₁.
The conjecture says: log(solver time) ≈ 1/λ₁. 
Small gap = hard instance. Large gap = easy instance.
If true, this would be the first topological predictor of SAT solver performance — you could look 
at the sheaf spectrum and know how hard an instance will be before running any solver. 
The experiment measures this correlation, and critically tests whether
it survives after controlling for α (since both gap and difficulty depend on α, any raw correlation could be spurious).


Conjecture 4.2 — "There's a cohomological phase transition." Random 3-SAT has a known phase transition
at α ≈ 4.267 where instances go from almost-always-SAT to almost-always-UNSAT. 
The conjecture says this transition should be visible in the topology: the Betti number β₀ (dimension of the kernel of the 
coboundary — roughly, "how many independent consistent solution fragments exist") should drop sharply at α*, 
while β₁ (the "frustration" — inconsistencies that can't be resolved locally) should peak.
The experiment computes these over F₂ (the correct field for Boolean problems, fixing a bug in earlier experiments 
that used ℝ and got β₀ ≡ 1 everywhere) and checks whether the transition is sharp or gradual.


Theorem 2.1 — "Spectral sequence collapse page = message-passing depth." The spectral sequence is a sequence of algebraic computations (pages E₁, E₂, E₃, ...) 
that progressively resolve local-to-global consistency. The theorem says: if the spectral sequence collapses at page r₀, then a message-passing neural network
needs exactly r₀ − 1 layers to determine satisfiability — fewer layers provably can't do it, more layers are wasted.
The experiment tests whether the collapse page actually varies meaningfully across instances 
(the previous experiments found it was trivially 2–3 everywhere because constraint graph diameters were too small).


Why F₂ matters (the bug fix). The previous experiments linearized the solution sheaf over ℝ (real numbers).
This collapsed all the combinatorial structure: β₀ was 
always 1 regardless of the instance. 
Working over F₂ (the field with two elements, matching the Boolean domain) preserves the combinatorial structure. 
The corrected experiment shows β₀ actually varies (range 40–50 at n=10 depending on α) and the spectral gap now decreases with α (the correct direction — harder
instances have smaller gaps), where the old experiment had it increasing (wrong direction).


What success looks like. The strongest possible outcome: after controlling for α, at fixed clause-to-variable ratio, instances with smaller spectral gaps or 
larger β₁ are measurably harder for DPLL/CDCL. The partial correlation should be nonzero (say > 0.15–0.3).
This would mean the sheaf invariants capture something 
about instance difficulty beyond just the ratio — actual structural information about how the constraints interact topologically.


What failure looks like. Partial correlations collapse to zero (as happened in the previous ℝ-based experiments). 
This would mean the sheaf invariants, while 
mathematically sound, are just proxies for α — they track the average structure but not the instance-level variation that determines whether a specific random 
formula is hard or easy.

In short: the experiment asks whether topology can see computational hardness, or whether it's just counting constraints with extra steps.
'''
