# Implement the user's algorithm "GENERATE-RANDOM-DBN(n, T, alpha)" and show results.
# - Uses a reproducible RNG seed for this demo (you can change it below).
# - Prints: D (density), edge counts, edge lists, densities, and a small sample of CPDs.
# - Visualizes the DBN with clearly visible arrowheads.
#
# You can tweak n, T, alpha, and seed in the CONFIG block.

import itertools
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

try:
    import networkx as nx
except Exception as e:
    raise RuntimeError("This demo requires the 'networkx' package, which should be available in this environment.")


Node = Tuple[str, int]  # ('X3', 1) means variable X3 at time slice 1


@dataclass
class Config:
    n: int
    T: int
    alpha: float
    seed: Optional[int] = None  #None - no fixed seed


def generate_random_dbn(cfg: Config):
    rng = random.Random(cfg.seed) if cfg.seed is not None else random.Random()

    n, T, alpha = cfg.n, cfg.T, cfg.alpha
    # Draw density parameter D ~ U(0,1)
    D = rng.random()

    # Max edges
    E_intra_max = n * (n - 1) // 2
    E_cross_max = n * n - n  # exclude self-links

    # Edge counts (integer)
    E_intra = int(round(D * E_intra_max))
    E_inter_cross = int(round(D * E_cross_max))

    # --- Nodes ---
    nodes = [(f"X{i+1}", t) for t in range(T) for i in range(n)]

    # --- Step 1: initialize subgraphs ---
    G_intra_template: List[Tuple[int, int]] = []  # edges within a slice, use indices i<j
    G_inter_template: List[Tuple[int, int]] = []  # edges from slice t to t+1, use indices i->j with i!=j

    # --- Step 2: Sample intra-slice structure at t=0, then replicate ---
    # Candidate intra edges to ensure acyclicity inside a slice: i<j (topological order X1..Xn)
    intra_candidates = [(i, j) for i in range(n) for j in range(i + 1, n)]
    G_intra_template = rng.sample(intra_candidates, E_intra) if E_intra > 0 else []

    # Build G_intra for each slice
    G_intra: Dict[int, List[Tuple[Node, Node]]] = defaultdict(list)
    for t in range(T):
        for (i, j) in G_intra_template:
            G_intra[t].append(((f"X{i+1}", t), (f"X{j+1}", t)))

    # --- Step 3: Add inter-slice structure for 0->1, then replicate ---
    # Self-link edges
    self_links = [(i, i) for i in range(n)]
    # Cross edges pool (i != j)
    cross_pool = [(i, j) for i in range(n) for j in range(n) if i != j]
    cross_edges = rng.sample(cross_pool, E_inter_cross) if E_inter_cross > 0 else []
    # Template (for 0->1)
    G_inter_template = self_links + cross_edges

    # Build inter edges for all t->t+1
    G_inter: Dict[int, List[Tuple[Node, Node]]] = defaultdict(list)
    for t in range(T - 1):
        for (i, j) in G_inter_template:
            G_inter[t].append(((f"X{i+1}", t), (f"X{j+1}", t + 1)))

    # --- Step 4: Assign CPDs (binary variables) ---
    # Parents per node
    parents: Dict[Node, List[Node]] = defaultdict(list)
    # Accumulate edges to also compute parents
    edges: List[Tuple[Node, Node]] = []

    for t in range(T):
        for (u, v) in G_intra[t]:
            edges.append((u, v))
            parents[v].append(u)

    for t in range(T - 1):
        for (u, v) in G_inter[t]:
            edges.append((u, v))
            parents[v].append(u)

    # CPDs: prior for t=0 nodes; conditional tables for t>0 nodes
    CPDs: Dict[Node, Dict[Tuple[int, ...], float]] = {}

    def skew(p: float) -> float:
        # p' = (1 - alpha)*p + alpha*1[p>0.5]
        return (1 - alpha) * p + (alpha if p > 0.5 else 0.0)

    # t=0 priors
    for i in range(n):
        node = (f"X{i+1}", 0)
        p = rng.random()
        p_prime = skew(p)
        # Represent as a CPD with empty parent tuple key
        CPDs[node] = {tuple(): p_prime}

    # t>0 conditionals
    for t in range(1, T):
        for i in range(n):
            node = (f"X{i+1}", t)
            pa = parents.get(node, [])
            # Sort parents for stable keying
            pa_sorted = sorted(pa, key=lambda z: (z[1], z[0]))
            k = len(pa_sorted)
            table = {}
            # For each parent assignment (0/1)^k
            for bits in itertools.product([0, 1], repeat=k):
                p = rng.random()
                p_prime = skew(p)
                table[bits] = p_prime
            CPDs[node] = table

    # --- Step 5: Aggregate ---
    V = nodes
    E = edges

    # Densities (relative to max possible in the unrolled graph)
    # Intra total max (replicated T times): T * n*(n-1)/2
    intra_total = sum(len(G_intra[t]) for t in range(T))
    intra_max_total = T * (n * (n - 1) // 2)
    # Inter total max (per layer all n^2 including self-links): (T-1)*n^2
    inter_total = sum(len(G_inter[t]) for t in range(T - 1))
    inter_max_total = (T - 1) * (n * n)

    total_edges = intra_total + inter_total
    total_max = intra_max_total + inter_max_total
    overall_density = total_edges / total_max if total_max else 0.0

    summary = {
        "D": D,
        "E_intra": intra_total,
        "E_intra_max": intra_max_total,
        "E_inter_total": inter_total,
        "E_inter_max": inter_max_total,
        "total_edges": total_edges,
        "total_max": total_max,
        "overall_density": overall_density,
    }

    return {
        "V": V,
        "E": E,
        "G_intra": G_intra,
        "G_inter": G_inter,
        "parents": parents,
        "CPDs": CPDs,
        "summary": summary,
        "cfg": cfg,
    }


def visualize_dbn(dbn, filename = None):
    V, E, cfg = dbn["V"], dbn["E"], dbn["cfg"]
    G = nx.DiGraph()
    G.add_nodes_from(V)
    G.add_edges_from(E)

    # Layered positions
    pos = {}
    n, T = cfg.n, cfg.T
    vars_order = [f"X{i+1}" for i in range(n)]
    for t in range(T):
        for i, x in enumerate(vars_order):
            pos[(x, t)] = (t, (n - 1 - i))

    plt.figure(figsize=(20, 16))
    # nx.draw_networkx_nodes(G, pos, node_size=200)
    # nx.draw_networkx_labels(G, pos, labels={node: f"$X_{{{node[0][1:]}}}^{{{node[1]}}}$" for node in G.nodes()}, font_size=6)
    # nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=15)

    # EDGES FIRST (curved), behind nodes
    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=32,
        connectionstyle='arc3,rad=0.16',  # gentle curve so edges don’t visually cross nodes
        alpha=0.9,
        width= 2.1,
        min_source_margin=15,              # keep arrow start a bit away from node center
        min_target_margin=15              # keep arrowhead from entering the node center
    )

    # NODES on top (white fill masks lines behind)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=1500,
        node_color='lightblue',
        edgecolors='black',
        linewidths=2.5
    )

    # LABELS last
    # Generate labels ensuring correct format for node types
    labels = {}
    for node in G.nodes():
        if isinstance(node, tuple) and len(node) == 2:
            labels[node] = f"$X_{{{node[0][1:]}}}^{{{node[1]}}}$"
        else:
            labels[node] = str(node) # Handle potential non-tuple nodes


    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=15,
        font_color='darkblue',
        font_weight='bold',
    )

    if filename:
      plt.savefig(filename, bbox_inches = 'tight')


    plt.axis('off')
    plt.title("Random DBN (per your algorithm)")
    plt.show()


# ------------------------
# CONFIG block: tweak as desired
# ------------------------
import random

# pick n and T randomly between 10 and 20
n = random.randint(4, 15)
# n = 3
T = random.randint(4, 15)
# T = 5

alpha = random.uniform(0.5, 1)
cfg = Config(n=n, T=T, alpha=alpha)   # no seed => full randomness


dbn = generate_random_dbn(cfg)

# Print summary
print("=== CONFIG ===")
print(cfg)
print("\n=== RANDOM D (density parameter) ===")
print(f"D = {dbn['summary']['D']:.4f}")

print("\n=== EDGE COUNTS & DENSITIES ===")
for k, v in dbn["summary"].items():
    if k == "D":
        continue
    if "density" in k:
        print(f"{k}: {v:.3f}")
    else:
        print(f"{k}: {v}")

# Show first few intra and inter edges (readable)
print("\n=== SAMPLE INTRA EDGES (t=0) ===")
for e in dbn["G_intra"][0][:min(10, len(dbn["G_intra"][0]))]:
    print(f"{e[0][0]}^{e[0][1]} -> {e[1][0]}^{e[1][1]}")

print("\n=== SAMPLE INTER EDGES (0->1) ===")
for e in dbn["G_inter"][0][:min(10, len(dbn["G_inter"][0]))]:
    print(f"{e[0][0]}^{e[0][1]} -> {e[1][0]}^{e[1][1]}")


# # Show a small CPD sample to avoid huge output
# some_node = (f"X1", 1)
# print(f"\n=== CPD SAMPLE for {some_node[0]}^{some_node[1]} (show up to 8 rows) ===")
# table = dbn["CPDs"][some_node]
# for i, (k, p) in enumerate(table.items()):
#     if i >= 10:
#         break
#     print(f"Parents={k} -> P({some_node[0]}^{some_node[1]}=1 | parents) = {p:.3f}")

PRINT_CPDS = False

if PRINT_CPDS:
    print("\n=== CPDs for all nodes (up to 8 rows each) ===")
    for node, table in dbn["CPDs"].items():
        print(f"\nCPD for {node[0]}^{node[1]} (show up to 8 rows):")
        for i, (parent_vals, p) in enumerate(table.items()):
            if i >= 8:
                break
            print(f"Parents={parent_vals} -> P({node[0]}^{node[1]}=1 | parents) = {p:.3f}")


# Visualize
import matplotlib
matplotlib.use('Agg')
visualize_dbn(dbn, filename = "dbn.png")
print("Saved DBN to dbn.png")