import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
from randomDBN import generate_random_dbn, Config, visualize_dbn

# --- Particle Filter class (copy or import from your other file) ---
class ParticleFilterDBN:
    def __init__(self, dbn, num_particles=1000):
        self.dbn = dbn
        self.num_particles = num_particles
        self.V = dbn['V']
        self.node_list = sorted([(n[0], n[1]) for n in self.V], key=lambda x: (x[1], x[0]))
        self.node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        self.parents = self.dbn['parents']
        self.CPDs = self.dbn['CPDs']
        self.T = dbn['cfg'].T

    def initialize_particles(self):
        self.particles = np.zeros((self.num_particles, len(self.node_list)), dtype=int)
        for idx, (node, t) in enumerate(self.node_list):
            if t == 0:
                p1 = self.CPDs[(node, 0)].get((), 0.5)
                self.particles[:, idx] = np.random.binomial(1, p1, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def transition_particles(self, t):
        for idx, (node, time) in enumerate(self.node_list):
            if time == t:
                parents = self.parents.get((node, t), [])
                parent_idxs = [self.node_to_idx[p] for p in sorted(parents, key=lambda x: (x[1], x[0]))]
                if parent_idxs:
                    parent_vals = self.particles[:, parent_idxs]
                    keys = [tuple(row) for row in parent_vals]
                else:
                    keys = [()] * self.num_particles
                p1s = np.array([self.CPDs[(node, t)].get(k, 0.5) for k in keys])
                self.particles[:, idx] = np.random.binomial(1, p1s)

    def update_weights(self, evidence, parent_subset, query_node):
        self.weights[:] = 1.0
        conditioning_nodes = set(parent_subset)
        conditioning_nodes.add(query_node)
        for node, obs in evidence.items():
            if node in conditioning_nodes:
                idx = self.node_to_idx[node]
                matches = self.particles[:, idx] == obs
                self.weights *= np.where(matches, 0.9, 0.1)
        s = self.weights.sum()
        if s > 0:
            self.weights /= s
        else:
            self.weights[:] = 1.0 / self.num_particles

    def resample_particles(self):
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0
        rs = np.random.uniform(0, 1, self.num_particles)
        idxs = np.searchsorted(cumsum, rs)
        self.particles = self.particles[idxs]
        self.weights[:] = 1.0 / self.num_particles

    def run_filter(self, evidences, query_node, parent_subset):
        self.initialize_particles()
        query_idx = self.node_to_idx[query_node]
        t = query_node[1]
        for step in range(t + 1):
            evidence = evidences[step]
            if step > 0:
                self.transition_particles(step)
            self.update_weights(evidence, parent_subset, query_node)
            self.resample_particles()
        marginal = np.average(self.particles[:, query_idx], weights=self.weights)
        return marginal

# --- Exact brute-force inference ---
def exact_marginal(dbn, query_node, evidence, parent_subset):
    nodes = dbn['V']
    parents = dbn['parents']
    CPDs = dbn['CPDs']

    conditioning_set = set(evidence.keys())
    hidden_vars = [node for node in nodes if node not in conditioning_set and node != query_node]

    total_prob = 0.0
    query_one_prob = 0.0

    for q_val in [0, 1]:
        for assignment in itertools.product([0, 1], repeat=len(hidden_vars)):
            state = dict(zip(hidden_vars, assignment))
            state.update(evidence)
            state[query_node] = q_val

            joint_prob = 1.0
            for node in nodes:
                node_cpd = CPDs[node]
                node_parents = parents.get(node, [])
                if node == query_node:
                    node_parents = parent_subset
                sorted_parents = sorted(node_parents, key=lambda x: (x[1], x[0]))
                parent_state = tuple(state[p] for p in sorted_parents) if sorted_parents else ()
                p1 = node_cpd.get(parent_state, 0.5)
                x = state[node]
                joint_prob *= p1 if x == 1 else 1 - p1

            total_prob += joint_prob
            if q_val == 1:
                query_one_prob += joint_prob

    if total_prob == 0.0:
        return 0.0
    return query_one_prob / total_prob

# --- main driver ---
def main():
    random.seed(42)
    np.random.seed(42)

    # Use existing config and DBN generator
    # pick n and T randomly between 10 and 20
    n = random.randint(3, 10)
    # n = 3
    T = random.randint(3, 10)
    # T = 5

    alpha = random.uniform(0.5, 1)
    cfg = Config(n=n, T=T, alpha=alpha)  # no seed => full randomness
    dbn = generate_random_dbn(cfg)

    # Visualize and save DBN plot (optionally, call here or in randomDBN.py)
    visualize_dbn(dbn, filename='dbn.png')
    print("Saved DBN to dbn.png")

    all_nodes = list(set(dbn['V']))
    query_node = random.choice(all_nodes)
    query_label = f"{query_node[0]}^{query_node[1]}"

    def format_nodes(nodes):
        return [f"{node[0]}^{node[1]}" for node in nodes]

    print(f"Random query node: {query_label}")

    full_parents = dbn['parents'].get(query_node, [])
    if full_parents:
        subset_size = min(len(full_parents), random.randint(1, len(full_parents)))
        parent_subset = random.sample(full_parents, subset_size)
    else:
        parent_subset = []

    print(f"Parent subset used for conditioning: {format_nodes(parent_subset) if parent_subset else 'None'}")

    node_names = set(n[0] for n in dbn['V'])
    observed_name = random.choice(list(node_names))

    evidences = []
    for t in range(cfg.T):
        obs = {}
        for node in dbn['V']:
            if node[1] == t and node[0] == observed_name:
                obs[node] = random.choice([0, 1])
        evidences.append(obs)

    observed_nodes_formatted = format_nodes([node for t in range(cfg.T) for node in evidences[t]])
    evidence_str = ", ".join(observed_nodes_formatted)

    parent_str = ", ".join(format_nodes(parent_subset)) if parent_subset else None
    if parent_str:
        query_str = f"P({query_label} | [{parent_str}], evidence = {{{evidence_str}}})"
    else:
        query_str = f"P({query_label} | evidence = {{{evidence_str}}})"

    print(f"\nQuery:\n{query_str}\n")

    merged_evidence = {}
    for step in range(query_node[1] + 1):
        merged_evidence.update(evidences[step])
    restricted_evidence = {k: v for k, v in merged_evidence.items() if k in parent_subset or k == query_node}

    particle_counts = [10000, 50000, 100000, 500000, 1000000, 5000000]
    pf_marginals = {}

    for N in particle_counts:
        print(f"\nRunning PF with {N} particles")
        pf = ParticleFilterDBN(dbn, num_particles=N)
        restricted_evidences = []
        for ev in evidences:
            filtered_ev = {k: v for k, v in ev.items() if k in parent_subset or k == query_node or k[0] == observed_name}
            restricted_evidences.append(filtered_ev)
        marg = pf.run_filter(restricted_evidences, query_node, parent_subset)
        pf_marginals[N] = marg
        print(f"PF approx {query_str} with N={N} = {marg:.4f}")

    exact_value = exact_marginal(dbn, query_node, restricted_evidence, parent_subset)
    print(f"Exact {query_str} = {exact_value:.4f}")

    pf_vals = np.array([pf_marginals[N] for N in particle_counts])
    mse = (pf_vals - exact_value) ** 2

    plt.figure(figsize=(8, 6))
    plt.plot(particle_counts, mse, marker='o')
    plt.xscale('log')
    plt.xticks(particle_counts, [f"{int(x/1000)}k" for x in particle_counts])
    plt.xlabel("Number of particles (N)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title(f"MSE: {query_str}\nExact={exact_value:.3f}")
    plt.grid(True)
    plt.savefig("mse_plot.png", bbox_inches="tight")
    print("Plot saved as mse_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
