from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

def dbn_to_pgmpy(dbn):
    # Create edges for all time slices as per dbn['parents']
    edges = []
    for child, parents in dbn['parents'].items():
        for p in parents:
            edges.append((p, child))

    model = BayesianModel(edges)

    # For each variable and time slice, add TabularCPDs
    cpds = []
    for node in dbn['V']:
        cpd_dict = dbn['CPDs'][node]
        # Extract parent states and probabilities
        # Build CPT as tabular numpy array in pgmpy format
        parents = sorted(dbn['parents'].get(node, []))
        n_parent_states = 2 ** len(parents)
        values = [0]* (2 * n_parent_states)  # binary in/out per parent config

        # Map parent parent state tuples to index, fill CPD values
        for i in range(n_parent_states):
            parent_state = tuple(((i >> j) & 1) for j in reversed(range(len(parents))))
            p1 = cpd_dict.get(parent_state, 0.5)
            values[i*2] = 1 - p1  # P(X=0 | parents)
            values[i*2+1] = p1    # P(X=1 | parents)

        # reshape to 2 x n_parent_states matrix as pgmpy expects
        cpd_values = [values[0::2], values[1::2]]
        var_name = f"{node[0]}_{node[1]}"
        parent_names = [f"{p[0]}_{p[1]}" for p in parents]

        cpd = TabularCPD(variable=var_name,
                         variable_card=2,
                         values=cpd_values,
                         evidence=parent_names if parent_names else None,
                         evidence_card=[2]*len(parent_names) if parent_names else None)
        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()
    return model

def exact_inference_pgmpy(dbn, query_node, evidence):
    model = dbn_to_pgmpy(dbn)

    infer = VariableElimination(model)

    query_var = f"{query_node[0]}_{query_node[1]}"

    # Convert evidence dict keys to pgmpy naming
    pgmpy_evidence = {f"{node[0]}_{node[1]}": val for node, val in evidence.items()}

    # Query marginal
    q = infer.query(variables=[query_var], evidence=pgmpy_evidence)
    return q.values[1]  # Probability X=1
