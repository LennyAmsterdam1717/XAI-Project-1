import json

def find_node(tree, target_name):
    if tree["name"] == target_name:
        return tree
    for child in tree.get("children", []):
        result = find_node(child, target_name)
        if result is not None:
            return result
    return None

def get_traces(node, beliefs):
    """Get all feasible traces, filtering by preconditions and updating beliefs."""
    node_type = node["type"]
    name = node["name"]
    pre = node.get("pre", [])

    # Check preconditions against current beliefs
    if not all(p in beliefs for p in pre):
        return []

    if node_type == "ACT":
        return [[name]]

    if node_type in ("SEQ", "AND"):
        traces = [[]]
        current_beliefs = list(beliefs)

        # Sort children by sequence number if present
        children = sorted(node.get("children", []), key=lambda c: c.get("sequence", 0))

        for child in children:
            child_traces = get_traces(child, current_beliefs)
            if not child_traces:
                return []  # If any step in a sequence is infeasible, the whole SEQ fails
            new_traces = []
            for existing_path in traces:
                for child_path in child_traces:
                    new_traces.append(existing_path + child_path)
            traces = new_traces

            # Update beliefs with postconditions from this child's actions
            update_beliefs(child, current_beliefs)

        return [[name] + path for path in traces]

    if node_type == "OR":
        all_traces = []
        for child in node["children"]:
            child_traces = get_traces(child, beliefs)
            for child_path in child_traces:
                all_traces.append([name] + child_path)
        return all_traces

    return []


def update_beliefs(node, beliefs):
    """Add postconditions of all ACT nodes in this subtree to beliefs."""
    if node.get("type") == "ACT":
        for p in node.get("post", []):
            if p not in beliefs:
                beliefs.append(p)
    for child in node.get("children", []):
        update_beliefs(child, beliefs)


def check_norm_violation(trace, norm, beliefs):
    """Returns True if the trace violates the norm."""
    if not norm:
        return False

    norm_type = norm.get("type", "")
    
    # Get the action(s)
    actions = norm.get("actions", [])
    if not actions and "action" in norm:
        actions = [norm["action"]]
    
    condition = norm.get("condition", [])
    trace_set = set(trace)

    if norm_type in ("O", "obligation"):
        # Obligation: ALL listed actions must appear in the trace
        if condition:
            condition_met = all(c in trace_set for c in condition)
            if not condition_met:
                return False
        for action in actions:
            if action not in trace_set:
                return True  # Missing obligated action = violation
        return False

    if norm_type in ("F", "prohibition"):
        # Forbidden: NONE of the listed actions may appear in the trace
        if condition:
            condition_met = all(c in trace_set for c in condition)
            if not condition_met:
                return False
        for action in actions:
            if action in trace_set:
                return True  # Forbidden action present = violation
        return False

    return False



def get_act_costs(tree, trace_names):
    """Collect costs for all ACT nodes whose names appear in the trace."""
    costs = {}
    if tree.get("name") in trace_names and tree.get("type") == "ACT":
        costs[tree["name"]] = tree.get("costs", [0, 0, 0])
    for child in tree.get("children", []):
        costs.update(get_act_costs(child, trace_names))
    return costs


def compute_weighted_cost(trace, json_tree, preferences):
    """Compute total weighted cost of a trace based on user preferences."""
    dims = preferences[0]      # ['quality', 'price', 'time']
    weights = preferences[1]   # [1, 2, 0]

    act_costs = get_act_costs(json_tree, set(trace))

    total = 0
    for dim_idx in range(len(dims)):
        dim_sum = sum(c[dim_idx] for c in act_costs.values())
        total += dim_sum * weights[dim_idx]
    return total

# Get all feasible traces (precondition-filtered)
all_traces = get_traces(json_tree, list(beliefs))

# Filter out traces that violate the norm
valid_traces = [t for t in all_traces if not check_norm_violation(t, norm, beliefs)]

# Pick the trace with the lowest weighted cost
best_trace = None
best_cost = float('inf')

for trace in valid_traces:
    cost = compute_weighted_cost(trace, json_tree, preferences)
    if cost < best_cost:
        best_cost = cost
        best_trace = trace

output = best_trace if best_trace else []