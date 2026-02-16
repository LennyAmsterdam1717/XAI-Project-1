from anytree import Node, PostOrderIter
import json


def build_tree_from_json(json_obj, parent=None):
    """Convert JSON tree to anytree Node structure."""
    node = Node(
        json_obj['name'],
        parent=parent,
        type=json_obj.get('type', 'ACT'),
        pre=json_obj.get('pre', []),
        post=json_obj.get('post', []),
        costs=json_obj.get('costs', []),
        link=json_obj.get('link', []),
        slink=json_obj.get('slink', []),
        sequence=json_obj.get('sequence', 0)
    )
    
    if 'children' in json_obj:
        for child_json in json_obj['children']:
            build_tree_from_json(child_json, parent=node)
    
    return node

# Assignment 3 - Generate Trace

def find_node_json(tree, target_name):
    if tree["name"] == target_name:
        return tree
    for child in tree.get("children", []):
        result = find_node_json(child, target_name)
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

    if norm_type in ("F", "prohibition", "P"):
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


# Assignment 4 - Explain Action

def explain_action(action_name: str,
                   root_node: Node,
                   trace: list[str],
                   beliefs: list[str],
                   norms: list[dict],
                   user_preferences: list) -> list[list]:
    """
    Compose the overall explanation for a given action.
    """
    if action_name not in trace:
        return []
    
    explanation = []
    
    current_beliefs = beliefs.copy()
    
    # Explain OR choices leading to this action
    or_explanations = explain_or_choices(action_name, root_node, trace, 
                                         current_beliefs, norms, user_preferences)
    explanation.extend(or_explanations)
    
    # 2. Explain preconditions for all actions in trace up to and including action_name
    current_beliefs = beliefs.copy()
    for node_name in trace:
        node = find_node(node_name, root_node)
        if node and node.type == "ACT":
            if hasattr(node, 'pre') and node.pre:
                satisfied = [p for p in node.pre if p in current_beliefs]
                if satisfied:
                    explanation.append(['P', node.name, satisfied])
            
            if hasattr(node, 'post') and node.post:
                current_beliefs.extend(node.post)
        
        if node_name == action_name:
            break
    
    # Explain links starting from the action
    link_explanations = explain_links(action_name, root_node)
    explanation.extend(link_explanations)
    
    #Explain goals (ancestors that are goal nodes)
    goal_explanations = explain_goals(action_name, root_node)
    explanation.extend(goal_explanations)
    
    #User preference
    explanation.append(['U', user_preferences])
    
    return explanation


def compare_costs_by_preference(costs1, costs2, user_preferences):
    """
    Compare two cost vectors according to lexicographic preference ordering.
    
    Returns:
    - 'less': costs1 is preferred (lower cost)
    - 'greater': costs2 is preferred (lower cost)  
    - 'equal': costs are identical
    """
    if not user_preferences or len(user_preferences) < 2:
        return 'equal'
    
    value_names, preference_order = user_preferences
    
    for priority_position in range(len(preference_order)):
        value_index = preference_order.index(priority_position)
        
        cost1 = costs1[value_index] if value_index < len(costs1) else 0.0
        cost2 = costs2[value_index] if value_index < len(costs2) else 0.0
        
        if cost1 < cost2:
            return 'less' 
        elif cost1 > cost2:
            return 'greater'  
        
    return 'equal'


def find_node(name, root_node):
    """Find a node by name in the tree using post-order traversal."""
    for node in PostOrderIter(root_node):
        if node.name == name:
            return node
    return None


def explain_or_choices(action_name, root_node, trace, beliefs, norms, user_preferences):
    """
    Compute explanation factors for OR nodes that lead to the given action.
    """
    explanations = []
    
    action_node = find_node(action_name, root_node)
    if action_node is None:
        return explanations
    try:
        action_index = trace.index(action_name)
    except ValueError:
        action_index = len(trace)
    
    or_nodes_from_trace = []
    for node_name in trace[:action_index]:
        node = find_node(node_name, root_node)
        if node and node.type == 'OR':
            or_nodes_from_trace.append(node)
    
    or_ancestors = [anc for anc in action_node.ancestors if anc.type == 'OR']
    
    all_or_nodes = []
    seen = set()
    
    for node in or_nodes_from_trace:
        if node.name not in seen:
            all_or_nodes.append(node)
            seen.add(node.name)
    
    for node in or_ancestors:
        if node.name not in seen:
            all_or_nodes.append(node)
            seen.add(node.name)
    
    for or_node in all_or_nodes:
        chosen_child = None
        for child in or_node.children:
            if child.name in trace:
                chosen_child = child
                break
        
        if chosen_child is None:
            continue
        
        satisfied_pre = []
        if hasattr(chosen_child, 'pre') and chosen_child.pre:
            satisfied_pre = [p for p in chosen_child.pre if p in beliefs]
        explanations.append(['C', chosen_child.name, satisfied_pre])
    
        for sibling in or_node.children:
            if sibling == chosen_child:
                continue
            
            norm_violation = check_norm_violation_in_subtree(sibling, norms, trace)
            if norm_violation:
                explanations.append(['N', sibling.name, norm_violation])
                continue
            
            if hasattr(sibling, 'pre') and sibling.pre:
                failed = [p for p in sibling.pre if p not in beliefs]
                if failed:
                    explanations.append(['F', sibling.name, failed])
                    continue
            
            chosen_costs = get_trace_costs(chosen_child, root_node, trace)
            sibling_costs = get_trace_costs(sibling, root_node, trace)
                
            comparison = compare_costs_by_preference(chosen_costs, sibling_costs, user_preferences)
            
            explanations.append(['V', chosen_child.name, chosen_costs, 
                                '>', sibling.name, sibling_costs])
    
    return explanations


def check_norm_violation_in_subtree(node, norms, trace):
    """
    Check if a node (alternative) violates a norm.
    Returns the norm string if violation found, None otherwise.
    """
    for descendant in PostOrderIter(node):
        if descendant.type == 'ACT':
            for norm in norms:
                if norm['type'] == 'P':
                    if descendant.name in norm.get('actions', []):
                        return f"P({descendant.name})"

    if node.type in ['OR', 'AND', 'SEQ']:
        for norm in norms:
            if norm['type'] == 'O':
                obligated_actions = norm.get('actions', [])
                
                actions_in_subtree = {
                    descendant.name 
                    for descendant in PostOrderIter(node) 
                    if descendant.type == 'ACT'
                }
                
                has_obligated_action = any(
                    act in actions_in_subtree 
                    for act in obligated_actions
                )
                
                if not has_obligated_action:
                    return f"O({', '.join(obligated_actions)})"
    
    return None


def get_trace_costs(node, root_node, trace):
    """
    Get the total costs for a path starting from a given node.
    This calculates what the costs WOULD BE if this path were chosen.
    """
    total_costs = None
    
    if node.type == 'ACT' and hasattr(node, 'costs'):
        return node.costs[:]
    
    if node.type in ['SEQ', 'AND']:
        for child in node.children:
            child_costs = get_trace_costs(child, root_node, trace)
            if child_costs:
                if total_costs is None:
                    total_costs = [0.0] * len(child_costs)
                for i, cost in enumerate(child_costs):
                    total_costs[i] += cost
    
    elif node.type == 'OR':
        chosen_child = None
        for child in node.children:
            if child.name in trace:
                chosen_child = child
                break
        
        if chosen_child:
            child_costs = get_trace_costs(chosen_child, root_node, trace)
            if child_costs:
                total_costs = child_costs
        else:
            for child in node.children:
                child_costs = get_trace_costs(child, root_node, trace)
                if child_costs:
                    total_costs = child_costs
                    break 
    
    return total_costs if total_costs else [0.0, 0.0, 0.0]


def explain_goals(action_name, root_node):
    """
    Compute goal contribution factors (D factors) for the action.
    Add a D factor for each ancestor that is a goal node.
    """
    explanations = []
    
    node = find_node(action_name, root_node)
    if node is None:
        return explanations
    
    #to ensure closest first
    ancestors_list = list(node.ancestors)
    ancestors_list.reverse() 
    
    for ancestor in ancestors_list:
        if ancestor.type in ['OR', 'AND', 'SEQ']:
            explanations.append(['D', ancestor.name])
    
    return explanations


def explain_links(action_name, root_node):
    """
    Compute linked action factors (L factors).
    Follow the chain of links starting from the action.
    """
    explanations = []
    
    node = find_node(action_name, root_node)
    if node is None or not hasattr(node, 'link') or not node.link:
        return explanations
    
    current_node = node
    visited = set() 
    
    while current_node and hasattr(current_node, 'link') and current_node.link:
        if current_node.name in visited:
            break
        visited.add(current_node.name)
        
        for linked_name in current_node.link:
            explanations.append(['L', current_node.name, '->', linked_name])
            current_node = find_node(linked_name, root_node)
            if current_node:
                break  
        else:
            break  
    
    return explanations


all_traces = get_traces(json_tree, list(beliefs))


valid_traces = [t for t in all_traces if not check_norm_violation(t, norm, beliefs)]

best_trace = None
best_cost = float('inf')

for trace in valid_traces:
    cost = compute_weighted_cost(trace, json_tree, preferences)
    if cost < best_cost:
        best_cost = cost
        best_trace = trace

selected_trace = best_trace if best_trace else []

root_node = build_tree_from_json(json_tree)
output = explain_action(action_to_explain, root_node, selected_trace, beliefs, [norm], preferences)
