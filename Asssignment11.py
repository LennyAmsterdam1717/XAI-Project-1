import json
def find_node(tree, target_name):
    if tree["name"] == target_name:
        return tree
  
    for child in tree.get("children", []):
        result=find_node(child, target_name)
        if result is not None:
            return result
    return None

def get_traces(node):
    node_type= node["type"]
    name =node["name"]
    if node_type == "ACT":
        return [[name]]

    if node_type in ("SEQ", "AND"):
        # Starting of with empty path
        traces = [[]]

        # For each child,take every existing path and extend it with 
        # all possible ways the child could execute
        for child in node["children"]:
            child_traces =get_traces(child)
            new_traces = []
            for existing_path in traces:
                for child_path in child_traces:
                    new_traces.append(existing_path + child_path)
            traces = new_traces

        return [[name] + path for path in traces]

    if node_type== "OR":
        all_traces= []
        
        for child in node["children"]:
            child_traces= get_traces(child)
            # Each child's path becomes a separate possibility
            for child_path in child_traces:
                all_traces.append([name] + child_path)
        
        return all_traces

# PrairieLearn provides json_tree and starting_node_name; do not open files
start = find_node(json_tree, starting_node_name)

output = get_traces(start)
