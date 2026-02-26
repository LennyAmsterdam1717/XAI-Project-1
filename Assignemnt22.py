import json
from anytree import RenderTree
from anytree.importer import DictImporter

def solve(tree_dict, norm):
    importer = DictImporter()
    root = importer.import_(tree_dict)
    
    norm_type = norm.get("type")
    norm_actions = norm.get("actions", [])
    
    def has_required_action(node):
        """Check if subtree contains at least one required action."""
        if node.type == "ACT":
            return node.name in norm_actions
        return any(has_required_action(child) for child in node.children)
    
    def annotate_violations(node):
        if node.type == "ACT":
            if norm_type == "P":
                node.violation = node.name in norm_actions
            else:  # Obligation
                node.violation = node.name not in norm_actions
        else:
            # First annotate children
            for child in node.children:
                annotate_violations(child)
            
            if norm_type == "P":
                # Prohibition: violates if ANY child violates
                if node.type == "OR":
                    node.violation = all(child.violation for child in node.children)
                else:  # SEQ/AND
                    node.violation = any(child.violation for child in node.children)
            else:  # Obligation
                # For obligation, check if subtree has at least one required action
                if node.type == "OR":
                    # OR: violates if ALL children violate
                    node.violation = all(child.violation for child in node.children)
                else:  # SEQ/AND
                    # SEQ/AND: violates if it doesn't contain ANY required action
                    node.violation = not has_required_action(node)
    
    annotate_violations(root)
    return str(RenderTree(root))

output = solve(json_tree, norm)
