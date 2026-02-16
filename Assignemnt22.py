import json
from anytree import RenderTree
from anytree.importer import DictImporter

def solve(tree_dict, norm):
    importer = DictImporter()
    root = importer.import_(tree_dict)
    
    norm_type = norm.get("type")
    norm_actions = norm.get("actions", [])
    
    def annotate_violations(node):
        if node.type == "ACT":
            if norm_type == "P":
                node.violation = node.name in norm_actions
            else:
                node.violation = node.name not in norm_actions
        else:
            for child in node.children:
                annotate_violations(child)
            
            if norm_type == "P":
                if node.type == "OR":
                    node.violation = all(child.violation for child in node.children)
                else:
                    node.violation = any(child.violation for child in node.children)
            else:
                if node.type == "OR":
                    node.violation = all(child.violation for child in node.children)
                else:
                    node.violation = any(child.violation for child in node.children)
    
    annotate_violations(root)
    return str(RenderTree(root))

output = solve(json_tree, norm)
