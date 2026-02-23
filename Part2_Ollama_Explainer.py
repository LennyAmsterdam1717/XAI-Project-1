"""
Natural Language Explanation Generator using Ollama
Part 2 - XAI Project

This script generates human-readable explanations for AI agent decisions
using Ollama LLM for natural language generation.
"""

import json
import re

class ExplanationGenerator:
    def __init__(self, json_tree, beliefs, norm, preferences):
        self.json_tree = json_tree
        self.beliefs = beliefs
        self.norm = norm
        self.preferences = preferences
    
    def find_node(self, tree, target_name):
        if tree["name"] == target_name:
            return tree
        for child in tree.get("children", []):
            result = self.find_node(child, target_name)
            if result:
                return result
        return None
    
    def get_traces(self, node, beliefs):
        node_type = node["type"]
        name = node["name"]
        pre = node.get("pre", [])
        
        if not all(p in beliefs for p in pre):
            return []
        
        if node_type == "ACT":
            return [[name]]
        
        if node_type in ("SEQ", "AND"):
            traces = [[]]
            current_beliefs = list(beliefs)
            children = sorted(node.get("children", []), key=lambda c: c.get("sequence", 0))
            
            for child in children:
                child_traces = self.get_traces(child, current_beliefs)
                if not child_traces:
                    return []
                new_traces = []
                for existing_path in traces:
                    for child_path in child_traces:
                        new_traces.append(existing_path + child_path)
                traces = new_traces
                self.update_beliefs(child, current_beliefs)
            
            return [[name] + path for path in traces]
        
        if node_type == "OR":
            all_traces = []
            for child in node["children"]:
                child_traces = self.get_traces(child, beliefs)
                for child_path in child_traces:
                    all_traces.append([name] + child_path)
            return all_traces
        
        return []
    
    def update_beliefs(self, node, beliefs):
        if node.get("type") == "ACT":
            for p in node.get("post", []):
                if p not in beliefs:
                    beliefs.append(p)
        for child in node.get("children", []):
            self.update_beliefs(child, beliefs)
    
    def check_norm_violation(self, trace):
        if not self.norm:
            return False
        
        norm_type = self.norm.get("type", "")
        actions = self.norm.get("actions", [])
        trace_set = set(trace)
        
        if norm_type in ("O", "obligation"):
            return not any(action in trace_set for action in actions)
        
        if norm_type in ("P", "F", "prohibition"):
            return any(action in trace_set for action in actions)
        
        return False
    
    def get_act_costs(self, tree, trace_names):
        costs = {}
        if tree.get("name") in trace_names and tree.get("type") == "ACT":
            costs[tree["name"]] = tree.get("costs", [0, 0, 0])
        for child in tree.get("children", []):
            costs.update(self.get_act_costs(child, trace_names))
        return costs
    
    def compute_weighted_cost(self, trace):
        dims = self.preferences[0]
        weights = self.preferences[1]
        act_costs = self.get_act_costs(self.json_tree, set(trace))
        
        total = 0
        for dim_idx in range(len(dims)):
            dim_sum = sum(c[dim_idx] for c in act_costs.values())
            total += dim_sum * weights[dim_idx]
        return total
    
    def humanize(self, name):
        """Convert camelCase to readable text."""
        result = re.sub('([A-Z])', r' \1', name).lower().strip()
        return result
    
    def get_actions_from_trace(self, trace):
        """Extract only action nodes from trace."""
        actions = []
        for name in trace:
            node = self.find_node(self.json_tree, name)
            if node and node.get("type") == "ACT":
                actions.append(name)
        return actions
    
    def generate_with_ollama(self):
        """Generate explanation using Ollama LLM."""
        try:
            import ollama
        except ImportError:
            return "Error: Please install ollama package: pip install ollama"
        
        # Get all feasible traces
        all_traces = self.get_traces(self.json_tree, list(self.beliefs))
        
        if not all_traces:
            return "I couldn't find any way to achieve the goal."
        
        # Filter norm-violating traces
        valid_traces = [t for t in all_traces if not self.check_norm_violation(t)]
        
        if not valid_traces:
            return "I couldn't find a way to achieve the goal without violating the rules."
        
        # Find best trace and alternative
        trace_costs = [(t, self.compute_weighted_cost(t)) for t in valid_traces]
        trace_costs.sort(key=lambda x: x[1])
        best_trace, best_cost = trace_costs[0]
        
        # Extract information for prompt
        chosen_actions = [self.humanize(a) for a in self.get_actions_from_trace(best_trace)]
        
        alternative_actions = []
        if len(trace_costs) > 1:
            alt_trace, alt_cost = trace_costs[1]
            alternative_actions = [self.humanize(a) for a in self.get_actions_from_trace(alt_trace)]
        
        # Get preferences
        dims = self.preferences[0]
        weights = self.preferences[1]
        important_prefs = [dims[i] for i in range(len(dims)) if weights[i] > 0]
        
        # Get norm info
        norm_info = ""
        if self.norm:
            norm_type = self.norm.get("type")
            norm_actions = [self.humanize(a) for a in self.norm.get("actions", [])]
            if norm_type in ("P", "F"):
                norm_info = f"I must avoid: {', '.join(norm_actions)}"
            elif norm_type == "O":
                norm_info = f"I must do: {', '.join(norm_actions)}"
        
        # Create prompt for LLM
        prompt = f"""You are an AI assistant explaining your decision to a non-technical user.

Your chosen actions: {', then '.join(chosen_actions)}
Alternative option: {', then '.join(alternative_actions) if alternative_actions else 'none'}
Your priorities: {', '.join(important_prefs) if important_prefs else 'none'}
Rules: {norm_info if norm_info else 'none'}

Write a brief, friendly explanation (2-3 sentences) of why you chose these actions.
Use simple language, no technical jargon. Be conversational and helpful."""

        # Call Ollama
        try:
            print("Generating explanation with Ollama (this may take a few seconds)...")
            response = ollama.chat(model='llama3', messages=[
                {'role': 'user', 'content': prompt}
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error calling Ollama: {str(e)}\nMake sure Ollama is running and llama3 model is installed."

def generate_explanation(json_tree, beliefs, norm, preferences):
    """Main function to generate explanation."""
    generator = ExplanationGenerator(json_tree, beliefs, norm, preferences)
    return generator.generate_with_ollama()

if __name__ == "__main__":
    # Test with coffee scenario
    json_tree = {
        "name": "getCoffee",
        "type": "OR",
        "children": [
            {"name": "getKitchenCoffee", "type": "SEQ", "pre": ["staffCardAvailable"],
             "children": [
                 {"name": "gotoKitchen", "type": "ACT", "sequence": 1, "post": ["atKitchen"], "costs": [0, 0, 2]},
                 {"name": "getCoffeeKitchen", "type": "ACT", "sequence": 2, "pre": ["atKitchen"], "post": ["haveCoffee"], "costs": [5, 0, 1]}
             ]},
            {"name": "getShopCoffee", "type": "SEQ", "pre": ["haveMoney"],
             "children": [
                 {"name": "gotoShop", "type": "ACT", "sequence": 1, "post": ["atShop"], "costs": [0, 0, 5]},
                 {"name": "payShop", "type": "ACT", "sequence": 2, "pre": ["haveMoney"], "post": ["paidShop"], "costs": [0, 3, 1]},
                 {"name": "getCoffeeShop", "type": "ACT", "sequence": 3, "pre": ["atShop", "paidShop"], "post": ["haveCoffee"], "costs": [0, 0, 3]}
             ]}
        ]
    }
    
    print("="*70)
    print("OLLAMA-BASED NATURAL LANGUAGE EXPLANATION GENERATOR")
    print("="*70)
    
    # Scenario 1
    print("\nSCENARIO 1: Prohibition of paying at shop")
    print("-"*70)
    beliefs = ["staffCardAvailable", "haveMoney"]
    norm = {"type": "P", "actions": ["payShop"]}
    preferences = [["quality", "price", "time"], [1, 2, 0]]
    
    explanation = generate_explanation(json_tree, beliefs, norm, preferences)
    print(f"\nExplanation:\n{explanation}\n")
    
    # Scenario 2
    print("="*70)
    print("SCENARIO 2: No norms, prioritize time")
    print("-"*70)
    beliefs = ["staffCardAvailable", "haveMoney"]
    norm = None
    preferences = [["quality", "price", "time"], [0, 0, 1]]
    
    explanation = generate_explanation(json_tree, beliefs, norm, preferences)
    print(f"\nExplanation:\n{explanation}\n")
    print("="*70)
