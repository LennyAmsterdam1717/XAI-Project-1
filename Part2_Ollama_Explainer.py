"""
Natural Language Explanation Generator using Ollama
Part 2 - XAI Project

This script generates human-readable explanations for AI agent decisions
using Ollama LLM for natural language generation.
"""

import json
import re
import random
from Assignment_4_prairielearn import build_tree_from_json, explain_action

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
        
        # Find best trace and cost (same optimisation as in Assignment 4)
        trace_costs = [(t, self.compute_weighted_cost(t)) for t in valid_traces]
        trace_costs.sort(key=lambda x: x[1])
        best_trace, best_cost = trace_costs[0]
        
        # === Use Assignment 4 algorithm to build a FORMAL explanation ===
        # Build anytree representation of the plan tree
        root_node = build_tree_from_json(self.json_tree)

        # Choose the concrete action to explain: last ACT in the best trace
        best_actions = self.get_actions_from_trace(best_trace)
        action_to_explain = best_actions[-1] if best_actions else best_trace[-1]

        # Norms are expected as a list in Assignment 4's explain_action
        norms_list = [self.norm] if self.norm else []

        formal_explanation = explain_action(
            action_to_explain,
            root_node,
            best_trace,
            list(self.beliefs),
            norms_list,
            self.preferences,
        )

        # Serialise the formal explanation structure
        formal_explanation_text = json.dumps(formal_explanation, indent=2)

        # Create prompt for LLM: translate formal explanation into very friendly first-person English
        prompt = f"""You are helping me explain my own decision in a simple, first-person way.
The text below is a formal description of how I chose what to do.
The person reading the explanation does not code and has never seen this assignment before.

You are given a FORMAL EXPLANATION of why I chose a particular sequence of actions.

The formal explanation is represented as a list of factors with codes:
- 'C': choice at an OR node (which branch was chosen and which preconditions held)
- 'F': failed preconditions for an alternative
- 'N': norms (obligations or prohibitions) that ruled out alternatives
- 'V': value and cost-based comparison between alternatives
- 'P': preconditions that had to hold for actions in the trace
- 'D': goals (ancestor nodes) that this action helps to achieve
- 'L': links between actions
- 'U': the user's preferences over quality, price, and time

Formal explanation (JSON-like list of factors):
{formal_explanation_text}

Task:
- Write a SHORT explanation in plain, everyday English (about 2–3 sentences).
- Use **first person**: always say "I decided..." or "I chose..." 
- Sound positive, friendly, and enthusiastic, as if you are helping a colleague understand what happened.
- Avoid technical terms (no words like 'trace', 'norm', 'cost function', or 'constraint').
- Clearly say, in simple words, what I decided to do, what other options I did NOT take, and why (because of rules and my priorities).
- End with one clear short concluding sentence that directly states why this choice was the best one for me in this situation.
- Do NOT mention the single‑letter codes ('C', 'F', 'N', 'V', 'P', 'D', 'L', 'U') or the word 'factor' at all.

Write your answer as if you are telling a short story to someone who knows nothing about programming or AI planning, but just wants to understand my decision."""
        
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
    # Load coffee planning problem from JSON file
    with open("coffee.json", "r") as f:
        json_tree = json.load(f)

    # Define possible belief atoms present in the coffee domain
    possible_beliefs = [
        "staffCardAvailable",
        "ownCard",
        "colleagueAvailable",
        "AnnInOffice",
        "haveMoney",
    ]

    # Candidate actions from the coffee domain for norms
    candidate_actions = [
        "payShop",
        "getOwnCard",
        "getOthersCard",
        "getCoffeeKitchen",
        "getCoffeeAnnOffice",
        "getCoffeeShop",
    ]

    # Re-sample random scenarios until we find one that has at least
    # one plan that achieves the goal *without* violating the norm.
    max_attempts = 50
    valid_scenario = None

    for _ in range(max_attempts):
        # Randomly sample initial beliefs by independently including each atom
        beliefs_candidate = [b for b in possible_beliefs if random.choice([True, False])]
        if not beliefs_candidate:
            beliefs_candidate = [random.choice(possible_beliefs)]

        # Randomly choose whether to have a norm and of what type
        norm_types = [None, "P", "O"]  # prohibition or obligation or no norm
        chosen_norm_type = random.choice(norm_types)
        norm_candidate = None
        if chosen_norm_type:
            norm_actions = random.sample(candidate_actions, k=1)
            norm_candidate = {"type": chosen_norm_type, "actions": norm_actions}

        # Randomly generate user preferences over [quality, price, time]
        value_names = ["quality", "price", "time"]
        raw_weights = [random.randint(0, 3) for _ in value_names]
        preferences_candidate = [value_names, raw_weights]

        # Use the ExplanationGenerator's own logic to check feasibility
        generator = ExplanationGenerator(json_tree, beliefs_candidate, norm_candidate, preferences_candidate)
        all_traces = generator.get_traces(json_tree, list(beliefs_candidate))
        valid_traces = [t for t in all_traces if not generator.check_norm_violation(t)]

        if valid_traces:
            valid_scenario = (beliefs_candidate, norm_candidate, preferences_candidate)
            break

    if valid_scenario is None:
        print("Could not generate a scenario that satisfies the rules within the attempt limit.")
    else:
        beliefs, norm, preferences = valid_scenario

        print("=" * 70)
        print("OLLAMA-BASED NATURAL LANGUAGE EXPLANATION GENERATOR")
        print("=" * 70)
        print("\nRANDOMLY GENERATED SCENARIO (RULE-COMPLIANT)")
        print("-" * 70)
        print(f"Beliefs: {beliefs}")
        print(f"Norm: {norm}")
        print(f"Preferences (values, weights): {preferences}")

        explanation = generate_explanation(json_tree, beliefs, norm, preferences)
        print(f"\nNatural-language explanation:\n{explanation}\n")