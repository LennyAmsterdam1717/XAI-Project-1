from pathlib import Path
import json

def load_input_file(path):
    ns = {}
    exec(Path(path).read_text(), ns)
    return {
        "name_json_tree_file": ns["name_json_tree_file"],
        "norm": ns["norm"],
        "beliefs": ns["beliefs"],
        "goal": ns["goal"],
        "preferences": ns["preferences"],
        "action_to_explain": ns["action_to_explain"],
    }

def run_assignment4(explain_path, input_vars):
    g = {
        "json_tree": json.loads(Path(input_vars["name_json_tree_file"]).read_text()),
        "norm": input_vars["norm"],
        "goal": input_vars["goal"],
        "beliefs": list(input_vars["beliefs"]),
        "preferences": input_vars["preferences"],
        "action_to_explain": input_vars["action_to_explain"],
    }
    exec(Path(explain_path).read_text(), g)
    return g.get("selected_trace", []), g.get("output", [])

def build_nl_payload(action_to_explain, selected_trace, formal):
    """
    Convert the technical explantions into a consive summary for the llm. 
    Only provide the action to explain, one chosen branch, 
    one goal context and exactly one rejected alternative and reasoning using priority.
    """
    payload = {
        "action": action_to_explain,
        "goal_context": None,
        "chosen": None,
        "alternative": None,
        "reason_type": None,   
        "reason_text": None,
        "preference_order": None
    }


    for f in formal:
        if f and f[0] == "C" and len(f) >= 2:
            payload["chosen"] = f[1]
            break

    # Goal context
    d_factors = [f for f in formal if f and f[0] == "D" and len(f) >= 2]
    if d_factors:
        payload["goal_context"] = d_factors[0][1]

    # Pick one rejected alternative with priority N > F > V
    n_factors = [f for f in formal if f and f[0] == "N"]
    f_factors = [f for f in formal if f and f[0] == "F"]
    v_factors = [f for f in formal if f and f[0] == "V"]

    if n_factors:
        x = n_factors[0]
        payload["alternative"] = x[1]
        payload["reason_type"] = "N"
        payload["reason_text"] = x[2] if len(x) > 2 else "rule conflict"
    elif f_factors:
        x = f_factors[0]
        payload["alternative"] = x[1]
        payload["reason_type"] = "F"
        payload["reason_text"] = x[2] if len(x) > 2 else []
    elif v_factors:
        x = v_factors[0]
        payload["alternative"] = x[4]
        payload["reason_type"] = "V"
        payload["reason_text"] = {"chosen_cost": x[2], "alt_cost": x[5]}

    for f in formal:
        if f and f[0] == "U" and len(f) >= 2:
            names, order = f[1]
            payload["preference_order"] = [names[i] for i in order]
            break

    return payload

def nl_from_payload(payload):
    import ollama, json

    prompt = f"""
You are writing a short, faithful explanation for a non-technical person.

You MUST use only the facts in this payload:
{json.dumps(payload, indent=2)}

Interpretation rules:
- "action" is the action being explained.
- "chosen" is the selected branch/option.
- "goal_context" is the nearby goal this supports.
- "alternative" is one option not chosen.
- "reason_type":
  - "N" => the alternative was blocked by a rule.
  - "F" => the alternative was not possible because required conditions were missing.
  - "V" => the alternative was less preferred based on my priorities.
- "reason_text" contains the exact evidence for that reason.
- "preference_order" lists what I care about from most important to least important.

The formal explanation is represented as a list of factors with codes:
- 'C': choice at an OR node (which branch was chosen and which preconditions held)
- 'F': failed preconditions for an alternative
- 'N': norms (obligations or prohibitions) that ruled out alternatives
- 'V': value and cost-based comparison between alternatives
- 'P': preconditions that had to hold for actions in the trace
- 'D': goals (ancestor nodes) that this action helps to achieve
- 'L': links between actions
- 'U': the user's preferences over quality, price, and time

Output requirements:
- Write EXACTLY 3 sentences.
- Use first person ("I chose...", "I did not choose...").
- Sentence 1: what I chose and what goal it supported.
- Sentence 2: one rejected alternative and the exact reason from payload.
- Sentence 3: my preference order in plain English and how it influenced my choice.
- Keep language simple and conversational.
- Do not mention technical labels, code letters, JSON, or internal structures (like (P(payShop))).
- Do not add facts that are not in the payload.
- No heading, no bullet points, no preface, no closing note.

Return only the 3 sentences.
"""
    resp = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]

if __name__ == "__main__":
    ASSIGNMENT4_PATH = "Assignment_4_prairielearn.py"
    #TREE_DIR = "."
    INPUT_FILES = [
        "input1.py",
        "input2.py",
        "input3.py",
        "input4.py",
        "input5.py",
    ]

    for inp in INPUT_FILES:
        iv = load_input_file(inp)
        selected_trace, formal = run_assignment4(ASSIGNMENT4_PATH, iv)
        payload = build_nl_payload(iv["action_to_explain"], selected_trace, formal)
        nl = nl_from_payload(payload)
        #nl = nl_from_formal(iv["action_to_explain"], selected_trace, formal)

        print("\n" + "=" * 70)
        print(f"Input: {Path(inp).name}")
        print(f"Action: {iv['action_to_explain']}")
        print(f"Selected trace: {selected_trace}")
        print(f"Formal output: {formal}")
        print(f"Natural language: {nl}")