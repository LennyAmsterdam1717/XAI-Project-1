from pathlib import Path
import json
import ollama

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


NAME_MAP = {
    "getCoffee": "getting coffee",
    "getKitchenCoffee": "getting coffee from the kitchen",
    "getAnnOfficeCoffee": "getting coffee from Ann's office",
    "getShopCoffee": "getting coffee from the shop",
    "getStaffCard": "getting a staff card",
    "getOwnCard": "using my own staff card",
    "getOthersCard": "borrowing a colleague's staff card",
    "gotoKitchen": "going to the kitchen",
    "getCoffeeKitchen": "getting the coffee from the kitchen machine",
    "gotoAnnOffice": "going to Ann's office",
    "getPod": "getting a coffee pod",
    "getCoffeeAnnOffice": "getting the coffee from Ann's machine",
    "gotoShop": "going to the coffee shop",
    "payShop": "paying at the shop",
    "getCoffeeShop": "getting the coffee from the shop",
    "staffCardAvailable": "a staff card is available",
    "ownCard": "I have my own staff card",
    "colleagueAvailable": "a colleague is available",
    "haveMoney": "I have money",
    "AnnInOffice": "Ann is in her office",
    "haveCard": "I have a staff card",
    "atKitchen": "I am at the kitchen",
    "havePod": "I have a coffee pod",
    "atAnnOffice": "I am at Ann's office",
    "atShop": "I am at the shop",
    "paidShop": "I have paid at the shop",
    "haveCoffee": "I have coffee",
}

def human_name(key):
    return NAME_MAP.get(key, key)

def build_nl_payload(action_to_explain, selected_trace, formal):
    if not formal:
        return None

    payload = {
        "action": human_name(action_to_explain),
        "goal_context": None,
        "chosen": None,
        "alternative": None,
        "reason_type": None,
        "reason_text": None,
        "preference_order": None
    }

    for f in formal:
        if f and f[0] == "C" and len(f) >= 2:
            payload["chosen"] = human_name(f[1])
            break

    d_factors = [f for f in formal if f and f[0] == "D" and len(f) >= 2]
    if d_factors:
        payload["goal_context"] = human_name(d_factors[-1][1])

    n_factors = [f for f in formal if f and f[0] == "N"]
    f_factors = [f for f in formal if f and f[0] == "F"]
    v_factors = [f for f in formal if f and f[0] == "V"]

    if n_factors:
        x = n_factors[0]
        payload["alternative"] = human_name(x[1])
        payload["reason_type"] = "N"
        raw = x[2] if len(x) > 2 else "rule conflict"
        if isinstance(raw, str) and raw.startswith("P(") and raw.endswith(")"):
            inner = raw[2:-1]
            raw = f"it is prohibited to {human_name(inner)}"
        elif isinstance(raw, str) and raw.startswith("O(") and raw.endswith(")"):
            inner = raw[2:-1]
            parts = [human_name(p.strip()) for p in inner.split(",")]
            raw = f"at least one of the following is required: {', '.join(parts)}"
        payload["reason_text"] = raw
    elif f_factors:
        x = f_factors[0]
        payload["alternative"] = human_name(x[1])
        payload["reason_type"] = "F"
        missing = [human_name(p) for p in x[2]] if len(x) > 2 else []
        payload["reason_text"] = f"the following required conditions were not available: {', '.join(missing)}"
    elif v_factors:
        x = v_factors[0]
        payload["alternative"] = human_name(x[4])
        payload["reason_type"] = "V"
        payload["reason_text"] = {"chosen_cost": x[2], "alt_cost": x[5]}

    for f in formal:
        if f and f[0] == "U" and len(f) >= 2:
            names, order = f[1]
            payload["preference_order"] = [names[i] for i in order]
            break

    return payload

def nl_from_payload(payload):

    if payload is None:
        return "This action was not part of the selected plan."

    prompt = f"""
You are writing a short, faithful explanation for a non-technical person.

You MUST use only the facts in this payload, do not add anything else:
{json.dumps(payload, indent=2)}

Interpretation rules:
- "action" is the action being explained.
- "chosen" is the branch that was selected.
- "goal_context" is the goal this action supports.
- "alternative" is one option that was NOT chosen.
- "reason_type":
  - "N" => the alternative was ruled out by a rule or prohibition. "reason_text" describes the rule.
  - "F" => the alternative was impossible because certain conditions were MISSING. "reason_text" lists what was missing.
  - "V" => the alternative had worse costs. "reason_text" shows the cost comparison.
- "preference_order" lists priorities from most to least important.

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
- Write EXACTLY 3 sentences. No more, no less.
- Use first person ("I chose...", "I did not choose...").
- Sentence 1: what action I chose and what goal it supported.
- Sentence 2: one rejected alternative and why — for "F" say the conditions were MISSING (not present), for "N" say it was not allowed, for "V" say it was less preferred.
- Sentence 3: state my preference order EXACTLY as given in "preference_order" from most to least important, then say only that this order guided my choice. Do not infer or add any reasoning beyond what is stated.
- Do NOT start with any preamble like "Here is the explanation" or "Here is a faithful explanation".
- Do NOT add opinions, flavor text, or facts not in the payload.
- Do NOT mention JSON, code letters, technical labels, or internal names.
- No headings, no bullet points, no closing note.
- Always write all 3 sentences. Never stop after 2, even if the explanation feels complete.

Return only the 3 sentences, nothing else.
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

