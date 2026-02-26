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
        "chosen": None,
        "goal_contexts": [],
        "rejections": [],
        "preference_order": None,
    }

    for f in formal:
        if f and f[0] == "C" and len(f) >= 2:
            payload["chosen"] = human_name(f[1])
            break

    payload["goal_contexts"] = [human_name(f[1]) for f in formal if f and f[0] == "D" and len(f) >= 2]

    for f in formal:
        if not f:
            continue
        if f[0] == "N" and len(f) >= 3:
            payload["rejections"].append({
                "type": "N",
                "alternative": human_name(f[1]),
                "reason": str(f[2]),
            })
        elif f[0] == "F" and len(f) >= 3:
            missing = ", ".join(human_name(x) for x in f[2]) if isinstance(f[2], list) else str(f[2])
            payload["rejections"].append({
                "type": "F",
                "alternative": human_name(f[1]),
                "reason": missing,
            })
        elif f[0] == "V" and len(f) >= 6:
            payload["rejections"].append({
                "type": "V",
                "alternative": human_name(f[4]),
                "reason": f"{f[2]} vs {f[5]}",
            })

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
- "goal_contexts" are the goal this action supports.
- "rejections" lists alternatives that were not chosen and why.
  - rejection type "N": not allowed by a rule.
  - rejection type "F": missing required conditions.
  - rejection type "V": less preferred based on priorities.
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
- Sentence 2: “Mention the rejected alternatives from "rejections" in one concise sentence; for "F" say the conditions were missing (not present), for "N" say it was not allowed, for "V" say it was less preferred.
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


