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
    "staffCardAvailable": "a staff card",
    "ownCard": "own staff card",
    "colleagueAvailable": "an available colleague",
    "haveMoney": "money",
    "AnnInOffice": "Ann being in her office",
    "haveCard": "a staff card in hand",
    "atKitchen": "being at the kitchen",
    "havePod": "a coffee pod",
    "atAnnOffice": "being at Ann's office",
    "atShop": "being at the shop",
    "paidShop": "having paid at the shop",
    "haveCoffee": "having coffee",
}


def human_name(key):
    return NAME_MAP.get(key, key)


def build_nl_payload(action_to_explain, selected_trace, formal, preferences):
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

    d_factors = [human_name(f[1]) for f in formal if f and f[0] == "D" and len(f) >= 2]
    payload["goal_context"] = d_factors[-1] if d_factors else None

    for f in formal:
        if not f:
            continue

        if f[0] == "N" and len(f) >= 3:
            raw = f[2]
            if isinstance(raw, str) and raw.startswith("P(") and raw.endswith(")"):
                inner = raw[2:-1]
                raw = f"a rule prohibits {human_name(inner)}"
            elif isinstance(raw, str) and raw.startswith("O(") and raw.endswith(")"):
                inner = raw[2:-1]
                parts = [human_name(p.strip()) for p in inner.split(",")]
                raw = f"at least one of the following is required: {', '.join(parts)}"
            payload["rejections"].append({
                "type": "N",
                "alternative": human_name(f[1]),
                "reason": raw,
            })

        elif f[0] == "F" and len(f) >= 3:
            missing = (
                ", ".join(human_name(x) for x in f[2])
                if isinstance(f[2], list)
                else str(f[2])
            )
            payload["rejections"].append({
                "type": "F",
                "alternative": human_name(f[1]),
                "reason": f"it was not possible because these conditions were absent: {missing}",
            })

        elif f[0] == "V" and len(f) >= 6:
            names, order = preferences
            top_dim = names[order[0]]
            payload["rejections"].append({
                "type": "V",
                "alternative": human_name(f[4]),
                "reason": f"it was less preferred based on {top_dim}",
            })

    for f in formal:
        if f and f[0] == "U" and len(f) >= 2:
            names, order = f[1]
            payload["preference_order"] = " > ".join(
                names[order[rank]] for rank in range(len(order))
            )
            break

    return payload


def nl_from_payload(payload, action_to_explain):

    if payload is None:
        action = human_name(action_to_explain)
        return f"I did not perform {action} because a different plan was selected to achieve my goal."
    
    rejection_count = len(payload["rejections"])

    rejection_lines = "\n".join(
    f"  {i+1}. {r['alternative']}: {r['reason']}"
    for i, r in enumerate(payload["rejections"])
)

    prompt = f"""
You are writing a short, faithful explanation for a non-technical person.

You MUST use only the facts in this payload. Do not add anything else.
{json.dumps(payload, indent=2)}

Interpretation rules:
- "action" is the action being explained.
- "chosen" is the branch that was selected.
- "goal_contexts" are the goals this action supports.
- "rejections" lists alternatives that were not chosen and why:
  - type "N": not allowed by a rule.
  - type "F": required conditions were absent (NOT present).
  - type "V": less preferred based on priorities.
- "preference_order" is a string showing priorities from most to least important, separated by ">". State it exactly as written.

Output requirements:
- Write EXACTLY 3 sentences. No more, no less. No paragraph breaks.
- Use first person ("I chose...", "I did not choose...").
- Sentence 1: what action I chose and what goal it supported.
- Sentence 2: Sentence 2 must cover exactly these {rejection_count} rejection(s) and no others:
{rejection_lines}. Combine them into one flowing sentence using "because", "nor", "and". For "F" say the conditions were absent, for "N" say it was not allowed, for "V" say it was less preferred. Only mention alternatives that appear in the "rejections" list. Do not invent or infer additional rejections.
- Sentence 3: state my preference order EXACTLY as given in "preference_order" from most to least important, then say only that this order guided my choice. Do not infer or add any reasoning beyond what is stated.
- Do NOT start with any preamble like "Here is the explanation" or "Here is a faithful explanation".
- Do NOT add opinions, flavor text, or facts not in the payload.
- Do NOT mention JSON, code letters, technical labels, or internal names.
- No headings, no bullet points, no paragraph breaks, no closing note.
- Always write all 3 sentences. Never stop after 2, even if the explanation feels complete.
- End sentence 2 immediately after the last rejection. Do not add any concluding remark or opinion after the final rejection.

Return only the 3 sentences as a single paragraph, nothing else.
"""
    resp = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    return resp["message"]["content"]


if __name__ == "__main__":
    ASSIGNMENT4_PATH = "Assignment_4_prairielearn.py"
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
        payload = build_nl_payload(
            iv["action_to_explain"], selected_trace, formal, iv["preferences"]
        )
        nl = nl_from_payload(payload, iv["action_to_explain"])

        print("\n" + "=" * 70)
        print(f"Input: {Path(inp).name}")
        print(f"Action: {iv['action_to_explain']}")
        print(f"Selected trace: {selected_trace}")
        print(f"Formal output: {formal}")
        print(f"Natural language: {nl}")



