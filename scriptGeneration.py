import json
import os
import re
import hashlib
import time
from typing import Dict, Any, Optional

from groq import Groq
from dotenv import load_dotenv

# ───────────────────────────── CONFIG ───────────────────────────── #

load_dotenv()

GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
MODEL_NAME          = "llama-3.3-70b-versatile"
TARGET_MIN_WORDS    = 200
TARGET_MAX_WORDS    = 400
MAX_REVISION_ROUNDS = 3
CACHE_FILE          = ".llm_cache.json"

# ───────────────────────────── CLIENT ───────────────────────────── #

client = Groq(api_key=GROQ_API_KEY)

# ───────────────────────────── CACHE ────────────────────────────── #

def _load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def call_llm_cached(prompt: str, system: str = "", bypass_cache: bool = False) -> str:
    cache_key = hashlib.md5((system + prompt).encode()).hexdigest()
    cache     = _load_cache()
    if not bypass_cache and cache_key in cache:
        print("  [cache hit — skipping API call]")
        return cache[cache_key]
    result = _call_groq(prompt, system)
    cache[cache_key] = result
    _save_cache(cache)
    return result

def _call_groq(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.9,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error (attempt {attempt+1}): {e}")
            if attempt == 2:
                raise
            time.sleep(2 ** attempt * 5)
    raise RuntimeError("Failed after 3 retries.")

# ───────────────────────────── DATA LOADING ─────────────────────── #

def load_facts(path: str = "aadu_facts.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ───────────────────────────── PROMPT ───────────────────────────── #

STYLE_EXAMPLE = """
EXAMPLE OUTPUT STYLE (follow this exact vibe):
[Scene 1 – Shaji Pappan Enters the Chat]
Shaji Pappan. One man. One goat. Infinite spine problems.
Pinky the goat arrives and somehow causes more drama than any human character.
Shaji's face: pure stress. The audience: howling.
[Smash cut to: every single person in Idukki knowing everyone else's business]
Classic. Expected. Delivered.
"""

def build_prompt(facts: Dict[str, Any]) -> str:
    a1 = facts["aadu1"]
    a2 = facts["aadu2"]
    def bullets(lines): return "\n".join(f"- {x}" for x in lines)

    return f"""
You are a chaotic, funny Malayalam movie recap narrator writing scripts for meme-style YouTube recap videos.

{STYLE_EXAMPLE}

YOUR TASK:
Write a {TARGET_MIN_WORDS}-{TARGET_MAX_WORDS} word script combining Aadu 1 and Aadu 2.

STYLE RULES:
- Tone: absurd, fast-paced, self-aware, meme-friendly
- Scene headings MUST use square brackets — e.g. [Scene 3 - Goat Goes Rogue]
- At least 6 scene headings spread across both films
- Mix punchy one-liners with brief explanatory sentences
- Highlight running gags every chance you get
- End on a single killer closing line

MANDATORY CONTENT:

[AADU 1 - PLOT POINTS]
{bullets(a1["plot_points"])}

[AADU 1 - RUNNING GAGS]
{bullets(a1["running_gags"])}

[AADU 2 - PLOT POINTS]
{bullets(a2["plot_points"])}

[AADU 2 - RUNNING GAGS]
{bullets(a2["running_gags"])}

Return ONLY a valid JSON object, no markdown fences, no explanation:
{{
  "scenes": [
    {{
      "id": 1,
      "heading": "Scene 1 - <title>",
      "narration": "<narration text>",
      "mood": "<chaotic|funny|dramatic|absurd>",
      "duration_seconds": 30,
      "visual_note": "<editing/visual suggestion>"
    }}
  ],
  "closing_line": "<final killer line>",
  "total_words": 0
}}
""".strip()

# ───────────────────────────── PARSING ──────────────────────────── #

def parse_structured_output(raw: str) -> Optional[Dict]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        data = json.loads(cleaned)
        all_narration = " ".join(s["narration"] for s in data.get("scenes", []))
        data["total_words"] = count_words(all_narration)
        return data
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}\n  Raw preview: {raw[:300]}")
        return None

# ───────────────────────────── VALIDATION ───────────────────────── #

def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))

def validate_script(data: Dict) -> tuple[bool, list[str]]:
    issues = []
    wc = data.get("total_words", 0)
    if wc < TARGET_MIN_WORDS:
        issues.append(f"too short ({wc} words) — expand to {TARGET_MIN_WORDS}-{TARGET_MAX_WORDS}")
    elif wc > TARGET_MAX_WORDS:
        issues.append(f"too long ({wc} words) — shorten to {TARGET_MIN_WORDS}-{TARGET_MAX_WORDS}")
    if len(data.get("scenes", [])) < 6:
        issues.append(f"only {len(data.get('scenes', []))} scenes — need at least 6")
    if not data.get("closing_line", "").strip():
        issues.append("missing closing_line")
    return len(issues) == 0, issues

# ───────────────────────────── REVISION ─────────────────────────── #

SYSTEM_PROMPT = "You write funny, chaotic Malayalam movie recap scripts. Always return valid JSON only."

def revise_script(original_raw: str, issues: list[str]) -> str:
    prompt = f"""
You previously generated this recap script JSON:

{original_raw}

Please revise it to fix these issues:
{"; ".join(issues)}

Keep the same chaotic Malayalam-meme style. Return ONLY valid JSON in the same format. No markdown, no explanation.
""".strip()
    return call_llm_cached(prompt, system=SYSTEM_PROMPT, bypass_cache=True)

# ───────────────────────────── FLAVOR ───────────────────────────── #

FLAVOR_GAGS = [
    "full-on kalipp mode",
    "Idukki-level chaos",
    "poor Pinky the goat",
    "Shaji's legendary back pain",
]

def add_flavor(data: Dict) -> Dict:
    if not data.get("scenes"):
        return data
    all_text = " ".join(s["narration"] for s in data["scenes"])
    missing  = [p for p in FLAVOR_GAGS if p.lower() not in all_text.lower()]
    if missing:
        data["scenes"][-1]["narration"] += " (Featuring: " + ", ".join(missing) + " — as always.)"
        data["total_words"] = count_words(" ".join(s["narration"] for s in data["scenes"]))
    return data

# ───────────────────────────── PIPELINE ─────────────────────────── #

def generate_aadu_script(facts_path: str = "aadu_facts.json") -> Dict:
    print("── Step 1: Loading facts ...")
    facts = load_facts(facts_path)

    print("── Step 2: Building prompt ...")
    prompt = build_prompt(facts)

    print("── Step 3: Calling Groq ...")
    raw = call_llm_cached(prompt, system=SYSTEM_PROMPT)

    print("── Step 4: Parsing output ...")
    data = parse_structured_output(raw)
    if data is None:
        raise ValueError("Initial generation failed to produce valid JSON.")

    print(f"           Words: {data['total_words']} | Scenes: {len(data.get('scenes', []))}")

    for round_num in range(1, MAX_REVISION_ROUNDS + 1):
        is_valid, issues = validate_script(data)
        if is_valid:
            print(f"── Validation passed (round {round_num})")
            break
        print(f"── Revision round {round_num}: {issues}")
        raw = revise_script(raw, issues)
        revised = parse_structured_output(raw)
        if revised:
            data = revised
            print(f"           Words: {data['total_words']} | Scenes: {len(data.get('scenes', []))}")
        else:
            print("   Revision produced invalid JSON — keeping previous.")
    else:
        print(f"Warning: did not fully pass after {MAX_REVISION_ROUNDS} rounds.")

    print("── Step 5: Adding flavor ...")
    data = add_flavor(data)
    print("── Done!")
    return data

# ───────────────────────────── OUTPUT HELPERS ───────────────────── #

def script_to_plaintext(data: Dict) -> str:
    lines = []
    for scene in data.get("scenes", []):
        lines.append(f"\n[{scene['heading']}]")
        lines.append(scene["narration"])
        lines.append(f"  -> Visual: {scene.get('visual_note', '')}  |  ~{scene.get('duration_seconds', '?')}s")
    lines.append(f"\n{data.get('closing_line', '')}")
    lines.append(f"\n-- Total words: {data.get('total_words', '?')} --")
    return "\n".join(lines)

def save_script(data: Dict, out_json: str = "final_script.json", out_txt: str = "final_script.txt") -> None:
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(script_to_plaintext(data))
    print(f"Saved -> {out_json}  |  {out_txt}")

# ───────────────────────────── ENTRYPOINT ───────────────────────── #

if __name__ == "__main__":
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set.\n"
            "1. Go to https://console.groq.com/keys\n"
            "2. Click 'Create API Key'\n"
            "3. Add to your .env file:  GROQ_API_KEY=gsk_..."
        )

    script_data = generate_aadu_script()
    print("\n============== GENERATED SCRIPT ==============\n")
    print(script_to_plaintext(script_data))
    save_script(script_data)