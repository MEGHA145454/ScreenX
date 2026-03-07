import json
import os
import re
from typing import Dict, Any

from openai import OpenAI  # pip install openai

# ------------- CONFIG ------------- #

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"  # or any chat model you have access to
TARGET_MIN_WORDS = 200
TARGET_MAX_WORDS = 400

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------- DATA LOADING ------------- #

def load_facts(path: str = "aadu_facts.json") -> Dict[str, Any]:
    """Load structured facts about Aadu 1 & 2."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------- PROMPT BUILDING ------------- #

def build_prompt(facts: Dict[str, Any]) -> str:
    """Turn the facts JSON into a single prompt string."""
    a1 = facts["aadu1"]
    a2 = facts["aadu2"]

    def bullets(lines):
        return "\n".join(f"- {x}" for x in lines)

    aadu1_plot = bullets(a1["plot_points"])
    aadu1_gags = bullets(a1["running_gags"])
    aadu2_plot = bullets(a2["plot_points"])
    aadu2_gags = bullets(a2["running_gags"])

    prompt = f"""
You are a chaotic, funny Malayalam movie recap narrator.

Task:
Write a {TARGET_MIN_WORDS}-{TARGET_MAX_WORDS} word script for a 3–5 minute recap video
that combines Aadu 1 and Aadu 2. The script will be narrated over fast, meme-style editing.

STYLE RULES:
- Keep the tone absurd, fast, and self-aware.
- Use scene headings in square brackets, like:
  [Scene 1 – Dude and Neelakoduveli]
  [Smash cut to: Shaji Pappan's goat problems]
- Mix short punchy lines with a few longer explanatory sentences.
- Highlight running jokes and character quirks.
- End with a strong, funny closing line.

MANDATORY CONTENT TO COVER:

[AADU 1 – CORE PLOT POINTS]
{aadu1_plot}

[AADU 1 – RUNNING GAGS]
{aadu1_gags}

[AADU 2 – CORE PLOT POINTS]
{aadu2_plot}

[AADU 2 – RUNNING GAGS]
{aadu2_gags}

Now generate the final script only. Do not explain what you are doing.
"""
    return prompt.strip()


# ------------- LLM CALL ------------- #

def call_llm(prompt: str) -> str:
    """Call the chat model and return the text content."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You write funny, chaotic movie recap scripts."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
    )
    return resp.choices[0].message.content.strip()


# ------------- VALIDATION + POST-PROCESSING ------------- #

def count_words(text: str) -> int:
    words = re.findall(r"\w+", text)
    return len(words)


def has_scene_markers(text: str, min_markers: int = 4) -> bool:
    # Very simple: count '[' ... ']' patterns
    markers = re.findall(r"\[.*?\]", text)
    return len(markers) >= min_markers


def add_extra_flavor(text: str) -> str:
    """Optional: inject a few recurring phrases if they are missing."""
    flavor_phrases = [
        "full-on kalipp mode",
        "Idukki-level chaos",
        "poor Pinky the goat",
        "Shaji's legendary back pain"
    ]
    for phrase in flavor_phrases:
        if phrase not in text:
            # Append at the end of a suitable sentence (very simple approach)
            text += f"\n\n(Yes, this is {phrase}.)"
    return text


def revise_with_llm(original_script: str, reason: str) -> str:
    """Ask the model to revise its own script based on constraints."""
    prompt = f"""
You previously wrote the following recap script:

\"\"\"{original_script}\"\"\"

Please revise this script with the following change:
{reason}

Keep the same style, language mix, and chaotic tone. Return only the revised script.
"""
    return call_llm(prompt)


def validate_and_fix(script: str) -> str:
    """Enforce word count and structure, with at most 2 revision rounds."""
    for _ in range(2):
        wc = count_words(script)
        ok_length = TARGET_MIN_WORDS <= wc <= TARGET_MAX_WORDS
        ok_markers = has_scene_markers(script)

        if ok_length and ok_markers:
            break

        # Prepare reason for revision
        reasons = []
        if not ok_length:
            if wc < TARGET_MIN_WORDS:
                reasons.append(
                    f"expand it to be between {TARGET_MIN_WORDS} and {TARGET_MAX_WORDS} words"
                )
            else:
                reasons.append(
                    f"shorten it to be between {TARGET_MIN_WORDS} and {TARGET_MAX_WORDS} words"
                )
        if not ok_markers:
            reasons.append(
                "add clear scene headings in square brackets for at least 4 scenes"
            )

        reason_text = " and ".join(reasons)
        script = revise_with_llm(script, reason_text)

    # Final flavor touch
    script = add_extra_flavor(script)
    return script


# ------------- PUBLIC ENTRYPOINT ------------- #

def generate_aadu_script() -> str:
    """Main function your pipeline will call."""
    facts = load_facts()
    prompt = build_prompt(facts)
    raw_script = call_llm(prompt)
    final_script = validate_and_fix(raw_script)
    return final_script


# ------------- MANUAL TEST ------------- #

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        raise RuntimeError("Please set OPENAI_API_KEY environment variable.")

    script = generate_aadu_script()
    print("\n=== GENERATED SCRIPT ===\n")
    print(script)
    print("\n=== WORD COUNT:", count_words(script), "===\n")
