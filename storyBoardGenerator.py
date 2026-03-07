import re
import json
import os
import time
import replicate
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# ───────────────────────────── CONFIG ───────────────────────────── #

REPLICATE_API_TOKEN  = os.getenv("REPLICATE_API_TOKEN")
TARGET_TOTAL_SECONDS = 240.0   # 4 minutes
MAX_PANELS           = 12
MIN_PANELS           = 6
OUTPUT_DIR           = "panels"
FLUX_MODEL           = "black-forest-labs/flux-2-pro"

# ───────────────────────────── STEP 1 – PARSE SCRIPT ────────────── #

def split_into_panels(script: str) -> List[Dict]:
    panels  = []
    current = None
    for line in script.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\[.*\]$", line):
            if current:
                panels.append(current)
            current = {
                "scene_title":     line.strip("[]"),
                "narration_lines": []
            }
        else:
            if current is None:
                current = {
                    "scene_title":     "Scene – Intro",
                    "narration_lines": []
                }
            current["narration_lines"].append(line)
    if current:
        panels.append(current)
    return panels


def merge_panels(panels: List[Dict]) -> List[Dict]:
    if len(panels) > MAX_PANELS:
        merged_narration = [
            line for p in panels[MAX_PANELS - 1:] for line in p["narration_lines"]
        ]
        panels = panels[:MAX_PANELS - 1] + [{
            "scene_title":     "Scene – Chaos Finale",
            "narration_lines": merged_narration
        }]
    return panels

# ───────────────────────────── STEP 2 – TIMING ──────────────────── #

def assign_timing(panels_raw: List[Dict], total_seconds: float) -> List[Dict]:
    total_words  = sum(
        len(" ".join(p["narration_lines"]).split()) for p in panels_raw
    )
    current_time = 0.0
    enriched     = []
    for idx, p in enumerate(panels_raw, start=1):
        text     = " ".join(p["narration_lines"])
        words    = len(text.split()) or 1
        duration = total_seconds * (words / total_words)
        start    = current_time
        end      = current_time + duration
        current_time = end
        enriched.append({
            "panel_id":       idx,
            "scene_title":    p["scene_title"],
            "narration_text": text,
            "start_time":     round(start,    2),
            "end_time":       round(end,      2),
            "duration":       round(duration, 2),
        })
    return enriched

# ───────────────────────────── STEP 3 – VISUALS ─────────────────── #

def build_visual_prompt(scene_title: str, narration_text: str) -> str:
    base       = f"{scene_title}, cinematic, colorful, comic Malayalam movie style, vivid colors, dramatic lighting"
    text_lower = narration_text.lower()
    extras     = []
    if "goat" in text_lower or "pinky" in text_lower:
        extras.append("goat in the frame")
    if "dude" in text_lower:
        extras.append("stylish gangster character")
    if "chase" in text_lower:
        extras.append("dynamic motion blur, chase scene")
    if "back pain" in text_lower:
        extras.append("comedic pose, holding lower back")
    if "crowd" in text_lower or "gang" in text_lower:
        extras.append("large crowd, chaos")
    if "forest" in text_lower or "jungle" in text_lower:
        extras.append("lush Kerala jungle background")
    if extras:
        base += ", " + ", ".join(extras)
    return base


def pick_transitions(scene_title: str):
    title_lower = scene_title.lower()
    if "smash cut" in title_lower:
        return "smash_cut", "whip_pan"
    if "intro" in title_lower or "scene 1" in title_lower:
        return "fade_in", "cut"
    if "final" in title_lower or "climax" in title_lower:
        return "cut", "glitch"
    return "cut", "cut"


def enrich_panels(panels_timed: List[Dict]) -> List[Dict]:
    enriched = []
    for p in panels_timed:
        vin, vout = pick_transitions(p["scene_title"])
        enriched.append({
            **p,
            "visual_prompt":  build_visual_prompt(p["scene_title"], p["narration_text"]),
            "transition_in":  vin,
            "transition_out": vout,
            "notes":          "Fast, chaotic editing; freeze-frame on punchlines.",
            "image_path":     None,
        })
    return enriched

# ───────────────────────────── STEP 4 – IMAGE GEN (FLUX) ─────────── #

def generate_panel_image(visual_prompt: str, panel_id: int) -> str:
    """Run FLUX.2 Pro on Replicate, save output as webp, return file path."""
    if not REPLICATE_API_TOKEN:
        print("  [skip] REPLICATE_API_TOKEN not set — no images generated.")
        return ""

    print(f"  Panel {panel_id} — generating with FLUX.2 Pro …")

    try:
        output = replicate.run(
            FLUX_MODEL,
            input={"prompt": visual_prompt}
        )

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"panel_{panel_id:02d}.webp")

        with open(path, "wb") as f:
            f.write(output.read())

        print(f"  Panel {panel_id} saved → {path}")
        return path

    except Exception as e:
        print(f"  Panel {panel_id} failed: {e}")
        return ""


def generate_all_images(storyboard: dict) -> dict:
    print(f"\nGenerating {storyboard['panel_count']} panel images via FLUX.2 Pro …")
    for panel in storyboard["panels"]:
        path = generate_panel_image(panel["visual_prompt"], panel["panel_id"])
        panel["image_path"] = path
    return storyboard

# ───────────────────────────── MAIN PIPELINE ────────────────────── #

def generate_storyboard(script: str, total_seconds: float = TARGET_TOTAL_SECONDS) -> dict:
    print("── Step 1: Parsing script into panels …")
    raw_panels = split_into_panels(script)
    raw_panels = merge_panels(raw_panels)
    print(f"           {len(raw_panels)} panels found.")

    print("── Step 2: Assigning timing …")
    timed = assign_timing(raw_panels, total_seconds)

    print("── Step 3: Building visual prompts & transitions …")
    full_panels = enrich_panels(timed)

    storyboard = {
        "total_duration": total_seconds,
        "panel_count":    len(full_panels),
        "panels":         full_panels,
    }

    print("── Step 4: Generating panel images …")
    storyboard = generate_all_images(storyboard)

    print("── Done!")
    return storyboard

# ───────────────────────────── ENTRYPOINT ───────────────────────── #

if __name__ == "__main__":
    script_file = "aadu_script.txt"
    if not os.path.exists(script_file):
        raise FileNotFoundError(
            f"'{script_file}' not found.\n"
            "Run aadu_script_generator.py first — it saves final_script.txt.\n"
            "Rename that to aadu_script.txt, or update script_file above."
        )

    with open(script_file, "r", encoding="utf-8") as f:
        script_text = f.read()

    sb = generate_storyboard(script_text, total_seconds=TARGET_TOTAL_SECONDS)

    out_path = "storyboard.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sb, f, ensure_ascii=False, indent=2)

    print(f"\nStoryboard saved → {out_path}")
    print(f"Panel images     → ./{OUTPUT_DIR}/panel_XX.webp")
    print(f"Total panels     : {sb['panel_count']}")
    print(f"Total duration   : {sb['total_duration']}s")