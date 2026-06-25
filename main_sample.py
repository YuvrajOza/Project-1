import glob
import os
import re
import sys
import json
import time
import base64
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# =========================
# 1. LOAD ENV + CLIENT
# =========================
load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# =========================
# 2. SYSTEM PROMPT (ENTERPRISE GRADE — v2 with Quality Gate)
# =========================
SYSTEM_PROMPT = """You are an advanced AI Insurance Damage Scanner used in real-world automobile claim processing.
You are responsible for analyzing vehicle images and ensuring image quality before damage detection.

## 🧠 TASK FLOW (IMPORTANT)
Step 1: Image Quality Check
- Detect if image is clear, blurred, dark, or unusable
- If image is not usable, set:
  "damage_visible": "UNKNOWN"
  and explain quality issue
Step 2: Damage Detection
- Identify visible damage only if image is clear enough
Step 3: Structured Output

## 📦 OUTPUT FORMAT (STRICT JSON ONLY)
{
  "image_quality": "Good | Blurry | Dark | Obstructed",
  "damage_visible": "YES | NO | UNKNOWN",
  "confidence_score": "0-100",
  "issue_type": "Scratch | Dent | Crack | Broken | Paint Damage | None",
  "object_part": "Bumper | Door | Hood | Windshield | Headlight | Other",
  "severity": "Minor | Moderate | Severe | Critical",
  "drivable_status": "Safe to Drive | Drive with Caution | Not Safe to Drive",
  "estimated_repair_cost_level": "Low | Medium | High | Very High",
  "explanation": "Short technical observation of image",
  "recommended_action": "Self repair | Workshop repair | Inspection required | Reject image"
}

## 🚨 CRITICAL RULES
- Output ONLY valid JSON
- No extra text, no markdown
- Do NOT guess damage in blurry images
- If image is unclear → set damage_visible = "UNKNOWN"
- Be conservative in severity estimation
- Use only visible evidence

## 🧠 IMAGE UNDERSTANDING RULES
- If image is dark → reduce confidence score
- If image is blurred → mark image_quality = "Blurry"
- If object is partially visible → mark "Obstructed"
- Prefer accuracy over assumption

## ⚡ SEVERITY LOGIC
- Minor → cosmetic scratches
- Moderate → visible dents or paint damage
- Severe → broken parts or structural risk
- Critical → safety risk (lights, windshield, structure)

## 🚫 DO NOT
- Do not hallucinate hidden damage
- Do not output incomplete JSON
- Do not add explanations outside JSON

Return ONLY structured JSON. Be precise, realistic, and insurance-grade."""

# =========================
# 3. EXPECTED SCHEMA
# =========================
REQUIRED_FIELDS = {
    "image_quality":               {"Good", "Blurry", "Dark", "Obstructed"},
    "damage_visible":              {"YES", "NO", "UNKNOWN"},
    "issue_type":                  {"Scratch", "Dent", "Crack", "Broken", "Paint Damage", "None"},
    "severity":                    {"Minor", "Moderate", "Severe", "Critical"},
    "estimated_repair_cost_level": {"Low", "Medium", "High", "Very High"},
    "drivable_status":             {"Safe to Drive", "Drive with Caution", "Not Safe to Drive"},
    "recommended_action":          {"Self repair", "Workshop repair", "Inspection required", "Reject image"},
}

# Quality states that block damage assessment — image must be re-submitted
UNUSABLE_QUALITY = {"Blurry", "Dark", "Obstructed"}

# =========================
# 4. IMAGE OPTIMIZATION
# =========================
# 256×256 @ quality 40 keeps the payload under ~15 KB of base64
# (384×384 @ 60 was ~40–50 KB — large enough to trigger NVIDIA 500s)
IMG_SIZE    = (256, 256)
IMG_QUALITY = 40

def compress_image(path: str) -> str:
    """Resize + compress image into a temp file. Returns temp file path."""
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)

    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, format="JPEG", quality=IMG_QUALITY)

    size_kb = os.path.getsize(tmp.name) / 1024
    print(f"   📦 Compressed to {IMG_SIZE[0]}×{IMG_SIZE[1]}px — {size_kb:.1f} KB on disk")
    return tmp.name


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# =========================
# 5. OUTPUT VALIDATION
# =========================
def validate_and_fix(data: dict) -> dict:
    """
    Three-path validation matching the new quality-gate flow:
      Path A — UNKNOWN  : image is unusable → zero out damage fields, flag for rejection
      Path B — NO       : image is clear but no damage found → safe defaults
      Path C — YES      : damage detected → enforce issue_type/damage_visible consistency

    All paths enforce enum values and clamp confidence score.
    Returns cleaned data + optional _validation_warnings list.
    """
    warnings = []

    # ── PATH A: UNKNOWN (bad image quality) ──────────────────────────
    if data.get("damage_visible") == "UNKNOWN" or \
       data.get("image_quality") in UNUSABLE_QUALITY:

        data["damage_visible"]              = "UNKNOWN"
        data["issue_type"]                  = "None"
        data["severity"]                    = "Minor"       # safest default
        data["estimated_repair_cost_level"] = "Low"
        data["drivable_status"]             = "Safe to Drive"
        data["recommended_action"]          = "Reject image"

        # Confidence should be low for bad images; cap it
        try:
            data["confidence_score"] = min(int(data.get("confidence_score", 20)), 30)
        except (ValueError, TypeError):
            data["confidence_score"] = 20
        warnings.append("Image marked unusable — all damage fields reset; action set to 'Reject image'")

    # ── PATH B: NO DAMAGE ────────────────────────────────────────────
    elif data.get("damage_visible") == "NO":
        if data.get("issue_type", "None") != "None":
            warnings.append("Forced issue_type to 'None' because damage_visible=NO")
            data["issue_type"] = "None"
        data["severity"]                    = "Minor"
        data["estimated_repair_cost_level"] = "Low"
        data["drivable_status"]             = "Safe to Drive"
        data["recommended_action"]          = "Self repair"

    # ── PATH C: DAMAGE VISIBLE ───────────────────────────────────────
    else:
        if data.get("issue_type", "None") not in ("None", ""):
            if data.get("damage_visible") != "YES":
                warnings.append("Forced damage_visible to 'YES' because issue_type is set")
                data["damage_visible"] = "YES"
        elif data.get("damage_visible") == "YES" and data.get("issue_type") == "None":
            # Model said YES but no issue_type — conservative fix
            warnings.append("damage_visible=YES but issue_type=None; forced damage_visible to 'NO'")
            data["damage_visible"] = "NO"

    # ── ENUM ENFORCEMENT (all paths) ─────────────────────────────────
    for field, valid_set in REQUIRED_FIELDS.items():
        if data.get(field) not in valid_set:
            # Pick the most conservative / safest option
            fallback = next(iter(sorted(valid_set)))
            warnings.append(f"Invalid '{field}': '{data.get(field)}' → defaulted to '{fallback}'")
            data[field] = fallback

    # ── CONFIDENCE SCORE CLAMP (for YES/NO paths only — UNKNOWN already clamped above) ──
    if data.get("damage_visible") != "UNKNOWN":
        try:
            data["confidence_score"] = max(0, min(100, int(data.get("confidence_score", 50))))
        except (ValueError, TypeError):
            data["confidence_score"] = 50
            warnings.append("confidence_score was invalid; defaulted to 50")

    # ── FALLBACKS ────────────────────────────────────────────────────
    if not data.get("object_part"):
        data["object_part"] = "Other"
    if not data.get("explanation"):
        data["explanation"] = "No additional details provided."

    if warnings:
        data["_validation_warnings"] = warnings

    return data

# =========================
# 6. JSON EXTRACTION UTILITY
# =========================
def extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from model output.

    Attempt order:
      1. Direct parse  — model returned clean JSON (ideal)
      2. Regex extract — JSON buried inside prose or markdown fences
      3. Failure dict  — return error with raw text for debugging

    Why regex over fence-stripping:
      Fence-stripping only handles ```...``` at the very start of the string.
      Regex finds the JSON block anywhere in the output, catching cases like:
        "Sure! Here is the result: { ... }"
        "```json\n{ ... }\n```" mid-string
        trailing explanation after the closing brace
    """
    # Pass 1: try a clean parse first (fastest path, no regex cost)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Pass 2: extract the first {...} block from anywhere in the string
    # re.DOTALL lets . match newlines inside multi-line JSON objects
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Pass 3: give up — return a structured error for the caller to handle
    return {"error": "Invalid JSON", "raw": text}


# =========================
# 7. RETRY UTILITY
# =========================
# These HTTP status codes are worth retrying — server-side transient failures.
# 400/401/404 are NOT retried (our fault, not theirs).
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

def call_with_retry(fn, max_retries: int = 3, base_delay: float = 2.0):
    """
    Call fn() with exponential backoff on retryable errors.

    Retry schedule (base_delay=2):  2s → 4s → 8s
    Raises the last exception if all attempts fail.

    Why exponential and not fixed delay:
      NVIDIA's inference server is usually recovering under load;
      hammering it every 2s prolongs the problem. Doubling the wait
      gives the backend time to shed load between attempts.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return fn()

        except Exception as e:
            last_error = e
            error_str  = str(e)

            # Extract HTTP status if the OpenAI client embedded it
            status = None
            if hasattr(e, "status_code"):
                status = e.status_code
            elif "500" in error_str:
                status = 500
            elif "429" in error_str:
                status = 429
            elif "502" in error_str:
                status = 502
            elif "503" in error_str:
                status = 503

            is_retryable = (status in RETRYABLE_STATUS_CODES) or (status is None)

            if not is_retryable:
                # Auth / bad request — retrying won't help
                print(f"   ❌ Non-retryable error (HTTP {status}): {error_str[:120]}")
                raise

            delay = base_delay * (2 ** (attempt - 1))   # 2 → 4 → 8

            if attempt < max_retries:
                print(f"   ⚠️  Attempt {attempt}/{max_retries} failed "
                      f"(HTTP {status or 'unknown'}) — retrying in {delay:.0f}s…")
                time.sleep(delay)
            else:
                print(f"   ❌ All {max_retries} attempts failed. Last error: {error_str[:120]}")

    raise last_error


# =========================
# 8. VISION MODEL CALL
# =========================
def analyze_single_image(image_path: str) -> dict:
    """Send one image to the vision model; return validated structured output."""
    tmp_path = None
    try:
        tmp_path      = compress_image(image_path)
        image_base64  = encode_image(tmp_path)

        def _api_call():
            return client.chat.completions.create(
                model="meta/llama-3.2-11b-vision-instruct",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this vehicle image and return ONLY the JSON schema defined in your instructions."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )

        # Retry up to 3 times on 500 / 502 / 503 / 429
        response    = call_with_retry(_api_call, max_retries=3, base_delay=2.0)
        raw_content = response.choices[0].message.content
        data        = extract_json(raw_content)

        if "error" in data:
            return {
                "error":      "Model returned invalid JSON",
                "raw_output": data.get("raw", raw_content)
            }

        return validate_and_fix(data)

    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# =========================
# 9. RISK + DECISION ENGINE
# =========================
SEVERITY_SCORE = {"Minor": 20, "Moderate": 50, "Severe": 80, "Critical": 95}
COST_BOOST     = {"Low": 0,   "Medium": 5,   "High": 10, "Very High": 15}

# Quality penalty: reduce effective confidence when image was degraded
QUALITY_CONFIDENCE_CAP = {"Good": 100, "Blurry": 25, "Dark": 35, "Obstructed": 45}

def decision_engine(data: dict) -> dict:
    """
    Three-path decision logic aligned with quality-gate flow:
      UNKNOWN  → image rejected, no risk score assigned
      NO       → no damage, no action needed
      YES      → score using severity + cost + quality-adjusted confidence
    """
    damage_visible = data.get("damage_visible")
    image_quality  = data.get("image_quality", "Good")

    # ── PATH A: Unusable image ────────────────────────────────────────
    if damage_visible == "UNKNOWN":
        return {
            "risk_score":         0,
            "status":             "🔄 IMAGE REJECTED — RESUBMIT REQUIRED",
            "action":             "Ask claimant to resubmit a clear, well-lit photo",
            "image_quality":      image_quality,
            "drivable_status":    "Unknown",
            "recommended_action": "Reject image"
        }

    # ── PATH B: No damage ─────────────────────────────────────────────
    if damage_visible == "NO":
        return {
            "risk_score":         0,
            "status":             "✅ NO DAMAGE DETECTED",
            "action":             "No action required",
            "image_quality":      image_quality,
            "drivable_status":    data.get("drivable_status", "Safe to Drive"),
            "recommended_action": "Self repair"
        }

    # ── PATH C: Damage detected ───────────────────────────────────────
    base_score  = SEVERITY_SCORE.get(data.get("severity", "Minor"), 10)
    cost_bonus  = COST_BOOST.get(data.get("estimated_repair_cost_level", "Low"), 0)

    # Cap model confidence by image quality
    raw_confidence   = data.get("confidence_score", 50)
    quality_cap      = QUALITY_CONFIDENCE_CAP.get(image_quality, 100)
    effective_conf   = min(raw_confidence, quality_cap) / 100

    raw_score   = base_score + cost_bonus
    final_score = round(raw_score * effective_conf)
    final_score = max(0, min(100, final_score))

    if final_score >= 80:
        status = "🚨 AUTO-APPROVED HIGH PRIORITY CLAIM"
        action = "Immediate inspection + tow if needed"
    elif final_score >= 60:
        status = "⚠️  MANUAL REVIEW REQUIRED"
        action = "Adjuster verification before approval"
    elif final_score >= 35:
        status = "🔧 WORKSHOP REPAIR RECOMMENDED"
        action = "Schedule professional repair"
    else:
        status = "🔨 LOW PRIORITY / SELF-REPAIR"
        action = "Owner may self-repair or minor workshop visit"

    return {
        "risk_score":          final_score,
        "status":              status,
        "action":              action,
        "image_quality":       image_quality,
        "drivable_status":     data.get("drivable_status", "Safe to Drive"),
        "recommended_action":  data.get("recommended_action", "Self repair")
    }

# =========================
# 10. BATCH PROCESSOR
# =========================
def analyze_multiple_images(image_list: list) -> list:
    results = []
    total   = len(image_list)

    print(f"\n🚗 Starting batch processing — {total} image(s) found...\n")
    print("=" * 60)

    for idx, img_path in enumerate(image_list, start=1):
        if not os.path.exists(img_path):
            print(f"⚠  [{idx}/{total}] SKIPPED (file not found): {img_path}")
            results.append({"image": img_path, "error": "File not found"})
            continue

        print(f"⚡ [{idx}/{total}] Analyzing: {img_path}")
        analysis = analyze_single_image(img_path)

        if "error" in analysis:
            print(f"   ❌ Error: {analysis['error']}")
            results.append({"image": img_path, "error": analysis})
            continue

        decision = decision_engine(analysis)
        quality  = analysis.get("image_quality", "?")
        vis      = analysis.get("damage_visible", "?")
        print(f"   ✅ Quality: {quality} | Damage: {vis} | "
              f"Severity: {analysis.get('severity')} | "
              f"Risk: {decision['risk_score']} | {decision['status']}")

        results.append({
            "image":    img_path,
            "analysis": analysis,
            "decision": decision
        })

    return results

# =========================
# 11. SUMMARY REPORT
# =========================
def generate_summary(results: list) -> dict:
    total      = len(results)
    errors     = [r for r in results if "error" in r]
    successful = [r for r in results if "analysis" in r]

    damaged    = [r for r in successful if r["analysis"].get("damage_visible") == "YES"]
    no_damage  = [r for r in successful if r["analysis"].get("damage_visible") == "NO"]
    rejected   = [r for r in successful if r["analysis"].get("damage_visible") == "UNKNOWN"]

    # Average risk score — only for images where damage was actually assessed
    assessed   = [r for r in successful if r["analysis"].get("damage_visible") in ("YES", "NO")]
    avg_risk   = (
        round(sum(r["decision"]["risk_score"] for r in assessed) / len(assessed), 1)
        if assessed else 0
    )

    severity_counts = {}
    for r in damaged:
        sev = r["analysis"].get("severity", "Unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    quality_counts = {}
    for r in successful:
        q = r["analysis"].get("image_quality", "Unknown")
        quality_counts[q] = quality_counts.get(q, 0) + 1

    high_priority = [r for r in successful if r["decision"].get("risk_score", 0) >= 80]

    return {
        "total_images_processed":  total,
        "successful_analyses":     len(successful),
        "errors":                  len(errors),
        # Quality gate breakdown
        "images_with_damage":      len(damaged),
        "images_no_damage":        len(no_damage),
        "images_rejected_quality": len(rejected),
        "rejected_image_paths":    [r["image"] for r in rejected],
        # Risk stats (assessed images only)
        "average_risk_score":      avg_risk,
        "severity_breakdown":      severity_counts,
        "image_quality_breakdown": quality_counts,
        # Priority escalation
        "high_priority_claims":    len(high_priority),
        "high_priority_images":    [r["image"] for r in high_priority]
    }

# =========================
# 12. MAIN EXECUTION
# =========================
def find_images(folder: str) -> list:
    """
    Return all .jpg / .jpeg / .png paths inside folder (non-recursive).
    Resolves to absolute paths so downstream code always knows exactly
    which file it's working on regardless of cwd.
    """
    exts   = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(set(images))          # deduplicate, stable order


if __name__ == "__main__":

    # ── Determine search folder ───────────────────────────────────────
    # Priority:
    #   1. CLI argument  →  python script.py /path/to/images
    #   2. Script's own directory  →  where insurance_damage_assessment.py lives
    #   3. Current working directory  →  fallback

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CWD        = os.getcwd()

    if len(sys.argv) > 1:
        search_folder = os.path.abspath(sys.argv[1])
        source_label  = "CLI argument"
    elif find_images(SCRIPT_DIR):
        search_folder = SCRIPT_DIR
        source_label  = "script directory"
    else:
        search_folder = CWD
        source_label  = "current working directory"

    # ── Diagnostics ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🔍 IMAGE SEARCH")
    print("=" * 60)
    print(f"  Source  : {source_label}")
    print(f"  Folder  : {search_folder}")
    print(f"  Script  : {SCRIPT_DIR}")
    print(f"  CWD     : {CWD}")

    if not os.path.isdir(search_folder):
        print(f"\n❌ Folder does not exist: {search_folder}")
        print("   Usage: python insurance_damage_assessment.py [path/to/images]")
        sys.exit(1)

    images = find_images(search_folder)

    if not images:
        print(f"\n❌ No images found in: {search_folder}")
        print("   Supported formats: .jpg  .jpeg  .png  (case-insensitive)")
        print("\n   Quick fixes:")
        print("   1. Put your images in the same folder as this script")
        print("   2. Run:  python insurance_damage_assessment.py /path/to/your/images")
        sys.exit(1)

    print(f"\n✅ Found {len(images)} image(s):")
    for img in images:
        size_kb = os.path.getsize(img) / 1024
        print(f"   • {os.path.basename(img)}  ({size_kb:.1f} KB)")

    # ── Run analysis ──────────────────────────────────────────────────
    final_results = analyze_multiple_images(images)
    summary       = generate_summary(final_results)

    print("\n" + "=" * 60)
    print("📋 BATCH SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print("🏆 FULL INSURANCE REPORT")
    print("=" * 60)
    print(json.dumps(final_results, indent=2))

    # Save report next to the script (not wherever cwd happens to be)
    report_path = os.path.join(SCRIPT_DIR, "insurance_report.json")
    with open(report_path, "w") as f:
        json.dump({"summary": summary, "results": final_results}, f, indent=2)
    print(f"\n💾 Report saved to: {report_path}")