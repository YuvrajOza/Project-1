import glob
import os
import json
import base64
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# =========================
# 1. SETUP
# =========================
load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# =========================
# 2. IMAGE PREPROCESSING
# =========================
def preprocess_image(path):
    img = Image.open(path)
    img = img.resize((384, 384))
    new_path = f"processed_{os.path.basename(path)}"
    img.save(new_path, quality=60)
    return new_path

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# =========================
# 3. QUALITY + DAMAGE SCANNER
# =========================
def scan_image(image_path):

    image_path = preprocess_image(image_path)
    image_base64 = encode_image(image_path)

    response = client.chat.completions.create(
        model="meta/llama-3.2-11b-vision-instruct",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """
You are an AI Insurance Damage Scanner.

Analyze image and return ONLY JSON:

{
  "image_quality": "Good | Blurry | Dark | Obstructed",
  "damage_visible": "YES | NO | UNKNOWN",
  "issue_type": "Scratch | Dent | Crack | Broken | None",
  "object_part": "Bumper | Door | Hood | Windshield | Other",
  "severity": "Minor | Moderate | Severe | Critical",
  "confidence_score": 0-100,
  "explanation": "short reasoning",
  "recommended_action": "Self repair | Workshop | Reject image"
}

Rules:
- If image is unclear → damage_visible = UNKNOWN
- Be conservative
- Output ONLY JSON
"""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }],
        max_tokens=150
    )

    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {
            "error": "Invalid JSON",
            "raw_output": response.choices[0].message.content
        }

# =========================
# 4. RISK ENGINE
# =========================
def risk_engine(data):

    severity_map = {
        "Minor": 20,
        "Moderate": 50,
        "Severe": 80,
        "Critical": 95
    }

    score = severity_map.get(data.get("severity", "Minor"), 10)

    if score >= 80:
        status = "HIGH PRIORITY CLAIM"
    elif score >= 50:
        status = "MANUAL REVIEW"
    else:
        status = "LOW PRIORITY"

    return {
        "risk_score": score,
        "status": status
    }

# =========================
# 5. BATCH SCANNER (MAIN SYSTEM)
# =========================
def scan_multiple_images(image_list):

    results = []

    print("\n🚗 STARTING MULTI-IMAGE SCAN...\n")

    for i, img in enumerate(image_list):

        print(f"⚡ Scanning {i+1}/{len(image_list)}: {img}")

        result = scan_image(img)

        if "error" not in result:
            decision = risk_engine(result)
        else:
            decision = {"status": "ERROR"}

        results.append({
            "image": img,
            "analysis": result,
            "decision": decision
        })

    return results

# =========================
# 6. FINAL REPORT
# =========================
if __name__ == "__main__":

    # 👇 automatically pick 4-5 images
    images = glob.glob("*.jpg")[:5]

    if not images:
        print("❌ No images found")
    else:
        final_report = scan_multiple_images(images)

        print("\n🏆 FINAL INSURANCE REPORT\n")
        print(json.dumps(final_report, indent=2))