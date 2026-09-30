import os
import io
import base64
import urllib.parse
import urllib.request
import json
import random
import requests
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from huggingface_hub import InferenceClient
from PIL import Image

app = Flask(__name__)
# ওয়েবসাইট থেকে যাতে কোনো কল ব্লক না হয়
CORS(app, resources={r"/*": {"origins": "*"}})

HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

def translate_to_english(text):
    """বাংলা থেকে ইংরেজিতে প্রম্পট অনুবাদ"""
    try:
        encoded_text = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={encoded_text}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode('utf-8'))
            return "".join([part[0] for part in result[0] if part[0]])
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

@app.route("/")
def home():
    return jsonify({"status": "Technography AI Engine is Live on Render!"})

# ==========================================
# ১. এআই ইমেজ জেনারেটর (/generate)
# ==========================================
@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        data = request.get_json(force=True, silent=True) or {}
        user_prompt = data.get("prompt", "").strip()
        seed = data.get("seed")
        if seed is None:
            seed = random.randint(1, 999999999)
        else:
            seed = int(seed)

        if not user_prompt:
            return jsonify({"success": False, "error": "প্রম্পট খালি রাখা যাবে না"}), 400

        english_prompt = translate_to_english(user_prompt)
        enhanced_prompt = f"{english_prompt}, cinematic, photorealistic, sharp focus, 8k resolution"

        image = client.text_to_image(
            prompt=enhanced_prompt,
            model="black-forest-labs/FLUX.1-schnell",
            seed=seed
        )

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return jsonify({
            "success": True,
            "image": f"data:image/jpeg;base64,{img_b64}",
            "seed": seed
        })

    except Exception as e:
        print(f"Generate Error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# ==========================================
# ২. এআই ব্যাকগ্রাউন্ড রিমুভার (/remove-bg)
# (র‍্যাম খরচ না করেই দ্রুতগতিতে চলবে)
# ==========================================
@app.route("/remove-bg", methods=["POST", "OPTIONS"])
def remove_bg():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "কোনো ফাইল পাওয়া যায়নি"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "ফাইল নির্বাচন করা হয়নি"}), 400

        image_bytes = file.read()

        # Hugging Face-এর ক্লাউড RMBG মডেল কল করা
        api_url = "https://api-inference.huggingface.co/models/briaai/RMBG-1.4"
        headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

        response = requests.post(api_url, headers=headers, data=image_bytes, timeout=40)

        if response.status_code == 200:
            return send_file(
                io.BytesIO(response.content),
                mimetype="image/png",
                as_attachment=False
            )
        else:
            return jsonify({
                "success": False, 
                "error": f"AI মডেল রেসপন্স দেয়নি (কোড {response.status_code})"
            }), 500

    except Exception as e:
        print(f"BG Remover Error: {str(e)}")
        return jsonify({"success": False, "error": f"সার্ভার ত্রুটি: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
