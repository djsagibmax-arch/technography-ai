import os
import io
import base64
import urllib.parse
import urllib.request
import json
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Render Environment Variable থেকে টোকেন
HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

def translate_to_english(text):
    """বাংলা, হিন্দি ইত্যাদি যেকোনো ভাষা থেকে ইংরেজিতে অনুবাদ"""
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
    return jsonify({"status": "Technography AI Engine Live on Render!"})

@app.route("/generate", methods=["POST", "OPTIONS"])
def generate():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        data = request.get_json(force=True, silent=True) or {}
        user_prompt = data.get("prompt", "").strip()
        # আগের ছবির সাথে মিল রাখার জন্য seed হ্যান্ডলিং
        seed = data.get("seed")
        if seed is None:
            seed = random.randint(1, 999999999)
        else:
            seed = int(seed)

        if not user_prompt:
            return jsonify({"success": False, "error": "প্রম্পট খালি রাখা যাবে না"}), 400

        # অনুবাদ করা
        english_prompt = translate_to_english(user_prompt)

        # ছবির নিখুঁত রূপ ও ধারাবাহিকতা বজায় রাখতে প্রম্পট সাজানো
        enhanced_prompt = f"{english_prompt}, cinematic, photorealistic, sharp focus, 8k resolution"

        # FLUX.1-schnell মডেলে seed পাস করে ইমেজ তৈরি
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
        print(f"Error occurred: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
