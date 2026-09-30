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
from PIL import Image, ImageOps

app = Flask(__name__)
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
# (সরাসরি ক্লাউড মাস্কিং - ৫১২MB র‍্যামের ভেতর চলবে)
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

        # ইনপুট ছবি লোড
        orig_image = Image.open(file.stream).convert("RGBA")
        
        # ছবির সাইজ সামঞ্জস্য করা (যাতে প্রসেসিং ফাস্ট হয়)
        orig_image.thumbnail((1200, 1200))
        
        rgb_image = orig_image.convert("RGB")
        img_byte_arr = io.BytesIO()
        rgb_image.save(img_byte_arr, format='JPEG', quality=95)
        raw_bytes = img_byte_arr.getvalue()

        # Hugging Face রাউটারে সরাসরি কল
        api_url = "https://router.huggingface.co/hf-inference/models/briaai/RMBG-1.4"
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "image/jpeg"
        }

        resp = requests.post(api_url, headers=headers, data=raw_bytes, timeout=45)

        if resp.status_code != 200:
            return jsonify({
                "success": False, 
                "error": f"HF API Error ({resp.status_code}): {resp.text[:100]}"
            }), 500

        # প্রাপ্ত মাস্ক দিয়ে মূল ছবির আলফা চ্যানেল আলাদা করা
        mask = Image.open(io.BytesIO(resp.content)).convert("L")
        mask = mask.resize(orig_image.size)
        
        # ট্রান্সপারেন্ট ছবি তৈরি
        final_image = orig_image.copy()
        final_image.putalpha(mask)

        out_buffer = io.BytesIO()
        final_image.save(out_buffer, format="PNG")
        out_buffer.seek(0)

        return send_file(
            out_buffer,
            mimetype="image/png",
            as_attachment=False
        )

    except Exception as e:
        print(f"BG Remover Error: {str(e)}")
        return jsonify({"success": False, "error": f"সার্ভার ত্রুটি: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
