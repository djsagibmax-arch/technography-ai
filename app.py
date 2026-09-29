import os
import io
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)
# সব ডোমেইন থেকে রিকোয়েস্ট অ্যালাও করার জন্য CORS কনফিগারেশন
CORS(app, resources={r"/*": {"origins": "*"}})

# Render-এর Environment Variable থেকে সুরক্ষিতভাবে টোকেন নেওয়া
HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

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

        if not user_prompt:
            return jsonify({"success": False, "error": "প্রম্পট খালি রাখা যাবে না"}), 400

        # FLUX.1-schnell মডেল দিয়ে ইমেজ তৈরি
        image = client.text_to_image(
            prompt=user_prompt,
            model="black-forest-labs/FLUX.1-schnell"
        )

        # ইমেজকে মেমোরি বাফারে নিয়ে Base64 এ কনভার্ট
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return jsonify({
            "success": True,
            "image": f"data:image/jpeg;base64,{img_b64}"
        })

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
