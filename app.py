import base64
import io
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from huggingface_hub import InferenceClient

app = Flask(__name__)
CORS(app)

# আপনার পাওয়া Hugging Face টোকেন
HF_TOKEN = "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"
client = InferenceClient(api_key=HF_TOKEN)


@app.route("/")
def home():
  return jsonify({"status": "Technography AI Engine Live on Render!"})


@app.route("/generate", methods=["POST"])
def generate():
  try:
    data = request.get_json() or {}
    user_prompt = data.get("prompt", "").strip()

    if not user_prompt:
      return jsonify({"success": False, "error": "প্রম্পট লিখুন"}), 400

    enhanced_prompt = (
        f"{user_prompt}, tack sharp focus, highly detailed skin pores, raw"
        " color photograph, masterpiece, 8k resolution, unblurred"
    )

    image = client.text_to_image(
        prompt=enhanced_prompt, model="black-forest-labs/FLUX.1-schnell"
    )

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify(
        {"success": True, "image": f"data:image/jpeg;base64,{img_b64}"}
    )

  except Exception as e:
    return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)
