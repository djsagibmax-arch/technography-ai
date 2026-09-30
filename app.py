import os
import io
import base64
import urllib.parse
import urllib.request
import json
import random

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from huggingface_hub import InferenceClient
from PIL import Image, ImageOps

# ==========================================
# Flask App
# ==========================================

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)

# ==========================================
# Environment
# ==========================================

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    print("WARNING: HF_TOKEN পাওয়া যায়নি!")

# Hugging Face current Inference Provider system
client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)

# ==========================================
# Basic Settings
# ==========================================

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# বর্তমান Background Removal Model
BG_MODEL = "briaai/RMBG-2.0"

# Image Generation Model
GEN_MODEL = "black-forest-labs/FLUX.1-schnell"


# ==========================================
# বাংলা -> English Translation
# ==========================================

def translate_to_english(text):
    """
    বাংলা/যেকোনো ভাষার Prompt ইংরেজিতে রূপান্তর করার চেষ্টা।
    Translation ব্যর্থ হলে original prompt ব্যবহার করবে।
    """

    if not text:
        return text

    try:
        encoded_text = urllib.parse.quote(text)

        url = (
            "https://translate.googleapis.com/translate_a/single"
            "?client=gtx"
            "&sl=auto"
            "&tl=en"
            "&dt=t"
            f"&q={encoded_text}"
        )

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(req, timeout=8) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

            translated = ""

            for part in result[0]:

                if part and part[0]:
                    translated += part[0]

            if translated.strip():
                return translated.strip()

    except Exception as e:

        print(
            "Translation Error:",
            str(e)
        )

    return text


# ==========================================
# Home / Health Check
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "success": True,
        "status": "online",
        "service": "Technography AI Engine",
        "image_generation": True,
        "background_removal": True,
        "background_model": BG_MODEL,
        "generation_model": GEN_MODEL
    })


# ==========================================
# Health Check
# ==========================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "hf_token": bool(HF_TOKEN)
    })


# ==========================================================
# ১. AI IMAGE GENERATOR
# ==========================================================

@app.route(
    "/generate",
    methods=["POST", "OPTIONS"]
)
def generate():

    if request.method == "OPTIONS":

        return jsonify({
            "status": "ok"
        }), 200

    try:

        # ------------------------------
        # Token Check
        # ------------------------------

        if not HF_TOKEN:

            return jsonify({
                "success": False,
                "error": "HF_TOKEN Render Environment Variable-এ পাওয়া যায়নি।"
            }), 500

        # ------------------------------
        # Request Data
        # ------------------------------

        data = request.get_json(
            force=True,
            silent=True
        ) or {}

        user_prompt = str(
            data.get("prompt", "")
        ).strip()

        seed = data.get("seed")

        # ------------------------------
        # Prompt Check
        # ------------------------------

        if not user_prompt:

            return jsonify({
                "success": False,
                "error": "প্রম্পট খালি রাখা যাবে না।"
            }), 400

        # ------------------------------
        # Seed
        # ------------------------------

        if seed is None:

            seed = random.randint(
                1,
                999999999
            )

        else:

            try:
                seed = int(seed)

            except Exception:

                seed = random.randint(
                    1,
                    999999999
                )

        # ------------------------------
        # Translation
        # ------------------------------

        english_prompt = translate_to_english(
            user_prompt
        )

        # ------------------------------
        # Enhanced Prompt
        # ------------------------------

        enhanced_prompt = (
            f"{english_prompt}, "
            "cinematic composition, "
            "photorealistic, "
            "high detail, "
            "sharp focus, "
            "professional lighting, "
            "high quality"
        )

        print(
            "Generating image:",
            enhanced_prompt
        )

        # ------------------------------
        # Generate Image
        # ------------------------------

        image = client.text_to_image(

            prompt=enhanced_prompt,

            model=GEN_MODEL,

            seed=seed,

            width=1024,

            height=1024
        )

        # ------------------------------
        # Convert Image
        # ------------------------------

        buffer = io.BytesIO()

        # PNG ব্যবহার করছি যাতে quality ভালো থাকে
        image.save(
            buffer,
            format="PNG"
        )

        buffer.seek(0)

        img_b64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        # ------------------------------
        # Response
        # ------------------------------

        return jsonify({

            "success": True,

            "image":
                "data:image/png;base64,"
                + img_b64,

            "seed": seed,

            "model": GEN_MODEL

        })

    except Exception as e:

        print(
            "Generate Error:",
            repr(e)
        )

        return jsonify({

            "success": False,

            "error":
                "ইমেজ তৈরি করতে সমস্যা হয়েছে: "
                + str(e)

        }), 500


# ==========================================================
# ২. AI BACKGROUND REMOVER
# ==========================================================

@app.route(
    "/remove-bg",
    methods=["POST", "OPTIONS"]
)
def remove_bg():

    if request.method == "OPTIONS":

        return jsonify({
            "status": "ok"
        }), 200

    try:

        # ------------------------------
        # HF Token Check
        # ------------------------------

        if not HF_TOKEN:

            return jsonify({

                "success": False,

                "error":
                    "HF_TOKEN Render Environment Variable-এ সেট করা নেই।"

            }), 500

        # ------------------------------
        # File Check
        # ------------------------------

        if "file" not in request.files:

            return jsonify({

                "success": False,

                "error":
                    "কোনো ছবি পাওয়া যায়নি।"

            }), 400

        file = request.files["file"]

        if not file:

            return jsonify({

                "success": False,

                "error":
                    "ফাইল খালি।"

            }), 400

        if file.filename == "":

            return jsonify({

                "success": False,

                "error":
                    "কোনো ছবি নির্বাচন করা হয়নি।"

            }), 400

        # ------------------------------
        # Read File
        # ------------------------------

        raw_input = file.read()

        if not raw_input:

            return jsonify({

                "success": False,

                "error":
                    "ছবির ডাটা পাওয়া যায়নি।"

            }), 400

        # ------------------------------
        # File Size
        # ------------------------------

        if len(raw_input) > MAX_IMAGE_SIZE:

            return jsonify({

                "success": False,

                "error":
                    "ছবির সর্বোচ্চ সাইজ ১০ MB।"

            }), 413

        # ------------------------------
        # Open Image
        # ------------------------------

        try:

            original_image = Image.open(
                io.BytesIO(raw_input)
            )

            original_image = ImageOps.exif_transpose(
                original_image
            )

            original_image.load()

        except Exception as e:

            return jsonify({

                "success": False,

                "error":
                    "ছবিটি সঠিকভাবে পড়া যাচ্ছে না: "
                    + str(e)

            }), 400

        # ------------------------------
        # Convert RGBA
        # ------------------------------

        original_image = original_image.convert(
            "RGBA"
        )

        # ------------------------------
        # Resize for AI
        # ------------------------------

        ai_image = original_image.copy()

        ai_image.thumbnail(
            (1600, 1600),
            Image.Resampling.LANCZOS
        )

        # ------------------------------
        # Convert to RGB
        # ------------------------------

        ai_rgb = ai_image.convert(
            "RGB"
        )

        # ------------------------------
        # Save Temporary Bytes
        # ------------------------------

        input_buffer = io.BytesIO()

        ai_rgb.save(
            input_buffer,
            format="JPEG",
            quality=92,
            optimize=True
        )

        input_buffer.seek(0)

        print(
            "Sending image to Hugging Face..."
        )

        # ==================================================
        # CURRENT HUGGING FACE IMAGE SEGMENTATION API
        # ==================================================

        results = client.image_segmentation(

            input_buffer.getvalue(),

            model=BG_MODEL,

            threshold=0.5,

            mask_threshold=0.5

        )

        # ------------------------------
        # Check Result
        # ------------------------------

        if not results:

            raise Exception(
                "Hugging Face কোনো mask ফেরত দেয়নি।"
            )

        print(
            "Segmentation result:",
            len(results)
        )

        # ------------------------------
        # Select Mask
        # ------------------------------

        best_result = results[0]

        # একাধিক result থাকলে score দেখে
        # সবচেয়ে ভালো mask নেওয়ার চেষ্টা
        try:

            best_result = max(
                results,
                key=lambda x: (
                    getattr(x, "score", 0)
                    or 0
                )
            )

        except Exception:

            pass

        mask = getattr(
            best_result,
            "mask",
            None
        )

        if mask is None:

            raise Exception(
                "AI mask পাওয়া যায়নি।"
            )

        # ------------------------------
        # Mask Convert
        # ------------------------------

        if not isinstance(mask, Image.Image):

            mask = Image.open(
                io.BytesIO(mask)
            )

        mask = mask.convert("L")

        # ------------------------------
        # Resize Mask
        # Original image size
        # ------------------------------

        mask = mask.resize(
            original_image.size,
            Image.Resampling.LANCZOS
        )

        # ------------------------------
        # Improve Mask
        # ------------------------------

        # Slight contrast enhancement
        mask = ImageOps.autocontrast(
            mask
        )

        # ------------------------------
        # Create Transparent Image
        # ------------------------------

        final_image = original_image.copy()

        final_image.putalpha(
            mask
        )

        # ------------------------------
        # PNG Output
        # ------------------------------

        output_buffer = io.BytesIO()

        final_image.save(
            output_buffer,
            format="PNG",
            optimize=True
        )

        output_buffer.seek(0)

        print(
            "Background removal completed."
        )

        # ------------------------------
        # Return PNG
        # ------------------------------

        return send_file(

            output_buffer,

            mimetype="image/png",

            as_attachment=False,

            download_name="technography_cutout.png"

        )

    except Exception as e:

        print(
            "BG Remover Error:",
            repr(e)
        )

        return jsonify({

            "success": False,

            "error":
                "সার্ভার ত্রুটি: "
                + str(e)

        }), 500


# ==========================================================
# Error Handler
# ==========================================================

@app.errorhandler(413)
def request_entity_too_large(error):

    return jsonify({

        "success": False,

        "error":
            "ফাইল অনেক বড়। সর্বোচ্চ ১০ MB ছবি দিন।"

    }), 413


# ==========================================================
# OPTIONS / CORS
# ==========================================================

@app.after_request
def after_request(response):

    response.headers["Access-Control-Allow-Origin"] = "*"

    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization"
    )

    response.headers["Access-Control-Allow-Methods"] = (
        "GET, POST, OPTIONS"
    )

    return response


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )
