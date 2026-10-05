"""Fridge Vision demo UI: upload a fridge photo, get inventory + freshness + recipes."""
import base64
import io
import os

from flask import Flask, render_template, request
from PIL import Image

from src.pipeline import analyze

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads


def _img_to_data_uri(pil_image, max_width=900):
    img = pil_image.copy()
    if img.width > max_width:
        img = img.resize((max_width, int(img.height * max_width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze_photo():
    upload = request.files.get("photo")
    if upload is None or not upload.filename:
        return render_template("index.html", error="Please choose a photo first.")
    try:
        image = Image.open(upload.stream).convert("RGB")
    except Exception:
        return render_template("index.html", error="Could not read that file as an image.")
    result = analyze(image)
    return render_template(
        "result.html",
        annotated=_img_to_data_uri(result["annotated"]),
        inventory=result["inventory"],
        recipes=result["recipes"],
        n_detections=len(result["detections"]),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
