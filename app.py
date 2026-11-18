from flask import Flask, render_template, request
from pathlib import Path
from werkzeug.utils import secure_filename
from PIL import Image
import uuid

from predict import predict_image

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    image_file = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            error = "Please select a leaf image."
        else:
            try:
                image = Image.open(file.stream).convert("RGB")

                filename = secure_filename(file.filename)
                extension = Path(filename).suffix.lower()

                if extension not in [".jpg", ".jpeg", ".png", ".webp"]:
                    error = "Please upload a JPG, PNG or WEBP image."
                else:
                    saved_name = f"{uuid.uuid4().hex}{extension}"
                    image_path = UPLOAD_FOLDER / saved_name
                    image.save(image_path)

                    prediction, confidence = predict_image(image_path)
                    image_file = f"uploads/{saved_name}"

            except Exception:
                error = "Could not process this image. Please try another one."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_file=image_file,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
    
