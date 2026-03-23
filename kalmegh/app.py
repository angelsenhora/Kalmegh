"""
Flask web application for Plant Disease Detection (Kalmegh).
"""

import os
import uuid

from flask import Flask, render_template, request, redirect, url_for, flash
from predict import predict

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXT   = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.secret_key = "kalmegh-secret-key-change-in-prod"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Detect TensorFlow availability once at startup
try:
    import tensorflow  # noqa: F401
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", cnn_available=CNN_AVAILABLE)


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files or request.files["file"].filename == "":
        flash("Please select an image file before submitting.")
        return redirect(url_for("index"))

    file = request.files["file"]

    if not allowed_file(file.filename):
        flash("Unsupported format. Please upload a JPG, PNG, or WEBP image.")
        return redirect(url_for("index"))

    ext       = file.filename.rsplit(".", 1)[1].lower()
    filename  = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    chosen = request.form.get("model", "svm")
    use_cnn = (chosen == "cnn")

    # For individual ML model selection, override best_model temporarily
    if chosen in ("svm", "knn", "decision_tree"):
        from predict import predict_ml_by_name
        try:
            result = predict_ml_by_name(save_path, chosen)
        except FileNotFoundError as e:
            flash(str(e))
            return redirect(url_for("index"))
        except Exception as e:
            flash(f"Prediction failed: {e}")
            return redirect(url_for("index"))
    else:
        try:
            result = predict(save_path, use_cnn=use_cnn)
        except FileNotFoundError as e:
            flash(str(e))
            return redirect(url_for("index"))
        except Exception as e:
            flash(f"Prediction failed: {e}")
            return redirect(url_for("index"))

    raw_conf   = result.get("confidence")
    confidence = round(float(raw_conf), 2) if raw_conf is not None else None

    return render_template(
        "result.html",
        image_url   = url_for("static", filename=f"uploads/{filename}"),
        prediction  = result.get("prediction", "Unknown"),
        confidence  = confidence,
        model_used  = result.get("model", "N/A"),
        error       = result.get("error"),
        cnn_available = CNN_AVAILABLE,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
