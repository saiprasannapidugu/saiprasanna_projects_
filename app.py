import os
import uuid

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
)
from PIL import Image, UnidentifiedImageError

import torch
from torch import nn
from torchvision import transforms
from torchvision.models import inception_v3

# ---------------- CONFIG ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_dr_model.pth")

# ✅ FIXED: separate uploads folder
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Index -> label (MUST match training order)
index_to_label = {
    0: "Mild",
    1: "Moderate",
    2: "No_DR",
    3: "Proliferate_DR",
    4: "Severe",
}

# ---------------- DR INFO ----------------
dr_info = {
    "No_DR": {
        "description": "Normal retina. No signs of diabetic retinopathy.",
        "precautions": [
            "Maintain normal blood sugar levels (HbA1c < 7%).",
            "Regular eye exam every 12 months.",
            "Healthy diet & regular exercise.",
            "Avoid smoking and alcohol."
        ]
    },
    "Mild": {
        "description": "Few microaneurysms (tiny swelling, early risk).",
        "precautions": [
            "Strict sugar control and healthy lifestyle.",
            "Eye checkup every 6–12 months.",
            "Reduce sugar & junk foods; include leafy vegetables.",
            "Monitor blood pressure regularly."
        ]
    },
    "Moderate": {
        "description": "Blood vessels may leak fluid. Higher risk progression.",
        "precautions": [
            "Eye checkup every 3–6 months.",
            "Consult eye specialist regarding laser or medication.",
            "Maintain BP (<140/90) & cholesterol.",
            "Seek medical attention if vision blurs."
        ]
    },
    "Severe": {
        "description": "Many blocked blood vessels. Retina not receiving enough blood.",
        "precautions": [
            "Urgent retina specialist consultation required.",
            "Laser photocoagulation therapy may be needed.",
            "Follow-up every 1–3 months.",
            "Strict diabetes and BP control."
        ]
    },
    "Proliferate_DR": {
        "description": "Most advanced stage. Abnormal blood vessels growing.",
        "precautions": [
            "Emergency retina specialist visit.",
            "Potential treatments: Laser therapy, Anti-VEGF injections, Vitrectomy surgery.",
            "Frequent monitoring every 2–4 weeks.",
            "Avoid strenuous exercise until cleared by doctor."
        ]
    }
}

# ---------------- FLASK APP ----------------

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------- MODEL ----------------

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = inception_v3(weights=None, aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, NUM_CLASSES)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # 🔥 FIX: remove AuxLogits mismatch keys
    state_dict = {k: v for k, v in state_dict.items() if "AuxLogits.fc" not in k}

    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()
    return model


model = load_model()
print("✅ Model loaded on:", DEVICE)

# ---------------- TRANSFORM ----------------

predict_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ---------------- HELPERS ----------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_image(image_path: str):
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        return None, "Invalid image file"

    img_tensor = predict_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    label = index_to_label.get(pred_idx, "Unknown")

    if label not in dr_info:
        return None, "Prediction error"

    description = dr_info[label]["description"]
    precautions = dr_info[label]["precautions"]

    return (label, confidence, description, precautions), None


# ---------------- ROUTES ----------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded.")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", error="No file selected.")

        if not allowed_file(file.filename):
            return render_template("index.html", error="Unsupported file format.")

        # ✅ FIX: unique filename
        filename = str(uuid.uuid4()) + ".jpg"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        result, err = predict_image(save_path)

        if err:
            return render_template("index.html", error=err)

        label, conf, description, precautions = result

        return render_template(
            "result.html",
            filename=filename,
            prediction=label,
            confidence=f"{conf * 100:.2f}%",
            description=description,
            precautions=precautions
        )

    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ---------------- RUN ----------------

if __name__ == "__main__":
    app.run(debug=True)