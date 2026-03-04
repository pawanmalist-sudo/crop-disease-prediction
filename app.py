import streamlit as st
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import oracledb
import os
import datetime

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="🌿 Crop Disease Predictor",
    page_icon="🌱",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main {
    background-color: #0f1a0f;
    color: #e8f5e9;
}

.stApp {
    background: linear-gradient(135deg, #0f1a0f 0%, #1a2e1a 50%, #0d1f1a 100%);
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
    color: #ffffff;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: #a8d5a2;
    text-align: center;
    margin-bottom: 0.2rem;
    text-shadow: 0 0 30px rgba(168, 213, 162, 0.3);
}

.hero-sub {
    text-align: center;
    color: #7aab73;
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2rem;
}

.result-card {
    background: linear-gradient(135deg, #1a2e1a, #1f3d1f);
    border: 1px solid #4a9e40;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    color: #ffffff !important;
}

.disease-badge-diseased {
    background: linear-gradient(135deg, #7a1f1f, #a33030);
    color: #ffd5d5;
    padding: 0.4rem 1.2rem;
    border-radius: 20px;
    font-weight: 500;
    font-size: 0.9rem;
    display: inline-block;
    margin-bottom: 0.5rem;
}

.disease-badge-healthy {
    background: linear-gradient(135deg, #1f5c2e, #2d8a42);
    color: #d5ffd5;
    padding: 0.4rem 1.2rem;
    border-radius: 20px;
    font-weight: 500;
    font-size: 0.9rem;
    display: inline-block;
    margin-bottom: 0.5rem;
}

.confidence-text {
    font-size: 2.5rem;
    font-weight: 700;
    color: #a8d5a2;
    font-family: 'Playfair Display', serif;
}

.divider {
    border: none;
    border-top: 1px solid #2d5a27;
    margin: 1.5rem 0;
}

.stFileUploader {
    border: 2px dashed #2d5a27 !important;
    border-radius: 12px !important;
    background: rgba(45, 90, 39, 0.1) !important;
}

footer {
    text-align: center;
    color: #4a7a44;
    font-size: 0.8rem;
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Oracle DB Connection ──────────────────────────────────
import oracledb

def get_db_connection():
    try:
        connection = oracledb.connect(
            user=st.secrets["oracle"]["user"],
            password=st.secrets["oracle"]["password"],
            dsn=st.secrets["oracle"]["dsn"], 
            mode=oracledb.AUTH_MODE_SYSDBA,
           
           
        )
        return connection
    except Exception as e:
        st.warning(f"⚠️ Database not connected: {e}")
        return None



def save_to_db(image_name, disease_name, confidence, status, top3_diseases, top3_confidences, image_path):
    conn = get_db_connection()
    if conn is None:
        return
    try:
        cursor = conn.cursor()
        pred_id_var = cursor.var(int)

        cursor.execute("""
             INSERT INTO PREDICTIONS (IMAGE_NAME, DISEASE_NAME, CONFIDENCE_SCORE, STATUS_TYPE)
           VALUES (:1, :2, :3, :4)

            RETURNING PREDICTION_ID INTO :5
        """, (image_name, disease_name, confidence, status, pred_id_var))

        prediction_id = pred_id_var.getvalue()[0]

        for rank, (disease, conf) in enumerate(zip(top3_diseases, top3_confidences), start=1):
            cursor.execute("""
                INSERT INTO TOP3_PREDICTIONS (PREDICTION_ID, RANK_NO, DISEASE_NAME, CONFIDENCE_SCORE)
                VALUES (:1, :2, :3, :4)
            """, (prediction_id, rank, disease, conf))

        cursor.execute("""
            INSERT INTO CROP_IMAGES (PREDICTION_ID, IMAGE_NAME, IMAGE_PATH)
            VALUES (:1, :2, :3)
        """, (prediction_id, image_name, image_path))

        conn.commit()
        cursor.close()
        conn.close()
        st.success("✅ Prediction saved to database!")
    except Exception as e:
        st.warning(f"⚠️ Could not save to DB: {e}")


# ─── Load Model & Class Names ──────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("model/best_model.h5"):
        return tf.keras.models.load_model("model/best_model.h5")
    return None


@st.cache_data
def load_class_names():
    if os.path.exists("class_names.json"):
        with open("class_names.json", "r") as f:
            return json.load(f)
    # Default PlantVillage class names
    return [
        "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
        "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
        "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
        "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight", "Grape___healthy",
        "Orange___Haunglongbing", "Peach___Bacterial_spot", "Peach___healthy",
        "Pepper___Bacterial_spot", "Pepper___healthy",
        "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
        "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch", "Strawberry___healthy",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites", "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
    ]


# ─── Prediction ────────────────────────────────────────────
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    return tf.expand_dims(img_array, axis=0)


def predict(image, model, class_names):
    img_tensor = preprocess_image(image)
    predictions = model.predict(img_tensor, verbose=0)[0]
    top3_indices = np.argsort(predictions)[::-1][:3]
    top3_diseases = [class_names[i] for i in top3_indices]
    top3_confidences = [round(float(predictions[i]) * 100, 2) for i in top3_indices]
    return top3_diseases, top3_confidences


# ─── Chart ─────────────────────────────────────────────────
def plot_confidence_chart(diseases, confidences):
    short_names = [d.split("___")[-1].replace("_", " ") for d in diseases]
    colors = ["#4caf50", "#81c784", "#c8e6c9"]

    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#1a2e1a")
    ax.set_facecolor("#1a2e1a")

    bars = ax.barh(short_names[::-1], confidences[::-1], color=colors[::-1],
                   height=0.5, edgecolor="none")

    for bar, conf in zip(bars, confidences[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{conf}%", va='center', ha='left',
                color="#a8d5a2", fontsize=11, fontweight='bold')

    ax.set_xlim(0, 115)
    ax.tick_params(colors="#7aab73", labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color("#2d5a27")
    ax.spines['left'].set_color("#2d5a27")
    ax.xaxis.label.set_color("#7aab73")
    ax.set_xlabel("Confidence (%)", color="#7aab73", fontsize=10)
    ax.set_title("Top 3 Predictions", color="#a8d5a2", fontsize=13, pad=12,
                 fontfamily='serif')
    plt.tight_layout()
    return fig


# ─── App UI ────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌿 Crop Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload a leaf image — get instant disease diagnosis powered by AI</div>',
            unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

model = load_model()
CLASS_NAMES = load_class_names()

if model is None:
    st.error("⚠️ Model not found! Please train the model first and place `best_model.h5` in the `model/` folder.")
    st.info("Run: `python src/train.py` to train the model.")
    st.stop()

# ─── Upload Section ────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear, close-up photo of the crop leaf"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image(image, caption="Uploaded Leaf", use_column_width=True)

    with st.spinner("🔍 Analyzing leaf..."):
        top3_diseases, top3_confidences = predict(image, model, CLASS_NAMES)

    disease_name = top3_diseases[0]
    confidence = top3_confidences[0]
    status = "Healthy" if "healthy" in disease_name.lower() else "Diseased"
    crop_name = disease_name.split("___")[0]
    display_disease = disease_name.split("___")[-1].replace("_", " ")

    with col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        badge_class = "disease-badge-healthy" if status == "Healthy" else "disease-badge-diseased"
        icon = "✅" if status == "Healthy" else "🔴"
        st.markdown(f'<span class="{badge_class}">{icon} {status}</span>', unsafe_allow_html=True)
        st.markdown(f"<p style='color:#ffffff; font-size:1.1rem;'>🌱 <b>Crop:</b> {crop_name}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#ffffff; font-size:1.1rem;'>🦠 <b>Disease:</b> {display_disease}</p>", unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-text">{confidence}%</div>', unsafe_allow_html=True)
        st.markdown("<small>confidence score</small>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─── Confidence Chart ──────────────────────────────────
    fig = plot_confidence_chart(top3_diseases, top3_confidences)
    st.pyplot(fig)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─── Save to DB Button ─────────────────────────────────
    if st.button("💾 Save to Database", use_container_width=True):
        save_dir = "uploads"
        os.makedirs(save_dir, exist_ok=True)
        image_path = os.path.join(save_dir, uploaded_file.name)
        image.save(image_path)

        save_to_db(
            image_name=uploaded_file.name,
            disease_name=disease_name,
            confidence=confidence,
            status=status,
            top3_diseases=top3_diseases,
            top3_confidences=top3_confidences,
            image_path=image_path
        )

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #4a7a44;">
        <div style="font-size: 4rem;">🍃</div>
        <div style="font-size: 1rem; margin-top: 1rem;">Upload a leaf image to get started</div>
        <div style="font-size: 0.85rem; margin-top: 0.5rem; color: #3a6a34;">
            Supports: Tomato, Potato, Corn, Apple, Grape, Pepper & more
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<footer>
    🌿 Crop Disease Predictor — Powered by MobileNetV2 + PlantVillage Dataset
</footer>
""", unsafe_allow_html=True)
