# 🌿 Crop Disease Predictor

AI-powered crop disease detection using MobileNetV2 + Streamlit + Oracle SQL.

---

## 📁 Project Structure

```
crop-disease-prediction/
│
├── app.py                    ← Main Streamlit app
├── requirements.txt          ← All dependencies
├── class_names.json          ← Disease class labels (auto-generated)
├── oracle_setup.sql          ← Run once in Oracle SQL Developer
│
├── model/
│   └── best_model.h5         ← Trained model (auto-generated)
│
├── src/
│   └── train.py              ← Model training script
│
├── dataset/                  ← PlantVillage dataset
│   ├── train/
│   └── valid/
│
├── uploads/                  ← Saved uploaded images
│
└── .streamlit/
    └── secrets.toml          ← Oracle DB credentials
```

---

## 🚀 Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download Dataset
Download PlantVillage from Kaggle:
```bash
kaggle datasets download -d emmarex/plantdisease
unzip plantdisease.zip -d dataset/
```

### Step 3 — Setup Oracle Database
Open Oracle SQL Developer and run:
```sql
-- Run oracle_setup.sql file
```

### Step 4 — Add Oracle Credentials
Edit `.streamlit/secrets.toml`:
```toml
[oracle]
user = "your_username"
password = "your_password"
dsn = "localhost/XEPDB1"
```

### Step 5 — Train the Model
```bash
cd src
python train.py
```

### Step 6 — Run the App
```bash
streamlit run app.py
```

---

## 📱 Access on Mobile
- **Local testing:** Use `ngrok http 8501`
- **Free hosting:** Deploy to [Streamlit Cloud](https://streamlit.io/cloud)

---

## 🌱 Supported Crops & Diseases
38 classes including Tomato, Potato, Corn, Apple, Grape, Pepper and more.
