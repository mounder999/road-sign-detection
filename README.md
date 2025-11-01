# 🚦 Road Sign Detection with YOLOv8 + Streamlit

This project is a **real-time Road Sign Detection app** built with [YOLOv8](https://github.com/ultralytics/ultralytics) and [Streamlit](https://streamlit.io/).  
It allows users to upload an image and instantly detect traffic or road signs using a trained YOLOv8 model.

---

## 🧠 Model Overview

- **Model:** YOLOv8n (fine-tuned on road sign dataset)
- **Framework:** Ultralytics YOLOv8
- **Accuracy:** mAP50 = ~0.96  
- **Classes Detected:** 21 different road signs (e.g., Stop, No Parking, Green Light, etc.)
- **Training Environment:** Google Colab with Tesla T4 GPU

---

## ⚙️ How to Run Locally

1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-username>/road-sign-detection.git
   cd road-sign-detection
