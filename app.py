import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile

# Streamlit page config
st.set_page_config(page_title="Road Sign Detector", page_icon="🚦", layout="wide")

# Enhanced CSS with road/traffic theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(to bottom, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Road-themed Header */
    .road-header {
        position: relative;
        text-align: center;
        padding: 3rem 2rem 4rem 2rem;
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 50%, #f39c12 100%);
        border-radius: 0 0 50% 50% / 0 0 20% 20%;
        margin-bottom: 3rem;
        box-shadow: 0 10px 40px rgba(231, 76, 60, 0.4);
        overflow: hidden;
    }
    
    .road-header::before {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 30px;
        background: repeating-linear-gradient(
            90deg,
            #fff 0px,
            #fff 40px,
            transparent 40px,
            transparent 80px
        );
        opacity: 0.3;
    }
    
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        color: white;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
    }
    
    .main-subtitle {
        font-size: 1.4rem;
        color: rgba(255,255,255,0.95);
        font-weight: 500;
    }
    
    /* Road Sign Icons */
    .sign-icon {
        font-size: 5rem;
        margin: 0 1rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    /* Feature Cards with Sign Shapes */
    .feature-card {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.9), rgba(192, 57, 43, 0.9));
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem;
        cursor: pointer;
        transition: all 0.4s ease;
        border: 3px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        transition: 0.5s;
    }
    
    .feature-card:hover::before {
        left: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 15px 50px rgba(231, 76, 60, 0.5);
        border-color: rgba(255,255,255,0.5);
    }
    
    .feature-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.3));
    }
    
    .feature-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 1rem;
        color: rgba(255,255,255,0.9);
    }
    
    /* Detection Results - Octagon Shape (like Stop Sign) */
    .result-container {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        padding: 2rem;
        margin: 2rem auto;
        max-width: 800px;
        clip-path: polygon(30% 0%, 70% 0%, 100% 30%, 100% 70%, 70% 100%, 30% 100%, 0% 70%, 0% 30%);
        box-shadow: 0 10px 40px rgba(231, 76, 60, 0.4);
        text-align: center;
    }
    
    .result-content {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        clip-path: polygon(30% 0%, 70% 0%, 100% 30%, 100% 70%, 70% 100%, 30% 100%, 0% 70%, 0% 30%);
    }
    
    /* Sign Detection Badge */
    .sign-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        padding: 1rem 2rem;
        margin: 0.5rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 700;
        box-shadow: 0 5px 15px rgba(243, 156, 18, 0.4);
        border: 3px solid white;
    }
    
    /* Image Display */
    .image-frame {
        border: 8px solid #e74c3c;
        border-radius: 20px;
        padding: 1rem;
        background: white;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin: 2rem 0;
    }
    
    /* Confidence Display */
    .confidence-display {
        background: linear-gradient(135deg, #27ae60, #229954);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        font-size: 1.3rem;
        font-weight: 700;
        box-shadow: 0 5px 20px rgba(39, 174, 96, 0.4);
        border: 3px solid rgba(255,255,255,0.3);
    }
    
    /* Upload Section - Circular like traffic light */
    .upload-zone {
        background: linear-gradient(135deg, #34495e, #2c3e50);
        padding: 3rem;
        border-radius: 50px;
        text-align: center;
        border: 5px dashed #e74c3c;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #f39c12;
        background: linear-gradient(135deg, #2c3e50, #34495e);
        transform: scale(1.02);
    }
    
    /* Traffic Light Style Status */
    .status-lights {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .light {
        width: 60px;
        height: 60px;
        border-radius: 50%;
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        border: 4px solid #2c3e50;
    }
    
    .light-green {
        background: radial-gradient(circle, #2ecc71, #27ae60);
        box-shadow: 0 5px 30px rgba(46, 204, 113, 0.6);
    }
    
    .light-yellow {
        background: radial-gradient(circle, #f1c40f, #f39c12);
    }
    
    .light-red {
        background: radial-gradient(circle, #e74c3c, #c0392b);
    }
    
    /* Footer Road Style */
    .road-footer {
        margin-top: 4rem;
        padding: 2rem;
        background: linear-gradient(to right, #2c3e50 0%, #34495e 50%, #2c3e50 100%);
        border-top: 5px solid #f39c12;
        text-align: center;
        color: white;
        position: relative;
    }
    
    .road-footer::before {
        content: '';
        position: absolute;
        top: -5px;
        left: 0;
        right: 0;
        height: 5px;
        background: repeating-linear-gradient(
            90deg,
            #fff 0px,
            #fff 30px,
            transparent 30px,
            transparent 60px
        );
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: 700;
        border-radius: 50px;
        box-shadow: 0 5px 20px rgba(231, 76, 60, 0.4);
        transition: all 0.3s ease;
        border: 3px solid rgba(255,255,255,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(231, 76, 60, 0.6);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Hero Header with Road Sign Theme
st.markdown("""
    <div class="road-header">
        <div>
            <span class="sign-icon">🛑</span>
            <span class="sign-icon">⚠️</span>
            <span class="sign-icon">🚸</span>
        </div>
        <div class="main-title">ROAD SIGN DETECTOR</div>
        <div class="main-subtitle">Advanced AI Traffic Sign Recognition System</div>
        <div class="status-lights">
            <div class="light light-green"></div>
            <div class="light light-yellow"></div>
            <div class="light light-red"></div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Feature Selection
st.markdown("<h2 style='text-align: center; color: white; margin: 3rem 0 2rem 0;'>Choose Detection Method</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📸</div>
            <div class="feature-title">Upload Images</div>
            <div class="feature-desc">Upload your road sign images for instant detection</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📹</div>
            <div class="feature-title">Use Webcam</div>
            <div class="feature-desc">Capture live images from your camera</div>
        </div>
    """, unsafe_allow_html=True)

# Method selection
detection_method = st.radio("", ["📸 Upload Images", "📹 Webcam"], label_visibility="collapsed", horizontal=True)

# Load model
@st.cache_resource
def load_model():
    try:
        model = YOLO("road_sign_best.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 Make sure 'road_sign_best.pt' is in the same folder as this script")
        return None

with st.spinner("🔄 Loading Detection System..."):
    model = load_model()
    
if model is None:
    st.stop()

# Display model info
with st.expander("🔧 Model Information & Debugging"):
    st.write(f"**Model Classes:** {model.names}")
    st.write(f"**Number of Classes:** {len(model.names)}")
    confidence_threshold = st.slider("🎯 Adjust Confidence Threshold (for testing)", 0.1, 1.0, 0.5, 0.05)
    st.info(f"Current threshold: {confidence_threshold*100:.0f}% - Lower this if no detections appear")

st.markdown("<br>", unsafe_allow_html=True)

# Upload Images Method
if detection_method == "📸 Upload Images":
    st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
            <h3 style="color: white; margin-bottom: 1rem;">Upload Your Images</h3>
            <p style="color: rgba(255,255,255,0.8);">Supports JPG, JPEG, PNG formats</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: rgba(46, 204, 113, 0.2); 
                        border-radius: 15px; margin: 2rem 0; border: 2px solid #27ae60;">
                <h3 style="color: #2ecc71; margin: 0;">✅ {len(uploaded_files)} Image(s) Ready for Detection</h3>
            </div>
        """, unsafe_allow_html=True)
        
        for idx, uploaded_file in enumerate(uploaded_files, 1):
            st.markdown("<hr style='border: 2px solid #e74c3c; margin: 3rem 0;'>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="text-align: center; margin: 2rem 0;">
                    <h2 style="color: #f39c12;">🖼️ Image {idx}: {uploaded_file.name}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            image = Image.open(uploaded_file)
            img_np = np.array(image)

            with st.spinner("🔍 Detecting Road Signs..."):
                results = model.predict(img_np, conf=0.5)
                annotated = results[0].plot()

            col_img, col_results = st.columns([3, 2])
            
            with col_img:
                st.markdown('<div class="image-frame">', unsafe_allow_html=True)
                st.image(annotated, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_results:
                boxes = results[0].boxes
                
                if len(boxes) > 0:
                    st.markdown(f"""
                        <div class="result-container">
                            <div class="result-content">
                                <h2 style="color: #e74c3c; margin-bottom: 1rem;">🎯 DETECTION RESULTS</h2>
                                <div style="font-size: 3rem; font-weight: 800; color: #27ae60; margin: 1rem 0;">
                                    {len(boxes)}
                                </div>
                                <h3 style="color: #2c3e50;">Sign(s) Detected</h3>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<h3 style='color: white; text-align: center; margin: 2rem 0;'>🚦 Detected Signs</h3>", unsafe_allow_html=True)
                    
                    for i, box in enumerate(boxes):
                        cls = int(box.cls[0])
                        conf_score = float(box.conf[0])
                        label = model.names[cls]

                        st.markdown(f"""
                            <div class="sign-badge">
                                {label}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                            <div class="confidence-display">
                                ✅ Confidence: {conf_score*100:.1f}%
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="result-container">
                            <div class="result-content">
                                <h2 style="color: #e67e22;">⚠️ NO SIGNS DETECTED</h2>
                                <p style="color: #7f8c8d; margin-top: 1rem;">
                                    No road signs found in this image
                                </p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

# Webcam Method
else:
    st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📷</div>
            <h3 style="color: white; margin-bottom: 1rem;">Capture from Webcam</h3>
            <p style="color: rgba(255,255,255,0.8);">Click the camera button below to take a photo</p>
        </div>
    """, unsafe_allow_html=True)
    
    camera_input = st.camera_input("📸 Take a picture", label_visibility="collapsed")

    if camera_input:
        st.markdown("""
            <div style="text-align: center; padding: 1rem; background: rgba(46, 204, 113, 0.2); 
                        border-radius: 15px; margin: 2rem 0; border: 2px solid #27ae60;">
                <h3 style="color: #2ecc71; margin: 0;">✅ Image Captured Successfully</h3>
            </div>
        """, unsafe_allow_html=True)
        
        img = Image.open(camera_input)
        img_np = np.array(img)
        
        with st.spinner("🔍 Analyzing Image for Road Signs..."):
            results = model.predict(img_np, conf=0.5)
            annotated = results[0].plot()
        
        col_img, col_results = st.columns([3, 2])
        
        with col_img:
            st.markdown('<div class="image-frame">', unsafe_allow_html=True)
            st.image(annotated, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_results:
            boxes = results[0].boxes
            
            if len(boxes) > 0:
                st.markdown(f"""
                    <div class="result-container">
                        <div class="result-content">
                            <h2 style="color: #e74c3c; margin-bottom: 1rem;">🎯 DETECTION RESULTS</h2>
                            <div style="font-size: 3rem; font-weight: 800; color: #27ae60; margin: 1rem 0;">
                                {len(boxes)}
                            </div>
                            <h3 style="color: #2c3e50;">Sign(s) Detected</h3>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<h3 style='color: white; text-align: center; margin: 2rem 0;'>🚦 Detected Signs</h3>", unsafe_allow_html=True)
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf_score = float(box.conf[0])
                    label = model.names[cls]
                    
                    st.markdown(f"""
                        <div class="sign-badge">
                            {label}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class="confidence-display">
                            ✅ Confidence: {conf_score*100:.1f}%
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="result-container">
                        <div class="result-content">
                            <h2 style="color: #e67e22;">⚠️ NO SIGNS DETECTED</h2>
                            <p style="color: #7f8c8d; margin-top: 1rem;">
                                No road signs found in this image
                            </p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="road-footer">
        <div style="font-size: 2rem; margin-bottom: 1rem;">🚗 🚦 🛑</div>
        <h3 style="color: white; margin-bottom: 0.5rem;">Road Sign Detection System</h3>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.9rem;">
            Keeping roads safer with AI-powered sign recognition
        </p>
    </div>
""", unsafe_allow_html=True)