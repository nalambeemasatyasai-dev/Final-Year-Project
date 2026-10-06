# -*- coding: utf-8 -*-
"""
🎓 Student Engagement Recognition AI — Interactive Streamlit Dashboard.
Supports:
- Uploading images of arbitrary sizes and dimensions.
- Real-time MediaPipe face detection & robust square cropping with Lanczos sinc resampling.
- Multi-model selection (Dual-Branch ResNet/EfficientNet, Image-Only ResNet/EfficientNet).
- Automatic archiving of test images and prediction results into 'Testing_Results/<test_session>/'.
- Real-time webcam snapshot testing.
- Past test history viewer and benchmark comparison.
"""

import os
import sys
import glob
import json
import time
import pandas as pd
from PIL import Image
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from inference_engine import (
    crop_face_and_extract_blendshapes,
    predict_engagement_level,
    save_test_session,
    MODEL_CONFIGS,
    CLASS_DESCRIPTIONS,
    TESTING_RESULTS_DIR
)

# Page configuration
st.set_page_config(
    page_title="Student Engagement AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    
    .badge-high {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.5rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 2px solid #10b981;
    }
    
    .badge-nominal {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 0.5rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 2px solid #3b82f6;
    }
    
    .badge-disengaged {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.5rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        border: 2px solid #ef4444;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    .storage-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 2.2rem; font-weight: 800;">🎓 Student Engagement Recognition AI</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.05rem; opacity: 0.95;">
        Real-time Multimodal Facial Expression Analysis — Google MediaPipe 52 FACS Muscle Blendshapes + Deep Learning
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Model Configuration")
    selected_model = st.selectbox(
        "Select Deep Learning Model",
        options=list(MODEL_CONFIGS.keys()),
        index=0
    )
    
    model_info = MODEL_CONFIGS[selected_model]
    st.info(f"**Architecture**: {model_info['description']}")
    
    st.divider()
    st.subheader("🔍 Preprocessing Settings")
    crop_margin = st.slider("Face Crop Margin (Padding)", min_value=0.10, max_value=0.60, value=0.20, step=0.05,
                            help="Extra margin around the detected face landmarks to capture forehead, chin, and hair context.")
    
    st.divider()
    st.subheader("📁 Output Directory")
    st.caption(f"All tested images and outputs are saved in:")
    st.code(TESTING_RESULTS_DIR, language="text")

# Tabs
tab1, tab2, tab3 = st.tabs(["📸 Test New Images (Upload & Predict)", "📁 Previous Test History", "📊 Model Benchmark & Ablations"])

# TAB 1: Test New Images
with tab1:
    st.subheader("Upload Any Photo for Real-Time Engagement Analysis")
    st.write("Upload an image of **any dimensions, aspect ratio, or resolution**. The engine will automatically detect the face, perform a square crop, extract 52 FACS muscle action units, and predict the engagement level.")
    
    input_method = st.radio("Choose Input Source:", ["Upload Local Image", "Use Webcam Camera", "Try Sample Benchmark Image"], horizontal=True)
    
    uploaded_image = None
    image_filename = "test_image.jpg"
    
    if input_method == "Upload Local Image":
        file_buffer = st.file_uploader("Choose an image (JPG, PNG, JPEG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
        if file_buffer is not None:
            uploaded_image = Image.open(file_buffer)
            image_filename = file_buffer.name
            
    elif input_method == "Use Webcam Camera":
        camera_buffer = st.camera_input("Take a snapshot of a student")
        if camera_buffer is not None:
            uploaded_image = Image.open(camera_buffer)
            image_filename = f"webcam_snapshot_{time.strftime('%H%M%S')}.jpg"
            
    elif input_method == "Try Sample Benchmark Image":
        c_col1, c_col2 = st.columns([1, 1.5])
        with c_col1:
            class_choice = st.selectbox(
                "Select Engagement Category:",
                [
                    "🟢 Class 2: Highly Engaged",
                    "🔵 Class 1: Nominally Engaged",
                    "🟠 Class 0: Disengaged"
                ]
            )
        
        if "Class 2" in class_choice:
            sample_options = {
                "Variation A: High Concentration & Active Response (img11_111.jpg)": os.path.join(BASE_DIR, "WACV data", "2", "img11_111.jpg"),
                "Variation B: Intense Focus & Eye Contact (img11_106.jpg)": os.path.join(BASE_DIR, "WACV data", "2", "img11_106.jpg"),
                "Variation C: Active Classroom Engagement (img10_16.jpg)": os.path.join(BASE_DIR, "WACV data", "2", "img10_16.jpg"),
                "Variation D: Direct Interactive Focus (img10_21.jpg)": os.path.join(BASE_DIR, "WACV data", "2", "img10_21.jpg"),
            }
        elif "Class 1" in class_choice:
            sample_options = {
                "Variation A: Normal Screen Viewing (img10_101.jpg)": os.path.join(BASE_DIR, "WACV data", "1", "img10_101.jpg"),
                "Variation B: Attentive Lecture Listening (img10_102.jpg)": os.path.join(BASE_DIR, "WACV data", "1", "img10_102.jpg"),
                "Variation C: Moderate Attention / Note Taking (img10_10.jpg)": os.path.join(BASE_DIR, "WACV data", "1", "img10_10.jpg"),
            }
        else:
            sample_options = {
                "Variation A: Looking Away / Distracted (img10_100.jpg)": os.path.join(BASE_DIR, "WACV data", "0", "img10_100.jpg"),
                "Variation B: Head Turned Away (img10_14.jpg)": os.path.join(BASE_DIR, "WACV data", "0", "img10_14.jpg"),
                "Variation C: Inattentive / Distracted Gaze (img10_34.jpg)": os.path.join(BASE_DIR, "WACV data", "0", "img10_34.jpg"),
            }
        
        with c_col2:
            sample_choice = st.selectbox("Select Image Variation to Test:", list(sample_options.keys()))
        sample_path = sample_options[sample_choice]
        if os.path.exists(sample_path):
            uploaded_image = Image.open(sample_path)
            image_filename = os.path.basename(sample_path)
        else:
            st.error(f"Sample benchmark image not found at: {sample_path}")

    # Process and Predict
    if uploaded_image is not None:
        st.divider()
        col_img1, col_img2 = st.columns([1, 1])
        
        with col_img1:
            st.markdown("#### 1. Original Uploaded Image")
            st.caption(f"Dimensions: {uploaded_image.size[0]} × {uploaded_image.size[1]} px")
            
        with col_img2:
            st.markdown("#### 2. Detected & Cropped Face (224×224)")
            st.caption("Standardized with Lanczos Sinc Interpolation")

        with st.spinner("Detecting face, extracting 52 FACS blendshapes, and running neural network..."):
            cropped_pil, annotated_pil, blendshapes_np, face_detected, bbox = crop_face_and_extract_blendshapes(
                uploaded_image,
                target_size=(224, 224),
                margin=crop_margin
            )
            
            pred_class, class_label, confidence, probs_dict, top_facs = predict_engagement_level(
                cropped_pil,
                blendshapes_np,
                model_key=selected_model
            )

            # Save test session into Testing_Results/
            session_dir, result_data = save_test_session(
                original_pil=uploaded_image,
                cropped_pil=cropped_pil,
                annotated_pil=annotated_pil,
                blendshapes_np=blendshapes_np,
                pred_class=pred_class,
                class_label=class_label,
                confidence=confidence,
                probs_dict=probs_dict,
                top_facs=top_facs,
                model_key=selected_model,
                filename=image_filename
            )

        with col_img1:
            st.image(annotated_pil, use_container_width=True)
            if face_detected:
                st.success("✅ MediaPipe Face Mesh successfully detected.")
            else:
                st.warning("⚠️ Face landmarks not detected with high confidence; used central crop fallback.")

        with col_img2:
            st.image(cropped_pil, width=224)

        # Prediction Results Card
        st.divider()
        st.subheader("🎯 Prediction Results & Confidence Analysis")
        
        res_col1, res_col2 = st.columns([1.2, 1])
        
        with res_col1:
            st.markdown("##### Predicted Engagement State")
            total_attentive = (probs_dict["Nominally Engaged (1)"] + probs_dict["Highly Engaged (2)"]) * 100.0
            
            if pred_class == 2:
                badge_html = f'<div class="badge-high">🟢 Highly Engaged (Class 2) — {confidence:.1f}% Confidence</div>'
            elif pred_class == 1:
                badge_html = f'<div class="badge-nominal">🔵 Nominally Engaged (Class 1) — {confidence:.1f}% Confidence</div>'
            else:
                badge_html = f'<div class="badge-disengaged">🟠 Disengaged (Class 0) — {confidence:.1f}% Confidence</div>'
            st.markdown(badge_html, unsafe_allow_html=True)
            st.caption(CLASS_DESCRIPTIONS[pred_class])
            
            # Overall Attentiveness Gauge
            st.markdown(f"""
            <div style="background-color: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; padding: 10px 14px; border-radius: 6px; margin: 10px 0;">
                <span style="font-size: 15px; font-weight: 600; color: #10b981;">Total Attentiveness: {total_attentive:.1f}%</span><br>
                <span style="font-size: 12px; color: #94a3b8;">
                    Combined attentive probability (Screen Listening: {probs_dict['Nominally Engaged (1)']*100:.1f}% + Active: {probs_dict['Highly Engaged (2)']*100:.1f}%)
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### Class Probability Distribution")
            for c_name, p_val in probs_dict.items():
                st.write(f"**{c_name}**: `{p_val * 100:.1f}%`")
                st.progress(float(p_val))

        with res_col2:
            st.markdown("##### 🧬 Top Active Facial Action Units (MediaPipe FACS)")
            st.caption("Active muscle contractions extracted from 3D facial landmarks:")
            
            df_facs = pd.DataFrame(top_facs)
            st.dataframe(df_facs.rename(columns={"name": "FACS Action Unit", "score": "Activation Score"}), use_container_width=True, hide_index=True)

        # File Storage Notification Card
        st.divider()
        st.markdown(f"""
        <div class="storage-box">
            <b>📁 Saved to Dedicated Test Directory:</b><br>
            <code>{session_dir}</code><br><br>
            <b>Saved Files in This Folder:</b><br>
            • <code>original_{os.path.splitext(os.path.basename(image_filename))[0]}.jpg</code> (Original image)<br>
            • <code>cropped_face_224.jpg</code> (Processed 224x224 input crop)<br>
            • <code>annotated_detection.jpg</code> (Visual detection with bounding box)<br>
            • <code>facs_blendshapes.csv</code> (All 52 FACS muscle action units)<br>
            • <code>prediction_result.json</code> (Machine-readable results)<br>
            • <code>report.txt</code> (Human-readable summary report)
        </div>
        """, unsafe_allow_html=True)
        
        with open(os.path.join(session_dir, "report.txt"), "r") as f:
            report_text = f.read()
            st.download_button("📥 Download Prediction Report (.txt)", data=report_text, file_name=f"engagement_report_{image_filename}.txt")


# TAB 2: Previous Test History
with tab2:
    st.subheader("📁 Previous Test Uploads & History")
    history_csv = os.path.join(TESTING_RESULTS_DIR, "all_test_history.csv")
    
    if os.path.exists(history_csv):
        df_history = pd.read_csv(history_csv)
        st.write(f"Total Tests Recorded: **{len(df_history)}**")
        st.dataframe(df_history, use_container_width=True)
        
        st.divider()
        st.markdown("#### Inspect Specific Test Session")
        session_folders = [f for f in os.listdir(TESTING_RESULTS_DIR) if os.path.isdir(os.path.join(TESTING_RESULTS_DIR, f))]
        if session_folders:
            chosen_session = st.selectbox("Select Past Test Session:", sorted(session_folders, reverse=True))
            sess_path = os.path.join(TESTING_RESULTS_DIR, chosen_session)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                orig_files = glob.glob(os.path.join(sess_path, "original_*.jpg"))
                if orig_files:
                    st.image(orig_files[0], caption="Original Image")
            with c2:
                crop_file = os.path.join(sess_path, "cropped_face_224.jpg")
                if os.path.exists(crop_file):
                    st.image(crop_file, caption="Cropped 224x224 Input")
            with c3:
                ann_file = os.path.join(sess_path, "annotated_detection.jpg")
                if os.path.exists(ann_file):
                    st.image(ann_file, caption="Detection Box")
                    
            json_file = os.path.join(sess_path, "prediction_result.json")
            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    st.json(json.load(f))
    else:
        st.info("No test history recorded yet. Upload an image in Tab 1 to generate your first test session!")


# TAB 3: Model Benchmark & Ablation Study
with tab3:
    st.subheader("📊 Deep Learning Benchmark & Ablation Study")
    st.write("Comparison of Dual-Branch Multimodal Fusion vs. Image-Only baselines on the 442 unseen holdout test split:")
    
    dl_eval_csv = os.path.join(BASE_DIR, "Results", "DL", "dl_models_evaluation.csv")
    if os.path.exists(dl_eval_csv):
        df_dl = pd.read_csv(dl_eval_csv, index_col=0)
        # Format percentages
        df_display = df_dl.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1"]:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda v: f"{float(v)*100:.2f}%")
        st.dataframe(df_display, use_container_width=True)
    
    st.divider()
    st.subheader("📈 Confusion Matrices & Training Loss Curves")
    cm_cols = st.columns(2)
    with cm_cols[0]:
        cm_dual = os.path.join(BASE_DIR, "Results", "DL", "confusion_matrix_resnet.png")
        if os.path.exists(cm_dual):
            st.image(cm_dual, caption="Dual-Branch ResNet-18 Confusion Matrix")
    with cm_cols[1]:
        cm_img = os.path.join(BASE_DIR, "Results", "DL", "confusion_matrix_resnet_image_only.png")
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Image-Only ResNet-18 Confusion Matrix")
            
    tc_cols = st.columns(2)
    with tc_cols[0]:
        tc_dual = os.path.join(BASE_DIR, "Results", "DL", "training_curves_resnet.png")
        if os.path.exists(tc_dual):
            st.image(tc_dual, caption="Dual-Branch ResNet-18 Training Curves")
    with tc_cols[1]:
        tc_img = os.path.join(BASE_DIR, "Results", "DL", "training_curves_resnet_image_only.png")
        if os.path.exists(tc_img):
            st.image(tc_img, caption="Image-Only ResNet-18 Training Curves")
