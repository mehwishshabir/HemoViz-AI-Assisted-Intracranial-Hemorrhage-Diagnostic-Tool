import os
import io
import base64
import gc
import json
import tempfile
import subprocess
import numpy as np
import nibabel as nib
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go
from PIL import Image
from jinja2 import Template
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.exposure import match_histograms
from skimage.transform import resize
from skimage.measure import marching_cubes
from datetime import datetime
import dicom_utils
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib import colors as rl_colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ====================== CONFIGURATION ====================== #
st.set_page_config(page_title="HemoViz", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Paths
CLS_MODEL_PATH = os.path.join(MODEL_DIR, "3d_Hemo_get_modelK7_classification.keras")
REF_NORMAL_PATH = os.path.join(BASE_DIR, "reference_normal.nii.gz")
REF_HEMO_PATH = os.path.join(BASE_DIR, "reference_hemo.nii.gz")
SEG_WORKER_PATH = os.path.join(BASE_DIR, "segmentation_worker.py")
LOGO_PATH = os.path.join(BASE_DIR, "hemo_logo_wordmark2.png")

# Constants
CLS_TARGET_SHAPE = (128, 128, 64)
SEG_TARGET_SHAPE = (256, 256, 32)
CLASS_NAMES = ["Background", "Intra-Axial", "Extra-Axial"]
THRESHOLD = 0.5

# Environment for Segmentation (neuroscan)
CONDA_ACTIVATE = r"C:\Users\user\miniconda3\Scripts\activate.bat"
SEG_ENV_NAME = "neuroscan"

# ====================== STYLING ====================== #
def load_css():
    css_path = os.path.join(ASSETS_DIR, "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback CSS
        st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #f0f4f8 0%, #e8f0f7 100%); color: #111827; }
        .premium-card { background: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e5e7eb; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
        .premium-header { background: linear-gradient(135deg, #106091 0%, #0d4d75 100%); padding: 16px 24px; border-radius: 12px 12px 0 0; font-weight: 700; color: #ffffff; margin-bottom: 20px; }
        .logo-subtitle { color: #D54A61; font-weight: 800; font-size: 1.2rem; margin-top: 10px; text-transform: uppercase; letter-spacing: 0.05em; }
        </style>
        """, unsafe_allow_html=True)

    # Reduce top padding
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    margin-top: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

# ====================== UTILS: CLASSIFICATION ====================== #
@st.cache_resource
def load_classification_model():
    return tf.keras.models.load_model(CLS_MODEL_PATH)

@st.cache_resource
def load_reference(path):
    ref = nib.load(path).get_fdata()
    ref = np.clip(ref, 0, 120) / 120.0
    ref_resized = resize(ref, CLS_TARGET_SHAPE, preserve_range=True, mode="constant")
    return ref_resized.astype(np.float32)

def resize_volume(img, target_shape):
    return resize(img, target_shape, mode="constant", preserve_range=True)

def window_ct(volume, level=60, width=120):
    lower, upper = level - (width / 2), level + (width / 2)
    volume = np.clip(volume, lower, upper)
    return ((volume - lower) / (upper - lower)).astype(np.float32)

def preprocess_classification(volume, ref_normal, ref_hemo):
    resized = resize_volume(volume, CLS_TARGET_SHAPE)
    windowed = window_ct(resized)
    
    # Histogram matching
    norm_matched = match_histograms(windowed, ref_normal)
    hemo_matched = match_histograms(windowed, ref_hemo)
    
    err_normal = np.mean((windowed - ref_normal) ** 2)
    err_hemo = np.mean((windowed - ref_hemo) ** 2)
    
    matched = norm_matched if err_normal < err_hemo else hemo_matched
    model_input = matched[..., np.newaxis][np.newaxis, ...]
    return model_input.astype(np.float32), windowed

def enhance_slice(slice_2d):
    slice_2d = np.clip(slice_2d, 0, 1)
    return (slice_2d * 255).astype(np.uint8)

def volume_to_base64_slices(volume):
    slices = []
    for i in range(volume.shape[2]):
        img = Image.fromarray(enhance_slice(volume[:, :, i]))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        slices.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return slices

# ====================== UTILS: SEGMENTATION ====================== #
def run_segmentation_subprocess(input_path):
    """
    Runs segmentation worker and returns (mask_path, classification_probs).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_out:
        output_path = tmp_out.name
    
    cmd = f'call "{CONDA_ACTIVATE}" {SEG_ENV_NAME} && python "{SEG_WORKER_PATH}" --input "{input_path}" --output "{output_path}"'
    
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    probs = None
    # Parse stdout for CLF_PROBS_JSON
    for line in stdout.splitlines():
        if "CLF_PROBS_JSON:" in line:
            try:
                json_str = line.split("CLF_PROBS_JSON:")[1]
                probs = json.loads(json_str)
            except:
                pass
    
    if process.returncode != 0:
        st.error(f"Segmentation failed:\n{stderr}")
        return None, None
    
    return output_path, probs

def normalize_slice(ct_slice):
    if ct_slice.max() > ct_slice.min():
        norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())
    else:
        norm = np.zeros_like(ct_slice)
    return (norm * 255).astype(np.uint8)

def prepare_segmentation_slices(vol, seg):
    intra_color = np.array([255, 0, 0], dtype=np.uint8)       # red
    extra_color = np.array([255, 200, 0], dtype=np.uint8)     # yellow/orange

    orig_slices, overlay_slices, mask_slices = [], [], []
    n_slices = seg.shape[2]

    for i in range(n_slices):
        ct = vol[:, :, i]
        mask = seg[:, :, i]

        ct_rgb = np.stack([normalize_slice(ct)] * 3, axis=-1)

        intra_mask = (mask & 1) > 0
        extra_mask = (mask & 2) > 0
        
        both = intra_mask & extra_mask
        intra_mask = intra_mask & ~both
        extra_mask = extra_mask | both

        overlay = ct_rgb.copy().astype(np.float32)
        overlay[intra_mask] = 0.5 * overlay[intra_mask] + 0.5 * intra_color
        overlay[extra_mask] = 0.5 * overlay[extra_mask] + 0.5 * extra_color
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        mask_only = np.zeros_like(ct_rgb)
        mask_only[mask > 0] = np.array([255, 255, 255], dtype=np.uint8)

        orig_slices.append(ct_rgb)
        overlay_slices.append(overlay)
        mask_slices.append(mask_only)

    return orig_slices, overlay_slices, mask_slices

def slices_to_base64(slices):
    b64 = []
    for img in slices:
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        b64.append(base64.b64encode(buf.getvalue()).decode())
    return b64

def plot_mask_3d(seg_mask):
    xs, ys, zs = np.where(seg_mask > 0)
    if len(xs) == 0: return None
    
    # Map colors: 1 (Intra) -> Red, 2 (Extra) -> Gold, 3 (Both) -> Orange
    vals = seg_mask[xs, ys, zs]
    colors = []
    for v in vals:
        if v == 1: colors.append('red')
        elif v == 2: colors.append('gold')
        else: colors.append('orange')
        
    fig = go.Figure(go.Scatter3d(
        x=xs, y=ys, z=zs, 
        mode="markers", 
        marker=dict(size=2, color=colors)
    ))
    fig.update_layout(scene=dict(aspectmode="data"), height=450, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='#ffffff', plot_bgcolor='#ffffff', font=dict(color='#1f2937'))
    return fig

def plot_full_3d_mesh(volume, seg):
    mask_intra = (seg & 1).astype(np.uint8)
    mask_extra = (seg & 2).astype(np.uint8)
    if not (np.any(mask_intra) or np.any(mask_extra)): return None

    vol_norm = (volume - np.min(volume)) / max(np.max(volume) - np.min(volume), 1e-5)
    vol_smooth = gaussian_filter(vol_norm, sigma=1)
    
    try:
        verts_brain, faces_brain, _, _ = marching_cubes(vol_smooth, level=0.25)
    except:
        verts_brain, faces_brain = np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    fig = go.Figure()
    if verts_brain.shape[0] > 0:
        fig.add_trace(go.Mesh3d(x=verts_brain[:,2], y=verts_brain[:,1], z=verts_brain[:,0],
                                i=faces_brain[:,0], j=faces_brain[:,1], k=faces_brain[:,2],
                                color="gray", opacity=0.1, name="Brain"))


    for mask, color, name in zip([mask_intra, mask_extra], ["red", "gold"], ["Intra-Axial", "Extra-Axial"]):
        if np.any(mask):
            try:
                verts_h, faces_h, _, _ = marching_cubes(mask, level=0.5)
                fig.add_trace(go.Mesh3d(x=verts_h[:,2], y=verts_h[:,1], z=verts_h[:,0],
                                        i=faces_h[:,0], j=faces_h[:,1], k=faces_h[:,2],
                                        color=color, opacity=0.5, name=name))
            except: pass
            
    fig.update_layout(scene=dict(aspectmode="data"), height=450, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='#ffffff', plot_bgcolor='#ffffff', font=dict(color='#1f2937'))
    return fig

# ====================== CLINICAL TOOLS: PATIENT INFO & GCS ====================== #
def render_patient_info():
    st.markdown("### 🏥 Patient Information")
    with st.container():
        st.markdown("""
        <style>
        .patient-info-box {
            background: #ffffff;
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            c1, c2, c3 = st.columns(3)
            with c1:
                st.text_input("Patient Name", placeholder="e.g. John Doe", key="p_name")
                st.number_input("Age", min_value=0, max_value=120, step=1, key="p_age")
            with c2:
                st.selectbox("Gender", ["Male", "Female", "Other"], key="p_gender")
                st.text_input("Blood Pressure", placeholder="e.g. 120/80 mmHg", key="p_bp")
            with c3:
                st.text_input("Rater Name", placeholder="Dr. Smith", key="p_rater")
                st.date_input("Date", datetime.now(), key="p_date")
    st.markdown("---")

def generate_report():
    # Gather Data
    p_name = st.session_state.get("p_name", "N/A")
    p_age = st.session_state.get("p_age", "N/A")
    p_gender = st.session_state.get("p_gender", "N/A")
    p_bp = st.session_state.get("p_bp", "N/A")
    p_rater = st.session_state.get("p_rater", "N/A")
    p_date = st.session_state.get("p_date", datetime.now()).strftime("%Y-%m-%d")
    
    # GCS
    gcs_eye = st.session_state.get("gcs_eye", 0)
    gcs_verbal = st.session_state.get("gcs_verbal", 0)
    gcs_motor = st.session_state.get("gcs_motor", 0)
    gcs_total = gcs_eye + gcs_verbal + gcs_motor
    
    # Analysis
    results = st.session_state.get("analysis_results", {})
    vol_intra = results.get("vol_intra", 0.0)
    vol_extra = results.get("vol_extra", 0.0)
    has_intra = vol_intra > 0
    has_extra = vol_extra > 0
    
    report = f"""==================================================
       HEMOVIZ - INTRACRANIAL HEMORRHAGE REPORT
==================================================

[PATIENT INFORMATION]
Name:           {p_name}
Age:            {p_age}
Gender:         {p_gender}
Blood Pressure: {p_bp}
Date:           {p_date}
Rater:          {p_rater}

--------------------------------------------------
[GLASGOW COMA SCALE (GCS)]
Total Score:    {gcs_total} / 15

Breakdown:
- Eye Response:    {gcs_eye}
- Verbal Response: {gcs_verbal}
- Motor Response:  {gcs_motor}

--------------------------------------------------
[AI ANALYSIS RESULTS]

1. Intra-Axial Hemorrhage:
   Status: {"DETECTED" if has_intra else "Not Detected"}
   Volume: {vol_intra:.2f} mL

2. Extra-Axial Hemorrhage:
   Status: {"DETECTED" if has_extra else "Not Detected"}
   Volume: {vol_extra:.2f} mL

==================================================
Generated by HemoViz AI Diagnostic Tool
"""
    return report

def generate_pdf_report():
    """Generate a comprehensive PDF report of the analysis"""
    # Gather Data
    p_name = st.session_state.get("p_name", "N/A")
    p_age = st.session_state.get("p_age", "N/A")
    p_gender = st.session_state.get("p_gender", "N/A")
    p_bp = st.session_state.get("p_bp", "N/A")
    p_rater = st.session_state.get("p_rater", "N/A")
    p_date = st.session_state.get("p_date", datetime.now()).strftime("%Y-%m-%d")
    
    # GCS
    gcs_eye = st.session_state.get("gcs_eye", 0)
    gcs_verbal = st.session_state.get("gcs_verbal", 0)
    gcs_motor = st.session_state.get("gcs_motor", 0)
    gcs_total = gcs_eye + gcs_verbal + gcs_motor
    
    # Analysis
    results = st.session_state.get("analysis_results", {})
    vol_intra = results.get("vol_intra", 0.0)
    vol_extra = results.get("vol_extra", 0.0)
    has_intra = vol_intra > 0
    has_extra = vol_extra > 0
    
    # Create PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=rl_colors.HexColor('#106091'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=rl_colors.HexColor('#D54A61'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # Title
    story.append(Paragraph("HemoViz - Intracranial Hemorrhage Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Patient Information
    story.append(Paragraph("Patient Information", heading_style))
    patient_data = [
        ['Name:', p_name, 'Age:', str(p_age)],
        ['Gender:', p_gender, 'Blood Pressure:', p_bp],
        ['Date:', p_date, 'Rater:', p_rater]
    ]
    patient_table = Table(patient_data, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), rl_colors.HexColor('#f3f4f6')),
        ('BACKGROUND', (2, 0), (2, -1), rl_colors.HexColor('#f3f4f6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), rl_colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, rl_colors.grey)
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.3*inch))
    
    # GCS Assessment
    story.append(Paragraph("Glasgow Coma Scale (GCS) Assessment", heading_style))
    gcs_severity = "Severe" if gcs_total <= 8 else "Moderate" if gcs_total <= 12 else "Mild"
    gcs_data = [
        ['Component', 'Score'],
        ['Eye Response', str(gcs_eye)],
        ['Verbal Response', str(gcs_verbal)],
        ['Motor Response', str(gcs_motor)],
        ['Total Score', f"{gcs_total} / 15"],
        ['Severity', gcs_severity]
    ]
    gcs_table = Table(gcs_data, colWidths=[3*inch, 2*inch])
    gcs_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#106091')),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, rl_colors.black)
    ]))
    story.append(gcs_table)
    story.append(Spacer(1, 0.3*inch))
    
    # AI Analysis Results
    story.append(Paragraph("AI Analysis Results", heading_style))
    analysis_data = [
        ['Hemorrhage Type', 'Status', 'Volume (mL)'],
        ['Intra-Axial', 'DETECTED' if has_intra else 'Not Detected', f"{vol_intra:.2f}"],
        ['Extra-Axial', 'DETECTED' if has_extra else 'Not Detected', f"{vol_extra:.2f}"]
    ]
    analysis_table = Table(analysis_data, colWidths=[2*inch, 2*inch, 1.5*inch])
    analysis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), rl_colors.HexColor('#106091')),
        ('TEXTCOLOR', (0, 0), (-1, 0), rl_colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (-1, -1), rl_colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, rl_colors.black)
    ]))
    story.append(analysis_table)
    story.append(Spacer(1, 0.4*inch))
    
    # Footer
    footer_text = f"<para align=center><font size=9 color='grey'>Generated by HemoViz AI-Assisted Diagnostic Tool on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</font></para>"
    story.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def render_gcs_calculator():

    st.markdown("<div class='premium-header'>📋 GLASGOW COMA SCALE (GCS) - Consciousness Assessment</div>", unsafe_allow_html=True)
    
    # Grid Layout: 2 Rows x 2 Columns
    
    # --- ROW 1: Eye & Verbal ---
    r1_c1, r1_c2 = st.columns(2)
    
    with r1_c1:
        st.markdown("**👁️ EYE OPENING**")
        eye_opts = {
            4: "4 - Spontaneous (Eyes open, not necessarily aware)",
            3: "3 - To speech (Non-specific response, not necessarily to command)",
            2: "2 - To pain (Pain from sternum/limb/supra-orbital pressure)",
            1: "1 - None (Even to supra-orbital pressure)"
        }
        eye_score = st.radio("Select Eye Response", list(eye_opts.keys()), format_func=lambda x: eye_opts[x], key="gcs_eye")
        
    with r1_c2:
        st.markdown("**🗣️ VERBAL RESPONSE**")
        verbal_opts = {
            5: "5 - Oriented (Converses and oriented)",
            4: "4 - Confused (Converses but confused, disoriented)",
            3: "3 - Inappropriate (Intelligible, no sustained sentences)",
            2: "2 - Incomprehensible (Moans/groans, no speech)",
            1: "1 - None (No verbalization of any type)"
        }
        verbal_score = st.radio("Select Verbal Response", list(verbal_opts.keys()), format_func=lambda x: verbal_opts[x], key="gcs_verbal")

    st.markdown("---")

    # --- ROW 2: Motor & Result ---
    r2_c1, r2_c2 = st.columns(2)
    
    with r2_c1:
        st.markdown("**💪 MOTOR RESPONSE**")
        motor_opts = {
            6: "6 - Obeys commands (Follows simple commands)",
            5: "5 - Localizes pain (Arm attempts to remove supra-orbital/chest pressure)",
            4: "4 - Withdrawal (Arm withdraws to pain, shoulder abducts)",
            3: "3 - Flexor response (Withdrawal response or assumption of hemiplegic posture)",
            2: "2 - Extension (Shoulder adducted and shoulder and forearm internally rotated)",
            1: "1 - None (To any pain; limbs remain flaccid)"
        }
        motor_score = st.radio("Select Motor Response", list(motor_opts.keys()), format_func=lambda x: motor_opts[x], key="gcs_motor")
        
    with r2_c2:
        # Calculate Score
        total_score = eye_score + motor_score + verbal_score
        
        # Determine Severity
        severity = "Severe Head Injury" if total_score <= 8 else "Moderate Head Injury" if total_score <= 12 else "Mild Head Injury"
        color = "red" if total_score <= 8 else "orange" if total_score <= 12 else "green"
        
        st.markdown("**📊 ASSESSMENT RESULT**")
        st.markdown(f"""
        <div style='text-align:center; background: #f9fafb; padding:20px; border-radius:12px; border:1px solid #e5e7eb; margin-top: 5px; box-shadow: inset 0 1px 2px 0 rgba(0, 0, 0, 0.05);'>
            <h1 style='margin:0; color: #111827; font-size: 3.5rem; border:none;'>{total_score}</h1>
            <p style='margin:0; color:#6b7280; font-size: 1rem; font-weight: 700; letter-spacing: 0.05em;'>TOTAL SCORE (3-15)</p>
            <hr style='margin: 15px 0; border-color: #e5e7eb;'>
            <h3 style='margin:0; color:{color}; font-weight: 800;'>{severity}</h3>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)



# ====================== MAIN APP ====================== #
def main():
    # Initialize DICOM environment
    dicom_utils.setup_dcm2niix()
    
    load_css()
    
    # Header with Logo (Centered)
    st.markdown("""
    <div style='text-align: center; padding: 0px;'>
    """, unsafe_allow_html=True)
    
    # Prepare Workflow Image (if exists)
    workflow_html = ""
    workflow_path = os.path.join(ASSETS_DIR, "workflow_diagram.png")
    if os.path.exists(workflow_path):
        with open(workflow_path, "rb") as f:
            wf_data = base64.b64encode(f.read()).decode("utf-8")
        workflow_html = f'<img src="data:image/png;base64,{wf_data}" style="max-width: 800px; width: 100%; margin: 20px 0; border-radius: 8px; border: 1px solid #e5e7eb;">'

    if os.path.exists(LOGO_PATH):
        # Read logo and encode to base64 for embedding in centered HTML div
        with open(LOGO_PATH, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{data}" width="300" style="margin-bottom: 5px;">
            <p class="logo-subtitle">AI-Assisted Intracranial Hemorrhage Diagnostic Tool</p>
            {workflow_html}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="logo-container">
            <h1 style='font-size: 4rem; margin: 0; border:none; color: #111827;'>🏥 HemoViz</h1>
            <p class="logo-subtitle">AI-Assisted Intracranial Hemorrhage Tool</p>
            {workflow_html}
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # 1. Patient Info & GCS Assessment
    render_patient_info()
    render_gcs_calculator()
    
    # 2. Upload Section
    st.markdown("### 📤 CT Scan Analysis")
    
    # --- RESET BUTTON ---
    col_upload, col_reset = st.columns([3, 1])
    with col_reset:
        if st.button("🔄 Start New Analysis", type="secondary", use_container_width=True):
            # Clear specific session state keys
            keys_to_clear = ["files_hash", "extract_dir", "series_dict", "nifti_path", "selected_uid"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Increment uploader key to reset the widget
            if "uploader_key" not in st.session_state:
                st.session_state["uploader_key"] = 0
            st.session_state["uploader_key"] += 1
            st.rerun()

    with col_upload:
        st.markdown("""
        **Upload Options:**
        1. **Drag & Drop a Folder:** Select all files in your DICOM folder (Ctrl+A) and drag them here.
        2. **Upload a ZIP:** Upload a single ZIP file containing the DICOM folder.
        3. **Upload NIfTI:** Upload a pre-processed `.nii` or `.nii.gz` file.
        """)
    
    # Initialize uploader key if not present
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
        
    uploaded_files = st.file_uploader(
        "Upload Files", 
        type=["nii", "nii.gz", "zip", "dcm", "ima"], 
        accept_multiple_files=True,
        key=f"uploader_{st.session_state['uploader_key']}"
    )

    if uploaded_files:
        # Check if new files are uploaded (reset state if needed)
        # We use a simple hash of filenames to detect change
        current_files_hash = hash(tuple(f.name for f in uploaded_files))
        if "files_hash" not in st.session_state or st.session_state["files_hash"] != current_files_hash:
            st.session_state["files_hash"] = current_files_hash
            st.session_state["extract_dir"] = None
            st.session_state["series_dict"] = None
            st.session_state["nifti_path"] = None
            st.session_state["selected_uid"] = None
            
        tmp_path = st.session_state.get("nifti_path")
        
        # If we don't have a NIfTI path yet, we need to process
        if not tmp_path:
            
            # 1. Handle Single NIfTI
            if len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith((".nii", ".nii.gz")):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
                    tmp.write(uploaded_files[0].read())
                    st.session_state["nifti_path"] = tmp.name
                    st.rerun()
            
            # 2. Handle DICOM (ZIP or Folder)
            else:
                # Extraction Step (Only if not done)
                if not st.session_state.get("extract_dir"):
                    extract_dir = tempfile.mkdtemp()
                    
                    with st.spinner("📦 Processing Uploaded Files..."):
                        if len(uploaded_files) == 1 and uploaded_files[0].name.lower().endswith(".zip"):
                            # ZIP Upload
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                                tmp_zip.write(uploaded_files[0].read())
                                dicom_utils.extract_zip(tmp_zip.name, extract_dir)
                        else:
                            # Folder Upload (Multiple Files)
                            for f in uploaded_files:
                                with open(os.path.join(extract_dir, f.name), "wb") as out:
                                    out.write(f.read())
                                    
                    st.session_state["extract_dir"] = extract_dir
                    
                # Scanning Step (Only if not done)
                if not st.session_state.get("series_dict"):
                    with st.spinner("🔍 Scanning for DICOM Series..."):
                        series_dict = dicom_utils.find_dicom_series(st.session_state["extract_dir"])
                        if not series_dict:
                            st.error("No valid DICOM series found.")
                            st.stop()
                        st.session_state["series_dict"] = series_dict
                
                # Selection Step
                series_dict = st.session_state["series_dict"]
                extract_dir = st.session_state["extract_dir"]
                
                # --- SERIES SELECTION UI ---
                st.info(f"📂 Found {len(series_dict)} DICOM series. Please select the series to analyze.")
                
                # Prepare options
                series_uids = list(series_dict.keys())
                def format_func(uid):
                    info = series_dict[uid]
                    return f"{info['description']} | {info['modality']} | {info['count']} Slices | {info['dims']}"

                # Use session state for radio selection
                selected_uid = st.radio(
                    "Select a Series:", 
                    series_uids, 
                    format_func=format_func,
                    key="series_selection_radio"
                )
                
                if selected_uid:
                    sel = series_dict[selected_uid]
                    st.markdown(f"""
                    **Selected Series Details:**
                    - **Description:** {sel['description']}
                    - **Slices:** {sel['count']}
                    - **Dimensions:** {sel['dims']}
                    - **Modality:** {sel['modality']}
                    """)
                    
                    if st.button("🚀 Load and Analyze Series", type="primary"):
                        st.write("Converting DICOM to NIfTI...")
                        nifti_out = os.path.join(extract_dir, "converted.nii.gz")
                        success, message = dicom_utils.convert_dicom_to_nifti(sel['files'], nifti_out)
                        
                        if success:
                            st.session_state["nifti_path"] = nifti_out
                            st.rerun()
                        else:
                            st.error(f"Conversion Failed: {message}")
                            st.stop()
                
                # Stop here if we haven't converted yet
                if not st.session_state.get("nifti_path"):
                    st.stop()

        # If we are here, we have a nifti_path in session_state
        tmp_path = st.session_state["nifti_path"]
        
        # Load NIfTI (Common Path)
        try:
            nifti_img = nib.load(tmp_path)
            raw_volume = nifti_img.get_fdata()
        except Exception as e:
            st.error(f"Error loading NIfTI file: {e}")
            return

        # ---------------- STEP 1: CLASSIFICATION ---------------- #
        st.divider()
        st.subheader("1️⃣ Automated Hemorrhage Detection")
        
        with st.spinner("Processing and analyzing scan..."):
            # Load resources
            cls_model = load_classification_model()
            ref_normal = load_reference(REF_NORMAL_PATH)
            ref_hemo = load_reference(REF_HEMO_PATH)
            
            # Preprocess and Predict
            cls_input, cls_processed = preprocess_classification(raw_volume, ref_normal, ref_hemo)
            cls_pred = cls_model.predict(cls_input, verbose=0)[0][0]
            
            is_hemorrhage = cls_pred >= 0.5
            confidence = float(cls_pred) if is_hemorrhage else 1 - float(cls_pred)
            
            if is_hemorrhage:
                # HEMORRHAGE DETECTED: Hide scan, show warning
                st.error(f"⚠️ **HEMORRHAGE SUSPECTED** (Confidence: {confidence:.2%})")
                st.markdown("""
                <div style='background-color: #3d0c0c; padding: 15px; border-radius: 8px; border: 1px solid #ff4b4b;'>
                    <p style='color: #ffcccc; margin: 0;'>
                        <b>Clinical Note:</b> The system has detected patterns consistent with intracranial hemorrhage. 
                        Visual inspection is suppressed until segmentation analysis is performed.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Button to trigger Step 2
                st.markdown("###")
                start_seg = st.button("🚀 Proceed to Segmentation Analysis", type="primary", use_container_width=True)
                
            else:
                # NORMAL: Show scan with JS Viewer
                st.success(f"✅ **NORMAL SCAN** (Confidence: {confidence:.2%})")
                st.markdown("No hemorrhage detected. Displaying scan for verification.")
                
                slices = volume_to_base64_slices(cls_processed)
                
                html_template = """
                <div class="viewer-container" style="text-align:center;">
                    <h4 class="viewer-header">Non-Contrast CT Scan</h4>
                    <img id="clsSliceImage" src="data:image/png;base64,{{ slices[0] }}" style="width:300px; border-radius:12px; border:1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.3);"/>
                    
                    <div style="margin-top:20px;">
                        <input type="range" min="0" max="{{ slices|length - 1 }}" value="0" id="clsSliceSlider" style="width:80%; accent-color: #3b82f6;"/>
                        <p style="color:#cbd5e1; margin-top:10px; font-size:16px; font-weight: 500;">Slice: <span id="clsSliceIndex">1</span> / {{ slices|length }}</p>
                    </div>
                </div>

                <script>
                const clsSlices = {{ slices | safe }};
                const clsSlider = document.getElementById("clsSliceSlider");
                const clsImage = document.getElementById("clsSliceImage");
                const clsIndex = document.getElementById("clsSliceIndex");
                clsSlider.oninput = function() {
                    const i = parseInt(this.value);
                    clsImage.src = "data:image/png;base64," + clsSlices[i];
                    clsIndex.innerText = i + 1;
                };
                </script>
                """
                rendered_html = Template(html_template).render(
                    slices=slices
                )
                st.components.v1.html(rendered_html, height=600, scrolling=True)
                
                start_seg = False

        # ---------------- STEP 2: SEGMENTATION ---------------- #
        if is_hemorrhage and start_seg:
            st.divider()
            st.subheader("2️⃣ Volumetric Segmentation & Localization")
            
            with st.spinner("Initializing Segmentation Protocol..."):
                mask_path, probs = run_segmentation_subprocess(tmp_path)
                
                if mask_path and os.path.exists(mask_path):
                    # Load result
                    seg_img = nib.load(mask_path)
                    seg_combined = seg_img.get_fdata().astype(np.uint8)
                    
                    # Display Classification Probabilities (Intra vs Extra)
                    if probs:
                        st.markdown("#### 📊 Classification & Volumetrics")
                        # p_intra, p_extra = probs["intra"], probs["extra"] # Removed as per user request
                        vol_intra = probs.get("vol_intra_ml", 0.0)
                        vol_extra = probs.get("vol_extra_ml", 0.0)
                        
                        # Logic Change: Use Volume > 0 instead of Probability > Threshold
                        has_intra = vol_intra > 0
                        has_extra = vol_extra > 0
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            # st.metric("Intra-Axial Probability", f"{p_intra:.3f}") # Removed
                            if has_intra: 
                                st.success(f"**Intra-Axial Hemorrhage Detected**\n\n**Volume:** {vol_intra:.2f} mL")
                                st.caption("Subtypes: Intraparenchymal (IPH), Intraventricular (IVH)")
                            else:
                                st.metric("Intra-Axial Volume", "0.00 mL")
                                st.caption("No Intra-Axial Hemorrhage Detected")

                        with c2:
                            # st.metric("Extra-Axial Probability", f"{p_extra:.3f}") # Removed
                            if has_extra: 
                                st.warning(f"**Extra-Axial Hemorrhage Detected**\n\n**Volume:** {vol_extra:.2f} mL")
                                st.caption("Subtypes: Epidural (EDH), Subdural (SDH), Subarachnoid (SAH)")
                            else:
                                st.metric("Extra-Axial Volume", "0.00 mL")
                                st.caption("No Extra-Axial Hemorrhage Detected")
                        
                        st.markdown("---")
                        
                        # Store results for report
                        st.session_state["analysis_results"] = {
                            "vol_intra": vol_intra,
                            "vol_extra": vol_extra
                        }
                    
                    # Prepare for visualization (resize raw to match segmentation output)
                    vol_prep = resize_volume(raw_volume, SEG_TARGET_SHAPE)
                    vol_prep = window_ct(vol_prep)
                    
                    # Generate slices for JS viewer
                    orig_slices, overlay_slices, mask_slices = prepare_segmentation_slices(vol_prep, seg_combined)
                    
                    orig_b64 = slices_to_base64(orig_slices)
                    overlay_b64 = slices_to_base64(overlay_slices)
                    mask_b64 = slices_to_base64(mask_slices)
                    n_slices = len(orig_slices)
                    
                    # --- JS SLICE VIEWER ---
                    st.markdown("#### 🖥️ Interactive Slice Viewer")
                    
                    html_template = """
                    <div class="viewer-container">
                        <div style="display:flex; justify-content:center; gap:20px; flex-wrap:wrap;">
                            <div style="text-align:center;">
                                <h4 class="viewer-header">Non-Contrast CT</h4>
                                <img id="origImg" src="data:image/png;base64,{{ orig[0] }}"
                                     style="width:250px; border-radius:12px; border:1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.3);"/>
                            </div>
                            <div style="text-align:center;">
                                <h4 class="viewer-header">Segmentation Overlay</h4>
                                <img id="overlayImg" src="data:image/png;base64,{{ overlay[0] }}"
                                     style="width:250px; border-radius:12px; border:1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.3);"/>
                            </div>
                            <div style="text-align:center;">
                                <h4 class="viewer-header">Binary Lesion Mask</h4>
                                <img id="maskImg" src="data:image/png;base64,{{ mask[0] }}"
                                     style="width:250px; border-radius:12px; border:1px solid #334155; box-shadow: 0 4px 6px rgba(0,0,0,0.3);"/>
                            </div>
                        </div>

                        <div style="text-align:center; margin-top:24px;">
                            <input type="range" min="0" max="{{ n_slices - 1 }}" value="0"
                                   id="sliceSlider" style="width:80%; accent-color: #3b82f6;"/>
                            <p style="color:#cbd5e1; margin-top:10px; font-size:16px; font-weight: 500;">Slice: <span id="sliceIndex">1</span> / {{ n_slices }}</p>
                        </div>
                    </div>

                    <script>
                    const orig = {{ orig|safe }};
                    const overlay = {{ overlay|safe }};
                    const mask = {{ mask|safe }};
                    const slider = document.getElementById("sliceSlider");
                    const origImg = document.getElementById("origImg");
                    const overlayImg = document.getElementById("overlayImg");
                    const maskImg = document.getElementById("maskImg");
                    const sliceIndex = document.getElementById("sliceIndex");

                    slider.oninput = function() {
                        let i = parseInt(this.value);
                        origImg.src = "data:image/png;base64," + orig[i];
                        overlayImg.src = "data:image/png;base64," + overlay[i];
                        maskImg.src = "data:image/png;base64," + mask[i];
                        sliceIndex.innerText = i + 1;
                    };
                    </script>
                    """
                    
                    st.components.v1.html(
                        Template(html_template).render(
                            orig=orig_b64,
                            overlay=overlay_b64,
                            mask=mask_b64,
                            n_slices=n_slices,
                        ),
                        height=450,
                        scrolling=False,
                    )
                    
                    # --- 3D VISUALIZATION ---
                    st.markdown("#### 3️⃣ Volumetric Analysis")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("Volumetric Lesion Scatter")
                        fig_box = plot_mask_3d(seg_combined)
                        if fig_box: st.plotly_chart(fig_box, use_container_width=True)
                    with c2:
                        st.caption("3D Lesion Surface Topology")
                        fig_mesh = plot_full_3d_mesh(vol_prep, seg_combined)
                        if fig_mesh: st.plotly_chart(fig_mesh, use_container_width=True)
                    
                    # --- DOWNLOAD REPORT ---
                    st.divider()
                    st.subheader("📥 Download Report")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        report_txt = generate_report()
                        st.download_button(
                            label="📄 Download Analysis Report (.txt)",
                            data=report_txt,
                            file_name=f"hemoviz_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with col2:
                        pdf_buffer = generate_pdf_report()
                        st.download_button(
                            label="📑 Download PDF Report (.pdf)",
                            data=pdf_buffer,
                            file_name=f"hemoviz_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True
                        )

                else:
                    st.error("Segmentation failed or returned no output.")

if __name__ == "__main__":
    main()
