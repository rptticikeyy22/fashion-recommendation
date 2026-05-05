import os
import io
import json
import random
import warnings
import torch
import faiss
import numpy as np
import streamlit as st
import google.generativeai as genai
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

warnings.filterwarnings("ignore")

# --- 1. SYSTEM CONFIGURATION ---
st.set_page_config(page_title="AI Fashion Search", page_icon="👗", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    html, body, [class*='css'] { font-family: 'Inter', sans-serif; }
    .main-title { font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 0px; padding-bottom: 10px; letter-spacing: -1px; }
    .sub-title { text-align: center; font-size: 1.1rem; opacity: 0.7; margin-bottom: 40px; }
    [data-testid='stImage'] img { border-radius: 12px; box-shadow: 0 6px 12px rgba(0,0,0,0.15); transition: all 0.3s ease-in-out; }
    [data-testid='stImage'] img:hover { transform: translateY(-5px) scale(1.02); box-shadow: 0 12px 24px rgba(0,0,0,0.3); }
    h3 { font-weight: 600 !important; letter-spacing: -0.5px; padding-bottom: 10px; border-bottom: 1px solid rgba(128,128,128,0.2); margin-bottom: 20px !important; }
    </style>
    <div class="main-title">Multimodal Fashion Recommendation</div>
    <div class="sub-title">Powered by Zero-shot Retrieval with CLIP, FAISS & Gemini AI</div>
""", unsafe_allow_html=True)

# --- 2. API & LOAD BALANCING SETUP ---
API_KEYS_POOL = [
    "AIzaSyAW48E6cfPQHe-Kn2_gSeDmGkF7_tNt8P4",
    "AIzaSyBmvfByjeM5VG8vA_bJphmOBRO9qlEfCas"
]

@st.cache_data(show_spinner=False)
def get_available_models(api_key):
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        vision_models = [m.replace("models/", "") for m in models if "flash" in m or "pro" in m]
        return vision_models if vision_models else ["gemini-1.5-flash"]
    except:
        return ["Invalid API Key"]

is_valid_key = API_KEYS_POOL[0] != "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY" and len(API_KEYS_POOL[0]) > 20
available_models = get_available_models(API_KEYS_POOL[0]) if is_valid_key else ["Vui lòng cập nhật API Key"]

with st.sidebar:
    st.header("⚙️ System Configuration")
    selected_model = st.selectbox("🤖 AI Model:", available_models, index=0)
    st.markdown("---")
    st.success("✅ Load Balancing: Active")
    st.success("✅ AI Caching: Active")

# --- 3. CORE AI INITIALIZATION (CLIP & FAISS) ---
@st.cache_resource(show_spinner=False)
def init_system():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    index = faiss.read_index("fashion_clip.index")
    image_paths = np.load("image_paths.npy")
    
    def get_embedding(text=None, image=None):
        with torch.no_grad():
            if text is not None:
                inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
                outputs = model.text_model(**inputs)
                embeds = outputs.pooler_output
                proj = model.text_projection
            else:
                inputs = processor(images=image, return_tensors="pt").to(device)
                outputs = model.vision_model(**inputs)
                embeds = outputs.pooler_output
                proj = model.visual_projection
                
            embeds = proj(embeds) if isinstance(proj, torch.nn.Module) else torch.matmul(embeds, proj)
            return embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            
    return index, image_paths, get_embedding

index, image_paths, get_embedding = init_system()

def search_faiss(query_features, top_k=9):
    distances, indices = index.search(query_features.cpu().numpy().astype('float32'), top_k)
    return [image_paths[i] for i in indices[0]]

# --- 4. GEMINI STYLIST MODULE ---
@st.cache_data(show_spinner=False)
def get_ai_advice(image_bytes, model_name):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((512, 512)) 
    
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    compressed_img = Image.open(buf)

    genai.configure(api_key=random.choice(API_KEYS_POOL))
    model_gemini = genai.GenerativeModel(model_name)
    
    prompt = (
        "You are an expert fashion stylist. Suggest ONE complementary item to make a great outfit.\n"
        "Return ONLY a valid JSON: {\"outfit_advice\": \"Vietnamese advice\", \"search_query\": \"[Gender] [Color] [Basic Item Name]\"}"
    )
    
    response = model_gemini.generate_content([prompt, compressed_img], generation_config={"response_mime_type": "application/json"})
    return json.loads(response.text)

# --- 5. USER INTERFACE ---
col_input, _, col_display = st.columns([1, 0.1, 2.5])

with col_input:
    st.markdown('<h3>🎮 Control Panel</h3>', unsafe_allow_html=True)
    mode = st.radio("Retrieval Mode:", ["Search by Text", "Search by Image", "✨ AI Stylist"], label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    results = []
    
    if mode == "Search by Text":
        query_text = st.text_input("Product description:", placeholder="e.g., women white blazer")
        if query_text:
            with st.spinner('Analyzing semantics...'):
                results = search_faiss(get_embedding(text=query_text))
                
    elif mode == "Search by Image":
        uploaded_file = st.file_uploader("Upload image:", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            st.image(Image.open(uploaded_file).convert("RGB"), caption="Query Image", width=200)
            with st.spinner('Scanning features...'):
                results = search_faiss(get_embedding(image=Image.open(uploaded_file).convert("RGB")))
                
    elif mode == "✨ AI Stylist":
        uploaded_file = st.file_uploader("Upload item for advice:", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image_bytes = uploaded_file.getvalue()
            st.image(Image.open(io.BytesIO(image_bytes)), caption="Your Item", width=200)
            
            if st.button("Get Styling Advice"):
                if not is_valid_key or selected_model == "Vui lòng cập nhật API Key":
                    st.error("⚠️ Invalid API Key. Please update API_KEYS_POOL.")
                else:
                    with st.spinner(f'Analyzing with {selected_model}...'):
                        try:
                            result_json = get_ai_advice(image_bytes, selected_model)
                            st.success("✨ AI Stylist Advice:")
                            st.write(result_json.get("outfit_advice", ""))
                            st.info(f"🔍 Searching: **{result_json.get('search_query', '')}**")
                            results = search_faiss(get_embedding(text=result_json.get("search_query", "")))
                        except Exception as e:
                            st.error("⏳ Rate Limit reached. Please swap model in sidebar or try again." if "429" in str(e) else f"Error: {e}")

with col_display:
    st.markdown('<h3>✨ Results</h3>', unsafe_allow_html=True)
    if results:
        cols = st.columns(3)
        for i, path in enumerate(results):
            if os.path.exists(path):
                cols[i % 3].image(path, use_container_width=True, caption=f"Top {i+1}")
            else:
                cols[i % 3].error("Image missing")
    else:
        st.info("System ready. Enter a query to begin.")