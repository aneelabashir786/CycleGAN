# ────────────────────────────────────────────────────────────────────────────
# CycleGAN Streamlit App - Sketch ↔ Photo Translation
# Deployed on Streamlit Cloud with models from Hugging Face
# ────────────────────────────────────────────────────────────────────────────

import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="CycleGAN - Sketch to Photo Translation",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other libraries
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import requests
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 128

# Check device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    st.sidebar.success("✅ GPU is available")
else:
    DEVICE = torch.device('cpu')
    st.sidebar.info("ℹ️ Using CPU (inference will be slower)")

# Hugging Face model URLs
MODEL_URLS = {
    'G_AB': 'https://huggingface.co/aneelaBashir22f3414/CycleGAN/resolve/main/cyc_G_AB_ep60.pth',
    'G_BA': 'https://huggingface.co/aneelaBashir22f3414/CycleGAN/resolve/main/cyc_G_BA_ep60.pth'
}

# ────────────────────────────────────────────────────────────────────────────
# Model Architecture (Must match training exactly)
# ────────────────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), 
            nn.Conv2d(c, c, 3, bias=False), 
            nn.InstanceNorm2d(c), 
            nn.ReLU(True),
            nn.ReflectionPad2d(1), 
            nn.Conv2d(c, c, 3, bias=False), 
            nn.InstanceNorm2d(c)
        )
    
    def forward(self, x): 
        return x + self.block(x)


class ResNetGen(nn.Module):
    """CycleGAN ResNet Generator with 6 residual blocks"""
    def __init__(self, in_c=3, out_c=3, ngf=64, n_res=6):
        super().__init__()
        L = [
            nn.ReflectionPad2d(3), 
            nn.Conv2d(in_c, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf), 
            nn.ReLU(True)
        ]
        
        # Downsample
        for m in [1, 2]:
            L += [
                nn.Conv2d(ngf*m, ngf*m*2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf*m*2), 
                nn.ReLU(True)
            ]
        
        # Residual blocks
        for _ in range(n_res): 
            L.append(ResBlock(ngf*4))
        
        # Upsample
        for m in [2, 1]:
            L += [
                nn.ConvTranspose2d(ngf*m*2, ngf*m, 3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf*m), 
                nn.ReLU(True)
            ]
        
        # Output layer
        L += [
            nn.ReflectionPad2d(3), 
            nn.Conv2d(ngf, out_c, 7), 
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*L)
    
    def forward(self, x): 
        return self.model(x)


# ────────────────────────────────────────────────────────────────────────────
# Download model from Hugging Face
# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def download_model(url):
    """Download model weights from Hugging Face"""
    try:
        with st.spinner(f"Downloading model from Hugging Face..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Create a temporary file
            temp_file = io.BytesIO()
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = st.progress(0)
            
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress_bar.progress(min(1.0, downloaded / total_size))
            
            progress_bar.progress(1.0)
            temp_file.seek(0)
            
            # Load with appropriate map location
            return torch.load(temp_file, map_location=DEVICE)
    except Exception as e:
        st.error(f"Failed to download model: {str(e)}")
        return None


@st.cache_resource
def load_models():
    """Load both generators from Hugging Face"""
    
    with st.spinner("📥 Loading models from Hugging Face..."):
        # Initialize models
        G_AB = ResNetGen().to(DEVICE)
        G_BA = ResNetGen().to(DEVICE)
        
        # Download weights
        weights_ab = download_model(MODEL_URLS['G_AB'])
        weights_ba = download_model(MODEL_URLS['G_BA'])
        
        if weights_ab is None or weights_ba is None:
            st.error("Failed to load models. Please check your internet connection.")
            return None, None
        
        # Load weights
        G_AB.load_state_dict(weights_ab)
        G_BA.load_state_dict(weights_ba)
        
        # Set to evaluation mode
        G_AB.eval()
        G_BA.eval()
        
        st.success("✅ Models loaded successfully from Hugging Face!")
        return G_AB, G_BA


# ────────────────────────────────────────────────────────────────────────────
# Image Processing Functions
# ────────────────────────────────────────────────────────────────────────────
def preprocess_image(image, img_size=IMG_SIZE):
    """Preprocess image for model input"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str):
        image = Image.open(image)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    return transform(image).unsqueeze(0).to(DEVICE)


def denormalize(tensor):
    """Denormalize from [-1, 1] to [0, 1]"""
    return (tensor * 0.5 + 0.5).clamp(0, 1)


def tensor_to_image(tensor):
    """Convert tensor to PIL Image"""
    img = denormalize(tensor.cpu()).squeeze(0)
    img = img.permute(1, 2, 0).numpy()
    return Image.fromarray((img * 255).astype(np.uint8))


def sketch_to_photo(G_AB, sketch_image):
    """Translate sketch to photo"""
    with torch.no_grad():
        input_tensor = preprocess_image(sketch_image)
        output_tensor = G_AB(input_tensor)
        return tensor_to_image(output_tensor)


def photo_to_sketch(G_BA, photo_image):
    """Translate photo to sketch"""
    with torch.no_grad():
        input_tensor = preprocess_image(photo_image)
        output_tensor = G_BA(input_tensor)
        return tensor_to_image(output_tensor)


# ────────────────────────────────────────────────────────────────────────────
# Main UI
# ────────────────────────────────────────────────────────────────────────────

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
        border: none;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🎨 CycleGAN: Unpaired Image-to-Image Translation")
st.markdown("""
    <p style='font-size: 18px; text-align: center;'>
    Transform <b>Sketches to Photos</b> and <b>Photos to Sketches</b> using CycleGAN
    </p>
""", unsafe_allow_html=True)

# Load models
G_AB, G_BA = load_models()

if G_AB is None or G_BA is None:
    st.error("Failed to load models. Please refresh the page or try again later.")
    st.stop()

# Main content - Two tabs for different translations
tab1, tab2 = st.tabs(["✏️ Sketch → Photo", "📸 Photo → Sketch"])

# Tab 1: Sketch to Photo
with tab1:
    st.header("Transform Your Sketch into a Realistic Photo")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("📝 Input Sketch")
        sketch_input = st.file_uploader(
            "Upload a sketch image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
            key="sketch_upload"
        )
        
        if sketch_input is not None:
            sketch_image = Image.open(sketch_input).convert('RGB')
            st.image(sketch_image, caption="Input Sketch", use_container_width=True)
    
    with col2:
        st.subheader("🎨 Generated Photo")
        if sketch_input is not None:
            if st.button("✨ Translate to Photo", key="sketch_to_photo_btn", use_container_width=True):
                with st.spinner("🎨 Generating photo... Please wait..."):
                    try:
                        photo_output = sketch_to_photo(G_AB, sketch_image)
                        st.image(photo_output, caption="Generated Photo", use_container_width=True)
                        st.success("✅ Translation complete!")
                        
                        # Download button
                        buf = io.BytesIO()
                        photo_output.save(buf, format="PNG")
                        st.download_button(
                            label="📥 Download Result",
                            data=buf.getvalue(),
                            file_name="generated_photo.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error during translation: {str(e)}")
        else:
            st.info("👈 Upload a sketch to begin translation")

# Tab 2: Photo to Sketch
with tab2:
    st.header("Convert Your Photo into an Artistic Sketch")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("📸 Input Photo")
        photo_input = st.file_uploader(
            "Upload a photo",
            type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
            key="photo_upload"
        )
        
        if photo_input is not None:
            photo_image = Image.open(photo_input).convert('RGB')
            st.image(photo_image, caption="Input Photo", use_container_width=True)
    
    with col2:
        st.subheader("✏️ Generated Sketch")
        if photo_input is not None:
            if st.button("✨ Convert to Sketch", key="photo_to_sketch_btn", use_container_width=True):
                with st.spinner("✏️ Generating sketch... Please wait..."):
                    try:
                        sketch_output = photo_to_sketch(G_BA, photo_image)
                        st.image(sketch_output, caption="Generated Sketch", use_container_width=True)
                        st.success("✅ Translation complete!")
                        
                        # Download button
                        buf = io.BytesIO()
                        sketch_output.save(buf, format="PNG")
                        st.download_button(
                            label="📥 Download Result",
                            data=buf.getvalue(),
                            file_name="generated_sketch.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error during translation: {str(e)}")
        else:
            st.info("👈 Upload a photo to begin translation")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
    <p>🎨 Built with CycleGAN | ResNet Generator | PatchGAN Discriminator</p>
    <p>🚀 Models hosted on <a href="https://huggingface.co/aneelaBashir22f3414/CycleGAN">Hugging Face</a></p>
    </div>
""", unsafe_allow_html=True)
