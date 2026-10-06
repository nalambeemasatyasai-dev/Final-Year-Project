"""
Explainable AI (XAI): Grad-CAM Visualizations for Multimodal Student Engagement Detection
Generates and saves visual heatmaps for ResNet-18 and EfficientNet-B1 across Engagement Classes.
"""

import os
import sys
import glob
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Set root directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from DL_models.dual_branch_network import get_upgraded_model
from Feature_extract.mediapipe_facs_extractor import get_face_landmarker, BLENDSHAPE_NAMES
import mediapipe as mp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using Device: {device}")
print(f"[INFO] Root Directory: {BASE_DIR}")

class DualBranchGradCAM:
    """
    Grad-CAM engine for Multimodal Dual-Branch Student Engagement Network.
    Hooks into the visual backbone's final convolutional layer.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, img_tensor, blend_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        logits = self.model(img_tensor, blend_tensor)
        probs = torch.softmax(logits, dim=1)
        
        if target_class is None:
            target_class = torch.argmax(probs, dim=1).item()
            
        score = logits[0, target_class]
        score.backward()
        
        # Global Average Pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam_relu = F.relu(cam)
        if torch.max(cam_relu) > 0:
            cam = cam_relu
        else:
            cam = torch.abs(cam)
        
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        
        # Min-max normalize between 0.0 and 1.0
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam, target_class, probs[0].detach().cpu().numpy()

def load_trained_model(backbone_type, ckpt_name):
    model = get_upgraded_model(backbone_type=backbone_type, num_classes=3).to(device)
    ckpt_path = os.path.join(BASE_DIR, 'DL_models', 'hyper-1', ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model

preprocess_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_MAP = {
    0: "Disengaged (Class 0)",
    1: "Nominally Engaged (Class 1)",
    2: "Highly Engaged (Class 2)"
}

def extract_blendshapes_from_image(pil_img):
    detector = get_face_landmarker()
    img_np = np.array(pil_img)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
    result = detector.detect(mp_img)
    
    blend_np = np.zeros(len(BLENDSHAPE_NAMES), dtype=np.float32)
    if result.face_blendshapes and len(result.face_blendshapes) > 0:
        b_list = result.face_blendshapes[0]
        name_to_idx = {name: i for i, name in enumerate(BLENDSHAPE_NAMES)}
        for item in b_list:
            if item.category_name in name_to_idx:
                blend_np[name_to_idx[item.category_name]] = item.score
            elif item.index < len(blend_np):
                blend_np[item.index] = item.score
    return blend_np

def save_gradcam_figure(img_path, gradcam_engine, model_title, output_save_path):
    pil_img = Image.open(img_path).convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
    img_np = np.array(pil_img) / 255.0
    
    blend_np = extract_blendshapes_from_image(pil_img)
    img_tensor = preprocess_transform(pil_img).unsqueeze(0).to(device)
    blend_tensor = torch.from_numpy(blend_np).unsqueeze(0).to(device)
    
    cam, pred_class, probs = gradcam_engine.generate(img_tensor, blend_tensor)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    
    overlay = 0.55 * heatmap + 0.45 * img_np
    overlay = overlay / np.max(overlay)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title(f"Original Face Crop\nFile: {os.path.basename(img_path)}", fontsize=11)
    axes[0].axis('off')
    
    im1 = axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(f"{model_title} Grad-CAM Heatmap\nActive Receptive Field", fontsize=11)
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    axes[2].imshow(overlay)
    conf = probs[pred_class] * 100.0
    axes[2].set_title(f"Superimposed XAI Attention\nPred: {CLASS_MAP[pred_class]} ({conf:.1f}%)", fontsize=11, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(
        f"Explainable AI Grad-CAM Analysis - {model_title}\n"
        f"Probabilities: Disengaged: {probs[0]*100:.1f}% | Nominally: {probs[1]*100:.1f}% | Highly: {probs[2]*100:.1f}%",
        fontsize=13, y=1.05
    )
    plt.tight_layout()
    plt.savefig(output_save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[OK] Saved visualization: {output_save_path}")

def main():
    out_dir = os.path.join(CURRENT_DIR, 'gradcam_outputs')
    os.makedirs(out_dir, exist_ok=True)
    
    print("[1/3] Loading Trained ResNet-18 Checkpoint...")
    model_resnet18 = load_trained_model('resnet', 'best_model_upgraded_resnet.pth')
    target_layer_resnet18 = model_resnet18.visual_backbone[-1][-1].conv2
    gradcam_resnet18 = DualBranchGradCAM(model_resnet18, target_layer_resnet18)
    
    print("[2/3] Loading Trained EfficientNet-B1 Checkpoint...")
    model_effnet_b1 = load_trained_model('efficientnet_b1', 'best_model_upgraded_efficientnet_b1.pth')
    target_layer_effnet_b1 = model_effnet_b1.visual_backbone[-1]
    gradcam_effnet_b1 = DualBranchGradCAM(model_effnet_b1, target_layer_effnet_b1)
    
    print("[3/3] Generating Class Visualizations...")
    # Class 0
    s0 = glob.glob(os.path.join(BASE_DIR, 'WACV data', '0', '*.jpg'))
    if s0:
        save_gradcam_figure(s0[0], gradcam_resnet18, "Dual-Branch ResNet-18", os.path.join(out_dir, "gradcam_class0_disengaged_resnet18.png"))
        save_gradcam_figure(s0[0], gradcam_effnet_b1, "Dual-Branch EfficientNet-B1", os.path.join(out_dir, "gradcam_class0_disengaged_effnet_b1.png"))
        
    # Class 1
    s1 = glob.glob(os.path.join(BASE_DIR, 'WACV data', '1', '*.jpg'))
    if s1:
        save_gradcam_figure(s1[0], gradcam_resnet18, "Dual-Branch ResNet-18", os.path.join(out_dir, "gradcam_class1_nominally_engaged_resnet18.png"))
        
    # Class 2
    s2 = glob.glob(os.path.join(BASE_DIR, 'WACV data', '2', '*.jpg'))
    if s2:
        save_gradcam_figure(s2[0], gradcam_resnet18, "Dual-Branch ResNet-18", os.path.join(out_dir, "gradcam_class2_highly_engaged_resnet18.png"))
        save_gradcam_figure(s2[0], gradcam_effnet_b1, "Dual-Branch EfficientNet-B1", os.path.join(out_dir, "gradcam_class2_highly_engaged_effnet_b1.png"))
        
    print("\n[SUCCESS] All Grad-CAM heatmaps saved to:", out_dir)

if __name__ == "__main__":
    main()
