import numpy as np
import cv2
import os
import uuid
from PIL import Image

def generate_heatmap(mean_pred, variance=None, original_image=None):
    """
    Generates a confidence heatmap with RGB overlay.
    
    Args:
        mean_pred: (H, W) array of mean probabilities (0-1)
        variance: (H, W) array of variance (uncertainty)
        original_image: Original RGB image for overlay
    
    Returns:
        Path to saved heatmap image
    """
    from config import Config
    
    H, W = mean_pred.shape
    
    # Create BGR heatmap (since OpenCV is used for saving)
    heatmap_bgr = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Red channel (BGR index 2): inversely proportional to confidence
    heatmap_bgr[:, :, 2] = ((1 - mean_pred) * 255).astype(np.uint8)
    # Green channel (BGR index 1): proportional to confidence  
    heatmap_bgr[:, :, 1] = (mean_pred * 255).astype(np.uint8)
    # Blue channel (BGR index 0): handles uncertainty visualization
    if variance is not None:
        heatmap_bgr[:, :, 0] = (variance * 255).astype(np.uint8)
    
    # If original image provided, create overlay
    if original_image is not None:
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Resize original to match heatmap size if needed
        if original_image.shape[:2] != (H, W):
            original_image = cv2.resize(original_image, (W, H))
        
        # Convert RGB to BGR for OpenCV
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        else:
            original_bgr = original_image
            
        # Blend: 60% original + 40% heatmap
        overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_bgr, 0.4, 0)
    else:
        overlay = heatmap_bgr
    
    # Save
    output_dir = Config.HEATMAP_FOLDER
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"heatmap_{uuid.uuid4().hex}.png"
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, overlay)
    
    return f"/heatmaps/{filename}"


def generate_heatmap_with_overlay(mean_pred, variance, original_image_path):
    """
    Full pipeline: Load image, generate overlay, save result.
    """
    # Load original image
    original = cv2.imread(original_image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    return generate_heatmap(mean_pred, variance, original)
