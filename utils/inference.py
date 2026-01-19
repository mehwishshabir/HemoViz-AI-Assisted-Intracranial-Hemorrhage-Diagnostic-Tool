# utils/inference.py
import numpy as np
from tensorflow.keras.models import load_model
import os

def load_segmentation_model(model_path, custom_objects=None):
    """
    Load Keras model with provided custom objects if required (losses/metrics).
    """
    if custom_objects:
        model = load_model(model_path, custom_objects=custom_objects)
    else:
        model = load_model(model_path)
    return model

def predict_mask(model, model_input, multiclass=True):
    """
    model_input: (1, H, W, D, 1)
    Returns:
      - label_mask: (H, W, D) integer labels 0..(C-1)
      - prob_map: (H, W, D, C) probabilities per class (if multiclass)
    """
    pred = model.predict(model_input, verbose=0)
    # Some models output tuple; handle common cases:
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    pred = np.asarray(pred)  # (1, H, W, D, C) or (1, H, W, D)
    if pred.ndim == 4:
        # (1,H,W,D) -> probably single-channel logits. Convert to binary
        prob_map = np.expand_dims(pred[0], axis=-1)
        label_mask = (prob_map[..., 0] > 0.5).astype(np.uint8)
    else:
        # (1,H,W,D,C)
        prob_map = pred[0]
        # If values appear logits, try softmax/sigmoid — assume model outputs probabilities
        label_mask = np.argmax(prob_map, axis=-1).astype(np.uint8)
    return label_mask, prob_map

def overlay_on_slice(ct_slice, label_slice, class_colors=None, alpha=0.5):
    """
    Create RGB overlay for a single axial slice (2D arrays).
    ct_slice: 2D float array (0..1)
    label_slice: 2D int labels (0..C-1)
    class_colors: dict label->(r,g,b) each 0..1. If None, default colors will be used.
    Returns RGB uint8 image.
    """
    import numpy as np
    H, W = ct_slice.shape
    rgb = np.stack([ct_slice]*3, axis=-1)
    if class_colors is None:
        # default two colors for two foreground classes
        default = {
            1: (1.0, 0.0, 0.0),  # red
            2: (0.0, 1.0, 0.0),  # green
            3: (0.0, 0.0, 1.0),  # blue
            4: (1.0, 1.0, 0.0),  # yellow
            5: (1.0, 0.0, 1.0),  # magenta
        }
        class_colors = default
    overlay = rgb.copy()
    for lbl, color in class_colors.items():
        mask = (label_slice == lbl)
        if mask.sum() == 0:
            continue
        for ch in range(3):
            overlay[..., ch] = np.where(mask, overlay[..., ch]*(1-alpha) + color[ch]*alpha, overlay[..., ch])
    # scale to 0..255
    overlay = np.clip(overlay*255, 0, 255).astype(np.uint8)
    return overlay
