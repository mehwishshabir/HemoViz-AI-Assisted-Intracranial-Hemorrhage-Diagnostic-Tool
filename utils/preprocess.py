import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def load_nifti(file_path):
    """
    Load NIfTI file and return data and affine.
    """
    img = nib.load(file_path)
    return img.get_fdata().astype(np.float32), img.affine

def window_ct(img, min_val=0, max_val=120):
    """
    Window CT values to a fixed range [min_val, max_val].
    """
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val)
    return img

def resize_volume(img, desired_shape=(256, 256, 32)):
    """
    Resize a 3D volume to desired shape.
    """
    current_shape = img.shape
    zoom_factors = [d / c for d, c in zip(desired_shape, current_shape)]
    return zoom(img, zoom_factors, order=1)

def preprocess_volume(volume, desired_shape=(256, 256, 32)):
    """
    Preprocess CT volume for model input.
    Steps: window -> resize -> normalize
    """
    volume = window_ct(volume)
    volume = resize_volume(volume, desired_shape)
    return volume.astype(np.float32)
