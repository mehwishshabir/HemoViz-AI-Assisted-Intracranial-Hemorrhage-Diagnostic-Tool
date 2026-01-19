import os
import shutil
import tempfile
import zipfile
import pydicom
import dicom2nifti
import numpy as np

def setup_dcm2niix():
    """
    Configures the environment for dcm2niix and dicom2nifti.
    """
    # 1. Check if dcm2niix is in PATH
    if not shutil.which("dcm2niix"):
        # Try to find it in common Conda paths
        # Based on user's environment: C:\Users\user\miniconda3\envs\neuro\Library\bin\dcm2niix.EXE
        possible_paths = [
            r"C:\Users\user\miniconda3\envs\neuro\Library\bin",
            r"C:\Users\user\miniconda3\Library\bin",
            os.path.join(os.environ.get("CONDA_PREFIX", ""), "Library", "bin")
        ]
        
        for p in possible_paths:
            if os.path.exists(os.path.join(p, "dcm2niix.exe")):
                print(f"Found dcm2niix at {p}, adding to PATH...")
                os.environ["PATH"] += os.pathsep + p
                break
    
    # 2. Configure dicom2nifti settings
    # Disable validation of slice increment to be more lenient with non-uniform slices
    try:
        dicom2nifti.settings.disable_validate_slice_increment()
        dicom2nifti.settings.enable_resampling()
        dicom2nifti.settings.set_gdcm_conv_path("") # Disable gdcm if not needed/available
    except:
        pass

def extract_zip(zip_path, extract_to):
    """Extracts a ZIP file to a target directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def find_dicom_series(root_dir):
    """
    Scans a directory for DICOM files and groups them by Series Instance UID.
    Returns a dictionary: 
    {
        SeriesUID: {
            "files": [list of file paths],
            "description": "Series Description",
            "modality": "CT",
            "count": 100,
            "dims": (rows, cols)
        }
    }
    """
    series_dict = {}
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Read only the header to speed up processing
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                
                # Check if it's a valid DICOM with SeriesUID
                if "SeriesInstanceUID" in ds:
                    uid = ds.SeriesInstanceUID
                    if uid not in series_dict:
                        series_dict[uid] = {
                            "files": [],
                            "description": ds.get("SeriesDescription", "Unknown"),
                            "modality": ds.get("Modality", "Unknown"),
                            "dims": (ds.get("Rows", 0), ds.get("Columns", 0)),
                            "patient_id": ds.get("PatientID", "Unknown"),
                            "study_date": ds.get("StudyDate", "Unknown"),
                            "manufacturer": ds.get("Manufacturer", "Unknown")
                        }
                    series_dict[uid]["files"].append(file_path)
            except:
                continue # Not a valid DICOM file
    
    # Update counts
    for uid in series_dict:
        series_dict[uid]["count"] = len(series_dict[uid]["files"])
        # Sort files to ensure order (optional but good practice)
        series_dict[uid]["files"].sort()
                
    return series_dict

def select_best_series(series_dict):
    """
    Selects the 'best' series for brain CT analysis.
    Criteria:
    1. Must have > 10 slices (to avoid scouts/localizers).
    2. Prefer 'Axial' orientation (if available in description).
    3. Prefer 'Soft Tissue' or standard kernels.
    
    Returns: (SeriesUID, list_of_files)
    """
    candidates = []
    
    for uid, files in series_dict.items():
        if len(files) < 10:
            continue
            
        # Read first file to check metadata
        try:
            ds = pydicom.dcmread(files[0], stop_before_pixels=True)
            desc = ds.get("SeriesDescription", "").lower()
            modality = ds.get("Modality", "").upper()
            
            if modality != "CT":
                continue
                
            score = 0
            if "axial" in desc: score += 2
            if "brain" in desc: score += 2
            if "bone" in desc: score -= 1 # We want soft tissue
            
            candidates.append({
                "uid": uid,
                "files": files,
                "score": score,
                "desc": desc,
                "count": len(files)
            })
        except:
            continue
            
    if not candidates:
        return None, None
        
    # Sort by score (desc), then by slice count (desc)
    candidates.sort(key=lambda x: (x["score"], x["count"]), reverse=True)
    
    best = candidates[0]
    return best["uid"], best["files"]

def convert_dicom_to_nifti(dicom_files, output_path):
    """
    Converts a list of DICOM files (representing one series) to a single NIfTI file.
    Returns: (success, message)
    """
    # Check for dcm2niix if possible (dicom2nifti uses it internally)
    # We can't easily check if dicom2nifti will fail until we try, 
    # but we can catch the specific error.
    
    # Create a temp directory for the specific series
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy files to temp dir (dicom2nifti expects a directory)
        try:
            for f in dicom_files:
                shutil.copy(f, temp_dir)
        except Exception as e:
            return False, f"Failed to copy DICOM files: {str(e)}"
            
        try:
            # dicom2nifti writes to a directory, we need to find the file
            dicom2nifti.convert_directory(temp_dir, temp_dir, compression=True, reorient=True)
            
            # Find the generated .nii.gz file
            generated_files = [f for f in os.listdir(temp_dir) if f.endswith(".nii.gz")]
            if generated_files:
                src = os.path.join(temp_dir, generated_files[0])
                shutil.move(src, output_path)
                return True, "Conversion successful"
            else:
                return False, "No NIfTI file generated. This often happens if dcm2niix is missing or the DICOM data is invalid."
        except Exception as e:
            error_msg = str(e)
            if "dcm2niix" in error_msg.lower() or "not found" in error_msg.lower():
                return False, "dcm2niix not found. Please install dcm2niix and add it to your PATH, or ensure dicom2nifti is correctly installed."
            return False, f"Conversion Error: {error_msg}"
