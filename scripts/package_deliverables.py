"""
=============================================================================
Script: package_deliverables.py
Project: AI-Driven RF Signal Classification & Interference Detection System
Utility: Automated Release Distribution & Deliverable Packager
=============================================================================

This utility creates a clean, complete .ZIP archive of the entire project,
ready for release distribution, customer deployment, or archiving.
"""

import os
import sys
import zipfile
import time

def create_deliverables_zip():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent_dir = os.path.dirname(base_dir)
    
    zip_filename = "AI_Driven_RF_Spectrum_Monitoring_System_Release.zip"
    zip_dest_path = os.path.join(parent_dir, zip_filename)
    
    print("=" * 80)
    print("PACKAGING SYSTEM DELIVERABLES INTO STANDALONE ZIP ARCHIVE")
    print("=" * 80)
    
    # Exclude temporary cache folders (__pycache__, .git, etc.)
    exclude_dirs = {"__pycache__", ".pytest_cache", ".git", ".venv", "venv"}
    exclude_extensions = {".pyc", ".pyo"}
    
    files_to_pack = []
    
    # 1. Project Directory files
    for root, dirs, files in os.walk(base_dir):
        # In-place modify dirs to skip excluded folders
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in exclude_extensions:
                continue
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, parent_dir)
            files_to_pack.append((full_path, rel_path))
            
    # 2. PowerPoint Presentation from parent_dir if exists
    pptx_path = os.path.join(parent_dir, "AI_Driven_RF_Spectrum_Monitoring_System.pptx")
    if os.path.exists(pptx_path):
        files_to_pack.append((pptx_path, "AI_Driven_RF_Spectrum_Monitoring_System.pptx"))
        
    print(f"Total files identified for packaging: {len(files_to_pack)}")
    
    with zipfile.ZipFile(zip_dest_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for full_p, arc_name in files_to_pack:
            zipf.write(full_p, arc_name)
            
    zip_size_mb = os.path.getsize(zip_dest_path) / (1024.0 * 1024.0)
    print("\n" + "=" * 80)
    print("PACKAGING COMPLETE!")
    print(f"Archive Destination : {zip_dest_path}")
    print(f"Archive File Size   : {zip_size_mb:.2f} MB")
    print("=" * 80)
    return zip_dest_path

if __name__ == "__main__":
    create_deliverables_zip()
