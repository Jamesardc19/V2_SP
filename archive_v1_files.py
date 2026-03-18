"""
Archive V1 Files Script
Moves old/intermediate files to Archive_V1 folder instead of deleting them
"""

import os
import shutil
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent
ARCHIVE_DIR = PROJECT_ROOT / "Archive_V1"

# Files and folders to ARCHIVE
TO_ARCHIVE = [
    # Duplicate/old scripts
    "apply_calibration.py",
    "apply_calibration_fixed.py",
    "improved_preprocessing.py",
    "improved_model_training.py",
    "data_preprocessing.py",
    "model_training_final.py",
    "run_improved_pipeline.py",
    
    # Intermediate results folders
    "Model_Results_Improved",
    "PREPROCESS_OUTPUT",
    "PREPROCESS_OUTPUT_IMPROVED",
    "Trained_Models",
    "Trained_Models_Improved",
    
    # Cache and temp files
    "__pycache__",
    "catboost_info",
    ".qodo",
    "processed_dataset.csv",
    
    # Empty folders
    "LIME_Explanations",
    "EDA_Results",
]

def get_size(path):
    """Get size of file or directory in MB"""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except:
                    pass
        return total / (1024 * 1024)
    return 0

def archive_files(dry_run=True):
    """
    Archive old files to Archive_V1 folder
    
    Args:
        dry_run: If True, only show what would be archived without actually moving
    """
    print("="*80)
    print("ARCHIVE V1 FILES")
    print("="*80)
    print(f"Mode: {'DRY RUN (no files will be moved)' if dry_run else 'ACTUAL ARCHIVING'}")
    print(f"Archive location: {ARCHIVE_DIR}")
    print("="*80)
    
    total_size = 0
    items_to_archive = []
    
    # Check what exists
    for item_name in TO_ARCHIVE:
        item_path = PROJECT_ROOT / item_name
        if item_path.exists():
            size = get_size(item_path)
            total_size += size
            item_type = "DIR " if item_path.is_dir() else "FILE"
            items_to_archive.append((item_path, size, item_type))
            print(f"  [{item_type}] {item_name} ({size:.2f} MB)")
    
    print(f"\n{'='*80}")
    print(f"Total to archive: {len(items_to_archive)} items ({total_size:.2f} MB)")
    print(f"{'='*80}\n")
    
    if not dry_run:
        # Create archive directory
        ARCHIVE_DIR.mkdir(exist_ok=True)
        print(f"Created archive directory: {ARCHIVE_DIR}\n")
        
        print("Moving files to archive...")
        moved_count = 0
        for item_path, size, item_type in items_to_archive:
            try:
                dest_path = ARCHIVE_DIR / item_path.name
                
                # If destination exists, remove it first
                if dest_path.exists():
                    if dest_path.is_dir():
                        shutil.rmtree(dest_path)
                    else:
                        dest_path.unlink()
                
                # Move to archive
                shutil.move(str(item_path), str(dest_path))
                print(f"  ✅ Archived: {item_path.name}")
                moved_count += 1
            except Exception as e:
                print(f"  ❌ Error archiving {item_path.name}: {e}")
        
        print(f"\n✅ Archive complete! Moved {moved_count} items ({total_size:.2f} MB)")
        print(f"   Files saved in: {ARCHIVE_DIR}")
    else:
        print("DRY RUN complete. Run with dry_run=False to actually archive files.")
    
    # Show what will be kept in main folder
    print(f"\n{'='*80}")
    print("CLEAN PROJECT STRUCTURE (after archiving):")
    print(f"{'='*80}")
    
    structure = [
        "",
        "V2_SP/",
        "├── Core Scripts:",
        "│   ├── improved_preprocessing_v2.py (final preprocessing)",
        "│   ├── improved_model_training_v2.py (final training)",
        "│   ├── revised_preprocessing.py (baseline)",
        "│   ├── model_training_revised.py (baseline)",
        "│   └── tabnet_wrapper.py (utility)",
        "│",
        "├── Documentation:",
        "│   ├── README.md",
        "│   ├── Conceptual_Framework.md",
        "│   ├── Model_Results_Interpretation.md",
        "│   └── Model_Improvement_Strategies.md",
        "│",
        "├── Results (Final - V2):",
        "│   ├── Model_Results_Improved_V2/",
        "│   ├── Trained_Models_Improved_V2/",
        "│   └── PREPROCESS_OUTPUT_IMPROVED_V2/",
        "│",
        "├── Results (Baseline):",
        "│   └── Model_Results/",
        "│",
        "├── Data:",
        "│   └── DATASETS/",
        "│",
        "└── Archive_V1/ (old files - not for GitHub)",
        "    ├── Old scripts (7 files)",
        "    ├── Intermediate results (5 folders)",
        "    ├── Cache files",
        "    └── Large CSV file",
        "",
    ]
    
    for line in structure:
        print(line)
    
    print(f"{'='*80}")
    print("\n💡 TIP: Add 'Archive_V1/' to .gitignore to exclude from GitHub")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    # Actually archive files
    archive_files(dry_run=False)
