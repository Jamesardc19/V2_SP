"""
Project Cleanup Script for GitHub
Removes unnecessary files and organizes the project structure
"""

import os
import shutil
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Files and folders to REMOVE
TO_REMOVE = [
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

# Optional files (comment out if you want to keep them)
OPTIONAL_REMOVE = [
    "analyze_features.py",
    "create_feature_mapping.py",
    "model_evaluation_enhanced.py",
]

def get_size(path):
    """Get size of file or directory in MB"""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    elif path.is_dir():
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        return total / (1024 * 1024)
    return 0

def cleanup_project(dry_run=True):
    """
    Clean up project files
    
    Args:
        dry_run: If True, only show what would be deleted without actually deleting
    """
    print("="*80)
    print("PROJECT CLEANUP FOR GITHUB")
    print("="*80)
    print(f"Mode: {'DRY RUN (no files will be deleted)' if dry_run else 'ACTUAL DELETION'}")
    print("="*80)
    
    total_size = 0
    items_to_remove = []
    
    # Check what exists
    for item_name in TO_REMOVE:
        item_path = PROJECT_ROOT / item_name
        if item_path.exists():
            size = get_size(item_path)
            total_size += size
            item_type = "DIR " if item_path.is_dir() else "FILE"
            items_to_remove.append((item_path, size, item_type))
            print(f"  [{item_type}] {item_name} ({size:.2f} MB)")
    
    print(f"\n{'='*80}")
    print(f"Total to remove: {len(items_to_remove)} items ({total_size:.2f} MB)")
    print(f"{'='*80}\n")
    
    if not dry_run:
        confirm = input("Are you sure you want to delete these files? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Cleanup cancelled.")
            return
        
        print("\nDeleting files...")
        for item_path, size, item_type in items_to_remove:
            try:
                if item_path.is_dir():
                    shutil.rmtree(item_path)
                else:
                    item_path.unlink()
                print(f"  ✅ Deleted: {item_path.name}")
            except Exception as e:
                print(f"  ❌ Error deleting {item_path.name}: {e}")
        
        print(f"\n✅ Cleanup complete! Freed {total_size:.2f} MB")
    else:
        print("DRY RUN complete. Run with dry_run=False to actually delete files.")
    
    # Show what will be kept
    print(f"\n{'='*80}")
    print("FILES TO KEEP:")
    print(f"{'='*80}")
    
    keep_files = [
        "Core Scripts:",
        "  - improved_preprocessing_v2.py (final preprocessing)",
        "  - improved_model_training_v2.py (final training)",
        "  - revised_preprocessing.py (baseline)",
        "  - model_training_revised.py (baseline)",
        "  - tabnet_wrapper.py (utility)",
        "",
        "Documentation:",
        "  - README.md",
        "  - Conceptual_Framework.md",
        "  - Model_Results_Interpretation.md",
        "  - Model_Improvement_Strategies.md",
        "",
        "Results:",
        "  - Model_Results_Improved_V2/ (final results)",
        "  - Trained_Models_Improved_V2/ (final models)",
        "  - PREPROCESS_OUTPUT_IMPROVED_V2/ (final data)",
        "  - Model_Results/ (baseline results)",
        "",
        "Data:",
        "  - DATASETS/ (original datasets)",
    ]
    
    for line in keep_files:
        print(line)
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    # First run in dry-run mode to see what would be deleted
    # cleanup_project(dry_run=True)
    
    # Actually delete files
    cleanup_project(dry_run=False)
