"""
Data splitting script for deep learning classification challenge.
Creates stratified splits for different experimental settings.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
import json
from typing import List, Dict, Tuple
import argparse


def collect_files(data_dir: str) -> pd.DataFrame:
    """
    Collect all image files from the data directory structure.
    
    Expected structure:
    data_dir/
    ├── center_1/
    │   ├── ndbe/  (class 0)
    │   └── neo/   (class 1)
    └── center_2/
        ├── ndbe/  (class 0)
        └── neo/   (class 1)
    
    Returns:
        DataFrame with columns: image_path, sample_id, center, class_name, target
    """
    data_path = Path(data_dir)
    files_data = []
    
    centers = ['center_1', 'center_2']
    class_mapping = {'ndbe': 0, 'neo': 1}
    
    for center in centers:
        center_path = data_path / center
        if not center_path.exists():
            print(f"Warning: {center_path} does not exist")
            continue
            
        for class_name, class_label in class_mapping.items():
            class_path = center_path / class_name
            if not class_path.exists():
                print(f"Warning: {class_path} does not exist")
                continue
                
            # Get all image files (common extensions)
            image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
            for ext in image_extensions:
                for file_path in class_path.glob(f'*{ext}'):
                    files_data.append({
                        'image_path': str(file_path.relative_to(data_path)),
                        'sample_id': str(file_path).split("/")[-1],
                        'center': center,
                        'class_name': class_name,
                        'target': class_label
                    })
                # Also check uppercase extensions
                for file_path in class_path.glob(f'*{ext.upper()}'):
                    files_data.append({
                        'image_path': str(file_path.relative_to(data_path)),
                        'sample_id': str(file_path).split("/")[-1],
                        'center': center,
                        'class_name': class_name,
                        'target': class_label
                    })
    
    df = pd.DataFrame(files_data)
    df = df.drop_duplicates(subset=['image_path'])  # Remove duplicates from case variations
    df = df.reset_index(drop=True)
    return df


def create_5fold_cv_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 5-fold cross-validation splits preserving class ratios across centers.
    """
    df_splits = df.copy()
    df_splits['split'] = -1
    
    # Create stratification key combining center and class
    df_splits['strat_key'] = df_splits['center'] + '_' + df_splits['class_name']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # split_col_idx = df_splits.columns.get_loc('split')
    
    for fold, (_, test_idx) in enumerate(skf.split(df_splits, df_splits['strat_key'])):
        df_splits.loc[list(test_idx), 'split'] = f'fold_{fold}'
    
    df_splits = df_splits.drop('strat_key', axis=1)
    return df_splits


def create_center_splits(df: pd.DataFrame, train_center: str, test_center: str) -> pd.DataFrame:
    """
    Create train/test splits based on centers.
    """
    df_splits = df.copy()
    df_splits['split'] = df_splits['center'].apply(
        lambda x: 'train' if x == train_center else 'test'
    )
    return df_splits


def create_holdout_cv_splits(df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
    """
    Create holdout test set (20%) and 5-fold CV on remaining 80%.
    """
    df_splits = df.copy()
    
    # Create stratification key
    df_splits['strat_key'] = df_splits['center'] + '_' + df_splits['class_name']
    
    # First split: 80% for CV, 20% for test
    train_cv_idx, test_idx = train_test_split(
        range(len(df_splits)),
        test_size=test_size,
        stratify=df_splits['strat_key'],
        random_state=42
    )
    
    df_splits['split'] = 'test'
    df_splits.loc[list(train_cv_idx), 'split'] = 'cv'
    
    # Create 5-fold CV on the training data
    cv_data = df_splits[df_splits['split'] == 'cv'].copy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (_, val_idx) in enumerate(skf.split(cv_data, cv_data['strat_key'])):
        global_idx = cv_data.iloc[val_idx].index
        df_splits.loc[global_idx, 'split'] = f'cv_fold_{fold}'
    
    # Set remaining CV data as training folds
    for fold in range(5):
        mask = df_splits['split'] == 'cv'
        fold_mask = df_splits['split'] == f'cv_fold_{fold}'
        train_mask = mask & ~fold_mask
        df_splits.loc[train_mask, 'split'] = df_splits.loc[train_mask, 'split'].apply(
            lambda x: f'cv_train_{fold}' if x == 'cv' else x
        )
    
    # Clean up: rename cv folds to be more intuitive
    df_splits['split'] = df_splits['split'].apply(lambda x: 
        f'fold_{x.split("_")[-1]}' if x.startswith('cv_fold_') else x
    )
    df_splits['split'] = df_splits['split'].apply(lambda x: 
        f'train_fold_{x.split("_")[-1]}' if x.startswith('cv_train_') else x
    )
    
    df_splits = df_splits.drop('strat_key', axis=1)
    return df_splits


def print_split_summary(df: pd.DataFrame, split_name: str):
    """Print summary statistics for a split."""
    print(f"\n=== {split_name} ===")
    print(f"Total files: {len(df)}")
    
    # Summary by split
    split_summary = df.groupby(['split', 'center', 'class_name']).size().unstack(fill_value=0)
    print(f"\nSplit summary:")
    print(split_summary)
    
    # Class distribution
    class_dist = df.groupby(['split', 'class_name']).size().unstack(fill_value=0)
    print(f"\nClass distribution by split:")
    print(class_dist)


def sanity_check_splits(
    file_names=None,
    data_dir="data/splits",
    expected_sample_count=3095,
    raise_on_error=True
):
    
    if file_names is None:
        # Find all .csv files in the directory
        file_names = [os.path.basename(f) for f in glob.glob(os.path.join(data_dir, "*.csv"))]

    if not file_names:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")


    for file in file_names:
        print(f"\n🔍 Checking file: {file}")
        file_path = os.path.join(data_dir, file)
        
        # Load file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            message = f"❌ Failed to read {file}: {e}"
            if raise_on_error:
                raise RuntimeError(message)
            else:
                print(message)
                continue

        if 'image_path' not in df.columns:
            message = f"❌ Missing 'image_path' column in {file}"
            if raise_on_error:
                raise ValueError(message)
            else:
                print(message)
                continue

        # Check total rows
        if len(df) != expected_sample_count:
            message = f"❌ {file}: Row count = {len(df)}, expected {expected_sample_count}"
            if raise_on_error:
                raise ValueError(message)
            else:
                print(message)
        else:
            print("✅ Row count matches expected")

        # Check uniqueness
        num_duplicates = len(df) - df['sample_id'].nunique()
        if num_duplicates > 0:
            message = f"❌ {file}: Found {num_duplicates} duplicate sample IDs"
            if raise_on_error:
                raise ValueError(message)
            else:
                print(message)
        else:
            print("✅ All sample IDs are unique")

        # Check leakage between splits
        if 'split' in df.columns:
            for fold in df['split'].unique():
                fold_ids = set(df.loc[df['split'] == fold, 'sample_id'])
                other_ids = set(df.loc[df['split'] != fold, 'sample_id'])
                overlap = fold_ids & other_ids
                if overlap:
                    message = (
                        f"❌ {file}: Data leakage in split '{fold}' "
                        f"– {len(overlap)} overlapping IDs (e.g., {list(overlap)[:5]})"
                    )
                    if raise_on_error:
                        raise ValueError(message)
                    else:
                        print(message)
                else:
                    print(f"✅ No leakage in split '{fold}'")
        else:
            print("⚠️ No 'split' column found - skipping split leakage check")

        print("=" * 40)

def main():
    parser = argparse.ArgumentParser(description='Create data splits for DL classification')
    parser.add_argument('--data_dir', type=str, default='data/train',
                       help='Path to the data directory')
    parser.add_argument('--output_dir', type=str, default='data/splits',
                       help='Directory to save split files')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect all files
    print("Collecting files...")
    df = collect_files(args.data_dir)
    
    if df.empty:
        print("No files found! Please check your data directory structure.")
        return
    
    print(f"Found {len(df)} total files")
    print(f"Centers: {df['center'].unique()}")
    print(f"Classes: {df['class_name'].unique()}")
    
    # Create different splits
    splits = {}
    
    # 1. 5-fold Cross Validation
    print("\nCreating 5-fold CV splits...")
    splits['5fold_cv'] = create_5fold_cv_splits(df)
    print_split_summary(splits['5fold_cv'], "5-Fold Cross Validation")
    
    # 2. Train on center_1, test on center_2
    print("\nCreating center_1 train / center_2 test splits...")
    splits['center1_train_center2_test'] = create_center_splits(df, 'center_1', 'center_2')
    print_split_summary(splits['center1_train_center2_test'], "Center 1 Train / Center 2 Test")
    
    # 3. Train on center_2, test on center_1
    print("\nCreating center_2 train / center_1 test splits...")
    splits['center2_train_center1_test'] = create_center_splits(df, 'center_2', 'center_1')
    print_split_summary(splits['center2_train_center1_test'], "Center 2 Train / Center 1 Test")
    
    # 4. 20% holdout test + 5-fold CV on remaining 80%
    print("\nCreating 20% holdout + CV splits...")
    splits['holdout_cv'] = create_holdout_cv_splits(df, test_size=0.2)
    print_split_summary(splits['holdout_cv'], "20% Holdout + 5-Fold CV")
    
    # Save all splits
    for split_name, split_df in splits.items():
        output_path = os.path.join(args.output_dir, f'{split_name}.csv')
        split_df.to_csv(output_path, index=False)
        print(f"Saved {split_name} to {output_path}")
    
    # Create metadata file
    metadata = {
        'total_files': len(df),
        'centers': df['center'].unique().tolist(),
        'classes': df['class_name'].unique().tolist(),
        'class_mapping': {'ndbe': 0, 'neo': 1},
        'split_files': {
            '5fold_cv.csv': 'Stratified 5-fold cross-validation preserving class ratios across centers',
            'center1_train_center2_test.csv': 'Train on center_1, test on center_2',
            'center2_train_center1_test.csv': 'Train on center_2, test on center_1',
            'holdout_cv.csv': '20% stratified holdout test set + 5-fold CV on remaining 80%'
        }
    }
    
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    sanity_check_splits()


if __name__ == "__main__":
    main()