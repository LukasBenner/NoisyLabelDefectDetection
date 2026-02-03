"""
Script to split dataset into train, test, and val directories based on splits_clusters.tsv file.
"""
import argparse
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm


def split_dataset(tsv_path: str, output_dir: str, copy: bool = True, verbose: bool = True):
    """
    Split dataset into train, test, and val directories based on TSV file.
    
    Args:
        tsv_path: Path to the splits_clusters.tsv file
        output_dir: Directory where to create train/test/val subdirectories
        copy: If True, copy files. If False, move files.
        verbose: If True, show progress bar and statistics
    """
    # Read the TSV file
    if verbose:
        print(f"Reading {tsv_path}...")
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Validate required columns
    required_cols = ['image_path', 'class_label', 'split']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get unique splits and class labels
    splits = df['split'].unique()
    class_labels = df['class_label'].unique()
    
    if verbose:
        print(f"\nFound {len(splits)} splits: {sorted(splits)}")
        print(f"Found {len(class_labels)} classes: {sorted(class_labels)}")
    
    # Create directory structure
    for split in splits:
        for class_label in class_labels:
            split_dir = output_path / split / class_label
            split_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy or move files
    operation = shutil.copy2 if copy else shutil.move
    operation_name = "Copying" if copy else "Moving"
    
    if verbose:
        print(f"\n{operation_name} files...")
    
    success_count = 0
    error_count = 0
    errors = []
    
    iterator = tqdm(df.iterrows(), total=len(df), desc=operation_name) if verbose else df.iterrows()
    
    for idx, row in iterator:
        source_path = Path(row['image_path'])
        dest_path = output_path / row['split'] / row['class_label'] / source_path.name
        
        try:
            if not source_path.exists():
                errors.append(f"Source file not found: {source_path}")
                error_count += 1
                continue
            
            # If destination exists and we're copying, skip to avoid overwriting
            if dest_path.exists() and copy:
                if verbose:
                    errors.append(f"Destination already exists (skipping): {dest_path}")
                continue
            
            operation(str(source_path), str(dest_path))
            success_count += 1
            
        except Exception as e:
            errors.append(f"Error processing {source_path}: {str(e)}")
            error_count += 1
    
    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Successfully {operation_name.lower()}: {success_count} files")
        print(f"  Errors: {error_count}")
        print(f"{'='*60}")
        
        # Print split statistics
        print(f"\nSplit statistics:")
        for split in sorted(splits):
            count = len(df[df['split'] == split])
            print(f"  {split}: {count} images")
        
        # Print class distribution per split
        print(f"\nClass distribution:")
        for split in sorted(splits):
            print(f"\n  {split}:")
            split_df = df[df['split'] == split]
            for class_label in sorted(class_labels):
                count = len(split_df[split_df['class_label'] == class_label])
                print(f"    {class_label}: {count}")
        
        # Print errors if any
        if errors and len(errors) <= 10:
            print(f"\nErrors:")
            for error in errors:
                print(f"  {error}")
        elif errors:
            print(f"\nFirst 10 errors:")
            for error in errors[:10]:
                print(f"  {error}")
            print(f"  ... and {len(errors) - 10} more errors")


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/test/val directories based on splits_clusters.tsv"
    )
    parser.add_argument(
        '--tsv',
        type=str,
        default='splits_clusters.tsv',
        help='Path to the splits_clusters.tsv file (default: splits_clusters.tsv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory where train/test/val subdirectories will be created'
    )
    parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying them (default: copy)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress bar and detailed output'
    )
    
    args = parser.parse_args()
    
    split_dataset(
        tsv_path=args.tsv,
        output_dir=args.output,
        copy=not args.move,
        verbose=not args.quiet
    )
    
    print(f"\nDataset split complete! Output directory: {args.output}")


if __name__ == '__main__':
    main()
