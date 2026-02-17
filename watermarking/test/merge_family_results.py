"""
merge_family_results.py

Merge results from individual family experiments into one combined CSV.
"""

import pandas as pd
import argparse
from pathlib import Path


def merge_results(family_dirs, output_path):
    """Merge CSV results from multiple family directories."""
    
    print("="*80)
    print("MERGING FAMILY RESULTS")
    print("="*80)
    
    dfs = []
    
    for family_dir in family_dirs:
        csv_path = Path(family_dir) / "base_family_overlap_results.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded: {csv_path} ({len(df)} rows)")
            dfs.append(df)
        else:
            print(f"⚠️  Not found: {csv_path}")
    
    if not dfs:
        print("\n❌ No result files found!")
        return
    
    # Merge
    combined = pd.concat(dfs, ignore_index=True)
    
    print(f"\n📊 Combined statistics:")
    print(f"   Total rows: {len(combined)}")
    print(f"   Positive samples: {len(combined[combined['sample_type'] == 'positive'])}")
    print(f"   Negative samples: {len(combined[combined['sample_type'] == 'negative'])}")
    
    # Check for duplicates
    duplicates = combined.duplicated(subset=['base_model', 'test_model', 'sample_type'], keep=False)
    num_duplicates = duplicates.sum()
    
    if num_duplicates > 0:
        print(f"\n⚠️  Found {num_duplicates} duplicate entries")
        print("   Removing duplicates...")
        combined = combined.drop_duplicates(subset=['base_model', 'test_model', 'sample_type'], keep='first')
        print(f"   After deduplication: {len(combined)} rows")
    else:
        print(f"\n✅ No duplicates found")
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved merged results to: {output_path}")
    
    # Summary by base model
    print(f"\n{'='*80}")
    print("PER-FAMILY SUMMARY")
    print(f"{'='*80}")
    
    for base in combined['base_model'].unique():
        family_data = combined[combined['base_model'] == base]
        pos = len(family_data[family_data['sample_type'] == 'positive'])
        neg = len(family_data[family_data['sample_type'] == 'negative'])
        print(f"\n{base}")
        print(f"  Positive: {pos}")
        print(f"  Negative: {neg}")
        print(f"  Total: {len(family_data)}")


def main():
    parser = argparse.ArgumentParser(description="Merge family experiment results")
    parser.add_argument(
        '--family_dirs',
        nargs='+',
        default=[
            'test_results_family_3',
            'test_results_family_4',
            'test_results_family_5',
            'test_results_family_6',
        ],
        help='List of family result directories'
    )
    parser.add_argument(
        '--output',
        default='test_results_combined/base_family_overlap_results_all.csv',
        help='Output path for merged CSV'
    )
    
    args = parser.parse_args()
    
    merge_results(args.family_dirs, args.output)


if __name__ == "__main__":
    main()
