"""
Script to fetch Hugging Face model download counts and add them to overlap CSV files.
Processes 4 CSV files and creates new versions with download counts and sorted by downloads.
"""

import pandas as pd
import requests
import time
from pathlib import Path
from typing import Dict, Optional
import json

def get_hf_downloads(model_name: str, max_retries: int = 3) -> Optional[int]:
    """
    Fetch download count for a model from Hugging Face API.
    
    Args:
        model_name: Model name in format "owner/model-name"
        max_retries: Maximum number of retry attempts
        
    Returns:
        Download count or None if failed
    """
    api_url = f"https://huggingface.co/api/models/{model_name}"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # Downloads field might be in different locations
                downloads = data.get('downloads', data.get('downloadsAllTime', 0))
                return downloads
            elif response.status_code == 404:
                print(f"  Model not found: {model_name}")
                return 0
            elif response.status_code == 429:
                # Rate limited, wait longer
                wait_time = (attempt + 1) * 5
                print(f"  Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  HTTP {response.status_code} for {model_name}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching {model_name}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            
    return None

def process_csv_file(input_path: str, output_path: str):
    """
    Process a single CSV file: add download counts and sort by downloads.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
    """
    print(f"\n{'='*80}")
    print(f"Processing: {Path(input_path).name}")
    print(f"{'='*80}\n")
    
    # Read CSV
    df = pd.read_csv(input_path)
    
    # Initialize new columns
    df['base_model_downloads'] = None
    df['derived_model_downloads'] = None
    
    # Get unique model names
    all_models = set(df['base_model_name'].unique()) | set(df['derived_model_name'].unique())
    
    # Cache for downloads to avoid duplicate API calls
    download_cache: Dict[str, Optional[int]] = {}
    
    print(f"Found {len(all_models)} unique models to fetch")
    print(f"Total rows: {len(df)}\n")
    
    # Fetch downloads for all unique models
    for i, model_name in enumerate(sorted(all_models), 1):
        if model_name not in download_cache:
            print(f"[{i}/{len(all_models)}] Fetching: {model_name}")
            downloads = get_hf_downloads(model_name)
            download_cache[model_name] = downloads
            
            if downloads is not None:
                print(f"  ✓ Downloads: {downloads:,}")
            else:
                print(f"  ✗ Failed to fetch")
            
            # Rate limiting: wait between requests
            time.sleep(0.5)
    
    print(f"\n{'='*80}")
    print("Populating download counts in dataframe...")
    print(f"{'='*80}\n")
    
    # Populate download counts
    df['base_model_downloads'] = df['base_model_name'].map(download_cache)
    df['derived_model_downloads'] = df['derived_model_name'].map(download_cache)
    
    # Sort by derived model downloads (descending), then by base model downloads
    df_sorted = df.sort_values(
        by=['derived_model_downloads', 'base_model_downloads'], 
        ascending=[False, False],
        na_position='last'
    )
    
    # Reorder columns to put downloads near the front
    cols = df_sorted.columns.tolist()
    # Move download columns after model names
    cols.remove('base_model_downloads')
    cols.remove('derived_model_downloads')
    cols.insert(2, 'base_model_downloads')
    cols.insert(3, 'derived_model_downloads')
    df_sorted = df_sorted[cols]
    
    # Save to new CSV
    df_sorted.to_csv(output_path, index=False)
    
    # Print statistics
    print(f"Statistics for {Path(input_path).name}:")
    print(f"  Total rows: {len(df_sorted)}")
    print(f"  Rows with base downloads: {df_sorted['base_model_downloads'].notna().sum()}")
    print(f"  Rows with derived downloads: {df_sorted['derived_model_downloads'].notna().sum()}")
    print(f"  Avg base downloads: {df_sorted['base_model_downloads'].mean():,.0f}")
    print(f"  Avg derived downloads: {df_sorted['derived_model_downloads'].mean():,.0f}")
    print(f"  Median derived downloads: {df_sorted['derived_model_downloads'].median():,.0f}")
    print(f"\nSaved to: {output_path}\n")

def main():
    """Main function to process all 4 CSV files."""
    
    base_dir = Path("/Users/kenzieluo/Desktop/columbia/course/model_linage/llm_fingerprint/result/result_12.15/wordlist")
    
    files = [
        ("lineage_bottomk_overlap_llama_diff.csv", "lineage_bottomk_overlap_llama_diff_with_downloads.csv"),
        ("lineage_bottomk_overlap_llama_same.csv", "lineage_bottomk_overlap_llama_same_with_downloads.csv"),
        ("lineage_bottomk_overlap_qwen_diff.csv", "lineage_bottomk_overlap_qwen_diff_with_downloads.csv"),
        ("lineage_bottomk_overlap_qwen_same.csv", "lineage_bottomk_overlap_qwen_same_with_downloads.csv"),
    ]
    
    print("\n" + "="*80)
    print("Hugging Face Model Download Fetcher")
    print("="*80)
    print("\nThis script will:")
    print("1. Fetch download counts from Hugging Face for all models")
    print("2. Add download counts to the CSV files")
    print("3. Sort by download counts (descending)")
    print("4. Save new CSV files with '_with_downloads' suffix")
    print("\n" + "="*80)
    
    start_time = time.time()
    
    for input_file, output_file in files:
        input_path = base_dir / input_file
        output_path = base_dir / output_file
        
        try:
            process_csv_file(str(input_path), str(output_path))
        except Exception as e:
            print(f"\n❌ Error processing {input_file}: {e}\n")
            continue
    
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print(f"✓ All files processed in {elapsed:.1f} seconds")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
