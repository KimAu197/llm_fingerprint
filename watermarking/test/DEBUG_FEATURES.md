# Debug Features Added

## New Timing & GPU Monitoring

The experiment script now includes detailed timing and GPU memory tracking for Colab debugging.

## What's Been Added

### 1. Timer Class
Automatically tracks and displays:
- Start time for each operation
- Elapsed time (formatted as HH:MM:SS)
- GPU memory before and after each operation

### 2. GPU Memory Monitoring
For each operation, logs:
- GPU device name
- Allocated memory (GB)
- Reserved memory (GB)
- Free memory (GB)
- Total GPU memory (GB)

### 3. Per-Family Timing
Tracks time for each of the 6 families:
- Time to load base model
- Time to generate fingerprints (default 5)
- Time to compute base model bottom-k vocab
- Time for all positive samples (derivatives vs base)
- Time for all negative samples (cross-family comparisons)
- Total time for the family

### 4. Experiment Summary
At the end, displays:
- Total experiment time
- Per-family breakdown
- Saves timing summary to JSON file

## Example Output

```
⏱️  [Load Base Model (meta-llama/Llama-3.1-8B)] Started at 14:23:15
  [GPU Start] Tesla T4
    Allocated: 0.00 GB
    Reserved:  0.00 GB
    Free:      15.00 GB / 15.00 GB
  Loading: meta-llama/Llama-3.1-8B
    ✓ Loaded in 23.45s
  [GPU After Load] Tesla T4
    Allocated: 14.52 GB
    Reserved:  14.62 GB
    Free:      0.48 GB / 15.00 GB
⏱️  [Load Base Model (meta-llama/Llama-3.1-8B)] Completed in 0:00:24
  [GPU End] Tesla T4
    Allocated: 14.52 GB
    Reserved:  14.62 GB
    Free:      0.48 GB / 15.00 GB

⏱️  [Generate 5 Fingerprints] Started at 14:23:40
  [GPU Start] Tesla T4
    Allocated: 14.52 GB
    Reserved:  14.62 GB
    Free:      0.48 GB / 15.00 GB
    Generating fingerprint 1/5
    Generating fingerprint 2/5
    ...
⏱️  [Generate 5 Fingerprints] Completed in 0:01:23
  [GPU End] Tesla T4
    Allocated: 14.58 GB
    Reserved:  14.62 GB
    Free:      0.42 GB / 15.00 GB
```

## Final Summary

```
================================================================================
EXPERIMENT SUMMARY
================================================================================

Total time: 1:23:45
Families processed: 6

Per-family breakdown:

  1. meta-llama/Llama-3.1-8B
     Time: 0:12:34
     Positive: 5, Negative: 25

  2. Qwen/Qwen2.5-1.5B
     Time: 0:11:23
     Positive: 5, Negative: 25

  ...

================================================================================
EXPERIMENT COMPLETE
================================================================================
Results saved to: test_results/base_family_overlap_results.csv
Timing summary saved to: test_results/experiment_timing_summary.json
```

## JSON Timing Summary

File: `test_results/experiment_timing_summary.json`

```json
{
  "total_time_seconds": 5025.3,
  "total_time_formatted": "1:23:45",
  "num_families": 6,
  "families": [
    {
      "family": "meta-llama/Llama-3.1-8B",
      "time": 754.2,
      "positive_count": 5,
      "negative_count": 25
    },
    ...
  ],
  "config": {
    "num_pairs": 5,
    "num_negative_samples": 5,
    "bottom_k_vocab": 2000,
    "device": "cuda"
  },
  "timestamp": "2026-02-09T14:23:15.123456"
}
```

## How to Use in Colab

### 1. Run Experiment
```python
!python3 test/run_base_family_experiment.py \
    --csv_path ../result/result_2.10/data/experiment_models_base_family.csv \
    --output_dir test_results \
    --num_pairs 5 \
    --num_negative_samples 5
```

### 2. Monitor GPU in Real-Time
The script will automatically log GPU usage after each model load/unload.

### 3. Check Timing Summary
```python
import json

with open('test_results/experiment_timing_summary.json') as f:
    timing = json.load(f)

print(f"Total time: {timing['total_time_formatted']}")
for family in timing['families']:
    print(f"{family['family']}: {family['time']:.1f}s")
```

## Debugging Tips

### Check GPU Memory Issues
Look for patterns in GPU memory:
- Does memory grow after each model?
- Is unload properly freeing memory?
- Which models use most memory?

### Identify Bottlenecks
From timing output, see:
- Which family takes longest?
- Is fingerprint generation slow?
- Are negative samples taking too much time?

### Colab-Specific
Monitor Colab's own GPU widget alongside script output to verify memory reporting is accurate.

## Key Changes Made

1. **Import changes**: Now imports from `utils` instead of `watermarking.utils`
2. **Added timing utilities**: Timer class, GPU memory logging
3. **Added timing to all operations**: Load, generate, compute, test
4. **Per-family statistics**: Track and report time for each family
5. **JSON summary export**: Save detailed timing to file for later analysis

## No Functional Changes

All fingerprinting logic remains the same - only added monitoring/logging!
