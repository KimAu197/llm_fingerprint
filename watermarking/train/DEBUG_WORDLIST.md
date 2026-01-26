# Debug Wordlist Changes

## 🔍 New Debug Features

The training script now saves detailed wordlist information at each evaluation step to help you understand why overlap is changing.

---

## 📁 Debug Files Generated

For each evaluation step, the script creates:

```
output_dir/
├── wordlist_debug_step_1.json     # Step 1 wordlists
├── wordlist_debug_step_100.json   # Step 100 wordlists
├── wordlist_debug_step_200.json   # Step 200 wordlists
└── ...
```

### File Format

```json
{
  "step": 100,
  "avg_overlap": 0.5234,
  "fingerprints": [
    {
      "fingerprint_idx": 0,
      "prompt": "First 100 chars of fingerprint...",
      "base_bottomk_full": [123, 456, 789, ...],     // All 2000 token IDs
      "ft_bottomk_full": [123, 457, 790, ...],       // All 2000 token IDs
      "base_bottomk_preview": [123, 456, ...],       // First 50 for quick view
      "ft_bottomk_preview": [123, 457, ...],         // First 50 for quick view
      "overlap": 0.5234
    },
    // ... more fingerprints
  ]
}
```

---

## 🔧 Analysis Script

Use `analyze_wordlist_changes.py` to understand what's changing:

### Basic Usage

```bash
python analyze_wordlist_changes.py --output_dir ./exp_output
```

### Analyze Specific Steps

```bash
python analyze_wordlist_changes.py \
    --output_dir ./exp_output \
    --steps 1 10 20 50 100 200
```

### Example Output

```
Available steps: [1, 2, 3, 4, 5, 10, 20, 50, 100]
Analyzing steps: [1, 10, 20, 50, 100]

================================================================================
STEP 1 (Avg Overlap: 1.0000)
================================================================================

Fingerprint 0:
  Prompt: The quick brown fox jumps...
  Overlap: 1.0000 (2000/2000)
  Removed: 0 tokens
  Added: 0 tokens

================================================================================
STEP 10 (Avg Overlap: 0.9213)
================================================================================

Fingerprint 0:
  Prompt: The quick brown fox jumps...
  Overlap: 0.9150 (1830/2000)
  Removed: 170 tokens
  Added: 170 tokens
  
  Removed examples: ['ъ', 'ř', 'ė', 'ų', 'ń', ...]
  Added examples: ['Wikipedia', 'article', 'the', 'and', ...]

================================================================================
STEP 100 (Avg Overlap: 0.5234)
================================================================================

Fingerprint 0:
  Prompt: The quick brown fox jumps...
  Overlap: 0.5100 (1020/2000)
  Removed: 980 tokens
  Added: 980 tokens
  
  Removed examples: ['ъ', 'ř', 'ė', 'ų', 'ń', 'ł', 'ş', 'ž', ...]
  Added examples: ['Wikipedia', 'article', 'the', 'and', 'of', 'in', ...]
```

---

## 🎯 What to Look For

### 1. Which Tokens Are Being Removed?

**Good signs (expected):**
- Rare characters: `ъ`, `ř`, `ė`, `ų` (non-English)
- Special symbols
- Uncommon punctuation

**Bad signs (unexpected):**
- Common English words: `the`, `and`, `of`
- Common punctuation: `.`, `,`, `!`

### 2. Which Tokens Are Being Added?

**Good signs (expected):**
- Domain-specific words (e.g., "Wikipedia", "article" for Wikipedia data)
- Common words in your training data
- Natural language tokens

**Bad signs (unexpected):**
- Random tokens
- Garbage characters
- Padding tokens

### 3. Rate of Change

**Normal:**
```
Step 1:   overlap = 1.000 (0 tokens changed)
Step 10:  overlap = 0.915 (170 tokens changed)
Step 20:  overlap = 0.850 (300 tokens changed)
Step 50:  overlap = 0.700 (600 tokens changed)
Step 100: overlap = 0.500 (1000 tokens changed)
```

**Too fast (problem):**
```
Step 1:   overlap = 1.000 (0 tokens changed)
Step 10:  overlap = 0.200 (1600 tokens changed)  ← Too many!
Step 20:  overlap = 0.050 (1900 tokens changed)
```

---

## 🔬 Understanding the Results

### Why Overlap Decreases

During fine-tuning, the model's logit distribution changes:

**Before training (Step 1):**
```
Token probabilities (last position):
  "the": logit = 5.2
  "ъ":   logit = -8.5  ← In bottom-k (low logit)
  "and": logit = 4.8
```

**After training on Wikipedia (Step 100):**
```
Token probabilities (last position):
  "the": logit = 6.5   ← Increased (Wikipedia has many "the")
  "ъ":   logit = -9.2  ← Decreased further (no "ъ" in English Wikipedia)
  "and": logit = 6.2   ← Increased
```

**Result:**
- Rare tokens (like `ъ`) stay in bottom-k or go even lower
- Common tokens (like `the`) might enter bottom-k if they become less probable relative to others
- **The bottom-k set changes → overlap decreases**

### What's Normal?

**For Wikipedia training:**
- Removed: Non-English characters, rare symbols
- Added: Common English words, Wikipedia-specific terms
- Rate: Gradual decrease over 100-1000 steps

---

## 📊 Debugging Your Training

### Step 1: Run Analysis

After training for a bit:

```bash
python analyze_wordlist_changes.py --output_dir ./your_output_dir
```

### Step 2: Check the Results

Look for:

1. **Are removed tokens reasonable?**
   - Should be rare/non-English characters
   - Not common English words

2. **Are added tokens reasonable?**
   - Should be domain-specific (Wikipedia terms)
   - Not random garbage

3. **Is the rate of change reasonable?**
   - Should be gradual, not sudden drop

### Step 3: Compare with Synthia-I

**Synthia-I final overlap: 0.5494**

Your training should show similar trajectory:
- Step 100: ~0.5-0.7
- Step 500: ~0.4-0.6
- Step 1000: ~0.3-0.5

---

## 🎯 Example Debug Session

```bash
# 1. Train for 100 steps
python train_and_eval_overlap.py \
    --output_dir ./debug_run \
    --max_steps 100 \
    --eval_steps 10

# 2. Analyze wordlist changes
python analyze_wordlist_changes.py \
    --output_dir ./debug_run \
    --steps 1 10 20 50 100

# 3. Check the output
# Look at which tokens are being removed/added
# Verify it makes sense for your training data
```

---

## 📝 Quick Reference

### Files Created

| File | Purpose |
|------|---------|
| `wordlist_debug_step_N.json` | Full wordlists at step N |
| `overlap_results.json` | Summary of all overlaps |
| `overlap_summary.csv` | CSV format for plotting |

### Analysis Commands

```bash
# Analyze all available steps
python analyze_wordlist_changes.py --output_dir ./exp

# Analyze specific steps
python analyze_wordlist_changes.py --output_dir ./exp --steps 1 100 500 1000

# Use different tokenizer
python analyze_wordlist_changes.py --output_dir ./exp --tokenizer "Qwen/Qwen2.5-0.5B"
```

---

## 🎉 Summary

**New features:**
1. ✅ Saves full wordlists at each evaluation step
2. ✅ Analysis script to compare wordlists
3. ✅ Shows which tokens are removed/added
4. ✅ Helps debug why overlap is changing

**Use this to verify:**
- Your training is working correctly
- Overlap changes are reasonable
- Token changes make sense for your data

Now you can see exactly what's happening to the bottom-k wordlist during training! 🔍
