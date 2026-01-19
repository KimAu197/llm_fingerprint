# CSV Data Format Guide

This guide explains how to prepare your own CSV files for fine-tuning with `train_and_eval_overlap.py`.

## Supported Formats

The script supports two CSV formats:

### Format 1: Simple Text Format

A single column containing the training text.

**CSV Structure:**
```csv
text
"Your training text here..."
"Another training example..."
"More training data..."
```

**Example:** `example_data_simple.csv`

**Usage:**
```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --csv_path "./example_data_simple.csv" \
    --text_column "text" \
    --output_dir "./my_experiment"
```

---

### Format 2: Instruction Format

Three columns for instruction-based training (like Alpaca format).

**CSV Structure:**
```csv
instruction,input,output
"Task description","Input context (optional)","Expected output"
"Another task","Another input","Another output"
```

**Columns:**
- `instruction`: The task or question
- `input`: Additional context (can be empty)
- `output`: The expected response

**Example:** `example_data_instruction.csv`

**Usage:**
```bash
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --csv_path "./example_data_instruction.csv" \
    --instruction_column "instruction" \
    --input_column "input" \
    --output_column "output" \
    --output_dir "./my_experiment"
```

**Formatted Output:**
The script automatically formats instruction data as:
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

If `input` is empty, it becomes:
```
### Instruction:
{instruction}

### Response:
{output}
```

---

## Creating Your Own CSV

### Option 1: Simple Text Format

1. Create a CSV file with a `text` column
2. Each row contains one training example
3. Text can be any length (will be truncated to `max_length`)

**Example:**
```csv
text
"Explain quantum computing."
"What is the theory of relativity?"
"How do neural networks work?"
```

**Python code to create:**
```python
import pandas as pd

data = {
    "text": [
        "Explain quantum computing.",
        "What is the theory of relativity?",
        "How do neural networks work?"
    ]
}

df = pd.DataFrame(data)
df.to_csv("my_training_data.csv", index=False)
```

---

### Option 2: Instruction Format

1. Create a CSV with `instruction`, `input`, `output` columns
2. `input` can be empty for some rows
3. Good for task-specific fine-tuning

**Example:**
```csv
instruction,input,output
"Translate to Spanish","Hello world","Hola mundo"
"Explain the concept","photosynthesis","Photosynthesis is the process by which plants..."
"Summarize","Long text here...","Brief summary here..."
```

**Python code to create:**
```python
import pandas as pd

data = {
    "instruction": [
        "Translate to Spanish",
        "Explain the concept",
        "Summarize"
    ],
    "input": [
        "Hello world",
        "photosynthesis",
        "Long text here..."
    ],
    "output": [
        "Hola mundo",
        "Photosynthesis is the process...",
        "Brief summary here..."
    ]
}

df = pd.DataFrame(data)
df.to_csv("my_instruction_data.csv", index=False)
```

---

## Custom Column Names

If your CSV uses different column names, specify them:

```bash
python train_and_eval_overlap.py \
    --csv_path "./my_data.csv" \
    --text_column "content" \              # Instead of "text"
    --instruction_column "task" \          # Instead of "instruction"
    --input_column "context" \             # Instead of "input"
    --output_column "response" \           # Instead of "output"
    --output_dir "./my_experiment"
```

---

## Data Quality Tips

### 1. **Size Recommendations**

| Purpose | Minimum Rows | Recommended Rows |
|---------|--------------|------------------|
| Quick test | 100 | 500 |
| Experiment | 1,000 | 5,000 |
| Serious training | 10,000 | 50,000+ |

### 2. **Text Length**

- Default `max_length` is 512 tokens
- Longer texts will be truncated
- Shorter texts will be padded
- Adjust with `--max_length` parameter

### 3. **Data Diversity**

- Include varied examples
- Cover different topics/tasks
- Avoid too much repetition
- Balance different types of content

### 4. **Data Cleaning**

- Remove HTML tags if present
- Fix encoding issues
- Remove duplicate entries
- Check for empty rows

**Python cleaning example:**
```python
import pandas as pd

# Load data
df = pd.read_csv("raw_data.csv")

# Remove duplicates
df = df.drop_duplicates(subset=['text'])

# Remove empty rows
df = df.dropna(subset=['text'])

# Remove very short texts
df = df[df['text'].str.len() > 20]

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)
```

---

## Example Workflows

### Workflow 1: Convert Existing Text Files

```python
import pandas as pd
import glob

# Read all text files from a directory
texts = []
for file_path in glob.glob("./texts/*.txt"):
    with open(file_path, 'r', encoding='utf-8') as f:
        texts.append(f.read())

# Create CSV
df = pd.DataFrame({"text": texts})
df.to_csv("compiled_training_data.csv", index=False)
```

### Workflow 2: Convert JSON to CSV

```python
import pandas as pd
import json

# Load JSON data
with open("data.json", 'r') as f:
    data = json.load(f)

# Convert to DataFrame
# Assuming JSON format: [{"text": "..."}, {"text": "..."}, ...]
df = pd.DataFrame(data)

# Or for instruction format:
# [{"instruction": "...", "input": "...", "output": "..."}, ...]
df = pd.DataFrame(data)

# Save as CSV
df.to_csv("training_data.csv", index=False)
```

### Workflow 3: Sample from Large Dataset

```python
import pandas as pd

# Load large CSV
df = pd.read_csv("large_dataset.csv")

# Sample 10,000 random rows
df_sample = df.sample(n=10000, random_state=42)

# Save sample
df_sample.to_csv("training_sample.csv", index=False)
```

---

## Testing Your CSV

Before running a full experiment, test your CSV:

```bash
# Quick test with limited samples
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --csv_path "./my_data.csv" \
    --output_dir "./test_run" \
    --max_steps 10 \
    --eval_steps 5 \
    --num_fingerprints 5 \
    --num_train_samples 100
```

This will:
- Use only 100 samples from your CSV
- Train for just 10 steps
- Verify the CSV format is correct
- Complete in a few minutes

---

## Common Issues and Solutions

### Issue 1: "Column not found"

**Error:** `KeyError: 'text'`

**Solution:** Specify the correct column name
```bash
--text_column "your_column_name"
```

### Issue 2: "Empty dataset"

**Error:** `ValueError: Dataset is empty`

**Solution:** 
- Check CSV has data rows (not just header)
- Verify encoding (should be UTF-8)
- Check for null values

### Issue 3: "Memory error"

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```bash
--num_train_samples 5000 \        # Limit dataset size
--per_device_train_batch_size 2 \  # Reduce batch size
--gradient_accumulation_steps 8    # Increase accumulation
```

### Issue 4: "Encoding error"

**Error:** `UnicodeDecodeError`

**Solution:** Save CSV with UTF-8 encoding
```python
df.to_csv("data.csv", index=False, encoding='utf-8')
```

---

## Complete Example

Here's a complete workflow from raw data to training:

```python
# Step 1: Prepare data
import pandas as pd

data = {
    "instruction": [
        "Explain the concept",
        "Translate to French",
        "Summarize the text"
    ],
    "input": [
        "machine learning",
        "Hello world",
        "Long article text..."
    ],
    "output": [
        "Machine learning is...",
        "Bonjour le monde",
        "Brief summary..."
    ]
}

df = pd.DataFrame(data)
df.to_csv("my_training_data.csv", index=False)
```

```bash
# Step 2: Train with overlap evaluation
python train_and_eval_overlap.py \
    --base_model_name "Qwen/Qwen2.5-0.5B" \
    --csv_path "./my_training_data.csv" \
    --instruction_column "instruction" \
    --input_column "input" \
    --output_column "output" \
    --output_dir "./my_experiment" \
    --max_steps 1000 \
    --eval_steps 100 \
    --bottom_k_vocab 2000 \
    --num_fingerprints 20
```

```bash
# Step 3: Visualize results
python plot_overlap_vs_steps.py \
    --result_dir "./my_experiment"
```

---

## Summary

**Simple Format:**
- Single `text` column
- Good for: general text, articles, conversations
- Example: `example_data_simple.csv`

**Instruction Format:**
- Three columns: `instruction`, `input`, `output`
- Good for: task-specific training, Q&A, instructions
- Example: `example_data_instruction.csv`

**Both formats:**
- Support custom column names
- Automatically formatted for training
- Can be any size (will be tokenized to `max_length`)

For more help, see:
- `QUICK_START_OVERLAP.md` - Quick start guide
- `OVERLAP_EXPERIMENT_README.md` - Detailed documentation
- Example files in this directory
