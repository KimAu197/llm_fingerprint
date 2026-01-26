"""
train_and_eval_overlap.py

Fine-tune a base model and evaluate wordlist overlap at different training steps.

This script extends the overlap testing methodology from run_bottomk_subspace_overlap_from_base.py
(in the 'new' directory) to track overlap changes during the fine-tuning process.

Key functions used from the existing codebase:
- compute_bottomk_vocab_for_model(): Compute bottom-k vocabulary (from utils)
- sample_fingerprint_prompt(): Generate fingerprint prompts (from utils)
- overlap_ratio(): Calculate overlap between two wordlists (same as in new/run_bottomk_subspace_overlap_from_base.py)

This script:
1. Fine-tunes a base model on a specified dataset
2. At specified checkpoint intervals, evaluates the overlap ratio between:
   - The fine-tuned model's bottom-k wordlist
   - The original base model's bottom-k wordlist
3. Saves results to track how overlap changes with training steps
4. Logs metrics to Weights & Biases (wandb) for visualization

Usage:
    # Using Wikipedia English with wandb
    python train_and_eval_overlap.py \
        --base_model_name "Qwen/Qwen2.5-0.5B" \
        --dataset_name "wikimedia/wikipedia" \
        --dataset_config "20231101.en" \
        --output_dir "./ft_overlap_experiment" \
        --max_steps 1000 \
        --eval_steps 100 \
        --use_wandb \
        --wandb_project "model-overlap" \
        --wandb_run_name "wiki_en_experiment"
    
    # Using Wikipedia Japanese
    python train_and_eval_overlap.py \
        --base_model_name "Qwen/Qwen2.5-0.5B" \
        --dataset_name "wikimedia/wikipedia" \
        --dataset_config "20231101.ja" \
        --output_dir "./ft_overlap_experiment_ja" \
        --max_steps 1000 \
        --eval_steps 100
    
    # Using custom CSV file
    python train_and_eval_overlap.py \
        --base_model_name "Qwen/Qwen2.5-0.5B" \
        --csv_path "./my_training_data.csv" \
        --text_column "text" \
        --output_dir "./ft_overlap_experiment" \
        --max_steps 1000 \
        --eval_steps 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[warn] wandb not installed. Install with: pip install wandb")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import overlap testing functions from the 'new' directory
# These are the same functions used in run_bottomk_subspace_overlap_from_base.py
try:
    from utils import (
        compute_bottomk_vocab_for_model,
        sample_fingerprint_prompt,
        set_seed,
    )
    print("[info] Successfully imported utils functions")
except ImportError:
    print("[warn] Could not import from utils, using fallback implementations")
    
    def set_seed(seed: int):
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def compute_bottomk_vocab_for_model(
        model,
        tokenizer,
        k: int = 2000,
        device: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> List[int]:
        """
        Compute bottom-k vocabulary for a model.
        
        This is the same function used in run_bottomk_subspace_overlap_from_base.py
        
        Args:
            model: HuggingFace causal LM
            tokenizer: Corresponding tokenizer
            k: Size of bottom-k vocabulary
            device: Device to run on (cuda/mps/cpu)
            prompt: Prompt text for context
            
        Returns:
            List of k token IDs with lowest logits
        """
        model.eval()
        
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = "cpu"
        
        if isinstance(device, torch.device):
            device = device.type
        
        # Use BOS token or default prompt
        if prompt is None:
            if tokenizer.bos_token is not None:
                prompt = tokenizer.bos_token
            else:
                prompt = "Fingerprint base prompt."
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
        
        vocab_size = logits.shape[0]
        k = min(k, vocab_size)
        
        # Get k tokens with lowest logits
        _, bottomk_indices = torch.topk(logits, k=k, largest=False)
        bottomk_ids = bottomk_indices.tolist()
        return bottomk_ids
    
    def sample_fingerprint_prompt(
        model,
        tokenizer,
        device: Optional[str] = None,
        l_random_prefix: int = 8,
        total_len: int = 64,
        k_bottom: int = 50,
    ) -> str:
        """
        Generate a fingerprint prompt using bottom-k sampling.
        
        This is the same function used in run_bottomk_subspace_overlap_from_base.py
        
        Args:
            model: HuggingFace causal LM
            tokenizer: Corresponding tokenizer
            device: Device to run on
            l_random_prefix: Length of random prefix
            total_len: Total length of fingerprint
            k_bottom: k for bottom-k sampling
            
        Returns:
            Fingerprint prompt string
        """
        import random
        import torch.nn.functional as F
        
        model.eval()
        if device is None:
            device = next(model.parameters()).device
        
        # Build allowed token set (exclude special tokens)
        special_ids = set()
        for attr in ['bos_token_id', 'eos_token_id', 'pad_token_id', 'unk_token_id']:
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                special_ids.add(tid)
        
        allowed = [tid for tid in range(tokenizer.vocab_size) if tid not in special_ids]
        allowed_t = torch.tensor(allowed, device=device)
        
        # Generate random prefix
        prefix_ids = random.choices(allowed, k=l_random_prefix)
        prompt_ids = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
        
        # Extend using bottom-k sampling
        while prompt_ids.shape[1] < total_len:
            with torch.no_grad():
                logits = model(prompt_ids).logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Mask special tokens
            masked = probs.clone()
            if len(allowed) < tokenizer.vocab_size:
                mask = torch.ones_like(masked, dtype=torch.bool)
                mask[:, allowed_t] = False
                masked[mask] = 1e9  # High value to exclude from bottom-k
            
            # Sort by probability (ascending) and sample from bottom-k
            _, sorted_idx = torch.sort(masked, dim=-1, descending=False)
            k_eff = min(k_bottom, sorted_idx.shape[1])
            bottomk_idx = sorted_idx[:, :k_eff]
            next_id = bottomk_idx[0, random.randrange(k_eff)].view(1, 1)
            prompt_ids = torch.cat([prompt_ids, next_id.to(device)], dim=1)
        
        return tokenizer.decode(prompt_ids.squeeze(0).cpu(), skip_special_tokens=True)


def overlap_ratio(set_a: List[int], set_b: List[int]) -> float:
    """
    Compute overlap ratio between two sets.
    
    Args:
        set_a: First set of token IDs
        set_b: Second set of token IDs
        
    Returns:
        |intersection| / |set_a|
    """
    sa, sb = set(set_a), set(set_b)
    if len(sa) == 0:
        return 0.0
    return len(sa.intersection(sb)) / float(len(sa))


def load_csv_dataset(
    csv_path: str,
    text_column: str = "text",
) -> Dataset:
    """
    Load dataset from CSV file.
    
    The CSV should have a text column containing the training text.
    No special formatting is applied - text is used as-is.
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of text column
        
    Returns:
        HuggingFace Dataset object
    """
    import pandas as pd
    
    print(f"[data] Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"[data] CSV columns: {list(df.columns)}")
    print(f"[data] Number of rows: {len(df)}")
    
    if text_column not in df.columns:
        raise ValueError(
            f"CSV must contain '{text_column}' column. Found: {list(df.columns)}"
        )
    
    print(f"[data] Using text column: {text_column}")
    dataset = Dataset.from_dict({"text": df[text_column].tolist()})
    
    return dataset


def load_and_prepare_dataset(
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    csv_path: Optional[str] = None,
    text_column: str = "text",
    tokenizer = None,
    max_length: Optional[int] = None,
    num_samples: Optional[int] = None,
):
    """
    Load and tokenize dataset for fine-tuning.
    
    No special formatting is applied - text is used as-is for training.
    This matches the evaluation approach where only the raw prompt is used.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "wikimedia/wikipedia")
        dataset_config: Dataset config/subset (e.g., "20231101.en" for English Wikipedia)
        csv_path: Path to custom CSV file
        text_column: Column name for text data
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (None = use tokenizer's model_max_length)
        num_samples: Limit number of samples (None = use all)
        
    Returns:
        Tokenized dataset ready for training
    """
    
    # Use tokenizer's default max_length if not specified
    if max_length is None:
        max_length = tokenizer.model_max_length
        # Some tokenizers have absurdly large max_length (e.g., 1000000000000000019884624838600)
        if max_length > 100000:
            max_length = 2048
            print(f"[data] Tokenizer max_length too large ({tokenizer.model_max_length}), capping at 2048")
        else:
            print(f"[data] Using tokenizer's model_max_length: {max_length}")
    else:
        print(f"[data] Using specified max_length: {max_length}")
    
    # Load from CSV or HuggingFace
    if csv_path:
        dataset = load_csv_dataset(
            csv_path=csv_path,
            text_column=text_column,
        )
    
    elif dataset_name:
        print(f"[data] Loading HuggingFace dataset: {dataset_name}")
        if dataset_config:
            print(f"[data] Using config: {dataset_config}")
        
        # Use streaming mode for large datasets to avoid downloading everything
        # This is especially important for Wikipedia which has 41 large files
        print(f"[data] Using streaming mode to avoid downloading entire dataset")
        
        if dataset_config:
            dataset_stream = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
        else:
            dataset_stream = load_dataset(dataset_name, split="train", streaming=True)
        
        # Check for text column (peek at first example)
        first_example = next(iter(dataset_stream))
        if "text" not in first_example:
            raise ValueError(
                f"Dataset must have a 'text' column. Found: {list(first_example.keys())}"
            )
        
        print(f"[data] Dataset columns: {list(first_example.keys())}")
        
        # Convert streaming dataset to regular dataset with limited samples
        # This avoids downloading the entire dataset
        if num_samples is None:
            # Calculate approximate samples needed based on training steps
            # Assume: steps * batch_size * gradient_accumulation_steps
            # For safety, use 2x the calculated amount
            print(f"[warn] No num_samples specified. This may download a lot of data.")
            print(f"[warn] Recommend setting --num_train_samples based on your max_steps")
            num_samples = 100000  # Default cap
        
        print(f"[data] Taking {num_samples} samples from streaming dataset")
        dataset = Dataset.from_dict({
            "text": [example["text"] for example, _ in zip(dataset_stream, range(num_samples))]
        })
        
        print(f"[data] Loaded {len(dataset)} samples")
    
    else:
        raise ValueError("Must provide either --dataset_name or --csv_path")
    
    # Limit samples if specified
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
        print(f"[data] Limited to {num_samples} samples")
    
    print(f"[data] Dataset size: {len(dataset)}")
    
    # Tokenize
    def tokenize_function(examples):
        """
        Tokenize text examples.
        
        Note: Using padding=False for dynamic padding by DataCollatorForLanguageModeling.
        This is more efficient than padding all sequences to max_length.
        """
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # Dynamic padding by data collator
        )
        return tokenized
    
    # Print sample data for debugging
    print(f"\n[data] Sample training data (first 3 examples):")
    for i in range(min(3, len(dataset))):
        sample_text = dataset[i]["text"]
        print(f"  Example {i+1}: {len(sample_text)} chars")
        print(f"    First 100 chars: {sample_text[:100]}")
        if len(sample_text) > 100:
            print(f"    Last 50 chars: ...{sample_text[-50:]}")
        print()
    
    print(f"[data] Tokenizing {len(dataset)} samples...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    
    # Print tokenization stats
    sample_lengths = [len(tokenized_dataset[i]["input_ids"]) for i in range(min(100, len(tokenized_dataset)))]
    avg_length = sum(sample_lengths) / len(sample_lengths)
    max_length_sample = max(sample_lengths)
    min_length_sample = min(sample_lengths)
    print(f"[data] Token length stats (first 100 samples):")
    print(f"  Average: {avg_length:.1f} tokens")
    print(f"  Min: {min_length_sample} tokens")
    print(f"  Max: {max_length_sample} tokens")
    
    return tokenized_dataset


class OverlapEvaluationCallback(TrainerCallback):
    """Callback to evaluate overlap ratio at specified intervals during training."""
    
    def __init__(
        self,
        base_model,
        base_tokenizer,
        base_bottomk_cache: Dict[str, List[int]],
        fingerprints: List[Dict[str, Any]],
        eval_steps: int,
        bottom_k_vocab: int,
        device: str,
        output_dir: str,
        use_wandb: bool = False,
    ):
        """
        Initialize overlap evaluation callback.
        
        Args:
            base_model: Reference base model (frozen)
            base_tokenizer: Tokenizer for base model
            base_bottomk_cache: Pre-computed bottom-k wordlists for base model
            fingerprints: List of fingerprint prompts
            eval_steps: Evaluate every N steps
            bottom_k_vocab: Size of bottom-k vocabulary
            device: Device to run evaluation on
            output_dir: Directory to save results
            use_wandb: Whether to log to wandb
        """
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.base_bottomk_cache = base_bottomk_cache
        self.fingerprints = fingerprints
        self.eval_steps = eval_steps
        self.bottom_k_vocab = bottom_k_vocab
        self.device = device
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.results = []
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        Evaluate overlap at specified step intervals.
        
        Called after each training step. Computes overlap ratio between
        fine-tuned model and base model at regular intervals.
        """
        current_step = state.global_step
        
        # Evaluate at specified intervals
        if current_step % self.eval_steps == 0 or current_step == 1:
            print(f"\n[eval] Evaluating overlap at step {current_step}...")
            
            # CRITICAL: Set model to eval mode for consistent evaluation
            model.eval()
            
            overlap_scores = []
            for idx, fp in enumerate(self.fingerprints):
                prompt_text = fp.get("x_prime", fp.get("prompt", ""))
                
                # Get base model's bottom-k for this prompt
                base_bottomk = self.base_bottomk_cache[prompt_text]
                
                # Compute fine-tuned model's bottom-k for this prompt
                ft_bottomk = compute_bottomk_vocab_for_model(
                    model, self.base_tokenizer, k=self.bottom_k_vocab,
                    device=self.device, prompt=prompt_text
                )
                
                # Calculate overlap ratio
                overlap = overlap_ratio(ft_bottomk, base_bottomk)
                overlap_scores.append(overlap)
            
            # Return model to training mode
            model.train()
            
            avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
            
            result = {
                "step": current_step,
                "avg_overlap_ratio": avg_overlap,
                "overlap_scores": overlap_scores,
            }
            self.results.append(result)
            
            print(f"[eval] Step {current_step}: Average overlap = {avg_overlap:.4f}")
            
            # Log to wandb if enabled
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "overlap/avg_overlap_ratio": avg_overlap,
                    "overlap/min_overlap": min(overlap_scores) if overlap_scores else 0.0,
                    "overlap/max_overlap": max(overlap_scores) if overlap_scores else 0.0,
                    "step": current_step,
                })
            
            # Save intermediate results
            results_path = os.path.join(self.output_dir, "overlap_results.json")
            with open(results_path, "w") as f:
                json.dump(self.results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune model and track overlap ratio with base model"
    )
    
    # Model arguments
    parser.add_argument(
        "--base_model_name", type=str, required=True,
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "--device", type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/mps/cpu)"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset_name", type=str, default=None,
        help="HuggingFace dataset name (e.g., 'wikimedia/wikipedia')"
    )
    parser.add_argument(
        "--dataset_config", type=str, default=None,
        help="Dataset config/subset (e.g., '20231101.en' for English Wikipedia, '20231101.ja' for Japanese)"
    )
    parser.add_argument(
        "--csv_path", type=str, default=None,
        help="Path to custom CSV file with training data"
    )
    parser.add_argument(
        "--text_column", type=str, default="text",
        help="Column name for text data in CSV or dataset (default: 'text')"
    )
    parser.add_argument(
        "--max_length", type=int, default=None,
        help="Maximum sequence length for training (None = use tokenizer's model_max_length, typically 2048-32768)"
    )
    parser.add_argument(
        "--num_train_samples", type=int, default=None,
        help="Limit number of training samples (None = use all)"
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for checkpoints and results"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000,
        help="Maximum number of training steps"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=100,
        help="Evaluate overlap every N steps"
    )
    parser.add_argument(
        "--save_steps", type=int, default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=4,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-6,
        help="Learning rate for optimizer (default: 5e-6, lower to prevent NaN)"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100,
        help="Number of warmup steps for learning rate scheduler"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100,
        help="Log training metrics every N steps (default: 100)"
    )
    
    # Wandb arguments
    parser.add_argument(
        "--use_wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="model-overlap",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None,
        help="Wandb run name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--wandb_api_key", type=str, default=None,
        help="Wandb API key (or set WANDB_API_KEY environment variable)"
    )
    
    # Overlap evaluation arguments
    parser.add_argument(
        "--bottom_k_vocab", type=int, default=2000,
        help="Size of bottom-k vocabulary for overlap computation"
    )
    parser.add_argument(
        "--num_fingerprints", type=int, default=5,
        help="Number of fingerprint prompts to generate for evaluation"
    )
    parser.add_argument(
        "--fingerprint_total_len", type=int, default=64,
        help="Total length of each fingerprint prompt"
    )
    parser.add_argument(
        "--k_bottom_random_prefix", type=int, default=50,
        help="k value for bottom-k sampling in fingerprint generation"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--load_fingerprints", type=str, default=None,
        help="Path to load pre-generated fingerprints (JSON file)"
    )
    parser.add_argument(
        "--save_fingerprints", type=str, default=None,
        help="Path to save generated fingerprints (JSON file)"
    )
    parser.add_argument(
        "--use_fp16", action="store_true",
        help="Use FP16 mixed precision training (may cause issues on some systems)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.dataset_name and not args.csv_path:
        raise ValueError("Must provide either --dataset_name or --csv_path")
    
    # Initialize wandb if requested
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("[error] wandb requested but not installed. Install with: pip install wandb")
            print("[error] Continuing without wandb...")
            args.use_wandb = False
        else:
            # Set API key if provided
            if args.wandb_api_key:
                os.environ["WANDB_API_KEY"] = args.wandb_api_key
            
            # Auto-generate run name if not provided
            if not args.wandb_run_name:
                dataset_name = args.dataset_name or "csv"
                if args.dataset_config:
                    dataset_name += f"_{args.dataset_config}"
                args.wandb_run_name = f"{dataset_name}_steps{args.max_steps}"
            
            # Initialize wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                tags=["overlap-experiment", args.base_model_name.split("/")[-1]],
            )
            print(f"[wandb] Initialized: {args.wandb_project}/{args.wandb_run_name}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save experiment arguments
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 80)
    print("Fine-tuning with Overlap Evaluation")
    print("=" * 80)
    print(f"Base model: {args.base_model_name}")
    if args.csv_path:
        print(f"Training data: {args.csv_path} (CSV)")
    else:
        dataset_info = args.dataset_name
        if args.dataset_config:
            dataset_info += f" (config: {args.dataset_config})"
        print(f"Training data: {dataset_info}")
    print(f"Max steps: {args.max_steps}")
    print(f"Eval steps: {args.eval_steps}")
    print(f"Bottom-k vocab size: {args.bottom_k_vocab}")
    print(f"Number of fingerprints: {args.num_fingerprints}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # ========== Step 1: Load base model (for reference) ==========
    print("\n[1/6] Loading base model for reference...")
    base_model_ref = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name,
        trust_remote_code=True,
    )
    
    # Setup padding token
    if base_tokenizer.pad_token_id is None:
        if base_tokenizer.eos_token_id is not None:
            base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
        else:
            base_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            base_model_ref.resize_token_embeddings(len(base_tokenizer))
    
    base_model_ref.to(args.device)
    base_model_ref.eval()
    
    # ========== Step 2: Generate or load fingerprints ==========
    print("\n[2/6] Generating/loading fingerprints...")
    if args.load_fingerprints and Path(args.load_fingerprints).exists():
        print(f"Loading fingerprints from: {args.load_fingerprints}")
        with open(args.load_fingerprints, "r") as f:
            fingerprints = json.load(f)
    else:
        print(f"Generating {args.num_fingerprints} fingerprints...")
        fingerprints = []
        for i in range(args.num_fingerprints):
            print(f"  Generating fingerprint {i+1}/{args.num_fingerprints}")
            x_prime = sample_fingerprint_prompt(
                base_model_ref,
                base_tokenizer,
                device=args.device,
                l_random_prefix=8,
                total_len=args.fingerprint_total_len,
                k_bottom=args.k_bottom_random_prefix,
            )
            fingerprints.append({"x_prime": x_prime})
        
        if args.save_fingerprints:
            with open(args.save_fingerprints, "w") as f:
                json.dump(fingerprints, f, indent=2)
            print(f"Saved fingerprints to: {args.save_fingerprints}")
    
    # ========== Step 3: Compute base model's bottom-k vocabulary ==========
    print("\n[3/6] Computing base model's bottom-k vocabulary...")
    base_bottomk_cache: Dict[str, List[int]] = {}
    for idx, fp in enumerate(fingerprints):
        prompt_text = fp.get("x_prime", fp.get("prompt", ""))
        print(f"  Computing bottom-k for fingerprint {idx+1}/{len(fingerprints)}")
        base_bottomk_ids = compute_bottomk_vocab_for_model(
            base_model_ref,
            base_tokenizer,
            k=args.bottom_k_vocab,
            device=args.device,
            prompt=prompt_text,
        )
        base_bottomk_cache[prompt_text] = base_bottomk_ids
    
    # Save base bottom-k cache
    cache_path = os.path.join(args.output_dir, "base_bottomk_cache.json")
    with open(cache_path, "w") as f:
        json.dump(base_bottomk_cache, f, indent=2)
    print(f"Saved base bottom-k cache to: {cache_path}")
    
    # ========== Step 4: Load training model ==========
    print("\n[4/6] Loading model for fine-tuning...")
    train_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    train_model.resize_token_embeddings(len(base_tokenizer))
    
    # ========== Step 5: Prepare dataset ==========
    print("\n[5/6] Preparing dataset...")
    train_dataset = load_and_prepare_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        csv_path=args.csv_path,
        text_column=args.text_column,
        tokenizer=base_tokenizer,
        max_length=args.max_length,
        num_samples=args.num_train_samples,
    )
    print(f"Training dataset size: {len(train_dataset)}")
    
    # ========== Step 6: Setup training ==========
    print("\n[6/6] Setting up training...")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=args.use_fp16,  # Only use FP16 if explicitly requested
        max_grad_norm=1.0,  # Gradient clipping to prevent NaN
        report_to="wandb" if args.use_wandb else "none",
        remove_unused_columns=True,
        dataloader_num_workers=2,  # Reduced workers to avoid issues
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=base_tokenizer,
        mlm=False,
    )
    
    # Create overlap evaluation callback
    overlap_callback = OverlapEvaluationCallback(
        base_model=base_model_ref,
        base_tokenizer=base_tokenizer,
        base_bottomk_cache=base_bottomk_cache,
        fingerprints=fingerprints,
        eval_steps=args.eval_steps,
        bottom_k_vocab=args.bottom_k_vocab,
        device=args.device,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
    )
    
    trainer = Trainer(
        model=train_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[overlap_callback],
    )
    
    # ========== Train ==========
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    trainer.train()
    
    # ========== Save final results ==========
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Save final overlap results
    results_path = os.path.join(args.output_dir, "overlap_results.json")
    with open(results_path, "w") as f:
        json.dump(overlap_callback.results, f, indent=2)
    print(f"Overlap results saved to: {results_path}")
    
    # Create summary CSV
    summary_path = os.path.join(args.output_dir, "overlap_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "avg_overlap_ratio"])
        for result in overlap_callback.results:
            writer.writerow([result["step"], result["avg_overlap_ratio"]])
    print(f"Summary CSV saved to: {summary_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("Overlap Ratio Summary:")
    print("=" * 80)
    for result in overlap_callback.results:
        print(f"Step {result['step']:5d}: {result['avg_overlap_ratio']:.4f}")
    
    # Calculate overlap decrease rate
    if len(overlap_callback.results) >= 2:
        first_overlap = overlap_callback.results[0]["avg_overlap_ratio"]
        last_overlap = overlap_callback.results[-1]["avg_overlap_ratio"]
        total_steps = overlap_callback.results[-1]["step"] - overlap_callback.results[0]["step"]
        
        if total_steps > 0:
            decrease_rate = (first_overlap - last_overlap) / total_steps
            print("\n" + "=" * 80)
            print(f"Initial overlap (step {overlap_callback.results[0]['step']}): {first_overlap:.4f}")
            print(f"Final overlap (step {overlap_callback.results[-1]['step']}): {last_overlap:.4f}")
            print(f"Total decrease: {first_overlap - last_overlap:.4f}")
            print(f"Decrease rate: {decrease_rate:.6f} per step")
            print("=" * 80)
    
    print("\n✓ Experiment completed successfully!")
    print(f"All results saved to: {args.output_dir}")
    
    # Finish wandb run
    if args.use_wandb and WANDB_AVAILABLE:
        # Log final summary to wandb
        if len(overlap_callback.results) >= 2:
            first_overlap = overlap_callback.results[0]["avg_overlap_ratio"]
            last_overlap = overlap_callback.results[-1]["avg_overlap_ratio"]
            total_steps = overlap_callback.results[-1]["step"] - overlap_callback.results[0]["step"]
            
            if total_steps > 0:
                decrease_rate = (first_overlap - last_overlap) / total_steps
                wandb.summary["initial_overlap"] = first_overlap
                wandb.summary["final_overlap"] = last_overlap
                wandb.summary["total_decrease"] = first_overlap - last_overlap
                wandb.summary["decrease_rate_per_step"] = decrease_rate
        
        wandb.finish()
        print("\n[wandb] Run finished and logged")


if __name__ == "__main__":
    main()
