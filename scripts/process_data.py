"""Data processing utilities.

Scripts for downloading, preprocessing, and generating datasets.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def generate_mqar_data(
    num_samples: int,
    vocab_size: int = 8192,
    num_kv_pairs: int = 4,
    num_queries: int = 4,
    context_length: int = 2048,
    power_a: float = 0.01,
    seed: int = 42,
) -> Dataset:
    """Generate MQAR (Multi-Query Associative Recall) synthetic dataset.
    
    MQAR tests the model's ability to store and retrieve key-value associations
    from earlier in the sequence.
    
    Args:
        num_samples: Number of samples to generate
        vocab_size: Size of vocabulary
        num_kv_pairs: Number of key-value pairs to insert
        num_queries: Number of queries to test
        context_length: Total sequence length
        power_a: Parameter for Zipf distribution
        seed: Random seed
        
    Returns:
        HuggingFace Dataset with MQAR samples
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    samples = []
    
    print(f"Generating {num_samples} MQAR samples...")
    
    for _ in tqdm(range(num_samples)):
        # Sample keys and values from vocabulary
        keys = np.random.randint(0, vocab_size, size=num_kv_pairs)
        values = np.random.randint(0, vocab_size, size=num_kv_pairs)
        
        # Create sequence
        sequence = []
        
        # Add key-value pairs at the beginning
        for k, v in zip(keys, values):
            sequence.extend([k, v])
        
        # Add filler tokens
        filler_length = context_length - len(sequence) - (num_queries * 2)
        filler = np.random.randint(0, vocab_size, size=filler_length)
        sequence.extend(filler.tolist())
        
        # Add queries (keys) - model should predict corresponding values
        query_positions = []
        for i in range(num_queries):
            query_idx = np.random.randint(0, num_kv_pairs)
            sequence.append(keys[query_idx])
            query_positions.append(len(sequence))
            sequence.append(values[query_idx])  # This is the target
        
        samples.append({
            "input_ids": sequence[:-1],  # All but last token
            "labels": sequence[1:],  # Shifted by 1
            "query_positions": query_positions,
        })
    
    return Dataset.from_list(samples)


def process_wikitext(
    output_dir: str,
    tokenizer_name: str = "gpt2",
    max_length: int = 2048,
):
    """Download and preprocess WikiText-103 dataset.
    
    Args:
        output_dir: Directory to save processed data
        tokenizer_name: Name of tokenizer to use
        max_length: Maximum sequence length
    """
    print("Loading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-v1")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            padding="max_length",
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )
    
    # Save to disk
    print(f"Saving to {output_dir}...")
    tokenized_dataset.save_to_disk(output_dir)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Data processing utilities")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["mqar", "wikitext", "c4"],
        help="Data processing task",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples (for MQAR)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    if args.task == "mqar":
        # Generate MQAR dataset
        dataset = generate_mqar_data(
            num_samples=args.num_samples,
            seed=args.seed,
        )
        
        # Save
        output_path = Path(args.output_dir) / "mqar"
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(output_path))
        print(f"MQAR dataset saved to {output_path}")
    
    elif args.task == "wikitext":
        process_wikitext(args.output_dir)
    
    elif args.task == "c4":
        print("C4 dataset is large and best used with streaming.")
        print("Use load_dataset('c4', 'en', streaming=True) directly in training.")


if __name__ == "__main__":
    main()
