# Audio to Token Converter

Convert audio files to SNAC discrete tokens and analyze patterns for BPE (Byte Pair Encoding) merging potential.

## Overview

This script processes audio files through a trained SNAC model to extract hierarchical discrete tokens. The tokens are saved as Parquet files for efficient analysis, and the script provides detailed statistics about token patterns to help determine if BPE-style merging would be beneficial.

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas pyarrow torchaudio tqdm
```

### 2. Tokenize Audio Dataset

```bash
# Using pretrained model
python audio_to_tokens.py \
    --input_dir /mnt/data/combine/train \
    --output_dir /mnt/data/tokens/train \
    --segment_length 4.0 \
    --batch_size 100

# Using fine-tuned checkpoint
python audio_to_tokens.py \
    --input_dir /mnt/data/combine/train \
    --output_dir /mnt/data/tokens/train \
    --checkpoint checkpoints/phase10_revolab_all/best_model.pt \
    --segment_length 4.0 \
    --batch_size 100

# For testing (limit number of files)
python audio_to_tokens.py \
    --input_dir /mnt/data/combine/train \
    --output_dir /mnt/data/tokens/test \
    --max_files 1000 \
    --segment_length 4.0
```

### 3. Analyze Token Patterns

After tokenization, the script automatically analyzes patterns. Or analyze existing tokens:

```bash
python audio_to_tokens.py \
    --analyze_only \
    --output_dir /mnt/data/tokens/train
```

## Output Files

### Parquet Files

Each batch is saved as `tokens_batch_XXXX.parquet` with columns:

- `file_path`: Original audio file path
- `duration`: Audio duration in seconds
- `tokens_scale_0`: Flattened tokens from scale 0 (coarsest)
- `tokens_scale_1`: Flattened tokens from scale 1
- `tokens_scale_2`: Flattened tokens from scale 2
- `tokens_scale_3`: Flattened tokens from scale 3 (finest)
- `n_tokens_scale_X`: Number of tokens at scale X
- `shape_scale_X`: Original shape before flattening

### Metadata File

`metadata.json` contains:
- Total files processed
- Total audio duration
- Token statistics per scale (min, max, unique count)
- Most common tokens per scale

## Understanding the Analysis

### 1. Frequency Distribution (Zipf's Law)

**Gini Coefficient**:
- 0.0 = Perfect equality (all tokens equally frequent)
- 1.0 = Perfect inequality (one token dominates)
- **> 0.8**: High skew → BPE very effective
- **0.6-0.8**: Moderate skew → BPE could help
- **< 0.6**: Low skew → BPE may not help much

**Top-K Coverage**:
- Shows what percentage of tokens are covered by top-K most frequent tokens
- High coverage indicates many rare tokens (good for compression)

### 2. Bigram Analysis

Shows patterns in adjacent token pairs:

- **Unique bigrams**: Number of distinct adjacent pairs
- **Bigram diversity**: Percentage of possible bigrams that actually occur
- **Top bigrams**: Most common adjacent token pairs

Low diversity (< 10%) means strong positional patterns → good for bigram merging.

### 3. Transition Sparsity

- **Sparsity**: 1.0 - (actual transitions / possible transitions)
- **> 0.95**: Very sparse → Excellent for BPE
- **0.8-0.95**: Moderately sparse → Some BPE benefit
- **< 0.8**: Dense → Limited BPE benefit

### 4. Repetition Analysis

- **Repetition rate**: Percentage of tokens that are repeated consecutively
- **Mean run length**: Average length of repeated token runs
- **Max run length**: Longest run of identical tokens

High repetition indicates strong patterns → BPE can compress well.

## BPE Recommendation

The script provides recommendations based on:

1. **Frequency skew** (Gini coefficient)
2. **Transition sparsity** (sparse transition matrix)
3. **Bigram patterns** (adjacent token relationships)
4. **Repetition patterns** (consecutive repeats)

**Good BPE Candidate:**
- Gini > 0.8
- Sparsity > 0.95
- Low bigram diversity
- High repetition rate

**Poor BPE Candidate:**
- Gini < 0.6
- Sparsity < 0.8
- High bigram diversity
- Low repetition rate

---

## Advanced: Weight-Free BPE Approaches

### Why Standard BPE Doesn't Work for SNAC

Our analysis shows:
- **7-gram diversity**: 0.0000% (almost every 7-gram is unique)
- **Top-10 coverage drops** from 8.21% (1-gram) to 1.19% (7-gram)
- **Conclusion**: Traditional BPE merging won't help - too much diversity

However, audio tokens have unique patterns that can be exploited with specialized BPE approaches:

### 1. Run-Length BPE (RL-BPE) ⭐ **RECOMMENDED**

Compress repeated tokens first, then apply BPE:

```python
from itertools import groupby

def run_length_bpe(tokens):
    """Apply run-length encoding before BPE."""
    # Step 1: Run-length encode repeated tokens
    compressed = []
    for token, count in groupby(tokens):
        if count > 1:
            compressed.append((token, count))  # Pair: (token_id, count)
        else:
            compressed.append(token)

    # Step 2: Apply standard BPE on compressed sequence
    # Now repeated tokens are single pairs: (2068, 5457)
    # This merges MUCH better than (2068, 2068, ..., 2068) [5457 times]
    return bpe_merge(compressed)
```

**Why it works:**
- Token 2068 appears 5457 times consecutively → becomes `(2068, 5457)` (1 pair!)
- Before: needed 5457 merge operations
- After: only 1 merge operation
- Handles the repetition patterns seen in SNAC tokens

**Expected compression**: 10-20x for repetitive audio

### 2. Cross-Scale BPE

Merge across SNAC's 4 scales at same time position:

```python
def cross_scale_bpe(codes_0, codes_1, codes_2, codes_3):
    """BPE on multi-scale tuples aligned by time."""
    # Align tokens by time position
    aligned = []
    for t in range(time_steps):
        # Create 4-scale tuple at each time step
        aligned.append((codes_0[t], codes_1[t], codes_2[t], codes_3[t]))

    # Standard BPE on these tuples
    # Example:
    # Before: (2068, 1736, 3422, 4095) + (2068, 1736, 3422, 4095)
    # After:  SUPERTOKEN_42 (representing this common 4-scale pattern)
    return bpe_merge(aligned)
```

**Why it works:**
- Natural temporal alignment (SNAC is hierarchical)
- Captures cross-scale correlations
- Each scale has different temporal resolution
- Still purely frequency-based (no weights)

**Expected compression**: 4x (one token instead of 4)

### 3. Thresholded BPE

Only merge pairs above minimum frequency threshold:

```python
def thresholded_bpe(tokens, min_freq=100):
    """Only merge pairs that appear at least min_freq times."""
    merge_rules = []

    while True:
        # Count all adjacent pairs
        pair_counts = count_pairs(tokens)

        # Filter by threshold (prevents over-merging rare patterns)
        frequent_pairs = {p: c for p, c in pair_counts.items()
                         if c >= min_freq}

        if not frequent_pairs:
            break  # No more frequent pairs to merge

        # Merge most frequent pair
        best_pair = max(frequent_pairs, key=frequent_pairs.get)
        tokens = merge_pair(tokens, best_pair)
        merge_rules.append(best_pair)

    return tokens, merge_rules
```

**Why it works:**
- Prevents merging rare patterns (adds noise)
- Focuses on high-frequency patterns only
- Results in more robust vocabulary
- Purely statistical (no weights)

**Expected compression**: 5-10x with better generalization

### 4. Hierarchical BPE (Two-Stage)

```python
def hierarchical_bpe(all_codes):
    """Learn BPE vocabularies in two stages."""
    # Stage 1: Learn per-scale vocabularies
    vocab_0 = bpe_train(all_codes[0], merges=1000)
    vocab_1 = bpe_train(all_codes[1], merges=1000)
    vocab_2 = bpe_train(all_codes[2], merges=1000)
    vocab_3 = bpe_train(all_codes[3], merges=1000)

    # Stage 2: Merge cross-scale patterns
    aligned = align_scales(vocab_0, vocab_1, vocab_2, vocab_3)
    final_vocab = bpe_train(aligned, merges=2000)

    return final_vocab
```

**Why it works:**
- Respects SNAC's hierarchical structure
- Per-scale vocabularies capture scale-specific patterns
- Cross-scale stage captures inter-scale relationships
- Two-stage approach is more efficient than trying to learn everything at once

**Expected compression**: 8-12x

### 5. Combined RL-BPE + Cross-Scale (Maximum Compression)

```python
def rl_cross_scale_bpe(all_codes):
    """Combine run-length and cross-scale BPE."""
    # Step 1: Run-length encode each scale separately
    compressed_scales = []
    for scale_codes in all_codes:
        compressed = run_length_encode(scale_codes)
        compressed_scales.append(compressed)

    # Step 2: Cross-scale BPE on compressed tuples
    aligned = align_by_time(compressed_scales)
    final_tokens, merge_rules = bpe_train(aligned, merges=5000)

    return final_tokens, merge_rules
```

**Why it works:**
- Exploits repetition (RL) within each scale
- Exploits cross-scale correlations
- Both techniques are purely statistical (no weights)
- Multiplies compression benefits

**Expected compression**: 20-40x combined

---

## BPE Approach Comparison

| Method | Weights | Complexity | Compression | Best For | Implementation |
|--------|---------|------------|-------------|----------|----------------|
| **RL-BPE** | ❌ | Low | 10-20x | Repetitive audio | Easy |
| **Cross-Scale** | ❌ | Medium | 4x | Multi-scale codecs | Medium |
| **Thresholded BPE** | ❌ | Low | 5-10x | Noisy data | Easy |
| **Hierarchical BPE** | ❌ | Medium | 8-12x | Structured data | Medium |
| **RL + Cross-Scale** | ❌ | Medium | 20-40x | Maximum compression | Hard |

---

## Implementation Recommendations

### For SNAC Audio Tokens (Based on Analysis)

Our analysis shows:
- ✅ High repetition: Token 2068 appears 5000+ times consecutively
- ✅ Low 7-gram diversity: 0.0000% (almost all unique)
- ✅ High sparsity: 97-99% of transitions never occur
- ✗ Low top-K coverage: Top-10 covers only 1-8%

**Recommended Approach: Start with RL-BPE**

```python
# Step 1: Implement run-length encoding
def run_length_encode(tokens):
    encoded = []
    i = 0
    while i < len(tokens):
        j = i + 1
        while j < len(tokens) and tokens[j] == tokens[i]:
            j += 1
        count = j - i
        if count > 1:
            encoded.append((tokens[i], count))
        else:
            encoded.append(tokens[i])
        i = j
    return encoded

# Step 2: Apply BPE on RLE-encoded tokens
# This will merge repeated tokens much more efficiently
```

**Next Steps:**
1. Implement RL-BPE on tokenized parquet files
2. Train language model on compressed tokens
3. Evaluate compression ratio and model performance
4. If needed, add cross-scale BPE for additional compression

---

## BPE vs Alternative Approaches

### When to Use BPE:

✅ **Use BPE if:**
- High repetition (run-length encoding helps)
- Strong adjacent-token patterns
- Want discrete tokens for language modeling
- Need interpretable merge rules

❌ **Don't use BPE if:**
- Token distribution is already uniform
- No repetition patterns
- Continuous representation works better
- Compression not critical

### Alternative: Vector Quantization on Windows

If BPE doesn't provide sufficient compression, consider VQ on token windows:

```python
class WindowVQ:
    def __init__(self, window_size=7, num_codes=4096):
        self.window_size = window_size
        self.vq = VectorQuantizer(num_codes)

    def encode(self, tokens):
        # Create windows of 7 tokens
        windows = tokens.unfold(1, 7, 7)
        # Vector quantization on windows
        codes = self.vq(windows)
        return codes  # Compressed representation
```

**Note**: This introduces weights (the VQ codebook), but provides more powerful compression.



## Usage Examples

### Example 1: Test on Small Dataset

```bash
# Tokenize 1000 files for quick analysis
python audio_to_tokens.py \
    --input_dir /mnt/data/combine/train \
    --output_dir /tmp/tokens_test \
    --max_files 1000 \
    --batch_size 50
```

### Example 2: Full Dataset Analysis

```bash
# Tokenize full training set
python audio_to_tokens.py \
    --input_dir /mnt/data/combine/train \
    --output_dir /mnt/data/tokens/train_full \
    --checkpoint checkpoints/phase10_revolab_all/best_model.pt \
    --segment_length 4.0 \
    --batch_size 200
```

### Example 3: Compare Pretrained vs Fine-tuned

```bash
# Pretrained model
python audio_to_tokens.py \
    --input_dir /mnt/data/combine/val \
    --output_dir /mnt/data/tokens/val_pretrained \
    --model hubertsiuzdak/snac_24khz

# Fine-tuned model
python audio_to_tokens.py \
    --input_dir /mnt/data/combine/val \
    --output_dir /mnt/data/tokens/val_finetuned \
    --checkpoint checkpoints/phase10_revolab_all/best_model.pt
```

## Interpreting Results

### Good for BPE:

```
✓ High frequency skew (gini=0.850)
→ BPE merging could be very effective

✓ Very sparse transitions (sparsity=0.980)
→ Good candidate for bigram merging

✓ High repetition rate (15.3%)
→ Many repeated patterns
```

### Not Good for BPE:

```
✗ Low frequency skew (gini=0.450)
→ BPE merging may not help much

✗ Dense transitions (sparsity=0.650)
→ Bigram merging may not help

✗ Low repetition rate (2.1%)
→ Few repeated patterns
```

## Advanced: Using the Parquet Files

```python
import pandas as pd
import numpy as np

# Load tokens
df = pd.read_parquet('/mnt/data/tokens/train/tokens_batch_0000.parquet')

# Access tokens at scale 0 (coarsest)
tokens_scale_0 = df['tokens_scale_0'].iloc[0]  # First file
print(f"Shape: {len(tokens_scale_0)} tokens")

# Analyze patterns across all files
all_tokens = []
for tokens_list in df['tokens_scale_0']:
    all_tokens.extend(tokens_list)

# Get unique tokens and frequencies
unique, counts = np.unique(all_tokens, return_counts=True)
print(f"Unique tokens: {len(unique)}")
print(f"Total tokens: {len(all_tokens)}")

# Save custom analysis
np.save('tokens_scale_0.npy', np.array(all_tokens))
```

## Troubleshooting

### CUDA Out of Memory

Reduce segment length or batch size:

```bash
python audio_to_tokens.py \
    --input_dir /path/to/audio \
    --output_dir /path/to/tokens \
    --segment_length 2.0 \
    --batch_size 50
```

### Slow Processing

Use more workers or reduce segment length for faster processing.

### Empty Output

Check that:
- Audio directory contains supported files (`.wav`, `.mp3`, `.flac`)
- Audio files are readable
- Sufficient disk space for output

## What This Tells You

After running this script, you'll know:

1. **Token distribution**: How balanced/imbbalanced the tokens are
2. **Pattern strength**: Whether there are strong adjacent-token patterns
3. **BPE potential**: Whether BPE-style merging would help compression
4. **Scale differences**: Which hierarchical scales have more patterns

This helps decide whether to:
- Apply BPE merging to create "compound tokens"
- Use different vocabulary sizes per scale
- Keep original discrete tokens
- Focus on specific scales for modeling

## Next Steps After BPE Analysis

### If BPE Looks Promising:

1. **Extract merge rules** from common bigrams
2. **Create new vocabulary** with merged tokens
3. **Re-encode data** with merged tokens
4. **Train language model** on merged tokens

### If BPE Doesn't Help:

1. **Keep original tokens** (they're already efficient)
2. **Focus on modeling** token sequences directly
3. **Consider hierarchical modeling** of different scales
4. **Use existing tokenization** (no changes needed)
