# Distance-Based Relation Score System

## Overview

This document describes the distance-based scoring system for measuring relationships between models in the Tukey fence outlier dataset.

## Distance Definitions

### 1. **Distance = 1** (Direct Fine-tuning)
- **Definition**: Direct parent-child fine-tuning relationship
- **Example**: `Qwen/Qwen2.5-1.5B → Gensyn/Qwen2.5-1.5B-Instruct`
- **Count**: 50 pairs (2.7% of all pairs)
- **Interpretation**: Strongest relationship - one model is directly fine-tuned from the other

### 2. **Distance = 2+** (Indirect Fine-tuning)
- **Definition**: Ancestor-descendant relationship through multiple fine-tuning steps
- **Distance Calculation**: Number of hops in the lineage path
- **Examples**:
  - Distance 2: `Qwen/Qwen2.5-7B → Qwen/Qwen2.5-7B-Instruct → Gensyn/Qwen2.5-7B-Instruct`
  - Distance 2: `Qwen/Qwen2.5-3B → Qwen/Qwen2.5-3B-Instruct → PowerInfer/SmallThinker-3B-Preview`
- **Count**: 18 pairs (1.0% of all pairs)
- **Interpretation**: Still related through direct lineage, but more distant

### 3. **Distance = 10** (Same Base Organization)
- **Definition**: Models whose base models come from the same organization
- **Example**: Two models both fine-tuned from different Qwen base models
- **Count**: 764 pairs (41.7% of all pairs)
- **Interpretation**: Loose relationship - shared organizational origin but no direct lineage

### 4. **Distance = null** (No Relation)
- **Definition**: No known relationship between the models
- **Example**: A Qwen-based model and a Llama-based model
- **Count**: 998 pairs (54.5% of all pairs)
- **Interpretation**: Unrelated models

## Statistics Summary

### Total Pairs: 1,830
- 61 models × 60 / 2 = 1,830 unique pairs

### Relation Distribution:
| Relation Type | Count | Percentage |
|---------------|-------|------------|
| No relation | 998 | 54.5% |
| Same base org | 764 | 41.7% |
| Direct FT | 50 | 2.7% |
| Indirect FT | 18 | 1.0% |

### Distance Distribution:
| Distance | Count | Percentage | Meaning |
|----------|-------|------------|---------|
| null | 998 | 54.5% | No relation |
| 1 | 50 | 2.7% | Direct fine-tuning |
| 2 | 18 | 1.0% | 2-hop fine-tuning |
| 10 | 764 | 41.7% | Same base org |

## Key Insights

1. **Most pairs are unrelated** (54.5%) - Models come from different organizational ecosystems

2. **Loose relations dominate among related pairs** (764/832 = 91.8%) - Most "related" models share organizational origin rather than direct lineage

3. **Direct lineage is rare** (68/832 = 8.2%) - Only a small fraction of related models have direct fine-tuning connections

4. **Multi-hop fine-tuning exists but is uncommon** (18 pairs) - Some models are fine-tuned from fine-tuned models

## Use Cases

This distance metric can be used for:

1. **Clustering Analysis**: Group models by relationship strength
2. **Similarity Prediction**: Use distance as a feature for predicting model similarity
3. **Lineage Validation**: Verify if empirical similarity matches lineage distance
4. **Network Analysis**: Construct weighted graphs with distance-based edge weights

## Files Generated

1. **pairwise_distances.json** - Complete distance data with relationship paths
2. **pairwise_distances.csv** - Tabular format for statistical analysis
3. **distance_matrix.csv** - 61×61 symmetric distance matrix

## Example Entries

### Direct Fine-tuning (distance=1):
```json
{
  "model1": "Qwen/Qwen2.5-1.5B",
  "model2": "Gensyn/Qwen2.5-1.5B-Instruct",
  "distance": 1,
  "relation_type": "direct_ft",
  "path": "Qwen/Qwen2.5-1.5B -> Gensyn/Qwen2.5-1.5B-Instruct"
}
```

### Indirect Fine-tuning (distance=2):
```json
{
  "model1": "Qwen/Qwen2.5-7B",
  "model2": "Gensyn/Qwen2.5-7B-Instruct",
  "distance": 2,
  "relation_type": "indirect_ft",
  "path": "Qwen/Qwen2.5-7B -> Qwen/Qwen2.5-7B-Instruct -> Gensyn/Qwen2.5-7B-Instruct"
}
```

### Same Base Organization (distance=10):
```json
{
  "model1": "Gensyn/Qwen2.5-1.5B-Instruct",
  "model2": "PowerInfer/SmallThinker-3B-Preview",
  "distance": 10,
  "relation_type": "same_base_org",
  "path": "Both from Qwen organization"
}
```

### No Relation (distance=null):
```json
{
  "model1": "Gensyn/Qwen2.5-1.5B-Instruct",
  "model2": "NousResearch/Hermes-3-Llama-3.1-8B",
  "distance": null,
  "relation_type": "none",
  "path": null
}
```
