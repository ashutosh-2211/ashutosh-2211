# ML / DL / LLM Fundamentals (5+ Years)

## Evaluation Metrics

### Q1
Your model shows 99.5% accuracy but business KPIs are poor. Explain.

**Expected Areas:** Class imbalance, Accuracy misleading, Precision/Recall, PR-AUC, Threshold tuning, Cost-sensitive evaluation

### Q2
When would F1 Score be a poor metric?

**Expected Areas:** Unequal business costs, Precision-focused use cases, Recall-focused use cases, Ranking problems

### Q3
When would you use Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC?

**Expected Areas:** Balanced datasets, Imbalanced datasets, Classification tradeoffs, Business objectives

## Feature Engineering

### Q4
You have 8000 features. How would you reduce them?

**Expected Areas:** Correlation removal, Mutual Information, L1 Regularization, RFE, Tree Importance, SHAP

### Q5
Feature Selection vs Feature Extraction?

**Expected Areas:** PCA, Autoencoders, Dimensionality reduction, Interpretability

## Deep Learning

### Q6
Why did RNNs fail for large-scale NLP?

**Expected Areas:** Vanishing gradients, Sequential processing, Long dependencies, Poor parallelization

### Q7
Walk through LSTM architecture.

**Expected Areas:** Cell state, Forget gate, Input gate, Output gate

### Q8
GRU vs LSTM?

**Expected Areas:** Fewer parameters, Faster training, Lower memory, Comparable performance

### Q9
Initialization strategies you have used?

**Expected Areas:** Xavier, He, Orthogonal, Gradient stability

### Q10
Exploding gradients? Solutions?

**Expected Areas:** Gradient clipping, Better initialization, BatchNorm, Residual connections

## Optimization

### Q11
Adam vs AdamW?

**Expected Areas:** Weight decay separation, Better generalization

### Q12
Learning rate schedulers?

**Expected Areas:** Warmup, Cosine decay, Step decay, Plateau scheduler

## LLM Fine-Tuning

### Q13
Explain LoRA.

**Expected Areas:** Frozen weights, Low-rank adapters, Efficient fine-tuning

### Q14
Rank and Alpha in LoRA?

**Expected Areas:** Adaptation capacity, Scaling factor, Memory tradeoffs, Overfit vs Underfit

### Q15
QLoRA vs LoRA?

**Expected Areas:** Quantization, 4-bit loading, Lower memory, Similar quality

## Retrieval

### Q16
Retrieval Precision@10 is high but answer quality is poor. Why?

**Expected Areas:** Chunk quality, Prompt quality, Context ordering, Contradictions, Context overflow

### Q17
How do you evaluate embeddings?

**Expected Areas:** Recall@K, MRR, NDCG, Latency, Relevance
