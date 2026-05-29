# ML / DL / LLM Fundamentals
# Easy → Hard | Ask in order, stop when candidate struggles

---

## Evaluation Metrics

**Q1. [Easy]** You trained a model to detect defective products on an assembly line. 98% of products are fine. Your model predicts "no defect" every time and gets 98% accuracy. Is this a good model?

> No. It never detects anything. Accuracy is useless when classes are imbalanced. Need Precision, Recall, or F1 on the defect class. A model that catches zero defects is worthless regardless of accuracy.

---

**Q2. [Easy]** You're building a system to detect if a patient has a rare disease. Missing a sick patient is far worse than a false alarm. Which metric do you care about most?

> Recall — minimize false negatives. It's fine to over-flag and run more tests. Missing a sick patient is the catastrophic failure mode.

---

**Q3. [Easy-Medium]** You're building a spam filter. Your manager says "don't ever send a real email to spam." Which metric matters more — precision or recall?

> Precision — you'd rather let spam through than wrongly block a real email. High precision means when you flag spam, you're almost always right.

---

**Q4. [Medium]** You trained two models for fraud detection. Model A catches 90% of fraud but flags 200 false alarms a day. Model B catches 70% of fraud but only 20 false alarms. The fraud team can manually review 50 cases a day. Which do you ship?

> Model B — the fraud team can handle 20 false alarms a day. Model A overwhelms them with 200 reviews, real fraud gets buried in noise. Business constraints dictate the threshold, not just the metric.

---

**Q5. [Medium]** Your RAG system retrieves top-10 chunks. Precision@10 is high but the final LLM answer is still wrong. What do you investigate?

> Chunk quality (too noisy/long), context ordering in prompt, contradictory chunks cancelling each other out, context window overflow, or the LLM ignoring retrieved content entirely. Debug each independently.

---

**Q6. [Hard]** Your retrieval system has Recall@10 = 0.9 but MRR = 0.3. What's the problem and what do you fix?

> Right documents are being retrieved but ranked 5th–10th, not 1st. The LLM sees the answer late in context and deprioritizes it (lost-in-the-middle problem). Fix the reranker or retrieval scoring — the embedding model is not the issue.

---

## Feature Engineering

**Q7. [Easy]** You have a dataset with 200 features. Two features have correlation 0.99. What do you do?

> Drop one — they carry nearly identical information. Keeping both adds noise, inflates dimensionality, and can cause multicollinearity in linear models.

---

**Q8. [Medium]** You have 8,000 features. Model overfits and training is slow. Walk me through your reduction approach.

> Step 1: Drop near-zero variance features. Step 2: Remove highly correlated pairs. Step 3: Rank remaining by mutual information or SHAP. Step 4: Apply L1 regularization or RFE to let the model decide. Validate each step doesn't drop validation performance.

---

**Q9. [Medium]** Your compliance team needs to explain every model prediction to regulators. A data scientist suggests PCA for dimensionality reduction. Is that okay?

> No — PCA creates synthetic components that have no real-world meaning. You can't explain "component 3 drove this prediction" to a regulator. Use feature selection (keep original features) instead. Interpretability and PCA don't mix.

---

## Deep Learning

**Q10. [Easy]** You're training a neural network and your loss stops improving after epoch 2. Training loss is low, validation loss is high. What's happening?

> Overfitting — model memorizes training data, fails to generalize. Add dropout, reduce model size, add regularization, or get more data.

---

**Q11. [Easy-Medium]** You're training an RNN on long documents. The model performs poorly and gradients near zero in early layers. What's the core problem?

> Vanishing gradients — the signal from early tokens dies as it backpropagates through many time steps. RNNs can't learn long-range dependencies. Switch to LSTM/GRU or Transformers.

---

**Q12. [Medium]** You have limited GPU budget. Task is sequence classification. LSTM or GRU — which do you start with?

> GRU — fewer parameters, faster to train, less memory. Comparable performance on most tasks. Escalate to LSTM only if GRU clearly underperforms on long sequences.

---

**Q13. [Medium]** Your model's training loss hits NaN after 3 epochs. What happened and how do you fix it?

> Exploding gradients. Fix: gradient clipping (norm threshold ~1.0), check weight init (He for ReLU, Xavier for tanh), add BatchNorm, lower learning rate. NaN = gradient blew up to infinity.

---

**Q14. [Hard]** Model diverges in the first 100 steps then suddenly stabilizes and trains fine. What's the likely cause?

> No learning rate warmup — LR starts too high for randomly initialized weights. The early instability destroys early layers before the model can learn anything meaningful. Add linear warmup for first N steps, then cosine decay.

---

## Optimization

**Q15. [Medium]** Your model overfits despite using Adam optimizer. What's the first optimizer change you try?

> AdamW — Adam conflates weight decay with gradient updates, weakening regularization. AdamW separates them correctly. Often improves generalization with no other changes.

---

## LLM Fine-Tuning

**Q16. [Medium]** You need to fine-tune a 7B LLM on a single A100. Full fine-tuning crashes with OOM. What do you do?

> LoRA — freeze the base model, add small trainable low-rank adapter matrices to attention layers. Trains ~1% of parameters with comparable quality. Much lower memory footprint.

---

**Q17. [Medium-Hard]** A teammate fine-tuned a model with LoRA rank=128 on 500 examples. It memorized the training set perfectly but fails on new inputs. What do they change?

> Lower the rank — rank controls adapter capacity. Rank 128 is massive overkill for 500 examples. Try rank 8 or 16. Reduce alpha proportionally. Add dropout to LoRA layers.

---

**Q18. [Hard]** When would you use QLoRA over LoRA, and when would you avoid it?

> QLoRA quantizes the base model to 4-bit, cutting VRAM significantly — use it when even LoRA doesn't fit in memory. Avoid it when numerical precision matters (scientific tasks, certain regression problems) or when quantization noticeably degrades output quality on your eval set.

---
