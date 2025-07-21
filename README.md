# Pruning + Quantization of DistilBERT for Sentiment Analysis

This project demonstrates:
1. Fine-tuning DistilBERT on the SST-2 sentiment dataset.
2. Structured pruning of linear layers based on L1 norm.
3. Dynamic quantization of the pruned model.
4. Accuracy comparison between original, pruned, and quantized versions.



## Dataset

- **SST-2 (GLUE benchmark)** — binary sentiment classification
- Loaded via HuggingFace Datasets

## Output

```bash
=== SST-2 Evaluation Results ===
            Model  accuracy
         Original     0.898
           Pruned     0.884
 Pruned+Quantized     0.877
```
