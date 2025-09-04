# EfficientAD (Modified for Custom Datasets)

## Overview
This repository provides a **practical and adapted implementation of EfficientAD**.  
Instead of strictly reproducing the pseudocode from the paper, the goal here is to **make EfficientAD easier to apply on real-world and custom datasets**.  
The implementation is based on [Original GitHub Repository](https://github.com/nelson1425/EfficientAD/tree/main), with additional modifications and personal insights added for better usability.

- **Original Paper:** [EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies](https://arxiv.org/abs/2303.14535)
- **Reference Implementation:** [Original GitHub Repository](https://github.com/nelson1425/EfficientAD/tree/main)  

---

## Main Modifications & Contributions

Compared with the reference repository, this version focuses on practical improvements for dataset adaptation and usability:

1. **Enhanced Training Loop with Epoch-Based Progress and Metrics**
    - Restructured the training process into a clear epoch-based loop instead of a step-based loop.
    - Metrics such as AUC and loss values are reported at the end of each epoch.
    - Student and autoencoder models are saved after each epoch, along with normalization parameters (teacher mean/std, quantile bounds, etc.) used for anomaly scoring.

2. **Dedicated Optimizers for Student and Autoencoder**
    - Introduced separate optimizers for the student and autoencoder networks.
    - This ensures gradient flow:
        - Teacher–Student loss updates only the student.
        - Student–Autoencoder imitation loss updates only the student (not the autoencoder).
        - Teacher–Autoencoder loss updates only the autoencoder.

3. **Flexible Quantile Scheduling for Hard Mining**
    - The original implementation fixed the hard-mining quantile at 99.9%, which was not always effective on real-world datasets.
    - Added support for quantile schedules: training can start with broader coverage (lower quantiles) and gradually focus on more difficult regions (higher quantiles).

4. **Multiple Scoring Strategies for Image-Level Anomaly Detection**
    - The original version used only the max value as the anomaly score.
    - This implementation provides additional scoring methods, including:
        - Top-k average (robust against noise)
        - Percentile-based score (e.g., p99)
        - Mean score (smooth alternative to max)
  
---

## Acknowledgements
- This repository is adapted from [Original GitHub Repository](https://github.com/nelson1425/EfficientAD/tree/main)
- Original paper: [EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies](https://arxiv.org/abs/2303.14535)
- Special thanks to the authors of both the paper and the reference implementation.