# PTQ-in-CNNs-Vision-Transformers-and-Language-Models

**Jinhee Kim, Tomohisa Kawakami, Yuxuan Wen, Sam Rivera**  
Department of Electrical and Computer Engineering  
Department of Biomedical Engineering  

## Overview

Post-training quantization (PTQ) has become an essential tool for reducing the memory and compute footprint of deep neural networks without requiring full finetuning. While numerous PTQ algorithms have been proposed, a systematic cross-architectural analysis of their behavior across convolutional networks, vision transformers, and language models remains limited.

In this work, we conduct a unified empirical study of learning-based PTQ applied to **ResNets**, **ViT-S**, and **OPT-125M**. We characterize quantization sensitivity through layer-wise reconstruction error, normalized error profiles, and activation distribution shifts, which are conserved across architectures.

Our findings reveal that PTQ difficulty aligns tightly with each architecture’s functional structure:

- **CNNs:** Downsampling layers form the primary bottleneck.  
- **Vision Transformers:** Attention projection matrices dominate quantization error.  
- **Language Models:** Similar attention-related sensitivity is observed.  
- **Across all models:** MLP layers remain consistently robust.

Additionally, we show that quantization disproportionately perturbs activation outliers while largely preserving central activation statistics.

Together, these results provide a **unified view of PTQ behavior** across major model families and highlight architectural factors that should guide future quantization methods toward generalizable, structure-aware solutions.

![Poster](./poster.jpg)
