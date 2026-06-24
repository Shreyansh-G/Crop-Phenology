# Crop Growth Stage Detection from Satellite Imagery

Sentinel-1 and Sentinel-2 are sourced from Google Earth Engine. The scripts for fetching them can be found in `fetch_data.ipynb` at project root.

The primary challenge in remote sensing lies in the nature of the data itself: agricultural environments are highly dynamic, and the satellite telemetry used to monitor them is inherently asynchronous and noisy. To capture the full spectrum of a winter wheat crop's growth cycle (Stages 0 through 4), this architecture relies on two independent data streams extracted via Google Earth Engine.

---

## Model 1: Dual Stream Hybrid Network

**Architecture Overview:**

- The data is passed through an async pipeline for pre-processing. It outputs Optical and Radar data. See `model_1.ipynb` Model 1 for details.
- The Optical network consists of a 1D CNN with time embedding and a CLS token with Multi-Head Attention.
- The Radar network consists of a 1D CNN with time embedding processed with a GRU.
- The outputs from both the networks are then concatenated (they run in parallel) in a fusion network for the final prediction.

### The Dual-Sensor Approach

The first stream is Sentinel-2 optical imagery, fetched via a custom automated pipeline that calculates specific vegetation indices such as NDVI, NDWI, NDRE, EVI, and SAVI at a 10-meter spatial resolution. While rich in chemical information regarding chlorophyll and leaf water content, this optical data is highly susceptible to atmospheric occlusion and cloud cover.

To counterbalance this, the pipeline integrates a second stream: Sentinel-1 Synthetic Aperture Radar (SAR). By capturing the Vertical-Vertical (VV) and Vertical-Horizontal (VH) backscatter, along with their derived ratio, the radar stream provides continuous, cloud-penetrating measurements of the physical canopy structure and biomass density.

### Asynchronous Temporal Alignment

To bridge the gap between this raw, asynchronous satellite telemetry and the rigid input requirements of standard neural networks, the data undergoes a temporal alignment process within the dataset generation phase. Traditional machine learning models often force multi-sensor data onto a shared, interpolated daily grid, which inevitably introduces hallucinated biological data during long cloudy periods.

Instead, this system utilizes a fully asynchronous timeline anchored directly to the ground truth observation, designated as Day 0. The pipeline looks backward over a 90-day window, extracting a maximum of 30 valid satellite events for each modality. Rather than interpolating, the system calculates a strict relative temporal distance, represented as a discrete `days_ago` integer, for every single reading.

Consequently, the dataset yields two entirely separate sets of tensors for every agricultural plot:

- A feature tensor containing the sensor values.
- A parallel time tensor containing the exact chronological placement of those values.

### The Optical Branch (Chemistry & Filtering)

Operating on this aligned temporal data, the first half of the architecture focuses on the chemical and biological signatures via the Optical Attention Branch.

The network ingests a tensor of shape $[\text{Batch}, 30, 6]$, representing the sequence of optical indices alongside the critical `cloud_pct` metric. In classical time-series analysis, researchers manually calculate temporal derivatives, such as the slope of NDVI, to determine how rapidly a crop is greening up. This approach is brittle and highly susceptible to sensor noise.

Instead, this architecture natively integrates a 1-Dimensional Convolutional Neural Network (1D CNN) with a sliding window of size 3 as its front-end feature extractor. As this convolutional window sweeps across the 30-day temporal sequence, it dynamically calculates local derivatives, smoothing minor atmospheric noise and extracting the biological momentum of the indices into a richer 64-dimensional latent space.

To ensure the network understands the massive, irregular gaps between these clear readings, the integer time tensor is passed through an $\text{Embedding}(120, 64)$ layer, generating a unique mathematical signature for the exact day the reading occurred. This temporal signature is added directly to the CNN output.

A learnable 64-dimensional classification token is then prepended to the sequence, expanding the temporal length to 31. This sequence is processed by a deep Transformer Encoder utilizing Multi-Head Attention. By analyzing the entire 90-day timeline simultaneously, the attention mechanism acts as a dynamic biological filter. When it encounters an event with a high cloud percentage and a consequently crashed NDVI, it mathematically drops the attention weight for that specific day to near zero.

The network actively routes around atmospheric corruption, mapping the long-term biological relationships of the clear days and compressing that knowledge entirely into the classification token, which serves as the final 64-dimensional optical summary vector.

> **Context Note:** The "Transformer Encoder" acts as a smart filter. Instead of stepping through the timeline day-by-day, it looks at the entire 3-month history at once. It learns to automatically mute data points that are corrupted by clouds, ensuring only clean biological trends are passed forward.

### The Radar Branch (Structure & Continuity)

In stark contrast to the optical stream, the second half of the network processes the Sentinel-1 SAR telemetry through a Recurrent Radar Branch.

The input here takes the shape of $[\text{Batch}, 30, 3]$, capturing the VV, VH, and VH/VV ratio. Because SAR data is inherently plagued by microwave interference known as speckle, a parallel 1D CNN acts as a learned speckle filter. It identifies short-term structural shapes, such as the steep drop in VH backscatter that reliably occurs just before harvest as the stalks dry out and lose physical volume.

After being fused with an identical 64-dimensional positional time embedding, the sequence is fed sequentially into a Gated Recurrent Unit (GRU) with 2 hidden layers. Because radar microwaves effortlessly punch through cloud cover, the Sentinel-1 timeline is highly continuous and rarely features the massive temporal gaps seen in the optical stream.

This continuous nature makes a recurrent memory network the mathematically optimal choice, allowing the GRU to step chronologically through the days, smoothly updating its internal hidden state to track the physical accumulation and eventual dry-down of the crop canopy.

The final hidden state is extracted as the 64-dimensional radar summary vector.

### Late-Fusion and Regularization

With the biological chemistry and physical structure independently encoded, the network employs a late-fusion strategy to make its final prediction. The optical and radar summary vectors are concatenated into a unified 128-dimensional multi-modal tensor.

This delayed integration is a critical architectural choice; by forcing the network to understand the optical curve and the radar curve entirely independently before combining them, the system prevents a highly noisy optical reading from preemptively corrupting the continuous radar memory state.

The fused manifold is finally passed through a Multi-Layer Perceptron classifier. This classifier scales the features down to 64 dimensions before outputting the final 5 class logits.

Crucially, this block utilizes Batch Normalization and a Dropout probability of 30 percent. This heavy regularization actively prevents the massive capacity of the dual-stream network from simply memorizing the specific layouts of the training plots, forcing it to generalize the underlying biophysical rules of crop phenology.

### Architectural Defenses and Physics Constraints

The structural integrity of this entire pipeline relies heavily on its ability to mathematically mirror physical realities, a trait that directly answers the most common technical critiques in satellite machine learning.

For instance, differing acquisition dates between sensors are handled flawlessly precisely because the sensors are never artificially forced onto the same temporal grid; the asynchronous embeddings ensure that an optical reading from 14 days ago and a radar reading from 12 days ago are treated as distinct physical events.

Furthermore, SAR backscatter is highly sensitive to the satellite's specific viewing geometry, particularly the incident angle and whether the orbit pass is ascending or descending. This architecture actively mitigates these geometry-induced variations by utilizing the VH/VV ratio as a core input feature, which mathematically cancels out much of the background soil moisture and baseline structural variation. Coupled with the local smoothing of the front-end CNN, this prevents the recurrent network from overreacting to sudden backscatter jumps caused purely by alternating orbit directions.

Finally, while lighter architectures like the Lightweight Temporal Attention Encoder rely on a single master query to extract temporal features, this system utilizes a full self-attention stack. By allowing every temporal observation to mathematically attend to every other observation within the lookback window, the network captures much deeper, non-linear chronological relationships, such as explicitly linking the exact day of peak vegetative green-up to the exact day the harvest senescence begins.

---

## Results

**Confusion Matrix:**

<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/96b4a82a-cf53-4994-9064-2d371b3818cf" />

**Latent Space:**

<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/64d13128-29eb-44d5-a677-4ca0bfae8dc6" />

<br>

---

## Model 2: Leakage-Aware Dual-Stream Network

**Architecture Overview:**

- Model 2 keeps the same multi-modal philosophy as Model 1, but restructures training and evaluation around strict plot-level isolation.
- The input pipeline builds two asynchronous sequences per event over a 90-day lookback, capped to 30 observations.
- The Optical branch uses a CNN + Transformer encoder stack with learned time embeddings and masked pooling.
- The Radar branch uses a CNN + bidirectional GRU stack with masked pooling and linear projection.
- The two modality embeddings are concatenated and passed through a regularized MLP classifier for 5-stage prediction.

<img width="500" height="800" alt="image" src="https://github.com/zen-zap/pheno_crop/blob/f7eb2b782625d9f2ddc008c5c2e5c96dfb04696d/Model_v2_flowchart.png">


### Leakage-Aware Data Protocol

Model 2 is built to reduce overly optimistic metrics caused by split leakage. Instead of random event-level splitting, the dataset is partitioned by `plot_id` using group-aware splitting logic. A plot can only appear in exactly one of train, validation, or test.

This structure directly targets the most common failure mode in agricultural time-series experiments, where near-identical temporal neighborhoods from the same field leak across splits. By forcing disjoint plot groups, Model 2 evaluates generalization on genuinely unseen fields rather than memorized local patterns.

### Temporal Representation

For each ground-truth event, the data loader creates separate Sentinel-1 and Sentinel-2 sequences anchored to the target date. Each observation is encoded with a relative `days_ago` index, preserving asynchronous sampling without interpolation.

The sequence builder also uses explicit masks to distinguish valid observations from padded positions. This allows downstream temporal layers to pool only real events and ignore synthetic padding.

### Optical Branch (Sentinel-2)

The optical encoder ingests a 9-feature tensor per time step: vegetation indices and cloud percentage metadata. A two-layer 1D CNN first extracts short-range temporal patterns and local trend structure.

The projected features are combined with a learned temporal embedding (`Embedding(120, 64)`) derived from `days_ago`, then passed through a 2-layer Transformer encoder (4 heads, feedforward 128, dropout 0.25).

The output is summarized with masked mean pooling, producing a fixed-size optical embedding that focuses on valid timestamps only.

### Radar Branch (Sentinel-1)

The radar encoder processes VV, VH, and VH/VV ratio through a two-layer 1D CNN front-end, then models sequence dynamics with a 2-layer bidirectional GRU.

As with the optical stream, masked mean pooling extracts a robust sequence summary from valid positions. A linear projection then maps the bidirectional GRU output back into a compact modality embedding.

### Fusion and Classifier Head

The optical and radar embeddings are concatenated into a fused feature vector and fed into a classifier head:

- Linear(128 -> 96)
- BatchNorm1d(96)
- ReLU
- Dropout(0.4)
- Linear(96 -> 5)

This late-fusion design preserves modality-specific temporal modeling before decision-level integration.

### Training Strategy

Model 2 uses class-balanced cross-entropy with label smoothing (`0.05`), AdamW optimization, ReduceLROnPlateau scheduling on validation macro-F1, and early stopping.

Validation and test reporting prioritize macro-F1 and per-class recall, alongside confusion matrices and detailed classification reports. This is better aligned with imbalanced stage distributions than relying on accuracy alone.

### Why Model 2 Is More Reliable

Model 2 improves trustworthiness through:

- strict group-based train/val/test partitioning by plot,
- explicit padded-step masking in both branches,
- stronger regularization in the classifier,
- evaluation centered on macro-F1 and class-wise behavior.

The result is a cleaner estimate of field-level generalization while preserving the dual-sensor architecture that performed well in this project.

## Results: 

- **Validation Accuracy**: 70.52% | **Macro-F1**: 0.6923

- **Test Accuracy**: 70.83% | **Macro-F1**: 0.6929

```
[Test (Unseen Plots)] Loss: 1.1031 | Accuracy: 70.83% | Macro-F1: 0.6929
Per-class recall:
  [0] Bare: 0.7741
  [1] Growth: 0.6705
  [2] Ripening: 0.7424
  [3] Seedling: 0.6426
  [4] Tillering: 0.7538
              precision    recall  f1-score   support

        Bare       0.28      0.77      0.41       332
      Growth       0.83      0.67      0.74       786
    Ripening       0.75      0.74      0.75       427
    Seedling       0.87      0.64      0.74       887
   Tillering       0.92      0.75      0.83      1133

    accuracy                           0.71      3565
   macro avg       0.73      0.72      0.69      3565
weighted avg       0.81      0.71      0.74      3565
```

**Confusion Matrix:**

<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/b154a55f-a425-4a2d-88c0-2ca4cac3873b" />

<br>

---


## Model 3: PhenoCrop-Presto

**Architecture Overview:**

- Model 3 is a custom adaptation of the Presto (Predictive Representation of Sensed Earth Observation) architecture, built entirely from scratch for the `pheno_crop_v2` dataset.
- Instead of using separate recurrent or convolutional networks for each sensor, it utilizes a unified Transformer Encoder to process all modalities simultaneously.
- Every satellite observation is treated as an independent token.
- Tokens are contextually enriched with three concatenated embeddings: a positional encoding (days ago), a channel embedding (Sensor ID), and a seasonal encoding (calendar month).
- Heavily corrupted or cloudy optical readings are explicitly masked, allowing the self-attention mechanism to seamlessly route information through available radar tokens.

<img width="500" height="700" alt="image" src="https://github.com/zen-zap/pheno_crop/blob/f7eb2b782625d9f2ddc008c5c2e5c96dfb04696d/model3_flowchrt.png">

### Unified Tokenization and Embedding

Previous models in this pipeline relied on modality-specific pathways (e.g., GRUs for radar, CNNs for optical) before late-stage fusion. PhenoCrop-Presto abandons this in favor of early-stage, unified tokenization. 

The 3-feature Sentinel-1 observations and 9-feature Sentinel-2 observations are independently projected via linear layers into a shared 128-dimensional latent space. To ensure the Transformer understands the origin and context of these now-uniform tokens, three distinct embeddings are injected:
1. **Positional Encoding:** A Presto-style sinusoidal embedding that maps the exact `days_ago` relative to the target prediction date, explicitly handling the asynchronous gaps between satellite passes.
2. **Channel Embedding:** A learnable scalar that identifies whether the token originated from the Synthetic Aperture Radar or the Optical multispectral imager.
3. **Seasonal Encoding:** A sinusoidal representation of the calendar month, giving the model native awareness of the agricultural season.

### Pure Attention Fusion

Once embedded, the tokens from both the Optical and Radar streams are concatenated into a single, flattened sequence and passed through a 4-layer Transformer Encoder. 

This architecture provides ultimate resilience to atmospheric interference. If Sentinel-2 is blinded by cloud cover for an extended period, the dataset generation phase masks those specific optical tokens. The Transformer's Multi-Head Attention mechanism naturally ignores the masked positions, relying instead on the temporal tags of the uninterrupted Sentinel-1 radar tokens to maintain its understanding of the crop's biological progression. 

The final sequence is condensed using masked mean-pooling, creating a single 128-dimensional feature vector that is passed to a regularized MLP classifier.

## Results:

```
[Test (Unseen Plots)] Loss: 0.5216 | Acc: 76.41% | Macro-F1: 0.7641
  [0] Bare: recall=0.8012
  [1] Growth: recall=0.7646
  [2] Ripening: recall=0.9696
  [3] Seedling: recall=0.7215
  [4] Tillering: recall=0.7087
              precision    recall  f1-score   support

        Bare       0.83      0.80      0.82       332
      Growth       0.87      0.76      0.81       786
    Ripening       0.43      0.97      0.60       427
    Seedling       0.84      0.72      0.78       887
   Tillering       0.97      0.71      0.82      1133

    accuracy                           0.76      3565
   macro avg       0.79      0.79      0.76      3565
weighted avg       0.84      0.76      0.78      3565
```

**Confusion Matrix:**

<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/f4c23b19-bc09-49c3-a7b8-3518f5cf5ba1" />

<br> 

---


## Model 4: BiMamba-Transformer Hybrid

**Architecture Overview:**

- Model 3 evolves the architecture to address the limitations of recurrent networks and pure self-attention on long, noisy sequences.
- The front-end CNN and GRU layers are replaced with Bidirectional State Space Models (BiMamba) to independently process the Optical and Radar streams.
- Positional, channel, and seasonal (month) embeddings are injected *before* the state-space modeling, allowing the network to natively understand asynchronous time gaps.
- The modality-specific sequences are concatenated and passed through a 4-layer Transformer Encoder for cross-modal fusion.
- The model is optimized using Focal Loss ($\gamma=1.5$) and dynamic data augmentations (Gaussian jitter and temporal masking) to prevent memorization and handle ambiguous class boundaries.

<img width="500" height="700" alt="image" src="https://github.com/zen-zap/pheno_crop/blob/f7eb2b782625d9f2ddc008c5c2e5c96dfb04696d/model4_flowchart.png">

### Bidirectional State Space Modeling (BiMamba)

While standard State Space Models (like Mamba) are highly efficient at processing long sequences, they are inherently causal—they sweep from past to future. In remote sensing, the entire historical window (up to 120 days) is available at once. 

To capture the necessary context, this architecture utilizes a `BiMambaBlock`. It runs one Mamba core forward and a second Mamba core backward, merging their outputs. This provides the speed and sequence-modeling power of an SSM while ensuring that future contextual data can inform the interpretation of past anomalies (e.g., using a harvest drop to contextualize a prior ripening signature).

### Pre-SSM Temporal Injection

In previous iterations, positional embeddings were added late in the pipeline. Model 3 shifts this paradigm by injecting Presto-style positional embeddings (representing `days_ago`) and seasonal embeddings (representing the calendar month) directly into the features *before* they enter the BiMamba block. 

Because optical satellite data is asynchronous due to cloud cover, standard sequence models can struggle with irregular temporal jumps. By injecting the exact time gaps prior to the sweep, the BiMamba block natively understands whether the jump between two readings was 2 days or 20 days, allowing it to accurately trace biological momentum.

### Cross-Modal Transformer Fusion

While Mamba excels at tracing individual temporal curves, it can struggle with cross-modal reasoning (e.g., comparing an optical NDVI curve directly against a radar VH backscatter curve). 

Model 3 solves this by using BiMamba strictly as a modality-specific feature extractor. Once the Optical and Radar curves are independently smoothed and embedded, they are concatenated and fed into a deep Transformer Encoder. The Transformer no longer has to filter raw, noisy satellite data; instead, it uses its Multi-Head Attention mechanism purely to fuse the refined chemical (Optical) and physical (Radar) representations into a unified predictive manifold.

### Robust Optimization: Focal Loss & Augmentations

The boundaries between continuous crop stages (such as the transition from Tillering to Stem Elongation) are incredibly subtle from space. Standard Cross-Entropy treats all misclassifications equally. Model 3 implements **Focal Loss**, which dynamically scales the penalty based on model confidence. This forces the neural network to focus its gradient updates on the hardest, most ambiguous boundary cases.

To further stabilize training on a scaled-up dataset (16,000+ training events), the data loader dynamically applies two augmentations:
1. **Gaussian Jitter:** Adds minor noise to the valid features, forcing the model to learn the structural shape of the growth curve rather than memorizing exact reflectance decimals.
2. **Temporal Masking:** Randomly drops up to 20% of valid timesteps during training, simulating severe cloud cover or sensor failure and teaching the model to interpolate missing biological data safely.

## Results:

```
[Test (Unseen Plots)] Loss=0.5926 | Acc=75.01% | Macro-F1=0.7543
  Bare: recall=0.7892
  Growth: recall=0.6539
  Ripening: recall=0.9696
  Seedling: recall=0.7136
  Tillering: recall=0.7511
              precision    recall  f1-score   support

        Bare       0.91      0.79      0.85       332
      Growth       0.91      0.65      0.76       786
    Ripening       0.41      0.97      0.57       427
    Seedling       0.84      0.71      0.77       887
   Tillering       0.90      0.75      0.82      1133

    accuracy                           0.75      3565
   macro avg       0.79      0.78      0.75      3565
weighted avg       0.83      0.75      0.77      3565
```

**Confusion Matrix:**

<img width="500" height="450" alt="image" src="https://github.com/user-attachments/assets/dd992563-f544-48f0-9392-c7a73501d06c" />


<br>

---

## Computational Complexity Profiling

To evaluate the feasibility of deploying these models, we benchmarked the computational complexity (parameter counts, FLOPs, and inference latency) across all standard baselines and our proposed architectures.

| Model | Parameters | Inference (ms) | Complexity |
|---|---|---|---|
| BiLSTM | 574.7K | 3.08 | O(T) |
| TempCNN | 58.2K | 1.0 | O(T) |
| TCN | 231.2K | 7.11 | O(T) |
| Vanilla Transformer | 407.9K | 2.21 | O(T²) |
| **PhenoCrop-Presto** | **540.5K** | **6.36** | **O(T²)** |
| **PhenoCrop-BiMamba** | **2.62M** | **60.75** | **O(T)** |

*(Note: Inference latency measured on CPU. BiMamba latency is currently CPU-bound; GPU inference is significantly faster.)*

The profiling script and raw JSON results are available in the [`scripts/`](scripts/) directory.

---

## References

- [Temporal Self Attention and Multi-Sensor Fusion](https://github.com/ellaampy/CropTypeMapping)
- [Cloud Imputation - SEN12MS-CR](https://patricktum.github.io/cloud_removal/sen12mscr/)
- [U-Net with Temporal Attention Encoder](https://github.com/Many98/Crop2Seg)
- [Implementations of Cross-Attention Layers](https://github.com/likyoo/awesome-multimodal-remote-sensing-classification)
- [L-TAE: Lightweight Temporal Attention Encoder](https://arxiv.org/abs/2007.00586)
