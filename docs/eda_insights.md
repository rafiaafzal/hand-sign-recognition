# EDA insights (auto-generated from the data)

- **Size:** 63,291 colour images kept after cleaning; 24 classes, 5 signers.
- **Class imbalance:** largest/smallest class ratio = 1.20; smallest class T (2516), largest W (3020).
- **Formats:** 12542 distinct resolutions, width 64–249 px, height 64–272 px, modes {'RGB': 63291}; resizing and mode conversion are required.
- **Signer effect:** mean brightness ranges from 78 (signer B) to 128 (signer D); median sharpness ranges from 1313 to 2424.
- **Skewness:** sharpness 0.4, brightness 0.5, contrast 0.0 (values above ~1 indicate a strongly right-skewed distribution).
- **Most correlated features:** mean_g/brightness (+1.00); mean_b/brightness (+1.00); mean_g/mean_b (+0.99).
- **Most similar letters (mean-image correlation):** R–V (0.92), N–O (0.91), D–R (0.90), L–R (0.90), P–Q (0.90).

## Implications for modelling
- Balance the classes (equal images per class) and report per-class precision/recall/F1, not just accuracy.
- Convert every image to RGB and resize to a fixed size; scale pixels to [0, 1] (then standardise with training-set mean/std if the model expects it).
- Split by signer at least once (signer-independent test) in addition to the random split, because frames from one signer are highly correlated.
- Expect confusion between visually similar letters; consider augmentation (small rotations, brightness jitter) to cover lighting differences between signers.