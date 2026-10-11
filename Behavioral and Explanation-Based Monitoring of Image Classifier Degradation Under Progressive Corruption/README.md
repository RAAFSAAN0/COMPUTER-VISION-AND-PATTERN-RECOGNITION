# Behavioral and Explanation-Based Monitoring of Image Classifier Degradation Under Progressive Corruption

A research project investigating how image classifier behavior and visual explanations change as input corruption progressively increases.

This project introduces a **Behavioral Degradation Index (BDI)** that combines prediction-level and explanation-level signals to characterize model degradation beyond classification accuracy alone. Using ResNet-50 on CIFAR-10 and CIFAR-10-C, the study evaluates behavioral changes across multiple corruption types and severity levels and investigates whether these changes can provide early warnings of performance failure.

## Research Overview

Deep learning models can achieve high accuracy on clean benchmark data but become less reliable when their inputs are corrupted. Conventional accuracy measurements describe performance degradation but may not fully capture changes in model confidence, prediction distributions, decision stability, or the visual evidence underlying predictions.

This study examines whether combining these behavioral signals provides a more comprehensive way to characterize progressive degradation.

The research focuses on three questions:

- How do prediction behavior and Grad-CAM explanations change under increasing image corruption?
- Can multiple behavioral signals be combined into an interpretable index that tracks corruption severity?
- Can adaptive thresholds identify performance failure before the classifier's accuracy falls below a predefined threshold?

## Project Files

| File | Description |
|---|---|
| `2.ipynb` | Jupyter notebook containing the experimental implementation, behavioral signal analysis, BDI construction, validation, and early-warning evaluation. |
| `Manuscirpt.pdf` | Research manuscript describing the motivation, methodology, experiments, results, limitations, and conclusions. |

## Methodology

### 1. Dataset and Experimental Setup

The experiments use the CIFAR-10 test set and the CIFAR-10-C corruption benchmark.

- **Dataset:** CIFAR-10 and CIFAR-10-C
- **Classification model:** ResNet-50
- **Clean test images:** 10,000
- **Corruption types:** 15
- **Severity levels:** 5 per corruption type
- **Experimental conditions:** 75 corruption–severity combinations
- **Reported clean accuracy:** 97.53%

The evaluated corruptions include brightness, contrast, defocus blur, elastic transformation, fog, frost, Gaussian blur, Gaussian noise, glass blur, impulse noise, JPEG compression, motion blur, pixelation, saturation, and shot noise.

### 2. Behavioral Degradation Signals

Five signals are used to characterize changes between clean and corrupted inputs.

| Signal | Purpose |
|---|---|
| Confidence Change | Measures changes in prediction confidence under corruption. |
| Jensen–Shannon Divergence | Quantifies differences between clean and corrupted predictive distributions. |
| Prediction Flip Rate | Measures how frequently predicted classes change under corruption. |
| Prediction-Margin Degradation | Characterizes changes in the classifier's prediction margins. |
| Grad-CAM Explanation Instability | Measures changes in normalized visual explanation maps between clean and corrupted inputs. |

For the explanation analysis, the clean image's predicted class is used as the Grad-CAM target for both the clean and corrupted images. Explanation instability is measured using the mean absolute difference between the normalized maps.

### 3. Behavioral Degradation Index

The five signals are standardized across the 75 experimental conditions and combined using fixed equal weights.

The resulting index is:

\[
\mathrm{BDI}=\frac{1}{5}\sum_{i=1}^{5}Z_i
\]

where \(Z_i\) represents one of the five standardized degradation signals.

The equal-weight formulation provides a transparent way to aggregate heterogeneous behavioral measurements without learning the component weights from the evaluation data.

In the primary analysis, the BDI is used as a descriptive measure of degradation rather than a deployment-ready predictive score.

### 4. Validation and Statistical Analysis

The experimental pipeline includes:

- Pearson, Spearman, and Kendall correlation analyses.
- Severity-wise BDI analysis and corruption-specific trajectories.
- A four-level degradation framework covering behavioral shift, decision instability, explanation shift, and performance failure.
- Adaptive warning thresholds based on the 90th, 95th, and 99th percentiles of severity-1 reference observations.
- Leave-one-corruption-out (LOCO) validation to evaluate transfer to an excluded corruption type.
- Pearson correlation and variance inflation factor (VIF) analysis to examine redundancy among signals.
- Multiple regression and leave-one-signal-out ablation analyses.

Performance failure is primarily defined as accuracy falling below 90% of clean accuracy. With the reported clean accuracy of 97.53%, this corresponds to a threshold of 87.777%.

## Key Findings

The manuscript reports the following findings:

| Evaluation | Result |
|---|---|
| BDI–severity Spearman correlation | 0.5239 |
| BDI–severity Pearson correlation | 0.5063 |
| BDI–severity Kendall correlation | 0.3962 |
| Strongest individual signal's Spearman correlation | Grad-CAM instability: 0.5387 |
| Mean BDI at severity 1 | -0.6517 |
| Mean BDI at severity 5 | 0.7504 |
| Corruption types exhibiting the defined performance failure | 13 out of 15 |
| Advance warnings under the primary Q95 protocol | 0 |

The results indicate that the BDI generally increases with corruption severity. Grad-CAM instability also shows a positive association with severity, suggesting that corruption can change the spatial evidence highlighted by the classifier.

However, the degradation pathways vary across corruption types. The primary Q95 warning protocol does not provide reliable advance warnings of the defined performance-failure threshold, including under the reported leave-one-corruption-out evaluation.

**Interpretation:** The findings support the BDI as a diagnostic and comparative measure of behavioral degradation. They do not establish it as a universally reliable early-warning predictor of future performance failure.

## Implementation and Tools

The notebook uses Python-based machine learning and data-analysis tools, including:

- Python
- PyTorch and Torchvision
- NumPy and Pandas
- Pillow
- SciPy
- scikit-learn
- Statsmodels
- Matplotlib
- tqdm

The implementation covers model inference, image preprocessing, clean–corrupted image correspondence recovery, Grad-CAM analysis, signal aggregation, statistical evaluation, and degradation-warning diagnostics.

## Running the Experiments

### Prerequisites

Install the required Python packages:

```bash
pip install torch torchvision numpy pandas pillow scipy scikit-learn statsmodels matplotlib tqdm
```

The appropriate PyTorch installation may depend on your operating system and whether CUDA is available.

### Data and Model Requirements

The notebook references external data and model artifacts, including:

- CIFAR-10 and CIFAR-10-C images.
- A trained ResNet-50 checkpoint named `resnet50_cifar10.pth`.
- CSV files containing previously generated behavioral degradation, divergence, margin, accuracy, and threshold-analysis results.

The notebook currently contains Kaggle-specific input paths. Before running it in another environment, update these paths to match the local dataset and artifact locations.

### Execution

1. Download or configure the required datasets, model checkpoint, and intermediate CSV files.
2. Update the input and output paths in `2.ipynb`.
3. Install the required dependencies.
4. Open the notebook in Jupyter Notebook, JupyterLab, or a compatible notebook environment.
5. Execute the cells in their intended sequence, ensuring that the required artifacts are available before running dependent analyses.

**Reproducibility note:** The notebook depends on external artifacts and intermediate results. A complete reproduction of the experiments requires access to the corresponding datasets, trained checkpoint, and generated CSV files. Results should be verified by executing the relevant cells rather than assuming that all experiments can run from the notebook alone.

## Limitations

The study has several limitations:

- Experiments focus on one model architecture and the CIFAR-10/CIFAR-10-C setting.
- The BDI uses fixed equal weights, which are not established as optimal.
- Primary warning thresholds are estimated from a small number of severity-1 reference observations.
- Corruption-specific degradation trajectories vary substantially.
- The evaluated warning mechanism does not consistently detect failure in advance.

These limitations motivate future work involving additional datasets and architectures, alternative signal-weighting strategies, more robust calibration, uncertainty-aware monitoring, and realistic distribution-shift settings.

## Future Research Directions

Potential extensions include:

- Evaluating generalization across different datasets and model architectures.
- Learning or adapting BDI component weights using independent validation data.
- Developing more reliable early-warning mechanisms for performance degradation.
- Integrating uncertainty estimation and probabilistic risk modeling.
- Comparing Grad-CAM with other explanation techniques.
- Evaluating behavioral monitoring under realistic temporal and distribution shifts.

## Citation and Research Context

For the complete methodology, experimental details, figures, statistical analyses, and references, consult `Manuscirpt.pdf`.

This repository documents an investigation into the relationship between classifier behavior, explanation instability, and performance degradation under progressive image corruption. The implementation and findings are intended to support further research into model reliability, explainable AI, and behavioral monitoring of deep learning systems.
