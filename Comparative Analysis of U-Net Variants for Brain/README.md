
# Medical Image Segmentation and Deep Learning Experiments

This directory contains a research paper and notebook-based implementations exploring deep learning for image segmentation and computer vision. The materials cover 3D brain tumor segmentation, attention-based retinal blood vessel segmentation, and transfer learning for face recognition.

The projects demonstrate experience with convolutional neural networks, U-Net architectures, image preprocessing, data augmentation, model training, and performance evaluation.

## Repository Contents

| File | Description |
|---|---|
| `final_assignment_paper.pdf` | Research paper comparing 3D U-Net architectures for brain tumor segmentation using the BraTS20 dataset. |
| `attention-unet.ipynb` | Implementation of an attention-based U-Net for retinal blood vessel segmentation, including image preprocessing and edge detection. |
| `transfer-learning.ipynb` | Notebook exploring transfer learning and fine-tuning for face recognition using a pretrained model and a four-class classification output. |

## 1. Comparative Analysis of U-Net Variants for Brain Tumor Segmentation

**Research area:** Medical image analysis, 3D semantic segmentation, and deep learning architectures.

The research paper compares three architectures for brain tumor segmentation using the BraTS20 dataset:

- **Baseline 3D U-Net:** An encoder–decoder architecture with 3D convolutions, max pooling, transposed convolutions, and skip connections.
- **Optimized 3D U-Net with Residual Blocks:** Incorporates residual connections to improve feature propagation through deeper networks.
- **U-Net 3+:** Uses full-scale skip connections to combine features across multiple encoder and decoder levels.

### Dataset

BraTS20 provides multimodal brain MRI scans, including:

- T1-weighted (T1)
- Contrast-enhanced T1-weighted (T1ce)
- T2-weighted (T2)
- Fluid-Attenuated Inversion Recovery (FLAIR)

### Evaluation Metrics

The paper examines segmentation performance using:

- Dice coefficient and Dice loss
- Jaccard index and Jaccard loss
- Mean Intersection over Union (Mean IoU)
- Precision, sensitivity, and specificity
- Class-specific Dice coefficients for necrotic tissue, edema, and enhancing tumor regions

### Reported Results

The paper reports the following test-set metrics:

| Model | Mean IoU | Dice coefficient |
|---|---:|---:|
| Baseline 3D U-Net | 0.8320 | 0.9889 |
| Optimized U-Net | 0.8725 | 0.4001 |
| U-Net 3+ | 0.9438 | 0.5042 |

These are the values reported in the manuscript. Their interpretation requires care because Mean IoU and Dice values show substantially different patterns across models. Further verification of metric definitions, class aggregation, and evaluation code would be appropriate before drawing conclusions about which architecture performs best overall.

For the complete methodology, architecture diagrams, training curves, validation results, and discussion, consult `final_assignment_paper.pdf`.

## 2. Attention U-Net for Retinal Blood Vessel Segmentation

**Research area:** Biomedical image processing, attention mechanisms, and semantic segmentation.

The `attention-unet.ipynb` notebook implements an attention-based U-Net pipeline for segmenting retinal blood vessels from images and corresponding segmentation masks.

### Methodology

The notebook includes the following steps:

1. **Image loading:** Reads retinal images and their corresponding masks from separate directories.
2. **Preprocessing:** Resizes images to 128 × 128 pixels, normalizes pixel values, converts images to grayscale for preprocessing, and applies histogram equalization.
3. **Edge detection:** Uses the Canny edge detector to extract edge information.
4. **Data augmentation:** Applies rotations, shifts, shearing, zooming, and horizontal flips.
5. **Attention mechanism:** Uses an attention block to generate spatial attention weights for encoder features.
6. **Model training:** Trains the segmentation model using paired images and masks.
7. **Evaluation and visualization:** Loads test images, evaluates the model, and visualizes input images, reference masks, and predicted masks.

### Technologies

- Python
- TensorFlow and Keras
- OpenCV
- NumPy
- Matplotlib
- Seaborn

### Running the Notebook

The notebook uses Kaggle-specific paths for the retinal blood vessel dataset. Update the dataset paths before running it in another environment.

The saved model is written as `unet_model.h5`. A complete reproduction requires the corresponding images and masks, compatible library versions, and the notebook's preprocessing and training steps.

## 3. Transfer Learning for Face Recognition

**Research area:** Computer vision, transfer learning, and image classification.

The `transfer-learning.ipynb` notebook explores a transfer-learning workflow for classifying face images into four classes.

### Methodology

The implementation includes:

- Inspecting class directories and visualizing sample images.
- Checking image readability and identifying potentially corrupted files.
- Converting images to grayscale during dataset preparation.
- Splitting images into training, validation, and testing directories.
- Loading an existing trained model from a saved Keras model file.
- Fine-tuning selected layers while keeping earlier layers frozen.
- Adding dense layers, batch normalization, ReLU activations, and dropout.
- Preparing image generators with resizing, normalization, and data augmentation.
- Tracking training and validation accuracy and loss.
- Generating predictions for test images.

The notebook uses images organized into identity-specific folders and defines a four-class softmax output.

### Technologies

- Python
- TensorFlow and Keras
- NumPy
- Matplotlib
- Pillow
- scikit-learn

### Running the Notebook

The notebook references Kaggle dataset and model paths. Before execution, configure the face image dataset, the pretrained model file, and the training, validation, and testing directories.

The pretrained model is referenced as `Final_Model.h5`. The notebook therefore depends on an existing model artifact rather than being entirely self-contained.

## Technical Skills Demonstrated

These materials cover several practical areas of machine learning development:

| Area | Techniques |
|---|---|
| Image preprocessing | Resizing, normalization, grayscale conversion, histogram equalization |
| Computer vision | Edge detection, image classification, semantic segmentation |
| Deep learning | CNNs, encoder–decoder networks, 3D convolutions |
| Model architectures | U-Net, attention mechanisms, residual blocks, U-Net 3+ |
| Transfer learning | Layer freezing, fine-tuning, pretrained model reuse |
| Regularization | Dropout and data augmentation |
| Evaluation | Dice coefficient, IoU, precision, sensitivity, specificity, accuracy and loss |
| Experimentation | Jupyter notebooks, training curves and prediction visualization |

## Reproducibility and Limitations

The notebooks depend on external datasets, local or Kaggle-specific paths, and, in the transfer-learning workflow, a pretrained model artifact. Consequently, cloning the repository alone may not be sufficient to reproduce every experiment.

Before running the notebooks:

1. Obtain the required datasets and model files.
2. Update all dataset and checkpoint paths.
3. Install compatible Python libraries.
4. Verify image-mask correspondence and class labels.
5. Re-run the experiments to validate the reported results.

The brain tumor segmentation manuscript and the two notebooks address different datasets and tasks. They should therefore be evaluated independently rather than treated as a single end-to-end experimental pipeline.

## Purpose

This collection documents practical work in deep learning, biomedical image segmentation, and computer vision. It provides a starting point for examining model architectures, preprocessing workflows, transfer learning, and quantitative evaluation, while identifying areas where further validation and reproducibility work may be needed.
