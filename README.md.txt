# Revealing the Spatiotemporal Dynamics of Subcellular POD Activity in Tomato under Salt Stress by Explainable Micro-Hyperspectral Imaging and Deep Learning

## Project Overview

This project focuses on the non-destructive detection of Peroxidase (POD) activity changes in tomato subcellular structures (chloroplasts, mitochondria, cytoplasm) under salt stress. Utilizing micro-hyperspectral imaging technology combined with explainable AI, this study evaluates the importance of characteristic wavelengths and develops predictive models. The core models employed are Competitive Adaptive Reweighted Sampling-Least Squares Support Vector Machine (CARS-LSSVM), Competitive Adaptive Reweighted Sampling-Multiple Linear Regression (CARS-MLR), and Competitive Adaptive Reweighted Sampling-Convolutional Neural Network (CARS-CNN). Furthermore, SHapley Additive exPlanations (SHAP) is applied to interpret model decisions, identifying critical feature wavelengths and providing physical explanations for the predictions. The ultimate goal is to establish a visualization map of POD activity distribution in subcellular structures, offering insights for monitoring salt stress levels and precision cultivation practices in tomato plants.

## Abstract

Peroxidase (POD), a pivotal enzyme in the plant antioxidant system, exhibits activity dynamics that are directly correlated with salt tolerance in tomato. This study, through micro-hyperspectral imaging technology combined with explainable AI, evaluates the importance of characteristic wavelengths in the Competitive Adaptive Reweighted Sampling-Least Squares Support Vector Machine (CARS-LSSVM), Competitive Adaptive Reweighted Sampling-Multiple Linear Regression (CARS-MLR), and Competitive Adaptive Reweighted Sampling-Convolutional Neural Network (CARS-CNN) models for detecting POD activity changes in tomato subcellular structures (chloroplasts, mitochondria, cytoplasm). Based on this, non-destructive detection of POD activity changes in tomato leaf subcellular structures was achieved. As results of model evaluation, good prediction performances for POD activity were obtained using CARS-LSSVM (RC=0.854, RP= 0.745) for chloroplast, using CARS-MLR(RC= 0.793, RP= 0.819) for mitochondria, and using CARS-CNN (RC=0.897, RP= 0.794) for cytoplasm. To clarify the prediction mechanisms of the CARS-LSSVM, CARS-MLR, and CARS-CNN models, this study applied SHapley Additive exPlanations (SHAP) to identify the most critical feature wavelengths contributing to the prediction of tomato POD activity. This provided a reliable physical explanation for the model decisions, confirming their effective integration of spectral and chemical information. Finally, a visualization map of POD activity distribution in subcellular structures was established. The results provide a method for monitoring salt stress levels in tomato plants, while also offering practical insights for precision cultivation practices.

**Keywords**: SHAP analysis; Machine learning; Microscopic hyperspectral imaging technology; Peroxidase activity; Salt stress; Tomato leaf

## Directory Structure



## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/spectral-modeling.git
    cd spectral-modeling
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Place your data:**
    *   Place your actual raw data into `data/raw/`.
    *   For testing with provided examples, the `data/sample/` directory already contains dummy `.xlsx` files. Ensure your actual data matches the expected format (features and target values as columns).

## How to Run

### 1. Configure Settings

Edit `configs/default.yaml` to specify data paths, feature columns, and model hyperparameters.

### 2. Train a Model

To train a specific model (e.g., LSSVM):

```bash
python src/train.py --model_type LSSVM
