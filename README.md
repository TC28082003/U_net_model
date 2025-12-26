# Ultrasound Image Denoising with U-Net Models

This project focuses on developing and comparing different U-Net architectures to remove speckle noise—a common characteristic in ultrasound and radar imaging. The document includes experiments with various U-net architectures based on different parameters.

##  Project Workflow

The data pipeline and processing follow these key steps:

1.  **Data Preprocessing**: Images are randomly cropped into $256 \times 256$ patches during training to optimize GPU VRAM and focus on local texture features.
2.  **Noise Simulation**: Realistic speckle noise is generated using the formula: $Noisy = Image \times (1 + Gaussian\_Noise)$ and additive Gaussian noise depend on each Set . (For example: Set 3 uses additive Gaussian noise).
3.  **Model Training**: Various U-Net configurations learn to map noisy inputs back to their clean targets.
4.  **Weighted Reconstruction**: Full images are reconstructed from overlapping patches using a Hann window and weighted average to eliminate boundary artifacts.
5.  **Natural Blending**: A post-processing step blends the model output with the original noisy image to preserve natural details in dark regions.

##  Comparison of Training Sets

The project evaluates three distinct configurations (Sets) to test different architectures and loss functions:
The following table summarizes the architectural and training parameters for each set, highlighting the evolution from baseline to specialized models:


| Category | Parameter | Set 1 | Set 2 | Set 3 | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Architecture** | Network type | Standard U-Net | Attention U-Net | Mini U-Net | Baseline encoder–decoder architecture |
| | CNN depth (levels) | 2 (4) | 2 | 1 | Number of encoder–decoder levels |
| | Total layers | 32 | 32 | 23 | Depends on network depth |
| | Initial filters | 32 | 32 | 32 | Doubled at each deeper level |
| | Kernel size | (3, 3) | (3, 3) | (3, 3) | Standard convolution kernel |
| | Activation | ReLU | LeakyReLU | LeakyReLU | Used in hidden layers |
| | Output activation | Sigmoid | Sigmoid | Sigmoid | Depends on output data range |
| | Skip connections | Concatenate | Enhanced (+Extra skip) | Concatenate (Choice: NoSkipU-Net) | Encoder–decoder feature fusion |
| **Training Parameters** | Loss function | MAE (L1 Loss) | Hybrid (MAE + SSIM) | MSE (L2 Loss)/RMSE | Compared under identical conditions |
| | SSIM weight ($\lambda$) | N/A | 0.8 | N/A | Only for Hybrid loss |
| | Optimizer | Adam ($10^{-3}, 10^{-4}, 10^{-5}$) | Adam ($10^{-3}$) | Adam ($10^{-3}$) | Same optimizer for all experiments |
| | Noise Type | Multiplicative (Speckle) | Hybrid (Speckle + Additive) | Additive (Speckle) | Type of noise used for training |
### Set 1: Baseline U-Net
* **Architecture**: `Baseline_UNet` with standard convolutional blocks (Conv -> BN -> ReLU).
* **Loss Function**: Mean Absolute Error (MAE/L1 Loss).
* **Noise Level**: High speckle noise ($\sigma = 20/255$).
* **Purpose**: Serves as the fundamental model to evaluate basic denoising performance.

### Set 2: Structure-Aware U-Net
* **Architecture**: `Attention_UNet` featuring enhanced convolutional blocks with double layers.
* **Loss Function**: Custom `structure_loss` combining $20\%$ MAE and $80\%$ SSIM (Structural Similarity Index).
* **Noise Level**: Variable noise levels ($\sigma = 10, 20, 30$).
* **Purpose**: Prioritizes edge preservation and surface texture, making ultrasound images appear more anatomically accurate after denoising.

### Set 3: Mini U-Net
* **Architecture**: `Mini_UNet`, a shallow and lightweight version of the U-Net architecture.
* **Loss Function**: Mean Squared Error (MSE).
* **Training Strategy**: Trained for a higher number of epochs (100–200) to compensate for the simpler architecture.
* **Purpose**: Optimized for lower computational costs while maintaining stable denoising quality.

##  Directory Structure
* `images/clean/`: Contains over 500 clean ultrasound images for training. (For security reasons, the images we use cannot be uploaded)
* `ready for label/`: Contains 47 images reserved for model testing.
* `results_inference/`: Stores denoised outputs and comparisons for each Set.

##  System Requirements
* **Python**: 3.11+
* **Core Libraries**: TensorFlow, OpenCV, Matplotlib, Scikit-image, Pandas.
* **Advanced Metrics**: LPIPS and PyTorch (for perceptual loss evaluation).



