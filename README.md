# Brain MRI Synthesis and Tumor Detection

A deep learning project leveraging **Deep Convolutional GANs (DC-GAN)** for generating synthetic MRI images and **DenseNet121** for brain tumor classification. This project addresses dataset limitations through synthetic image augmentation, significantly improving classification accuracy.

---

## Overview

This project employs:
1. **DC-GAN**: Generates high-quality synthetic MRI images to augment the dataset.
2. **DenseNet121**: A pre-trained model fine-tuned for binary classification (tumor or no tumor).

By combining synthetic data and deep learning classification, the project achieves state-of-the-art accuracy in tumor detection.

---

## Dataset

- **Source**: [Brain MRI Images for Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Structure**:
  - `yes/`: MRI images with tumors.
  - `no/`: MRI images without tumors.
- **Preprocessing**:
  - Resized images to \(128 \times 128\).
  - Normalized pixel values to the range [0, 1].
  - Augmented data using rotation, flipping, and zooming to enhance the dataset.

---

## Technologies Used

- **Programming Language**: Python 3.8+
- **Deep Learning Frameworks**: TensorFlow, Keras
- **Model Architectures**:
  - DC-GAN for synthetic MRI image generation.
  - DenseNet121 for tumor classification.
- **Visualization**: Matplotlib, Seaborn, OpenCV

---

## How to Use

### **1. Training the DC-GAN**
To train the GAN for generating synthetic MRI images:
```python
# Import required libraries
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Reshape, Conv2DTranspose, BatchNormalization, Activation, LeakyReLU

# Define the generator and discriminator
generator = define_generator()
discriminator = define_discriminator()

# Compile the GAN
gan = compile_gan(generator, discriminator)

# Train the GAN
train_gan(generator, discriminator, gan, real_images, epochs=50, batch_size=32)
