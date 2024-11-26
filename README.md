# Brain-Mri-Synthesis-and-Tumor-detection

```markdown
# Brain MRI Synthesis and Tumor Detection

A deep learning project leveraging **Deep Convolutional GANs (DC-GAN)** for generating synthetic MRI images and **DenseNet121** for brain tumor classification. The project addresses dataset limitations using synthetic image augmentation, significantly improving classification accuracy.

---

## Overview

This project employs:
1. **DC-GAN**: Generates high-quality synthetic MRI images to augment the dataset.
2. **DenseNet121**: A pre-trained model fine-tuned to classify MRI images into tumor or no-tumor categories.

By combining synthetic data and deep learning classification, the project achieves state-of-the-art accuracy in tumor detection.

---

## Dataset

- **Source**: [Brain MRI Images for Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Structure**:
  - `yes/`: MRI images with tumors.
  - `no/`: MRI images without tumors.
- **Preprocessing**:
  - Resized images to \(128 \times 128\).
  - Normalized pixel values to [0, 1].
  - Applied augmentations (rotation, flipping, zooming) to enhance the dataset.

---

## Technologies Used

- **Programming Language**: Python 3.8+
- **Frameworks**: TensorFlow, Keras
- **Model Architectures**:
  - DC-GAN for MRI synthesis.
  - DenseNet121 for classification.
- **Visualization**: Matplotlib, Seaborn, OpenCV

---

## How to Use

### **1. Training the DC-GAN**
To train the GAN for synthetic MRI image generation:
```python
# Import required libraries
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Reshape, Conv2DTranspose, BatchNormalization, Activation, LeakyReLU

# Define generator and discriminator
generator = define_generator()
discriminator = define_discriminator()

# Compile the GAN
gan = compile_gan(generator, discriminator)

# Train the GAN
train_gan(generator, discriminator, gan, real_images, epochs=50, batch_size=32)
```

---

### **2. Generating Synthetic MRI Images**
Generate new MRI images using the trained GAN:
```python
# Generate synthetic images
noise = np.random.normal(0, 1, (10, noise_dim))
generated_images = generator.predict(noise)

# Display generated images
plot_images(generated_images, title="Synthetic MRI Images")
```

---

### **3. Training the Tumor Classifier**
Fine-tune the DenseNet121 model for binary tumor classification:
```python
# Load DenseNet121 with pre-trained weights
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential

base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

# Add custom classification layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the classifier
model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=20, batch_size=32)
```

---

### **4. Evaluating the Classifier**
Evaluate the model on the test set:
```python
# Evaluate the model
results = model.evaluate(test_data, test_labels)
print(f"Test Accuracy: {results[1]*100:.2f}%")
```

---

## Results

- **Synthetic Data Quality**:
  - Evaluated using Frechet Inception Distance (FID).
- **Tumor Classification**:
  - **Accuracy**: 96%
  - **Precision**: 93%
  - **Recall**: 97%
  - **F1-Score**: 95%

Sample classification performance:
```plaintext
| Metric       | Value  |
|--------------|--------|
| Accuracy     | 96%    |
| Precision    | 93%    |
| Recall       | 97%    |
| F1-Score     | 95%    |
```

---

## Future Improvements

- Explore advanced GAN architectures like StyleGAN for better synthesis.
- Extend classification to multi-class tumor types.
- Integrate explainable AI techniques for model interpretability.
- Optimize for real-time inference using edge devices.

---

## Acknowledgments

- Kaggle for providing the dataset.
- TensorFlow and Keras for the deep learning framework.
- The academic and research community for inspiration.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request.
```

This updated README replaces **VGGNet** with **DenseNet121**, reflecting its architecture and usage in your project. Let me know if further modifications are needed!
