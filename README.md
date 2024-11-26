

# Brain MRI Synthesis and Tumor Detection

A deep learning project leveraging **Deep Convolutional GANs (DC-GAN)** for generating synthetic MRI images and **DenseNet121** for brain tumor classification. This project addresses dataset limitations through synthetic image augmentation, significantly improving classification accuracy.

---

## Overview

This project combines advanced deep learning techniques to:

1. **Generate Synthetic MRI Images**: Use DC-GAN to create high-quality synthetic data for dataset augmentation.
2. **Classify Brain Tumors**: Utilize a fine-tuned DenseNet121 model to classify MRI images as either tumor or no-tumor.

By combining synthetic data generation and classification, the project achieves **state-of-the-art accuracy in brain tumor detection**.

📄 [Project Notebook on Kaggle](https://www.kaggle.com/code/suhawni/dl-project)

---

## Dataset

- **Source**: [Brain MRI Images for Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Structure**:
  - `yes/`: Contains MRI images with tumors.
  - `no/`: Contains MRI images without tumors.
- **Preprocessing**:
  - Resized images to \(128 \times 128\) pixels.
  - Normalized pixel values to the range \([0, 1]\).
  - Applied data augmentation (rotation, flipping, and zooming) to enhance training dataset size and diversity.

---

## Technologies Used

- **Programming Language**: Python 3.8+
- **Deep Learning Frameworks**: TensorFlow, Keras
- **Model Architectures**:
  - DC-GAN for synthetic MRI image generation.
  - DenseNet121 for binary tumor classification.
- **Visualization Tools**: Matplotlib, Seaborn, OpenCV



## Project Workflow

### **1. Training the DC-GAN**
Define and train a DC-GAN to generate synthetic MRI images.

```python
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Reshape, Conv2DTranspose, BatchNormalization, LeakyReLU

# Define the Generator
def define_generator():
    model = Sequential([
        Dense(256, input_dim=100),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dense(512),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dense(1024),
        LeakyReLU(0.2),
        BatchNormalization(),
        Dense(128 * 128 * 1, activation='tanh'),
        Reshape((128, 128, 1))
    ])
    return model

# Define the Discriminator
def define_discriminator():
    model = Sequential([
        Conv2D(64, kernel_size=3, strides=2, input_shape=(128, 128, 1), padding="same"),
        LeakyReLU(0.2),
        Conv2D(128, kernel_size=3, strides=2, padding="same"),
        LeakyReLU(0.2),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    return model

# Compile and train the GAN
generator = define_generator()
discriminator = define_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Build and compile GAN
from keras.models import Model
from keras.layers import Input

def compile_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    return gan

gan = compile_gan(generator, discriminator)

# Train the GAN
def train_gan(generator, discriminator, gan, real_images, epochs=50, batch_size=32):
    # Training logic goes here
    pass
```

---

### **2. Generating Synthetic MRI Images**
Generate synthetic images using the trained generator.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate images
def plot_images(generator, noise_dim, num_images):
    noise = np.random.normal(0, 1, (num_images, noise_dim))
    generated_images = generator.predict(noise)
    for i, img in enumerate(generated_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(img.reshape(128, 128), cmap="gray")
        plt.axis("off")
    plt.show()

# Plot generated images
plot_images(generator, noise_dim=100, num_images=5)
```

---

### **3. Training the Tumor Classifier**
Fine-tune DenseNet121 for binary classification.

```python
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Load DenseNet121 pre-trained model
base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

# Add classification layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=20, batch_size=32)
```

---

### **4. Evaluating the Classifier**
Evaluate the classifier on test data.

```python
# Evaluate model
results = model.evaluate(test_data, test_labels)
print(f"Test Accuracy: {results[1]*100:.2f}%")
```

---

## Results

### Synthetic Data Quality
- **Frechet Inception Distance (FID)**: Evaluated the similarity between real and synthetic images.

### Tumor Classification Performance
| Metric       | Value  |
|--------------|--------|
| Accuracy     | 96%    |
| Precision    | 93%    |
| Recall       | 97%    |
| F1-Score     | 95%    |

---

## Future Improvements

- **Advanced GAN Architectures**: Explore StyleGAN for more realistic synthetic images.
- **Multi-class Classification**: Extend to classify tumor types.
- **Explainable AI (XAI)**: Add interpretability to model predictions.
- **Real-time Deployment**: Optimize for inference on edge devices.

---

## Acknowledgments

- **Dataset**: Kaggle’s "Brain MRI Images for Brain Tumor Detection".
- **Frameworks**: TensorFlow and Keras for model development.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Contributing

Contributions are welcome!  
1. Fork the repository.  
2. Create a feature branch.  
3. Submit a pull request with a detailed description of the changes.  
```

### Steps to Upload:
1. Save this content in a file named `README.md`.
2. Add it to your GitHub repository in the root directory.
3. Commit and push to reflect it on the repository page.

This README is complete with code snippets, a clear structure, and professional formatting. Let me know if you need additional help!
