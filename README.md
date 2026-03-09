Brain Tumor MRI Classification
This notebook demonstrates the process of building, training, and evaluating a Convolutional Neural Network (CNN) for classifying brain MRI images into different tumor types.

1. Setup and Imports
First, necessary libraries such as TensorFlow, Keras, NumPy, and Matplotlib are imported. Warnings are also ignored for a cleaner output.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
2. Dataset Download and Inspection
The brain MRI dataset is downloaded from Kaggle using kagglehub. The dataset structure (Training and Testing directories) is inspected.

import kagglehub
import os

path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")

train_dir = os.path.join(path, 'Training')
test_dir = os.path.join(path, 'Testing')

print(f"Training directory: {train_dir}")
print(f"Testing directory: {test_dir}")
3. Data Loading and Preprocessing
Image datasets for training and testing are loaded using tf.keras.utils.image_dataset_from_directory. Images are resized to 256x256 pixels, batched, and labels are set to 'categorical' for multi-class classification.

from tensorflow.keras.utils import image_dataset_from_directory

image_size = (256, 256)
batch_size = 32

train_ds = image_dataset_from_directory(
    train_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

test_ds = image_dataset_from_directory(
    test_dir,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='categorical'
)

for images, labels in train_ds.take(1):
    print("Shape of images batch:", images.shape)
    print("Shape of labels batch:", labels.shape)
    print("Class names:", train_ds.class_names)
4. Model Definition (CNN Architecture)
A Sequential CNN model is defined with multiple Conv2D and MaxPooling2D layers for feature extraction, followed by Flatten and Dense layers for classification. The final Dense layer uses a softmax activation for multi-class probability output.

model = Sequential()
model.add(layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Conv2D(128, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Conv2D(256, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
5. Model Training
The model is trained using the train_ds for 50 epochs. The training history, including accuracy and loss, is recorded.

history = model.fit(train_ds, epochs = 50)
6. Visualization of Training Accuracy
The training accuracy over epochs is plotted to visualize the model's learning progress.

plt.plot(history.history['accuracy'])
plt.title('Model Accuracy during Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
7. Model Prediction and Evaluation
Predictions are made on the test_ds. A function plot_image_predictions is defined to visualize individual predictions, showing the predicted label versus the true label for a sample of images.

prediction = model.predict(test_ds)

def plot_image_predictions(images, true_labels_indices, prediction, class_names):
    plt.figure(figsize=(12, 12))
    for i in range(min(25, len(images))):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].astype("uint8"))

        predicted_label_index = np.argmax(prediction[i])
        true_label_index = true_labels_indices[i]

        color = 'blue' if predicted_label_index == true_label_index else 'red'
        plt.xlabel(f"{class_names[predicted_label_index]} ({class_names[true_label_index]})", color=color)
    plt.tight_layout()
    plt.show()

class_names = test_ds.class_names

all_test_images = []
all_test_labels_one_hot = []
for images_batch, labels_batch in test_ds:
    all_test_images.append(images_batch.numpy())
    all_test_labels_one_hot.append(labels_batch.numpy())

all_test_images = np.concatenate(all_test_images, axis=0)
all_test_labels_one_hot = np.concatenate(all_test_labels_one_hot, axis=0)

all_test_labels_indices = np.argmax(all_test_labels_one_hot, axis=1)

plot_image_predictions(all_test_images, all_test_labels_indices, prediction, class_names)
8. Save and Download Model
The trained model is saved in HDF5 format and then downloaded to the local machine.

model.save('brain_tumor_model.h5')

from google.colab import files
files.download('brain_tumor_model.h5')
