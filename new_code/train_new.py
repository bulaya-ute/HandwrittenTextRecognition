import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import os
import matplotlib.pyplot as plt


# Load and preprocess the dataset
def load_data(image_dir, target_size=(64, 64)):
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        image_dir,
        target_size=target_size,
        color_mode='grayscale',  # Ensure images are loaded as grayscale
        class_mode='categorical',
        batch_size=32,
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        image_dir,
        target_size=target_size,
        color_mode='grayscale',  # Ensure images are loaded as grayscale
        class_mode='categorical',
        batch_size=32,
        subset='validation'
    )

    return train_gen, val_gen


# Build the CNN model
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model


# Load the data
train_gen, val_gen = load_data('path/to/dataset')

# Define the input shape and number of classes
input_shape = (64, 64, 1)  # Grayscale images of size 64x64
num_classes = len(train_gen.class_indices)  # Number of word classes

# Build the model
model = build_cnn_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen
)

# Save the model after training
model.save('word_classification_model.h5')

# Plot training & validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# Function to classify a word image using the saved model
def classify_word_image(image_path):
    # Load the saved model
    model = load_model('word_classification_model.h5')

    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64), color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class


# Example usage
image_path = 'path/to/new/word_image.jpg'
predicted_class = classify_word_image(image_path)
print("Predicted class:", predicted_class)
