import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define constants
IMG_HEIGHT = 32
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20

# Define paths
data_dir = 'Datasets/IAM_Words'
words_file = os.path.join(data_dir, 'words.txt')
images_dir = os.path.join(data_dir, 'words')

# Read and process words.txt
def process_words_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 9:
                    image_id = parts[0]
                    word = parts[-1]
                    data.append((image_id, word))
    return pd.DataFrame(data, columns=['image_id', 'word'])

df = process_words_file(words_file)

# Split the data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Create a dictionary mapping words to integer labels
word_to_index = {word: idx for idx, word in enumerate(df['word'].unique())}
num_classes = len(word_to_index)

# Custom data generator
class WordDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, word_to_index, batch_size, img_size, images_dir, is_training=True):
        self.dataframe = dataframe
        self.word_to_index = word_to_index
        self.batch_size = batch_size
        self.img_size = img_size
        self.images_dir = images_dir
        self.is_training = is_training
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_df = self.dataframe.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.zeros((len(batch_df), self.img_size[0], self.img_size[1], 1))
        batch_y = np.zeros((len(batch_df), len(self.word_to_index)))

        for i, (_, row) in enumerate(batch_df.iterrows()):
            img_path = os.path.join(self.images_dir, row['image_id'].split('-')[0],
                                    '-'.join(row['image_id'].split('-')[:2]),
                                    f"{row['image_id']}.png")
            img = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale',
                                                        target_size=self.img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize
            batch_x[i] = img_array
            batch_y[i, self.word_to_index[row['word']]] = 1

        if self.is_training:
            # Apply data augmentation here if needed
            pass

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.is_training:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)

# Create data generators
train_generator = WordDataGenerator(train_df, word_to_index, BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH), images_dir)
val_generator = WordDataGenerator(val_df, word_to_index, BATCH_SIZE, (IMG_HEIGHT, IMG_WIDTH), images_dir, is_training=False)

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

# Save the model
model.save('word_classification_model.h5')

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('word_classification_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Training completed and models saved.")