import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

# Function to load images from a directory
def load_images_from_directory(base_dir):
    data = []
    labels = []
    shapes = ['circle', 'kite', 'parallelogram', 'rectangle', 'rhombus', 'square', 'trapezoid', 'triangle']
    
    for shape in shapes:
        shape_dir = os.path.join(base_dir, shape)
        for filename in os.listdir(shape_dir):
            img_path = os.path.join(shape_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                data.append(img)
                labels.append(shape)
    
    return np.array(data), np.array(labels)

# Load training dataset
data_train_dir = '../dataset/train'
data_train, labels_train = load_images_from_directory(data_train_dir)

# Preprocess training data
data_train = data_train.reshape((data_train.shape[0], 128, 128, 1)) / 255.0
label_encoder = LabelEncoder()
labels_train = to_categorical(label_encoder.fit_transform(labels_train))

# Load testing dataset
data_test_dir = '../dataset/test'
data_test, labels_test = load_images_from_directory(data_test_dir)

# Preprocess testing data
data_test = data_test.reshape((data_test.shape[0], 128, 128, 1)) / 255.0
labels_test = to_categorical(label_encoder.transform(labels_test))

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(data_train)

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with data augmentation
model.fit(datagen.flow(data_train, labels_train, batch_size=32), 
          epochs=50, 
          validation_split=0.2, 
          callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(data_test, labels_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save('/model.h5')
