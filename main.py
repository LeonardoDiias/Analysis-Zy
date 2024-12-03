import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


img_width, img_height = 128, 128
batch_size = 32
epochs = 20
dataset_path = 'dataset'


data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2) 

train_gen = data_gen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',  # Como é uma classificação binária
    subset='training',
    shuffle=True
)

validation_gen = data_gen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',  
    subset='validation'
)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid para classificação binária
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
]


model.fit(
    train_gen,
    epochs=epochs,
    validation_data=validation_gen,
    callbacks=callbacks
)


model.save('pneumonia_detection_model.h5')
class_labels = {v: k for k, v in train_gen.class_indices.items()}
np.save('class_labels.npy', class_labels)
print("Model and class labels saved.")

