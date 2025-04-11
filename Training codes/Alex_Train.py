import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import os

def create_alexnet(input_shape=(224, 224, 3), num_classes=38):
    model = Sequential([
        Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=input_shape),
        MaxPooling2D((3, 3), strides=2),

        Conv2D(256, (5, 5), padding="same", activation="relu"),
        MaxPooling2D((3, 3), strides=2),

        Conv2D(384, (3, 3), padding="same", activation="relu"),
        Conv2D(384, (3, 3), padding="same", activation="relu"),
        Conv2D(256, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((3, 3), strides=2),

        Flatten(),
        Dense(4096, activation="relu"),
        Dropout(0.5),
        Dense(4096, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")  # Multi-class classification
    ])
    return model

train_dir = "C:/Crop_disease_prediction/New Plant Diseases Dataset/train"
val_dir = "C:/Crop_disease_prediction/New Plant Diseases Dataset/valid"

IMG_SIZE = (224, 224) 
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Get number of classes
num_classes = len(train_generator.class_indices)

# Compile and train the model
alexnet_model = create_alexnet(num_classes=num_classes)
alexnet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

history = alexnet_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=20,
    verbose=1
)

alexnet_model.save(r'C:\Crop_disease_prediction\Project1\Trained models\AlexNet.keras')
print("Training Completed")