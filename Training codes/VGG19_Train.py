import tensorflow as tf
from tensorflow.keras.applications import VGG19 # type: ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 64

TRAIN_DIR = "C:/Crop_disease_prediction/New Plant Diseases Dataset/train"
VALID_DIR = "C:/Crop_disease_prediction/New Plant Diseases Dataset/valid"

train_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = val_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

NUM_CLASSES = len(train_generator.class_indices)

for layer in base_model.layers:
    layer.trainable = False

# Add Custom Layers
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

# Compile Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=20,
    validation_data=valid_generator
)

model.save(r'C:\Crop_disease_prediction\Project1\Trained models\VGG19.keras')
print("Training Completed")