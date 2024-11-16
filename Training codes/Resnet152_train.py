import os
import tensorflow as tf
import keras
from tensorflow.keras.applications import ResNet152V2 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
import warnings

warnings.filterwarnings("ignore", message="Palette images with Transparency")

# datasets
train_dir = r"C:\Crop_disease_prediction\New Plant Diseases Dataset\train"
val_dir = r"C:\Crop_disease_prediction\New Plant Diseases Dataset\valid"

IMG_SIZE = (224, 224)  
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = len(os.listdir(train_dir))  

train_datagen = ImageDataGenerator(rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))

base_model.trainable = False

# Initialize the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# assigning checkpoints
checkpoint = ModelCheckpoint("resnet-10.keras", monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint, early_stopping]
)

#fine tuning the model
base_model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint, early_stopping]
)

model.save("final_resnet-10.keras")

print ("training complete")