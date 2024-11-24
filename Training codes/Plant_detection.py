import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications import EfficientNetV2B0 # type: ignore
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore


# datasets
train_dir = r"C:\Crop_disease_prediction\plant detection\New_data"

IMG_SIZE = (224, 224) 
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

base_model.trainable = False

# Adding layers on the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  
output_layer = Dense(train_generator.num_classes, activation='softmax')(x)

# Creating the final model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compileing the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


checkpoint = ModelCheckpoint('Plant_CPU.keras', monitor='accuracy', verbose=1, save_best_only=True, mode='max')

early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

base_model.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_history = model.fit(
    train_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)

model.save('Plant_detect_cpu.keras')

print("Model training and saving completed.")
