import tensorflow as tf
from tensorflow.keras.applications import InceptionV3   # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import os

train_dir = r"C:\Crop_disease_prediction\New Plant Diseases Dataset\train"
val_dir = r"C:\Crop_disease_prediction\New Plant Diseases Dataset\valid"

IMG_SIZE = (299, 299) 
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

base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(299, 299, 3))

# Freeze base model layers
base_model.trainable = False

# Add Custom Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Prevent overfitting
output_layer = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the Model
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile the Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10
)

base_model.trainable = True
for layer in base_model.layers[:150]:  # Freeze first 150 layers
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10
)

model.save(r'C:\Crop_disease_prediction\Project1\Trained models\GoogleNetV4.keras')
print("Training Completed")