# ecosnap/train_model.py

from tensorflow.keras.applications import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
import os

# Data
img_size = (224, 224)
batch_size = 16
data_dir = "dataset"

# Image Generator
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(train_gen.num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

# Freeze base
for layer in base_model.layers:
    layer.trainable = False

# Compile
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# Train
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model/food_model.h5")

# Save class names
os.makedirs("data", exist_ok=True)
with open("data/class_names.txt", "w") as f:
    f.write("\n".join(train_gen.class_indices.keys()))

print("✅ Training complete. Model and class names saved.")
