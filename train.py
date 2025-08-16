import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import os
import numpy as np

train_dir = "dataset/train"
val_dir   = "dataset/val"


classes = ['fake', 'real']
counts = [len(os.listdir(os.path.join(train_dir, c))) for c in classes]

class_weights_array = compute_class_weight(
    'balanced',
    classes=np.arange(len(classes)),
    y=np.repeat(np.arange(len(classes)), counts)
)
class_weight_dict = {0: class_weights_array[0], 1: class_weights_array[1]}
print("Class weights:", class_weight_dict)


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    shear_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode="binary",
    color_mode="rgb"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode="binary",
    color_mode="rgb"
)


base_model = ResNet50(include_top=False, input_shape=(128,128,3), weights="imagenet")
base_model.trainable = False  

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


checkpoint = ModelCheckpoint("deepfake_detector.keras", monitor="val_accuracy", save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-7)


history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    callbacks=[checkpoint, early_stop, lr_reduce],
    class_weight=class_weight_dict
)


base_model.trainable = True
for layer in base_model.layers[:50]: 
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy",
              metrics=["accuracy"])

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[checkpoint, early_stop, lr_reduce],
    class_weight=class_weight_dict
)

print("✅ Training completed, model saved as deepfake_detector.keras")


def plot_history(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']

    epochs = range(1, len(acc)+1)

    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b-', label="Training Accuracy")
    plt.plot(epochs, val_acc, 'r-', label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")

    # Loss
    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b-', label="Training Loss")
    plt.plot(epochs, val_loss, 'r-', label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")

    plt.tight_layout()
    plt.savefig("training_plot.png")  
    plt.show()

plot_history(history1, history2)
