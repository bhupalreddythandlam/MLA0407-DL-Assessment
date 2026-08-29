import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# 1. Load and Preprocess Data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, y_train = x_train[:1000], y_train[:1000]
x_test, y_test = x_test[:200], y_test[:200]

x_train = np.repeat(x_train[..., np.newaxis] / 255.0, 3, axis=-1)
x_test = np.repeat(x_test[..., np.newaxis] / 255.0, 3, axis=-1)
x_train = tf.image.resize(x_train, [32, 32]).numpy()
x_test = tf.image.resize(x_test, [32, 32]).numpy()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Base Model setup
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 3. Phase 1: Train frozen model
print("Phase 1: Training top layers...")
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2, batch_size=32)

# 4. Phase 2: Fine-Tuning
print("Phase 2: Fine-tuning top layers of base model...")
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Keep mostly frozen, unfreeze last 10 layers
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2, batch_size=32, validation_data=(x_test, y_test))