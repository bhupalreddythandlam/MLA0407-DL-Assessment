import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

# 1. Load Tiny Dataset Subset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, y_train = x_train[:1000], y_train[:1000]
x_test, y_test = x_test[:200], y_test[:200]

# 2. Preprocess: Convert 1-channel to 3-channel RGB and resize to 32x32
x_train = np.repeat(x_train[..., np.newaxis] / 255.0, 3, axis=-1)
x_test = np.repeat(x_test[..., np.newaxis] / 255.0, 3, axis=-1)
x_train = tf.image.resize(x_train, [32, 32]).numpy()
x_test = tf.image.resize(x_test, [32, 32]).numpy()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. Build Model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
base_model.trainable = False  # Freeze base model

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compile and Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Training Transfer Learning Model...")
model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test))