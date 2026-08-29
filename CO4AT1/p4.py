import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np

# 1. Generate Synthetic Medical Data (Zero download, runs instantly)
print("Generating synthetic 224x224 medical images...")
num_samples = 100
x_data = np.random.rand(num_samples, 224, 224, 3).astype('float32')
y_data = np.random.randint(0, 2, size=(num_samples, 1))  # 0 = Normal, 1 = Pneumonia

# Split into train and test
x_train, x_test = x_data[:80], x_data[80:]
y_train, y_test = y_data[:80], y_data[80:]

# 2. Build Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=predictions)

# 3. Compile and Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training Medical Image Classifier...")
model.fit(x_train, y_train, epochs=3, batch_size=16, validation_data=(x_test, y_test))