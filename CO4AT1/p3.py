import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import fashion_mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, y_train = x_train[:500], y_train[:500]
x_test, y_test = x_test[:100], y_test[:100]

x_train = np.repeat(x_train[..., np.newaxis] / 255.0, 3, axis=-1)
x_test = np.repeat(x_test[..., np.newaxis] / 255.0, 3, axis=-1)
x_train = tf.image.resize(x_train, [32, 32]).numpy()
x_test = tf.image.resize(x_test, [32, 32]).numpy()

# 2. Extract Features
print("Extracting Features using VGG16...")
feature_extractor = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

train_features = feature_extractor.predict(x_train)
train_features_flat = train_features.reshape(train_features.shape[0], -1)

test_features = feature_extractor.predict(x_test)
test_features_flat = test_features.reshape(test_features.shape[0], -1)

# 3. Train Machine Learning Classifier
print("Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(train_features_flat, y_train)

# 4. Evaluate
predictions = rf.predict(test_features_flat)
print(f"Random Forest Accuracy: {accuracy_score(y_test, predictions):.4f}")