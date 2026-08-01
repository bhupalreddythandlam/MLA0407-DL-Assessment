from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time

features = [10, 100, 1000, 10000]

for n in features:
    X, y = make_classification(n_samples=1000, n_features=n, n_informative=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LogisticRegression(max_iter=1000)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    acc = model.score(X_test, y_test)
    print(f"Features: {n:5} | Time: {train_time:.4f}s | Accuracy: {acc:.4f}")