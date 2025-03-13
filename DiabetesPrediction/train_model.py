import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
df = pd.read_csv(url, names=columns)

# Split data
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1. Deep Learning Model
def create_model():
    model = Sequential([
        Dense(128, input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        Dropout(0.4),
        Dense(64),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        Dropout(0.3),
        Dense(32),
        BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001), 
                  loss="binary_crossentropy", 
                  metrics=["accuracy"])
    return model

model = create_model()
early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=16, 
          validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)
dl_accuracy = model.evaluate(X_test, y_test)[1]
model.save("diabetes_model.h5")
joblib.dump(scaler, "scaler.pkl")

# 2. kNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_accuracy = accuracy_score(y_test, knn.predict(X_test))
joblib.dump(knn, "knn_model.pkl")

# 3. SVM Linear
svm_linear = SVC(kernel="linear", probability=True)
svm_linear.fit(X_train, y_train)
svm_linear_accuracy = accuracy_score(y_test, svm_linear.predict(X_test))
joblib.dump(svm_linear, "svm_linear_model.pkl")

# 4. SVM RBF
svm_rbf = SVC(kernel="rbf", probability=True)
svm_rbf.fit(X_train, y_train)
svm_rbf_accuracy = accuracy_score(y_test, svm_rbf.predict(X_test))
joblib.dump(svm_rbf, "svm_rbf_model.pkl")

# 5. Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
logreg_accuracy = accuracy_score(y_test, logreg.predict(X_test))
joblib.dump(logreg, "logreg_model.pkl")

# Print accuracies
print(f"Deep Learning Accuracy: {dl_accuracy:.4f}")
print(f"kNN Accuracy: {knn_accuracy:.4f}")
print(f"SVM Linear Accuracy: {svm_linear_accuracy:.4f}")
print(f"SVM RBF Accuracy: {svm_rbf_accuracy:.4f}")
print(f"Logistic Regression Accuracy: {logreg_accuracy:.4f}")

# Plot accuracy comparison
accuracies = [dl_accuracy, knn_accuracy, svm_linear_accuracy, svm_rbf_accuracy, logreg_accuracy]
models = ["Deep Learning", "kNN", "SVM Linear", "SVM RBF", "Logistic Regression"]

plt.figure(figsize=(10, 6))
plt.plot(models, accuracies, marker="o", linestyle="-", color="b")
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.savefig("accuracy_comparison.png")
plt.show()