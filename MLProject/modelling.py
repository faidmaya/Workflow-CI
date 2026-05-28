import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Mengaktifkan autolog MLflow
mlflow.sklearn.autolog()

# Load dataset preprocessing
df = pd.read_csv("titanic_preprocessing.csv")

# Memisahkan fitur dan target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Split data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Start MLflow run
with mlflow.start_run():

    # Membuat model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Training model
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")