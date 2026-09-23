# Imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf

df = pd.read_csv("data/trimmedTitanic.csv")
x = df[["Pclass", "Sex", "Age"]]
y= df["Survived"]

XTrain, XVal, yTrain, yVal = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)  
# Preprocessing for numerical data
numericFeatures = ["Age"]
categoricalFeatures = ["Pclass", "Sex"]

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numericFeatures),
    ("cat", Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categoricalFeatures)
])

xTrainProcessed = preprocessor.fit_transform(XTrain)
xValProcessed = preprocessor.transform(XVal)

def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model

model = build_model(xTrainProcessed.shape[1])

earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    xTrainProcessed, yTrain,
    validation_data=(xValProcessed, yVal),
    epochs=100,
    batch_size=32,
    callbacks=[earlyStop],
    verbose=1
)

# Evaluate the model
loss, acc = model.evaluate(xValProcessed, yVal, verbose=0)
print("TensorFlow accuracy:", acc)