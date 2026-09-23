# Imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# import tensorflow as tf

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

# Model
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(xTrainProcessed, yTrain)

# Evaluations
preds = rf.predict(xValProcessed)
print("Accuracy:", accuracy_score(yVal, preds))
print(classification_report(yVal, preds))