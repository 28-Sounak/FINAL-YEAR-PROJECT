import os

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report

df = pd.read_csv("./data/water_potability.csv")

X, y = df.drop(columns = "Potability"), df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, stratify = y, random_state = 42
)

pipe = Pipeline([
    ("impute", SimpleImputer(strategy = "median")),
    ("gnb", GaussianNB()),
])

grid = GridSearchCV(
    pipe,
    {"gnb__var_smoothing": np.logspace(0, -9, 10)},
    cv = 5, scoring = "f1", n_jobs = -1,
)

grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

report_text = classification_report(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict = True)

print(grid.best_params_)

print(report_text)

#Saving the output terminal

os.makedirs("results", exist_ok = True)
with open("results/gaussian_nb_redults.txt", "w") as f:
    f.write(f"Best params: {grid.best_params_}\n\n{report_text}")

#Saving the key numbers for the final comparison table

row = {
    "model": "Gaussian Naive Bayes",
    "accuracy": report["accuracy"],
    "precision_potable": report["1"]["precision"],
    "recall_potable": report["1"]["recall"],
    "f1_potable": report["1"]["f1-score"],
}

pd.DataFrame([row]).to_csv("results/gaussian_nb_row.csv", index=False)