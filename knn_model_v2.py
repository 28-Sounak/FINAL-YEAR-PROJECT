import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

df = pd.read_csv("data/water_potability.csv")

X = df.drop(columns = "Potability")

y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

pipe = Pipeline([
    ("imputer", SimpleImputer()),

    ("scaler", StandardScaler()),

    ("knn", KNeighborsClassifier()),
])

param_grid = {
    "imputer": [SimpleImputer(strategy = "median"),
                SimpleImputer(strategy = "mean"),
                KNNImputer(n_neighbors = 5)],
    
    "knn__n_neighbors": list(range(3, 52, 2)),

    "knn__weights": ["uniform", "distance"],

    "knn__p": [1, 2],
}

cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

grid = GridSearchCV(pipe, param_grid, cv = cv, scoring = "accuracy", n_jobs = 1)

grid.fit(X_train, y_train)


print(grid.best_params_)

print(f"CV accuracy: {grid.best_score_:.3f}")

print(classification_report(y_test, grid.predict(X_test)))