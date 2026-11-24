import pandas as pd

from sklearn.model_selection import  train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

df = pd.read_csv("./data/water_potability.csv")

X, y = df.drop(columns = "Potability"), df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

pipe = Pipeline([
    ("inpute", SimpleImputer(strategy = "median")),
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier()),
])

grid = GridSearchCV(
    pipe,
    {"knn__n_neighbors" : range(3, 26, 2), "knn__weights" : ["uniform", "distance"]},
    cv = 5, scoring = "f1", n_jobs = 1,
)

grid.fit(X_train, y_train)

print(grid.best_params_)

print(classification_report(y_test, grid.predict(X_test)))
