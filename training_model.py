# =====================================
# MODEL TRAINING AND COMPARISON
# =====================================
import training_Set
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Load processed data
X_train = training_Set.X_train
X_test = training_Set.X_test
y_train = training_Set.y_train
y_test = training_Set.y_test
# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100,random_state=42),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(n_neighbors=5)}
results = {}
# Train models
for name, model in models.items():
    print("\n===============================")
    print("Training:", name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print("Accuracy:", accuracy)
# Compare models
print("\n===============================")
print("MODEL COMPARISON")
results_df = pd.DataFrame(
    list(results.items()),
    columns=["Model","Accuracy"])
print(results_df)