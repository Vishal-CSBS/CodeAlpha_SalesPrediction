
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("Starting Machine Learning Model Training...")

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_sales.csv")

# 2. Separate input features and target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# 4. Define machine learning models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=4
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
}

# 5. Train and evaluate each model
results = []
trained_models = {}

print("\nMODEL PERFORMANCE RESULTS")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model

    print(f"\n{name}")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

# 6. Compare model results
results_df = pd.DataFrame(results)

# Rank by R2 score, using MAE to break ties
results_df = results_df.sort_values(
    by=["R2 Score", "MAE"],
    ascending=[False, True]
).reset_index(drop=True)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# 7. Select best model
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nBEST MODEL:", best_model_name)

# 8. Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 9. Save best model
joblib.dump(best_model, "models/best_sales_model.pkl")

# 10. Save model comparison results
results_df.to_csv(
    "outputs/model_results.csv",
    index=False
)

print("\nBest model saved: models/best_sales_model.pkl")
print("Model results saved: outputs/model_results.csv")

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY!")
