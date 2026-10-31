
import joblib
import pandas as pd

print("====================================")
print("       SALES PREDICTION SYSTEM")
print("====================================")

# 1. Load the trained model
model = joblib.load("models/best_sales_model.pkl")

# 2. Get advertising budget inputs
try:
    tv = float(input("\nEnter TV advertising budget: "))
    radio = float(input("Enter Radio advertising budget: "))
    newspaper = float(input("Enter Newspaper advertising budget: "))

    # 3. Validate inputs
    if tv < 0 or radio < 0 or newspaper < 0:
        print("\nError: Advertising budgets cannot be negative.")

    else:
        # 4. Prepare input data
        input_data = pd.DataFrame({
            "TV": [tv],
            "Radio": [radio],
            "Newspaper": [newspaper]
        })

        # 5. Predict sales
        prediction = model.predict(input_data)

        # 6. Display result
        print("\n====================================")
        print("         PREDICTION RESULT")
        print("====================================")

        print(f"TV Budget: {tv:.2f}")
        print(f"Radio Budget: {radio:.2f}")
        print(f"Newspaper Budget: {newspaper:.2f}")

        print("------------------------------------")
        print(f"Predicted Sales: {prediction[0]:.2f}")
        print("====================================")

except ValueError:
    print("\nError: Please enter valid numerical values.")

print("\nPrediction process completed.")
