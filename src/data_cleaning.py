
import pandas as pd
import os

print("Starting Data Cleaning Process...")

# 1. Load original dataset
file_path = "data/Advertising.csv"
df = pd.read_csv(file_path)

print("\nOriginal Dataset Shape:", df.shape)

# 2. Remove unnecessary index columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("\nRemoved unnecessary columns.")

# 3. Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# 4. Remove duplicate records
df = df.drop_duplicates()

print("\nDuplicate records removed.")

# 5. Convert columns to numerical data types
columns = ["TV", "Radio", "Newspaper", "Sales"]

for col in columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 6. Handle missing values
df = df.dropna(subset=columns)

# 7. Validate advertising expenditure
# Advertising expenditure cannot be negative
for col in ["TV", "Radio", "Newspaper"]:
    df = df[df[col] >= 0]

# Sales should not be negative
df = df[df["Sales"] >= 0]

# 8. Check the cleaned dataset
print("\nCleaned Dataset Shape:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Records After Cleaning:")
print(df.duplicated().sum())

# 9. Display cleaned data
print("\nCleaned Dataset Preview:")
print(df.head())

# 10. Save cleaned dataset
output_folder = "data/processed"
os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(output_folder, "cleaned_sales.csv")

df.to_csv(output_file, index=False)

print("\nCleaned dataset saved successfully!")
print("Output Location:", output_file)

print("\nDATA CLEANING COMPLETED!")
