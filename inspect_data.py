
import pandas as pd

# Load the advertising dataset
file_path = "data/Advertising.csv"

df = pd.read_csv(file_path)

print("\n========== FIRST 5 RECORDS ==========")
print(df.head())

print("\n========== DATASET SIZE ==========")
print("Rows and Columns:", df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())

print("\nDataset inspection completed successfully!")
