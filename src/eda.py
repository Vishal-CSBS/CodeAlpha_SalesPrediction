
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Starting Exploratory Data Analysis...")

# 1. Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_sales.csv")

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# 2. Create output folder
output_folder = "outputs/charts"
os.makedirs(output_folder, exist_ok=True)

sns.set_theme(style="whitegrid")

# 3. Sales Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Sales"], bins=20, kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{output_folder}/sales_distribution.png")
plt.close()

# 4. TV Advertising vs Sales
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x="TV", y="Sales")
plt.title("TV Advertising vs Sales")
plt.tight_layout()
plt.savefig(f"{output_folder}/tv_vs_sales.png")
plt.close()

# 5. Radio Advertising vs Sales
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x="Radio", y="Sales")
plt.title("Radio Advertising vs Sales")
plt.tight_layout()
plt.savefig(f"{output_folder}/radio_vs_sales.png")
plt.close()

# 6. Newspaper Advertising vs Sales
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x="Newspaper", y="Sales")
plt.title("Newspaper Advertising vs Sales")
plt.tight_layout()
plt.savefig(f"{output_folder}/newspaper_vs_sales.png")
plt.close()

# 7. Correlation Heatmap
plt.figure(figsize=(8, 5))
correlation = df.corr(numeric_only=True)
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Advertising and Sales Correlation")
plt.tight_layout()
plt.savefig(f"{output_folder}/correlation_heatmap.png")
plt.close()

# 8. Pairplot
pair_plot = sns.pairplot(df)
pair_plot.savefig(f"{output_folder}/pairplot.png")
plt.close("all")

print("\nEDA charts saved successfully!")

print("\nGenerated Files:")
for file in os.listdir(output_folder):
    print(file)

print("\nEXPLORATORY DATA ANALYSIS COMPLETED!")
