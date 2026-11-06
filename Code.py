import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

students = 100

df = pd.DataFrame({
    "Study_Hours": np.random.randint(1, 10, students),
    "Attendance": np.random.randint(50, 101, students),
    "Sleep_Hours": np.random.randint(5, 10, students),
    "Assignments": np.random.randint(40, 101, students),
    "Previous_Score": np.random.randint(40, 91, students)
})

df["Final_Score"] = (
    df["Study_Hours"] * 3
    + df["Attendance"] * 0.2
    + df["Assignments"] * 0.2
    + df["Previous_Score"] * 0.3
    + np.random.normal(0, 5, students)
)

df["Final_Score"] = df["Final_Score"].clip(0, 100).round(2)

print("First 5 students:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nStatistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())


plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Final_Score"
)
plt.title("Study Hours vs Final Score")
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Attendance",
    y="Final_Score"
)
plt.title("Attendance vs Final Score")
plt.xlabel("Attendance")
plt.ylabel("Final Score")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(
    df["Final_Score"],
    bins=15,
    kde=True
)
plt.title("Final Score Distribution")
plt.xlabel("Final Score")
plt.show()


plt.figure(figsize=(8, 6))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Matrix")
plt.show()


def performance(score):
    if score >= 80:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Average"
    else:
        return "Poor"


df["Performance"] = df["Final_Score"].apply(performance)

print("\nPerformance:")
print(df["Performance"].value_counts())


plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="Performance"
)
plt.title("Student Performance")
plt.show()


print("\nAverage Final Score:")
print(
    df.groupby("Performance")["Final_Score"]
    .mean()
    .round(2)
)


print("\nTop 10 Students:")
print(
    df.sort_values(
        "Final_Score",
        ascending=False
    ).head(10)
)


df.to_csv(
    "student_performance.csv",
    index=False
)

print("\nAnalysis completed successfully!")
