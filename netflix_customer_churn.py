import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import shap

# Load dataset
df = pd.read_csv(r"C:\Users\neha\Downloads\netflix_customer_churn_data.csv")
print(df.head())

# Shape and info
print(df.shape)
print(df.info())
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Churn distribution
sns.countplot(x="churned", data=df)
plt.title("Churn Distribution")
plt.savefig("churn_distribution.png")
plt.close()

# Drop duplicates and handle missing values
df = df.drop_duplicates()
df = df.dropna()  # Or impute if needed

# Define features and target
X = df.drop(["customer_id", "churned"], axis=1)
y = df["churned"]

# Identify categorical and numerical columns
categorical_cols = ["gender", "subscription_type", "region", "device", "payment_method", "favorite_genre"]
numerical_cols = ["age", "watch_hours", "last_login_days", "monthly_fee", "number_of_profiles", "avg_watch_time_per_day"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply one-hot encoding to categorical columns
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Align X_train and X_test to ensure they have the same columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Scale numerical columns
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Verify shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_log))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

# ROC-AUC
print("Logistic ROC-AUC:", roc_auc_score(y_test, y_pred_log))
print("Random Forest ROC-AUC:", roc_auc_score(y_test, y_pred_rf))

# SHAP analysis
explainer = shap.Explainer(rf_model, X_train)   # fit explainer on train
shap_values = explainer(X_test)                 # explain test set

# Summary plot
shap.summary_plot(shap_values, X_test)

# Get churn probabilities for the entire dataset
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X, _ = X.align(X_train, join='left', axis=1, fill_value=0)  # Align with X_train columns
X[numerical_cols] = scaler.transform(X[numerical_cols])  # Use the same scaler
df["churn_prob"] = rf_model.predict_proba(X)[:, 1]

# Segmentation
df["segment"] = pd.cut(
    df["churn_prob"],
    bins=[0, 0.3, 0.7, 1],
    labels=["Loyal", "At Risk", "Dormant"]
)

# Check segment distribution
print(df["segment"].value_counts())

# Save the updated DataFrame (optional)
df.to_csv("netflix_customer_churn_segmented.csv", index=False)

sns.countplot(x="segment", data=df, order=["Loyal","At Risk","Dormant"])
plt.title("Customer Segments")
plt.show()
