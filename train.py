import numpy as np, pandas as pd, joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(42)
n = 5000

df = pd.DataFrame({
    "worker_type": np.random.choice([0, 1], n),          # 0 = farmer, 1 = gig worker
    "monthly_income": np.random.normal(15000, 5000, n).clip(4000, 50000),
    "income_volatility": np.random.uniform(0.05, 0.7, n), # higher = more irregular
    "upi_txns_per_month": np.random.poisson(30, n),
    "recharge_regularity": np.random.uniform(0.3, 1, n),
    "bill_payment_ratio": np.random.uniform(0.2, 1, n),
    "age": np.random.randint(21, 65, n),
    "dependents": np.random.randint(0, 6, n),
    "land_acres": np.random.uniform(0, 8, n),
    "active_days": np.random.randint(8, 30, n),
    "tenure_months": np.random.randint(1, 60, n),
})

# Farmers have no gig activity; gig workers have no land
df.loc[df.worker_type == 1, "land_acres"] = 0
df.loc[df.worker_type == 0, ["active_days", "tenure_months"]] = [20, 24]

# Hidden risk logic that produces the default label
risk = (
    -0.00008 * df.monthly_income
    + 3.0 * df.income_volatility
    - 2.0 * df.bill_payment_ratio
    - 1.5 * df.recharge_regularity
    - 0.02 * df.upi_txns_per_month
    + 0.25 * df.dependents
    - 0.15 * df.land_acres
    + 2.5
)
prob = 1 / (1 + np.exp(-risk))
df["default"] = (np.random.rand(n) < prob).astype(int)

X, y = df.drop(columns="default"), df["default"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1)

model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.08)
model.fit(X_tr, y_tr)
print("AUC:", round(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]), 3))

joblib.dump((model, list(X.columns)), "model.pkl")
df.to_csv("synthetic_data.csv", index=False)