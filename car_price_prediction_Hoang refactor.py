# ============================================
# CAR PRICE PREDICTION - REFACTORED VERSION (WITH XGBOOST)
# ============================================

# 1. IMPORT LIBRARIES
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import StackingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# NEW: XGBOOST
from xgboost import XGBRegressor

# ============================================
# 2. LOAD DATA
# ============================================

df = pd.read_csv('data.csv')

# ============================================
# 3. FEATURE ENGINEERING
# ============================================

def feature_engineering(df):
    df = df.copy()
    
    if 'year' in df.columns:
        df['car_age'] = 2026 - df['year']
    
    if 'price' in df.columns:
        df['log_price'] = np.log1p(df['price'])
    
    return df


df = feature_engineering(df)

# ============================================
# 4. SPLIT DATA
# ============================================

target = 'price'

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
# 5. PREPROCESSING PIPELINE
# ============================================

num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# ============================================
# 6. DEFINE MODELS
# ============================================

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

# ============================================
# 7. TRAIN MODELS
# ============================================

trained_models = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    trained_models[name] = pipeline

# ============================================
# 8. ENSEMBLE MODEL (STACKING)
# ============================================

stack_model = StackingRegressor(
    estimators=[
        ('lr', LinearRegression()),
        ('ridge', Ridge()),
        ('lasso', Lasso()),
        ('xgb', XGBRegressor(n_estimators=200, random_state=42))
    ],
    final_estimator=LinearRegression()
)

stack_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', stack_model)
])

stack_pipeline.fit(X_train, y_train)
trained_models['Stacking'] = stack_pipeline

# ============================================
# 9. EVALUATION FUNCTION
# ============================================

def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

# ============================================
# 10. EVALUATE ALL MODELS
# ============================================

results = {}

for name, model in trained_models.items():
    pred = model.predict(X_test)
    results[name] = evaluate(y_test, pred)

results_df = pd.DataFrame(results).T
print(results_df)

# ============================================
# 11. CROSS VALIDATION
# ============================================

for name, model in trained_models.items():
    scores = cross_val_score(
        model, X, y,
        cv=5,
        scoring='neg_root_mean_squared_error'
    )
    print(f"{name} CV RMSE:", -scores.mean())

# ============================================
# 12. SAVE BEST MODEL
# ============================================

best_model_name = results_df['RMSE'].idxmin()
best_model = trained_models[best_model_name]

import joblib
joblib.dump(best_model, 'best_model.pkl')

print(f"Best model: {best_model_name}")
