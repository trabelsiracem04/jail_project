from random import random
import pandas as pd
import numpy as np      
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("jail.csv", 
                 encoding='latin-1', 
                 low_memory=False)
# drop rows where COMMITMENT_TERM is missing or empty (whitespace-only)
if 'COMMITMENT_TERM' in df.columns:
    empty_mask = df['COMMITMENT_TERM'].isna() | df['COMMITMENT_TERM'].astype(str).str.strip().eq('')
    n_drop = int(empty_mask.sum())
    if n_drop > 0:
        print(f"Dropping {n_drop} rows with empty COMMITMENT_TERM")
        df = df.loc[~empty_mask].reset_index(drop=True)

# drop rows where COMMITMENT_UNIT contains 'Term' or is weight/volume units
if 'COMMITMENT_UNIT' in df.columns:
    unit_str = df['COMMITMENT_UNIT'].astype(str).str.strip()
    
    # mask for 'Term' (case-insensitive)
    term_mask = df['COMMITMENT_UNIT'].astype(str).str.contains('Term', case=False, na=False)
    
    # mask for weight/volume units
    weight_units_mask = unit_str.isin(['Pounds', 'Ounces', 'Kilos', 'Grams'])
    
    # combine both conditions
    drop_mask = term_mask | weight_units_mask
    n_drop = int(drop_mask.sum())
    
    if n_drop > 0:
        print(f"Dropping {n_drop} rows where COMMITMENT_UNIT contains 'Term' or is ['Pounds', 'Ounces', 'Kilos', 'Grams']")
        df = df.loc[~drop_mask].reset_index(drop=True)

print( df.shape, )        # number of rows and columns
# df.info(),        # data types, missing values
# df.describe() ,   # stats for numeric columns
# df.head(),        # quick preview
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
# print(num_cols)
df_num=df[num_cols]
# print( df_num.shape,         # number of rows and columns
# df_num.info(),        # data types, missing values
# df_num.describe() ,   # stats for numeric columns
# df_num.head(),        # quick preview
# )

# missing value report
miss = df.isnull().sum()
miss_perc = (miss / len(df)) * 100
report = pd.DataFrame({'missing_count': miss, 'missing_perc': miss_perc})
report = report.sort_values('missing_perc', ascending=False)
print('\n=== MISSING VALUES ===')
print(report.head(50))

# drop "CHARGE_DISPOSITION_REASON" column
df = df.drop(columns=['CHARGE_DISPOSITION_REASON'])
print("Dropped column: CHARGE_DISPOSITION_REASON")

# --- add single JAIL_DAYS column (vectorized) ---
# safe references to source columns
term_raw = df['COMMITMENT_TERM'] if 'COMMITMENT_TERM' in df.columns else pd.Series(np.nan, index=df.index)
unit_raw = df['COMMITMENT_UNIT'] if 'COMMITMENT_UNIT' in df.columns else pd.Series(np.nan, index=df.index)

# parse numeric portion of COMMITMENT_TERM: try direct numeric then extract digits
term_num = pd.to_numeric(term_raw, errors='coerce')
extracted = term_raw.astype(str).str.extract(r'([+-]?\d*\.?\d+)')[0]
term_num = term_num.fillna(pd.to_numeric(extracted, errors='coerce'))

# map units to day multipliers
unit = unit_raw.astype(str).str.lower().fillna('')
conds = [
    unit.str.contains(r'year|\byr\b|yrs|\by\b', na=False),
    unit.str.contains(r'month|\bmo\b|mos', na=False),
    unit.str.contains(r'week|\bwk\b|wks', na=False),
    unit.str.contains(r'day|\bd\b|days', na=False),
    unit.str.contains(r'hour|hr', na=False),
]
choices = [365.25, 30.44, 7, 1, 1.0 / 24.0]
multipliers = np.select(conds, choices, default=np.nan)

# compute JAIL_DAYS; leave NaN when parsing fails or unit unknown
df['JAIL_DAYS'] = term_num * multipliers

# --- special case: rows where COMMITMENT_UNIT indicates Natural Life ---
# For those rows, set JAIL_DAYS = numeric(COMMITMENT_TERM) * 250
# This block is intentionally self-contained.
nl_mask = unit_raw.astype(str).str.contains(r'natural\s+life', case=False, na=False)
if nl_mask.any():
    # use term_num (already attempted numeric coercion + regex extraction)
    nl_terms = term_num[nl_mask]
    # compute replacement values (will be NaN for non-numeric terms)
    nl_days = nl_terms * 250
    df.loc[nl_mask, 'JAIL_DAYS'] = nl_days
    # print(f"Adjusted {int(nl_mask.sum())} 'Natural Life' rows: set JAIL_DAYS = COMMITMENT_TERM * 250")
    # show a small sample of affected rows for verification
    sample_cols = [c for c in ('COMMITMENT_UNIT', 'COMMITMENT_TERM', 'JAIL_DAYS') if c in df.columns]
    # print(df.loc[nl_mask, sample_cols].head(5))


#plot distribution of sentence length (in days)
# for y in num_cols:
#     plt.figure(figsize=(6, 4))
#     sns.histplot(df[y].dropna(), kde=True)
#     plt.title(f'Distribution: {y}')
#     plt.tight_layout()
#     plt.show() 

corr_matrix = df.corr(numeric_only=True)

# print(corr_matrix)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features", fontsize=14)
plt.show()
print(df.head())

#convert JAIL_DAYS to integer (where not NaN):
df['JAIL_DAYS'] = df['JAIL_DAYS'].dropna().astype(int)

# save the current dataframe to CSV as requested:
out_file = 'jail_filtred.csv'
df.to_csv(out_file, index=False, encoding='utf-8')
# print(f"Saved dataframe to {out_file}")
print(df.shape)

# ===== RANDOM FOREST IMPLEMENTATION =====
print("\n=== APPLYING RANDOM FOREST REGRESSOR ===")

# Remove rows with NaN JAIL_DAYS
df_model = df.dropna(subset=['JAIL_DAYS']).copy()
print(f"Dataset shape after removing NaN JAIL_DAYS: {df_model.shape}")

# Separate target and features
y = df_model['JAIL_DAYS']
X = df_model.drop(columns=['JAIL_DAYS'])

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")

# Encode categorical variables
label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")
rf_model.fit(X_train, y_train)
print("Model training complete!")

# Make predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Evaluate on training set
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)

# Evaluate on test set
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)

print("\n=== RANDOM FOREST PERFORMANCE ===")
print(f"\nTraining Set:")
print(f"  MAE:  {train_mae:.4f}")
print(f"  RMSE: {train_rmse:.4f}")
print(f"  R²:   {train_r2:.4f}")

print(f"\nTest Set:")
print(f"  MAE:  {test_mae:.4f}")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  R²:   {test_r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== TOP 15 IMPORTANT FEATURES ===")
print(feature_importance.head(15))

# Save feature importance
feature_importance.to_csv('catboost_outputs/rf_feature_importance.csv', index=False)

# Save test predictions
predictions_df = pd.DataFrame({
    'actual': y_test.values,
    'predicted': y_pred_test,
    'error': y_test.values - y_pred_test
})
predictions_df.to_csv('catboost_outputs/rf_test_predictions.csv', index=False)

# Visualize feature importance
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'].values)
plt.yticks(range(len(top_features)), top_features['feature'].values)
plt.xlabel('Importance')
plt.title('Random Forest - Top 15 Feature Importance')
plt.tight_layout()
plt.savefig('catboost_outputs/rf_feature_importance.png', dpi=300)
plt.show()

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual JAIL_DAYS')
plt.ylabel('Predicted JAIL_DAYS')
plt.title('Random Forest: Predictions vs Actual')
plt.tight_layout()
plt.savefig('catboost_outputs/rf_predictions_vs_actual.png', dpi=300)
plt.show()

print("\n=== FILES SAVED ===")
print("✓ rf_feature_importance.csv")
print("✓ rf_test_predictions.csv")
print("✓ rf_feature_importance.png")
print("✓ rf_predictions_vs_actual.png")

