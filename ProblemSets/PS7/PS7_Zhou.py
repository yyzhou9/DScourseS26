"""
Econ 5253 - Spring 2026
Problem Set 7
Yeyang Zhou
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import warnings
warnings.filterwarnings('ignore')

# Step 4: Load data
wages_path = '/Users/zhouyeyang/Univ. of Oklahoma Dropbox/Yeyang Zhou/Y1S2/DScourseS26/ProblemSets/PS7/wages.csv'
df = pd.read_csv(wages_path)
print("=== Original Data ===")
print(f"Total observations: {len(df)}")
print("\nMissing values per column:")
print(df.isnull().sum())

# Convert string columns to dummy
df['college'] = (df['college'] == 'college grad').astype(float)
df['married'] = (df['married'] == 'married').astype(float)
for col in ['logwage','hgc','tenure','age']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 5: Drop obs where hgc or tenure are missing
df = df.dropna(subset=['hgc', 'tenure'])
print(f"\nAfter dropping missing hgc/tenure: {len(df)} obs")

# Step 6: Summary statistics + missing rate
miss_n = df['logwage'].isna().sum()
miss_rate = df['logwage'].isna().mean()
print(f"\nMissing logwage: {miss_n} obs ({miss_rate:.4f}, i.e. {miss_rate*100:.1f}%)")

print("\n=== Summary Statistics (LaTeX) ===")
summary_stats = df.describe().T[['count','mean','std','min','max']]
print(summary_stats.round(4).to_latex(float_format="%.4f"))

# Step 7: Four imputation methods
def get_X(data):
    X = data[['hgc','college','tenure','age','married']].copy().astype(float)
    X['tenure2'] = X['tenure'] ** 2
    X = X[['hgc','college','tenure','tenure2','age','married']]
    return sm.add_constant(X)

# Method 1: Complete cases
df_cc = df.dropna(subset=['logwage']).copy()
model_cc = sm.OLS(df_cc['logwage'].astype(float), get_X(df_cc)).fit()

# Method 2: Mean imputation
df_mean = df.copy()
df_mean['logwage'] = df_mean['logwage'].fillna(df_mean['logwage'].mean())
model_mean = sm.OLS(df_mean['logwage'].astype(float), get_X(df_mean)).fit()

# Method 3: Predicted value imputation
df_pred = df.copy()
predicted = model_cc.predict(get_X(df_pred))
df_pred.loc[df_pred['logwage'].isna(), 'logwage'] = predicted[df_pred['logwage'].isna()]
model_pred = sm.OLS(df_pred['logwage'].astype(float), get_X(df_pred)).fit()

# Method 4: Multiple imputation (5 imputations, Rubin's rules)
np.random.seed(42)
n_imp = 5
mi_params, mi_bse = [], []
for i in range(n_imp):
    df_mi = df.copy()
    noise = np.random.normal(0, model_cc.resid.std(), len(df_mi))
    df_mi.loc[df_mi['logwage'].isna(), 'logwage'] = (
        predicted[df_mi['logwage'].isna()] +
        noise[df_mi['logwage'].isna().values]
    )
    m = sm.OLS(df_mi['logwage'].astype(float), get_X(df_mi)).fit()
    mi_params.append(m.params)
    mi_bse.append(m.bse)

pooled_params = pd.DataFrame(mi_params).mean()
W = pd.DataFrame(mi_bse).pow(2).mean()
B = pd.DataFrame(mi_params).var()
pooled_se = np.sqrt(W + (1 + 1/n_imp) * B)
model_mi_display = sm.OLS(df_mi['logwage'].astype(float), get_X(df_mi)).fit()

# Regression table
print("\n=== Regression Table (LaTeX) ===")
table = summary_col(
    [model_cc, model_mean, model_pred, model_mi_display],
    model_names=['Complete\nCases', 'Mean\nImputation', 'Predicted\nImputation', 'Multiple\nImputation'],
    stars=True,
    float_format='%0.4f',
    info_dict={
        'N': lambda x: f"{int(x.nobs)}",
        'R2': lambda x: f"{x.rsquared:.4f}"
    }
)
print(table.as_latex())

print("\n=== KEY RESULTS ===")
print(f"Missing logwage rate:     {miss_rate:.4f}  ({miss_rate*100:.1f}%)")
print(f"N after dropping hgc/tenure missing: {len(df)}")
print(f"N complete cases:         {int(model_cc.nobs)}")
print(f"\nTrue beta1:               0.0930")
print(f"Complete cases:           {model_cc.params['hgc']:.4f}  (se={model_cc.bse['hgc']:.4f})")
print(f"Mean imputation:          {model_mean.params['hgc']:.4f}  (se={model_mean.bse['hgc']:.4f})")
print(f"Predicted imputation:     {model_pred.params['hgc']:.4f}  (se={model_pred.bse['hgc']:.4f})")
print(f"Multiple imputation:      {pooled_params['hgc']:.4f}  (se={pooled_se['hgc']:.4f})")
