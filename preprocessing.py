"""
=============================================================
GROUP NO. 25 — ML Customer Segmentation | EASTC BDS Year III
Phase 2: Data Preprocessing + Feature Engineering
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_excel('/mnt/user-data/uploads/survey_dataset_1523-2.xlsx')

df.columns = [
    'Timestamp','Age','Gender','District','Occupation','Network',
    'Years_Used','Monthly_Spend','MoMo_Transactions','Call_Minutes_Week',
    'SMS_Week','Primary_Service','Recharge_Frequency','Changed_Network',
    'Reason_Changed','Satisfaction','Heard_AI','AI_for_Telecom','Agree_Personalized'
]

print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# 2. HANDLE MISSING VALUES
# ─────────────────────────────────────────────
# MoMo_Transactions: missing = respondent doesn't use MoMo → fill with "None/0"
df['MoMo_Transactions'] = df['MoMo_Transactions'].fillna('None')

# SMS_Week: missing → fill with "None"  
df['SMS_Week'] = df['SMS_Week'].fillna('None')

# Satisfaction: 3 missing → fill with median (mode is also acceptable)
df['Satisfaction'] = df['Satisfaction'].fillna(df['Satisfaction'].median())

# Years_Used: 1 missing → fill with mode
df['Years_Used'] = df['Years_Used'].fillna(df['Years_Used'].mode()[0])

# AI_for_Telecom & Agree_Personalized: 1-2 missing → fill with mode
df['AI_for_Telecom'] = df['AI_for_Telecom'].fillna(df['AI_for_Telecom'].mode()[0])
df['Agree_Personalized'] = df['Agree_Personalized'].fillna(df['Agree_Personalized'].mode()[0])

# Reason_Changed: 576 missing because they said "No" to changing network → fill "N/A"
df['Reason_Changed'] = df['Reason_Changed'].fillna('N/A')

print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 3. ORDINAL ENCODING (for ML clustering)
# ─────────────────────────────────────────────

# Age groups → ordinal numbers
age_order = {'Under 18': 0, '18–24': 1, '25–34': 2, '35–44': 3, '45–54': 4, '55+': 5}
df['Age_Num'] = df['Age'].map(age_order)

# Monthly spend → ordinal (proxy for ARPU)
spend_order = {
    'Less than TZS 5,000': 1,
    '5,000–10,000': 2,
    '10,001–20,000': 3,
    '20,001–50,000': 4,
    'More than 50,000': 5
}
df['Spend_Num'] = df['Monthly_Spend'].map(spend_order)

# MoMo Transactions → ordinal (Frequency of MoMo usage)
momo_order = {'None': 0, '1–10': 1, '11–30': 2, '31–60': 3, 'More than 60': 4}
df['MoMo_Num'] = df['MoMo_Transactions'].map(momo_order)

# Call Minutes per week → ordinal (Voice usage)
call_order = {
    'Less than 30': 1, '30–60': 2, '61–120': 3, '121–300': 4, 'More than 300': 5
}
df['Call_Num'] = df['Call_Minutes_Week'].map(call_order)

# SMS per week → ordinal
sms_order = {'None': 0, '1–10': 1, '11–30': 2, 'More than 30': 3}
df['SMS_Num'] = df['SMS_Week'].map(sms_order)

# Years used → ordinal (loyalty/tenure)
tenure_order = {
    'Less than 1 year': 1, '1–2 years': 2, '3–5 years': 3, 'More than 5 years': 4
}
df['Tenure_Num'] = df['Years_Used'].map(tenure_order)

# Recharge frequency → ordinal
recharge_order = {'Occasionally': 1, 'Monthly': 2, 'Weekly': 3, 'Daily': 4}
df['Recharge_Num'] = df['Recharge_Frequency'].map(recharge_order)

# ─────────────────────────────────────────────
# 4. ONE-HOT ENCODING (categorical variables)
# ─────────────────────────────────────────────
df_encoded = pd.get_dummies(df, columns=['Gender', 'District', 'Occupation',
                                          'Network', 'Primary_Service',
                                          'Changed_Network', 'Heard_AI',
                                          'AI_for_Telecom', 'Agree_Personalized'],
                             drop_first=False)

print(f"After encoding: {df_encoded.shape[1]} columns")

# ─────────────────────────────────────────────
# 5. RFM FEATURE ENGINEERING
# ─────────────────────────────────────────────
"""
RFM = Recency · Frequency · Monetary
In telecom survey context:
  R (Recency)   → Recharge_Num (how often they recharge)
  F (Frequency) → Call_Num + MoMo_Num + SMS_Num (activity frequency)
  M (Monetary)  → Spend_Num (monthly airtime/data spend)
"""

df['RFM_Recency']   = df['Recharge_Num']
df['RFM_Frequency'] = (df['Call_Num'] + df['MoMo_Num'] + df['SMS_Num']) / 3
df['RFM_Monetary']  = df['Spend_Num']

# Composite engagement score
df['Engagement_Score'] = (
    df['Spend_Num'] * 0.35 +
    df['Call_Num']  * 0.25 +
    df['MoMo_Num']  * 0.20 +
    df['Recharge_Num'] * 0.10 +
    df['SMS_Num']   * 0.10
)

# Loyalty indicator (tenure + satisfaction)
df['Loyalty_Score'] = (df['Tenure_Num'] * 0.6 + df['Satisfaction'] * 0.4)

print("\nNew features created:")
print(df[['RFM_Recency','RFM_Frequency','RFM_Monetary',
          'Engagement_Score','Loyalty_Score']].describe().round(2))

# ─────────────────────────────────────────────
# 6. FINAL ML FEATURE MATRIX
# ─────────────────────────────────────────────
ml_features = [
    'Age_Num', 'Spend_Num', 'MoMo_Num', 'Call_Num', 'SMS_Num',
    'Tenure_Num', 'Recharge_Num', 'Satisfaction',
    'RFM_Recency', 'RFM_Frequency', 'RFM_Monetary',
    'Engagement_Score', 'Loyalty_Score'
]

X = df[ml_features].copy()
print(f"\nML Feature Matrix shape: {X.shape}")
print("No missing values in feature matrix:", X.isnull().sum().sum() == 0)

# ─────────────────────────────────────────────
# 7. FEATURE SCALING (Min-Max Normalization)
# ─────────────────────────────────────────────
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=ml_features)

print("\nScaled feature matrix (first 3 rows):")
print(X_scaled.head(3).round(3))

# ─────────────────────────────────────────────
# 8. SAVE CLEANED DATA
# ─────────────────────────────────────────────
df.to_csv('/home/claude/cleaned_survey_data.csv', index=False)
X_scaled.to_csv('/home/claude/ml_features_scaled.csv', index=False)
print("\n✓ Saved: cleaned_survey_data.csv")
print("✓ Saved: ml_features_scaled.csv")

# ─────────────────────────────────────────────
# 9. DESCRIPTIVE STATS PLOT
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Telecom Survey — Descriptive Analysis\nDar es Salaam (n=1,523)',
             fontsize=14, fontweight='bold')

# Network distribution
ax = axes[0, 0]
net_counts = df['Network'].value_counts()
colors = ['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0','#00BCD4']
bars = ax.bar(net_counts.index, net_counts.values, color=colors[:len(net_counts)])
ax.set_title('Network Provider Distribution')
ax.set_ylabel('Number of Respondents')
ax.tick_params(axis='x', rotation=30)
for bar, val in zip(bars, net_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(val), ha='center', fontsize=9)

# Age distribution
ax = axes[0, 1]
age_order_list = ['Under 18','18–24','25–34','35–44','45–54','55+']
age_counts = df['Age'].value_counts().reindex(age_order_list)
ax.bar(age_counts.index, age_counts.values, color='#42A5F5')
ax.set_title('Age Group Distribution')
ax.set_ylabel('Count')
ax.tick_params(axis='x', rotation=30)

# Primary service used
ax = axes[0, 2]
service_counts = df['Primary_Service'].value_counts()
wedges, texts, autotexts = ax.pie(
    service_counts.values,
    labels=service_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette('Set2', len(service_counts))
)
ax.set_title('Primary Service Used')

# Monthly spend
ax = axes[1, 0]
spend_order_list = ['Less than TZS 5,000','5,000–10,000','10,001–20,000',
                     '20,001–50,000','More than 50,000']
spend_counts = df['Monthly_Spend'].value_counts().reindex(spend_order_list)
ax.bar(range(len(spend_counts)), spend_counts.values, color='#66BB6A')
ax.set_xticks(range(len(spend_counts)))
ax.set_xticklabels(['<5k','5-10k','10-20k','20-50k','>50k'], rotation=0)
ax.set_title('Monthly Spend (TZS)')
ax.set_ylabel('Count')

# Satisfaction rating
ax = axes[1, 1]
sat_counts = df['Satisfaction'].value_counts().sort_index()
ax.bar(sat_counts.index.astype(int), sat_counts.values,
       color=['#EF5350','#FF7043','#FFA726','#66BB6A','#26C6DA'])
ax.set_title('Overall Satisfaction (1–5)')
ax.set_xlabel('Rating')
ax.set_ylabel('Count')
ax.set_xticks([1,2,3,4,5])

# Engagement Score distribution
ax = axes[1, 2]
ax.hist(df['Engagement_Score'], bins=20, color='#AB47BC', edgecolor='white')
ax.set_title('Engagement Score Distribution')
ax.set_xlabel('Score')
ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('/home/claude/01_descriptive_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_descriptive_analysis.png")

print("\n✅ PHASE 2 COMPLETE — Ready for Clustering (Phase 3)")
