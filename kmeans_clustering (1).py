"""
=============================================================
GROUP NO. 24 — ML Customer Segmentation | EASTC BDS Year III
SCRIPT 1 OF 3: K-MEANS CLUSTERING
=============================================================
Input : cleaned_survey_data.csv  (1,523 respondents, 13 ML features)
Output: KMeans_clustered_data.csv, KMeans_cluster_profiles.csv,
        KMeans_Elbow_PCA.png
=============================================================
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# ── 1. Load data & build the 13-feature ML matrix ──
df = pd.read_csv("cleaned_survey_data.csv")

FEATURES = ['Age_Num', 'Spend_Num', 'MoMo_Num', 'Call_Num', 'SMS_Num',
            'Tenure_Num', 'Recharge_Num', 'Satisfaction', 'RFM_Recency',
            'RFM_Frequency', 'RFM_Monetary', 'Engagement_Score', 'Loyalty_Score']

X = MinMaxScaler().fit_transform(df[FEATURES])
print(f"Feature matrix: {X.shape[0]} respondents x {X.shape[1]} features")

# ── 2. Elbow Method + Silhouette to select k (2-10) ──
inertia, sil_scores = [], []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300).fit(X)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X, km.labels_))
    print(f"  k={k:2d} | Inertia={km.inertia_:8.2f} | Silhouette={sil_scores[-1]:.4f}")

# ── 3. Fit final model (k=4) ──
BEST_K = 4
kmeans = KMeans(n_clusters=BEST_K, random_state=42, n_init=20, max_iter=500)
df['KMeans_Cluster'] = kmeans.fit_predict(X)

sil = silhouette_score(X, df['KMeans_Cluster'])
dbi = davies_bouldin_score(X, df['KMeans_Cluster'])
ch = calinski_harabasz_score(X, df['KMeans_Cluster'])
print(f"\nK-Means (k=4) -> Silhouette={sil:.4f} | Davies-Bouldin={dbi:.4f} | Calinski-Harabasz={ch:.1f}")

# ── 4. Segment labels & profiling ──
SEGMENT_NAMES = {
    0: 'High-Spend Infrequent Rechargers',
    1: 'Low-Spend Daily Rechargers',
    2: 'Budget Passive Users',
    3: 'High-Value Heavy Users'
}
df['Segment_Label'] = df['KMeans_Cluster'].map(SEGMENT_NAMES)

profile = df.groupby('KMeans_Cluster')[FEATURES].mean().round(3)
profile.insert(0, 'N', df['KMeans_Cluster'].value_counts().sort_index())
profile.insert(1, 'Pct', (profile['N'] / len(df) * 100).round(1))
print("\nCluster profiles:\n", profile)

# ── 5. Plot: Elbow curve + PCA scatter ──
X_pca = PCA(n_components=2, random_state=42).fit_transform(X)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(range(2, 11), sil_scores, 'go-')
axes[0].axvline(BEST_K, color='red', linestyle='--', label=f'k={BEST_K}')
axes[0].set(title='Silhouette Score vs k', xlabel='k', ylabel='Silhouette Score')
axes[0].legend()

for c in range(BEST_K):
    mask = df['KMeans_Cluster'] == c
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], s=12, alpha=0.6, label=f'C{c}')
axes[1].set(title='K-Means Clusters (PCA)', xlabel='PC1', ylabel='PC2')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("KMeans_Elbow_PCA.png", dpi=150)
print("  saved KMeans_Elbow_PCA.png")

# ── 6. Save results ──
df.to_csv("KMeans_clustered_data.csv", index=False)
profile.to_csv("KMeans_cluster_profiles.csv")
print("\nDone. Saved KMeans_clustered_data.csv and KMeans_cluster_profiles.csv")
