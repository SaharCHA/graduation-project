import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell('# Аналіз алгоритмів кластеризації відеоконтенту\nЦей ноутбук містить порівняння алгоритмів K-Means та DBSCAN, розрахунок метрик якості (Silhouette, Davies-Bouldin, Calinski-Harabasz) та графік для методу ліктя.'))

nb.cells.append(nbf.v4.new_code_cell('''import os
import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# Завантаження даних
PROCESSED_DIR = '../data/processed/'
X = joblib.load(os.path.join(PROCESSED_DIR, 'features.joblib'))
print(f'Shape of features: {X.shape}')'''))

nb.cells.append(nbf.v4.new_markdown_cell('## 1. Метод ліктя (Elbow Method) для K-Means'))

nb.cells.append(nbf.v4.new_code_cell('''inertias = []
silhouettes = []
K_range = range(2, 20)

# Для економії часу можна взяти випадкову підвибірку (якщо даних дуже багато)
X_sample = X
if X.shape[0] > 10000:
    from sklearn.utils import resample
    X_sample = resample(X, n_samples=10000, random_state=42)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_sample)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_sample, labels))

fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:blue'
ax1.set_xlabel('Кількість кластерів (k)')
ax1.set_ylabel('Інерція (WCSS)', color=color)
ax1.plot(K_range, inertias, marker='o', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Silhouette Score', color=color)
ax2.plot(K_range, silhouettes, marker='s', color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Метод Ліктя та Силуету для K-Means')
plt.show()'''))

nb.cells.append(nbf.v4.new_markdown_cell('## 2. Порівняння K-Means та DBSCAN'))

nb.cells.append(nbf.v4.new_code_cell('''# Оптимальне k (наприклад, 15 з нашого main.py)
k = 15
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_sample)

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_sample)

def evaluate_clustering(name, labels, X_data):
    unique_labels = set(labels)
    if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
        print(f"{name}: Алгоритм не зміг виділити більше 1 кластера.")
        return
    
    # Фільтруємо шум для чесного підрахунку метрик DBSCAN
    mask = labels != -1
    if hasattr(X_data, "tocsr"):
        X_filtered = X_data.tocsr()[mask]
    else:
        X_filtered = X_data[mask]
    labels_filtered = labels[mask]
    
    if len(set(labels_filtered)) > 1:
        sil = silhouette_score(X_filtered, labels_filtered)
        db = davies_bouldin_score(X_filtered.toarray(), labels_filtered)
        ch = calinski_harabasz_score(X_filtered.toarray(), labels_filtered)
        print(f"--- {name} ---")
        print(f"Кількість знайдених кластерів: {len(set(labels_filtered))}")
        print(f"Точок у шумі: {np.sum(labels == -1)}")
        print(f"Silhouette Score: {sil:.4f}")
        print(f"Davies-Bouldin: {db:.4f}")
        print(f"Calinski-Harabasz: {ch:.4f}\\n")

evaluate_clustering("K-Means", kmeans_labels, X_sample)
evaluate_clustering("DBSCAN", dbscan_labels, X_sample)'''))

nb.cells.append(nbf.v4.new_markdown_cell('## 3. Візуалізація кластерів (PCA)'))

nb.cells.append(nbf.v4.new_code_cell('''pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sample.toarray())

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='tab20', alpha=0.5, s=10)
plt.title(f"K-Means (k={k})")

plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='tab20', alpha=0.5, s=10)
plt.title("DBSCAN")
plt.show()'''))

nbf.write(nb, 'c:/универ/диплом + практика/диплом/project/base/notebooks/clustering_analysis.ipynb')
