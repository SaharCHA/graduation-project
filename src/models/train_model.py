import joblib
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def find_optimal_k(X, k_range=range(2, 21), output_path="data/processed/"):
    """Метод ліктя: обчислює інерцію K-Means для різних значень k та зберігає графік."""
    inertias = []
    print("Метод ліктя: обчислення інерції для k від 2 до 20...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        km.fit(X)
        inertias.append(km.inertia_)
        print(f"  k={k}: inertia={km.inertia_:.2f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=6)
    ax.set_xlabel('Кількість кластерів (k)', fontsize=12)
    ax.set_ylabel('Інерція (SSE)', fontsize=12)
    ax.set_title('Метод ліктя для вибору оптимального k', fontsize=14)
    ax.grid(True, alpha=0.3)
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(os.path.join(output_path, 'elbow_method.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"Графік методу ліктя збережено: {os.path.join(output_path, 'elbow_method.png')}")
    return inertias


def train_and_evaluate(features_path="data/processed/features.joblib", k_clusters=15):
    """
    Завантажує векторизовані дані, навчає K-Means, DBSCAN та AgglomerativeClustering,
    оцінює якість кожного алгоритму та будує порівняльну таблицю метрик.
    """
    print(f"Завантаження матриці ознак з {features_path}...")
    X = joblib.load(features_path)

    model_dir = os.path.dirname(features_path)
    if not model_dir:
        model_dir = "data/processed/"

    results = {}

    # ------------------------------------------------------------------ #
    # МЕТОД ЛІКТЯ — обґрунтування вибору k
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("МЕТОД ЛІКТЯ (Elbow Method)")
    print("=" * 60)
    find_optimal_k(X, k_range=range(2, 21), output_path=model_dir)

    # ------------------------------------------------------------------ #
    # K-MEANS
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"K-MEANS (n_clusters={k_clusters})")
    print("=" * 60)
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
    kmeans_labels = kmeans.fit_predict(X)

    X_dense = X.toarray() if hasattr(X, 'toarray') else np.array(X)

    sil_km = silhouette_score(X, kmeans_labels, sample_size=2000, random_state=42)
    db_km = davies_bouldin_score(X_dense, kmeans_labels)
    ch_km = calinski_harabasz_score(X_dense, kmeans_labels)

    print(f" - Silhouette Score:        {sil_km:.4f}  (ближче до 1 — краще)")
    print(f" - Davies-Bouldin Index:    {db_km:.4f}  (менше — краще)")
    print(f" - Calinski-Harabasz Index: {ch_km:.2f} (більше — краще)")

    results['K-Means'] = {
        'Silhouette Score': round(sil_km, 4),
        'Davies-Bouldin Index': round(db_km, 4),
        'Calinski-Harabasz Index': round(ch_km, 2),
        'Кількість кластерів': k_clusters,
    }

    # ------------------------------------------------------------------ #
    # PCA-зниження розмірності для DBSCAN та ієрархічної кластеризації
    # (TF-IDF матриця ~1003-вимірна; "прокляття розмірності" робить
    #  евклідові відстані неінформативними без попередньої редукції)
    # ------------------------------------------------------------------ #
    print("\nЗниження розмірності (PCA до 50 компонент) для DBSCAN та ієрархічної кластеризації...")
    pca_50 = PCA(n_components=50, random_state=42)
    X_pca = pca_50.fit_transform(X_dense)
    explained = pca_50.explained_variance_ratio_.sum()
    print(f" Пояснена дисперсія: {explained * 100:.1f}%")
    joblib.dump(pca_50, os.path.join(model_dir, 'pca_model.joblib'))

    # ------------------------------------------------------------------ #
    # DBSCAN (після PCA — параметри підібрані для 50-вимірного простору)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("DBSCAN (eps=3.0, min_samples=5, після PCA-50)")
    print("=" * 60)
    dbscan = DBSCAN(eps=3.0, min_samples=5, metric='euclidean', n_jobs=-1)
    dbscan_labels = dbscan.fit_predict(X_pca)

    unique_labels = set(dbscan_labels)
    n_clusters_db = len(unique_labels - {-1})
    n_noise = int((dbscan_labels == -1).sum())
    print(f" Знайдено кластерів: {n_clusters_db} | Шумових точок: {n_noise} ({n_noise/len(dbscan_labels)*100:.1f}%)")

    if n_clusters_db > 1:
        mask = dbscan_labels != -1
        X_db_valid = X_pca[mask]
        labels_db_valid = dbscan_labels[mask]
        sil_db = silhouette_score(X_db_valid, labels_db_valid,
                                  sample_size=min(2000, mask.sum()), random_state=42)
        db_db = davies_bouldin_score(X_db_valid, labels_db_valid)
        ch_db = calinski_harabasz_score(X_db_valid, labels_db_valid)
        print(f" - Silhouette Score:        {sil_db:.4f}")
        print(f" - Davies-Bouldin Index:    {db_db:.4f}")
        print(f" - Calinski-Harabasz Index: {ch_db:.2f}")
        results['DBSCAN'] = {
            'Silhouette Score': round(sil_db, 4),
            'Davies-Bouldin Index': round(db_db, 4),
            'Calinski-Harabasz Index': round(ch_db, 2),
            'Кількість кластерів': n_clusters_db,
        }
    else:
        print(" DBSCAN не виділив достатньо кластерів — метрики недоступні.")
        results['DBSCAN'] = {
            'Silhouette Score': None,
            'Davies-Bouldin Index': None,
            'Calinski-Harabasz Index': None,
            'Кількість кластерів': n_clusters_db,
        }

    # ------------------------------------------------------------------ #
    # ІЄРАРХІЧНА КЛАСТЕРИЗАЦІЯ (AgglomerativeClustering, Ward linkage)
    # Використовуємо підвибірку для керованого споживання пам'яті
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"ІЄРАРХІЧНА КЛАСТЕРИЗАЦІЯ — Ward (n_clusters={k_clusters})")
    print("=" * 60)

    sample_size_hier = min(3000, X_pca.shape[0])
    rng = np.random.RandomState(42)
    indices_hier = rng.choice(X_pca.shape[0], sample_size_hier, replace=False)
    X_hier = X_pca[indices_hier]

    print(f" Навчання на підвибірці {sample_size_hier} точок (з {X_pca.shape[0]})...")
    agglo = AgglomerativeClustering(n_clusters=k_clusters, linkage='ward')
    hier_labels = agglo.fit_predict(X_hier)

    sil_hier = silhouette_score(X_hier, hier_labels,
                                sample_size=min(2000, sample_size_hier), random_state=42)
    db_hier = davies_bouldin_score(X_hier, hier_labels)
    ch_hier = calinski_harabasz_score(X_hier, hier_labels)

    print(f" - Silhouette Score:        {sil_hier:.4f}")
    print(f" - Davies-Bouldin Index:    {db_hier:.4f}")
    print(f" - Calinski-Harabasz Index: {ch_hier:.2f}")

    results['Hierarchical (Ward)'] = {
        'Silhouette Score': round(sil_hier, 4),
        'Davies-Bouldin Index': round(db_hier, 4),
        'Calinski-Harabasz Index': round(ch_hier, 2),
        'Кількість кластерів': k_clusters,
    }

    # ------------------------------------------------------------------ #
    # ПОРІВНЯЛЬНА ТАБЛИЦЯ
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("ПОРІВНЯЛЬНА ТАБЛИЦЯ АЛГОРИТМІВ КЛАСТЕРИЗАЦІЇ")
    print("=" * 60)
    comparison_df = pd.DataFrame(results).T
    comparison_df.index.name = 'Алгоритм'
    print(comparison_df.to_string())

    csv_path = os.path.join(model_dir, 'clustering_comparison.csv')
    comparison_df.to_csv(csv_path, encoding='utf-8-sig')
    print(f"\nТаблицю збережено: {csv_path}")

    # ------------------------------------------------------------------ #
    # ЗБЕРЕЖЕННЯ МОДЕЛЕЙ
    # ------------------------------------------------------------------ #
    joblib.dump(kmeans, os.path.join(model_dir, 'kmeans_model.joblib'))
    joblib.dump(agglo, os.path.join(model_dir, 'agglo_model.joblib'))
    print("\nМоделі K-Means та AgglomerativeClustering збережено.")

    return kmeans_labels


if __name__ == "__main__":
    train_and_evaluate(
        features_path="c:/универ/диплом + практика/диплом/project/base/data/processed/features.joblib",
        k_clusters=15,
    )
