import joblib
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd
import os

def train_and_evaluate(features_path="data/processed/features.joblib", k_clusters=4):
    """
    Завантажує векторизовані дані, навчає моделі K-Means та DBSCAN,
    і оцінює їхню якість за допомогою метрик зі звіту.
    """
    print(f"Завантаження матриці ознак з {features_path}...")
    X = joblib.load(features_path)
    
    results = {}
    
    # --- K-MEANS ---
    print(f"\nНавчання моделі K-Means (n_clusters={k_clusters})...")
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
    kmeans_labels = kmeans.fit_predict(X)
    
    # Оцінка K-Means
    print("Оцінка K-Means:")
    sil_score_km = silhouette_score(X, kmeans_labels)
    db_score_km = davies_bouldin_score(X.toarray(), kmeans_labels)
    ch_score_km = calinski_harabasz_score(X.toarray(), kmeans_labels)
    
    print(f" - Silhouette Score: {sil_score_km:.4f} (ближче до 1 - краще)")
    print(f" - Davies-Bouldin Index: {db_score_km:.4f} (менше - краще)")
    print(f" - Calinski-Harabasz Index: {ch_score_km:.2f} (більше - краще)")
    
    # --- DBSCAN ---
    # Оскільки DBSCAN дуже чутливий до параметрів eps та min_samples, 
    # ми задамо базові значення, але вони можуть потребувати тюнінгу.
    print("\nНавчання моделі DBSCAN (eps=1.5, min_samples=3)...")
    dbscan = DBSCAN(eps=1.5, min_samples=3)
    dbscan_labels = dbscan.fit_predict(X)
    
    # Оцінюємо DBSCAN тільки якщо він знайшов більше 1 кластера (і не все є шумом -1)
    unique_labels = set(dbscan_labels)
    if len(unique_labels) > 1 and not (len(unique_labels) == 2 and -1 in unique_labels):
        print("Оцінка DBSCAN:")
        # Фільтруємо шум (-1) для коректнішого Silhouette score, або рахуємо з ним
        sil_score_db = silhouette_score(X, dbscan_labels)
        db_score_db = davies_bouldin_score(X.toarray(), dbscan_labels)
        print(f" - Silhouette Score: {sil_score_db:.4f}")
        print(f" - Davies-Bouldin Index: {db_score_db:.4f}")
    else:
        print("DBSCAN відніс всі точки до одного кластера або до шуму (потрібен тюнінг eps/min_samples).")
        dbscan_labels = None
        
    # Зберігаємо найкращу модель (K-Means для прикладу)
    model_dir = "data/processed/"
    joblib.dump(kmeans, os.path.join(model_dir, 'kmeans_model.joblib'))
    print("\nМодель K-Means збережено успішно.")
    
    return kmeans_labels

if __name__ == "__main__":
    train_and_evaluate(features_path="c:/универ/диплом + практика/диплом/project/base/data/processed/features.joblib")
