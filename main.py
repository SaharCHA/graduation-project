import os
import sys
import pandas as pd
from src.features.build_features import preprocess_and_vectorize
from src.models.train_model import train_and_evaluate

def main():
    # Fix for printing unicode characters in Windows console
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("="*50)
    print("Ініціалізація системи рекомендацій відеоконтенту...")
    print("="*50)
    
    raw_data_path = "data/raw/USvideos.csv"
    if not os.path.exists(raw_data_path):
        print(f"Помилка: Файл {raw_data_path} не знайдено!")
        return
        
    # 2. Попередня обробка та векторизація
    print("\n[КРОК 1] Попередня обробка реального датасету (TF-IDF)...")
    processed_dir = "data/processed/"
    df, features = preprocess_and_vectorize(input_path=raw_data_path, output_path=processed_dir)
    
    # 3. Навчання моделей кластеризації (K-Means)
    print("\n[КРОК 2] Кластеризація відеоконтенту...")
    features_path = os.path.join(processed_dir, 'features.joblib')
    # Для реальних даних збільшимо кількість кластерів
    labels = train_and_evaluate(features_path=features_path, k_clusters=15)
    
    # 4. Демонстрація результатів
    print("\n[КРОК 3] Демонстрація результатів кластеризації...")
    df['cluster'] = labels
    
    print("\nПриклад відео з Кластера 0:")
    print(df[df['cluster'] == 0][['title', 'channel_title']].head(5))
    
    print("\nПриклад відео з Кластера 1:")
    print(df[df['cluster'] == 1][['title', 'channel_title']].head(5))
    
    print("\nУспішно завершено!")

if __name__ == "__main__":
    main()
