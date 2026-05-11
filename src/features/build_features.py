import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib
import os

def preprocess_and_vectorize(input_path="data/raw/USvideos.csv", output_path="data/processed/"):
    """
    Зчитує сирі дані, обробляє пропущені значення, застосовує NLP (TF-IDF) 
    до тексту та стандартизацію до числових даних.
    """
    print(f"Завантаження даних з {input_path}...")
    # Реальний датасет може містити биті рядки, тому використовуємо on_bad_lines='skip'
    df = pd.read_csv(input_path, on_bad_lines='skip')
    
    # Видаляємо дублікати відео (бо в датасеті трендів одне відео може бути кілька днів підряд)
    if 'video_id' in df.columns:
        print(f"Видалення дублікатів. Було записів: {len(df)}")
        # Залишаємо лише останній запис про відео (з максимальною кількістю переглядів)
        df = df.sort_values('views').drop_duplicates('video_id', keep='last')
        print(f"Залишилось унікальних відео: {len(df)}")
        
    # Для пришвидшення демонстрації та навчання візьмемо випадкові 10,000 відео
    if len(df) > 10000:
        print("Випадкова вибірка 10,000 відео для швидкодії...")
        df = df.sample(n=10000, random_state=42).reset_index(drop=True)
    
    # 1. Обробка тексту
    print("Обробка тексту...")
    # Деякі відео можуть не мати опису або тегів (NaN)
    df['description'] = df['description'].fillna('')
    df['tags'] = df['tags'].fillna('')
    df['title'] = df['title'].fillna('')
    
    # Об'єднуємо назву, теги та опис для багатшого семантичного контексту
    df['text_content'] = df['title'] + " " + df['tags'] + " " + df['description']
    
    print("Векторизація тексту за допомогою TF-IDF...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    text_features = tfidf.fit_transform(df['text_content'])
    
    # 2. Обробка числових даних (перегляди, лайки, коментарі)
    print("Масштабування числових ознак...")
    scaler = StandardScaler()
    
    # Перевіряємо чи є пропущені значення в числах (заповнюємо нулями)
    for col in ['views', 'likes', 'comment_count']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0 # Fallback якщо колонки немає
    
    # Логарифмуємо для згладжування великих розкидів
    df['log_views'] = np.log1p(df['views'])
    df['log_likes'] = np.log1p(df['likes'])
    df['log_comments'] = np.log1p(df['comment_count'])
    
    numeric_features = scaler.fit_transform(df[['log_views', 'log_likes', 'log_comments']])
    
    # 3. Об'єднання текстових (розріджених) та числових ознак
    print("Об'єднання матриць...")
    final_features = hstack([text_features, numeric_features])
    
    # Зберігаємо оброблені дані та моделі
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(final_features, os.path.join(output_path, 'features.joblib'))
    joblib.dump(tfidf, os.path.join(output_path, 'tfidf_model.joblib'))
    joblib.dump(scaler, os.path.join(output_path, 'scaler_model.joblib'))
    
    print("Передобробку завершено. Вектори збережено.")
    
    return df, final_features

if __name__ == "__main__":
    preprocess_and_vectorize(input_path="c:/универ/диплом + практика/диплом/project/base/data/raw/USvideos.csv",
                             output_path="c:/универ/диплом + практика/диплом/project/base/data/processed/")
