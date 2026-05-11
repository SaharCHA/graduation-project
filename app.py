import streamlit as st
import pandas as pd
import joblib
import os
import sys

# Налаштування сторінки
st.set_page_config(page_title="Система рекомендацій відео", layout="wide")

st.title("🎥 Інтелектуальна система рекомендацій відеоконтенту")
st.markdown("Ця система використовує алгоритми кластеризації (навчання без вчителя) для групування схожих відео та видачі персоналізованих рекомендацій на основі метаданих.")

# Шляхи до збережених даних та моделей
PROCESSED_DATA_DIR = "data/processed/"
RAW_DATA_PATH = "data/raw/USvideos.csv"

@st.cache_data
def load_data_and_models():
    """Завантажує датасет та навчену модель кластеризації (кешується для швидкодії)"""
    try:
        # Завантажуємо оригінальний датасет (фільтруємо так само, як при навчанні)
        df = pd.read_csv(RAW_DATA_PATH, on_bad_lines='skip')
        
        # Видаляємо дублікати, як при навчанні
        if 'video_id' in df.columns:
            df = df.sort_values('views').drop_duplicates('video_id', keep='last')
            
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42).reset_index(drop=True)
            
        # Завантажуємо навчену модель та матрицю ознак
        kmeans_model = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'kmeans_model.joblib'))
        features = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'features.joblib'))
        
        # Прив'язуємо кластери до датафрейму
        df['cluster'] = kmeans_model.labels_
        
        return df, kmeans_model, features
    except Exception as e:
        st.error(f"Помилка завантаження даних або моделей. Спочатку запустіть main.py! Деталі: {e}")
        return None, None, None

df, model, features = load_data_and_models()

if df is not None:
    st.sidebar.header("Налаштування")
    st.sidebar.info(f"Завантажено відео: {len(df)}")
    st.sidebar.info(f"Кількість виділених кластерів: {len(df['cluster'].unique())}")
    
    # Вибір відео користувачем
    st.subheader("Виберіть відео, яке ви дивитесь:")
    video_titles = df['title'].tolist()
    
    # Для зручності показуємо назву та канал
    video_options = df['title'] + " (Канaл: " + df['channel_title'] + ")"
    selected_option = st.selectbox("Пошук відео:", video_options)
    
    if selected_option:
        # Знаходимо вибране відео в датафреймі
        idx = video_options[video_options == selected_option].index[0]
        selected_video = df.loc[idx]
        user_cluster = selected_video['cluster']
        
        st.write(f"**Опис вибраного відео:** {str(selected_video['description'])[:200]}...")
        
        # Перевірка наявності video_id для посилання
        if 'video_id' in selected_video:
            video_url = f"https://www.youtube.com/watch?v={selected_video['video_id']}"
            st.markdown(f"[📺 Дивитись оригінальне відео на YouTube]({video_url})")
            
        st.info(f"Це відео система віднесла до **Кластера №{user_cluster}**")
        
        st.markdown("---")
        st.subheader("💡 Система рекомендує вам також переглянути:")
        
        # Рекомендуємо відео з того ж кластера (відкидаємо те, що зараз вибрано)
        recommendations = df[(df['cluster'] == user_cluster) & (df.index != idx)]
        
        # Сортуємо рекомендації за популярністю (переглядами)
        recommendations = recommendations.sort_values(by='views', ascending=False).head(5)
        
        if not recommendations.empty:
            for _, row in recommendations.iterrows():
                with st.container():
                    col1, col2 = st.columns([1, 10])
                    with col1:
                        # Замість картинки ставимо заглушку або емодзі
                        st.markdown("### ▶️")
                    with col2:
                        if 'video_id' in row:
                            url = f"https://www.youtube.com/watch?v={row['video_id']}"
                            st.markdown(f"#### [{row['title']}]({url})")
                        else:
                            st.markdown(f"#### {row['title']}")
                            
                        st.caption(f"Канал: {row['channel_title']} | Переглядів: {row['views']:,} | Лайків: {row['likes']:,}")
        else:
            st.warning("На жаль, у цьому кластері більше немає відео для рекомендації.")

        st.markdown("---")
        st.subheader("📊 Візуалізація кластерів (2D Проєкція)")
        with st.spinner("Генеруємо візуалізацію (PCA)..."):
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Знаходимо позиційний індекс вибраного відео
            pos_idx = df.index.get_loc(idx)
            
            # Для швидкодії беремо підвибірку точок
            sample_size = min(2000, features.shape[0])
            indices = np.random.choice(features.shape[0], sample_size, replace=False)
            
            # Обов'язково додаємо наше відео у вибірку
            if pos_idx not in indices:
                indices[0] = pos_idx
                
            if hasattr(features, "tocsr"):
                X_sample = features.tocsr()[indices].toarray()
            else:
                X_sample = features[indices].toarray()
                
            clusters_sample = df.iloc[indices]['cluster'].values
            
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_sample)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Всі інші кластери
            mask_other = clusters_sample != user_cluster
            ax.scatter(X_pca[mask_other, 0], X_pca[mask_other, 1], c='lightgray', alpha=0.5, label='Інші кластери', s=20)
            
            # Наш кластер
            mask_user = clusters_sample == user_cluster
            ax.scatter(X_pca[mask_user, 0], X_pca[mask_user, 1], c='tab:blue', alpha=0.7, label=f'Кластер {user_cluster}', s=30)
            
            # Вибране відео
            selected_idx_in_sample = np.where(indices == pos_idx)[0][0]
            ax.scatter(X_pca[selected_idx_in_sample, 0], X_pca[selected_idx_in_sample, 1], c='red', marker='*', s=200, label='Вибране відео', edgecolor='black')
            
            ax.set_title("2D Проєкція простору ознак (PCA)", fontsize=14)
            ax.legend()
            ax.axis('off')
            
            st.pyplot(fig)

