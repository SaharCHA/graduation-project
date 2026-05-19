import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.decomposition import PCA

st.set_page_config(page_title="Система рекомендацій відео", layout="wide")
st.title("Інтелектуальна система рекомендацій відеоконтенту")
st.markdown("Система використовує алгоритми кластеризації (навчання без вчителя) для групування схожих відео та генерації персоналізованих рекомендацій на основі метаданих.")

PROCESSED_DATA_DIR = "data/processed/"
RAW_DATA_PATH = "data/raw/USvideos.csv"


@st.cache_data
def load_data_and_models():
    try:
        df = pd.read_csv(RAW_DATA_PATH, on_bad_lines='skip')
        if 'video_id' in df.columns:
            df = df.sort_values('views').drop_duplicates('video_id', keep='last')
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42).reset_index(drop=True)
        kmeans_model = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'kmeans_model.joblib'))
        features = joblib.load(os.path.join(PROCESSED_DATA_DIR, 'features.joblib'))
        df['cluster'] = kmeans_model.labels_
        return df, kmeans_model, features
    except Exception as e:
        st.error(f"Помилка завантаження. Спочатку запустіть main.py! Деталі: {e}")
        return None, None, None


df, model, features = load_data_and_models()

if df is not None:
    st.sidebar.header("Інформація")
    st.sidebar.info(f"Завантажено відео: {len(df)}")
    st.sidebar.info(f"Кількість кластерів: {len(df['cluster'].unique())}")

    tab1, tab2 = st.tabs(["Рекомендації", "Порівняння алгоритмів"])

    # ──────────────────────────────────────────────
    # ВКЛАДКА 1 — РЕКОМЕНДАЦІЇ
    # ──────────────────────────────────────────────
    with tab1:
        st.subheader("Виберіть відео, яке ви дивитесь:")
        video_options = df['title'] + " (Канал: " + df['channel_title'] + ")"
        selected_option = st.selectbox("Пошук відео:", video_options)

        if selected_option:
            idx = video_options[video_options == selected_option].index[0]
            selected_video = df.loc[idx]
            user_cluster = selected_video['cluster']

            st.write(f"**Опис:** {str(selected_video['description'])[:200]}...")
            if 'video_id' in selected_video:
                video_url = f"https://www.youtube.com/watch?v={selected_video['video_id']}"
                st.markdown(f"[Дивитись на YouTube]({video_url})")
            st.info(f"Це відео система віднесла до **Кластера №{user_cluster}**")

            st.markdown("---")
            st.subheader("Система рекомендує переглянути:")
            recommendations = df[(df['cluster'] == user_cluster) & (df.index != idx)]
            recommendations = recommendations.sort_values(by='views', ascending=False).head(5)

            if not recommendations.empty:
                for _, row in recommendations.iterrows():
                    with st.container():
                        col1, col2 = st.columns([1, 10])
                        with col1:
                            st.markdown("### ▶️")
                        with col2:
                            if 'video_id' in row:
                                url = f"https://www.youtube.com/watch?v={row['video_id']}"
                                st.markdown(f"#### [{row['title']}]({url})")
                            else:
                                st.markdown(f"#### {row['title']}")
                            st.caption(
                                f"Канал: {row['channel_title']} | "
                                f"Переглядів: {row['views']:,} | Лайків: {row['likes']:,}"
                            )
            else:
                st.warning("У цьому кластері більше немає відео для рекомендації.")

            st.markdown("---")
            st.subheader("Візуалізація кластерів (2D PCA-проєкція)")
            with st.spinner("Генеруємо візуалізацію..."):
                pos_idx = df.index.get_loc(idx)
                sample_size = min(2000, features.shape[0])
                indices = np.random.choice(features.shape[0], sample_size, replace=False)
                if pos_idx not in indices:
                    indices[0] = pos_idx

                X_sample = features.tocsr()[indices].toarray() if hasattr(features, "tocsr") \
                    else features[indices].toarray()
                clusters_sample = df.iloc[indices]['cluster'].values

                pca2 = PCA(n_components=2)
                X_pca2 = pca2.fit_transform(X_sample)

                fig, ax = plt.subplots(figsize=(10, 6))
                mask_other = clusters_sample != user_cluster
                mask_user  = clusters_sample == user_cluster
                ax.scatter(X_pca2[mask_other, 0], X_pca2[mask_other, 1],
                           c='lightgray', alpha=0.5, label='Інші кластери', s=20)
                ax.scatter(X_pca2[mask_user, 0], X_pca2[mask_user, 1],
                           c='tab:blue', alpha=0.7, label=f'Кластер {user_cluster}', s=30)
                sel_pos = np.where(indices == pos_idx)[0][0]
                ax.scatter(X_pca2[sel_pos, 0], X_pca2[sel_pos, 1],
                           c='red', marker='*', s=200, label='Вибране відео', edgecolor='black')
                ax.set_title("2D Проєкція простору ознак (PCA)", fontsize=14)
                ax.legend()
                ax.axis('off')
                st.pyplot(fig)

    # ──────────────────────────────────────────────
    # ВКЛАДКА 2 — ПОРІВНЯННЯ АЛГОРИТМІВ
    # ──────────────────────────────────────────────
    with tab2:
        st.subheader("Порівняльний аналіз алгоритмів кластеризації")
        st.markdown(
            "Система навчила три алгоритми кластеризації та оцінила їх за трьома незалежними метриками. "
            "Результати зведено у таблицю нижче."
        )

        # ── Чому K-Means ──
        st.markdown("#### Чому для рекомендацій обрано K-Means?")
        st.markdown(
            """
            У системі реалізовано три алгоритми кластеризації, однак лише **K-Means** використовується
            для безпосередньої генерації рекомендацій. Вибір обумовлений такими причинами:
            """
        )

        col_why1, col_why2, col_why3 = st.columns(3)
        with col_why1:
            st.success(
                "**Підтримка predict()**\n\n"
                "K-Means зберігає навчені центроїди і може миттєво визначити кластер "
                "для будь-якого нового відео без повторного навчання."
            )
        with col_why2:
            st.success(
                "**Масштабованість**\n\n"
                "Навчається на повній матриці ознак 10 000×1003 без попереднього "
                "зниження розмірності. Лінійна складність O(n·k·I)."
            )
        with col_why3:
            st.success(
                "**Детерміновані кластери**\n\n"
                "Фіксований random_state=42 гарантує однакові результати при кожному "
                "запуску — критично важливо для відтворюваності рекомендацій."
            )

        st.markdown("#### Порівняльна таблиця характеристик алгоритмів")
        algo_compare = pd.DataFrame({
            'Характеристика': [
                'Потребує задати k апріорі',
                'Підтримує predict() для нових точок',
                'Працює на повній матриці (1003 вим.)',
                'Виявляє шумові точки',
                'Знаходить кластери довільної форми',
                'Використовується для рекомендацій',
                'Обчислювальна складність',
            ],
            'K-Means': ['Так', 'Так', 'Так', 'Ні', 'Ні', 'Так', 'O(n·k·I)'],
            'DBSCAN': ['Ні', 'Ні', 'Ні (потрібен PCA)', 'Так', 'Так', 'Ні', 'O(n²)'],
            'Hierarchical (Ward)': ['Так', 'Ні', 'Ні (потрібен PCA)', 'Ні', 'Частково', 'Ні', 'O(n² log n)'],
        })
        st.dataframe(algo_compare.set_index('Характеристика'), use_container_width=True)

        st.info(
            "DBSCAN та Hierarchical реалізовані для порівняльного аналізу метрик якості. "
            "Обидва потребують попередньої PCA-редукції до 50 компонент через прокляття розмірності. "
            "Hierarchical додатково навчається лише на підвибірці з 3 000 точок через квадратичну складність."
        )
        st.markdown("---")

        if st.button("Оновити дані (очистити кеш)"):
            st.cache_data.clear()
            st.rerun()

        # ── Таблиця метрик ──
        csv_path = os.path.join(PROCESSED_DATA_DIR, 'clustering_comparison.csv')
        if os.path.exists(csv_path):
            comparison_df = pd.read_csv(csv_path, index_col=0, encoding='utf-8-sig')
            st.markdown("#### Таблиця метрик якості")
            st.dataframe(comparison_df, use_container_width=True)

            st.markdown(
                """
                **Інтерпретація метрик:**
                - **Silhouette Score** — від −1 до 1, чим вище — тим краще відокремлені кластери
                - **Davies-Bouldin Index** — чим менше — тим компактніші та відокремленіші кластери
                - **Calinski-Harabasz Index** — чим більше — тим щільніші та краще розділені кластери
                """
            )

            # ── Стовпчасті діаграми метрик ──
            st.markdown("#### Візуальне порівняння метрик")
            numeric_cols = [c for c in comparison_df.columns if c != 'Кількість кластерів']
            num_metrics = len(numeric_cols)

            fig_bar, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))
            if num_metrics == 1:
                axes = [axes]

            colors = ['#4C72B0', '#DD8452', '#55A868']
            algorithms = comparison_df.index.tolist()

            for ax_b, metric in zip(axes, numeric_cols):
                values = pd.to_numeric(comparison_df[metric], errors='coerce')
                bars = ax_b.bar(algorithms, values, color=colors[:len(algorithms)], edgecolor='white', width=0.5)
                ax_b.set_title(metric, fontsize=11, fontweight='bold')
                ax_b.set_ylabel(metric)
                ax_b.tick_params(axis='x', rotation=15)
                for bar, val in zip(bars, values):
                    if not np.isnan(val):
                        ax_b.text(bar.get_x() + bar.get_width() / 2,
                                  bar.get_height() * 1.01,
                                  f"{val:.3f}", ha='center', va='bottom', fontsize=9)

            fig_bar.tight_layout()
            st.pyplot(fig_bar)
        else:
            st.warning("Файл clustering_comparison.csv не знайдено. Запустіть main.py для навчання моделей.")

        # ── Графік методу ліктя ──
        st.markdown("#### Метод ліктя (вибір оптимального k для K-Means)")
        elbow_path = os.path.join(PROCESSED_DATA_DIR, 'elbow_method.png')
        if os.path.exists(elbow_path):
            st.image(elbow_path, caption="Залежність інерції (SSE) від кількості кластерів k", use_container_width=True)
        else:
            st.warning("Графік методу ліктя не знайдено. Запустіть main.py.")

        # ── PCA 2D — всі кластери K-Means ──
        st.markdown("#### Розподіл відео у просторі ознак (PCA 2D, K-Means, усі кластери)")
        with st.spinner("Будуємо PCA-візуалізацію..."):
            sample_size2 = min(3000, features.shape[0])
            idx2 = np.random.choice(features.shape[0], sample_size2, replace=False)
            X2 = features.tocsr()[idx2].toarray() if hasattr(features, "tocsr") else features[idx2].toarray()
            labels2 = df.iloc[idx2]['cluster'].values

            pca_all = PCA(n_components=2)
            X_pca_all = pca_all.fit_transform(X2)

            fig_all, ax_all = plt.subplots(figsize=(10, 7))
            scatter = ax_all.scatter(
                X_pca_all[:, 0], X_pca_all[:, 1],
                c=labels2, cmap='tab20', alpha=0.6, s=15
            )
            plt.colorbar(scatter, ax=ax_all, label='Кластер')
            ax_all.set_title("Розподіл відео за кластерами K-Means (PCA 2D)", fontsize=13)
            ax_all.axis('off')
            st.pyplot(fig_all)
