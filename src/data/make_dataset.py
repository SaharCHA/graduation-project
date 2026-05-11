import pandas as pd
import numpy as np
import random
import os

def generate_mock_data(num_records=200, output_path="data/raw/mock_videos.csv"):
    """
    Генерує синтетичний набір даних відеороликів для тестування рекомендаційної системи.
    Створює приховані 'кластери' (категорії), щоб алгоритми могли їх знайти.
    """
    
    np.random.seed(42)
    random.seed(42)
    
    categories = ['Programming', 'Cooking', 'Gaming', 'Travel Vlog']
    
    # Словники для генерації текстів
    vocab = {
        'Programming': {
            'titles': ['How to learn Python in 2024', 'React vs Angular', 'Machine Learning basics', 'Build an API with FastAPI', 'CSS Flexbox tutorial', 'Understanding Pointers in C', 'Docker for beginners', 'Git and GitHub crash course'],
            'desc': ['A comprehensive guide to coding', 'Learn software development step by step', 'Programming tutorial for beginners', 'Advanced software engineering concepts', 'Code along with me']
        },
        'Cooking': {
            'titles': ['Gordon Ramsay perfect steak', '10 minute easy breakfast', 'How to bake a chocolate cake', 'Vegan dinner recipes', 'Italian pasta from scratch', 'Healthy meal prep ideas', 'Making sushi at home'],
            'desc': ['Delicious and easy recipes', 'Cook a perfect meal for your family', 'Culinary masterclass', 'Quick food preparation tutorial', 'Tasty and healthy dishes']
        },
        'Gaming': {
            'titles': ['Elden Ring boss fight', 'Minecraft speedrun record', 'GTA 5 funny moments', 'CS:GO pro tips', 'The Witcher 3 lore explained', 'Valorant best crosshairs', 'Cyberpunk 2077 walkthrough'],
            'desc': ['Epic gaming moments and highlights', 'Walkthrough and tips for gamers', 'Let us play this awesome game', 'Top tier gameplay analysis', 'Funny glitches and fails']
        },
        'Travel Vlog': {
            'titles': ['Exploring the streets of Tokyo', 'Backpacking across Europe', 'Top 10 places to visit in Italy', 'My trip to Bali', 'Hiking in the Swiss Alps', 'Solo travel guide', 'Best street food in Mexico'],
            'desc': ['Travel vlog and adventure', 'Discovering new cultures and places', 'Beautiful cinematic travel video', 'Tips for budget traveling', 'Wandering around the world']
        }
    }
    
    data = []
    
    for i in range(num_records):
        # Вибираємо випадкову категорію
        category = random.choice(categories)
        
        # Генеруємо назву та опис на основі категорії
        title = f"{random.choice(vocab[category]['titles'])} - Part {random.randint(1, 5)}"
        description = f"{random.choice(vocab[category]['desc'])}. {random.choice(['Subscribe for more!', 'Leave a like.', 'Enjoy the video!'])}"
        
        # Генеруємо метрики (перегляди, лайки). 
        # Додаємо трохи логіки: ігрові відео можуть мати більше переглядів тощо, 
        # але загалом робимо логнормальний розподіл для реалістичності.
        views = int(np.random.lognormal(mean=10, sigma=1.5))
        
        # Лайки зазвичай корелюють з переглядами (наприклад, 1-10% від переглядів)
        like_ratio = random.uniform(0.01, 0.1)
        likes = int(views * like_ratio)
        
        # Тривалість відео в секундах (від 1 хв до 60 хв)
        duration_sec = random.randint(60, 3600)
        
        data.append({
            'video_id': f'vid_{i:04d}',
            'title': title,
            'description': description,
            'views': views,
            'likes': likes,
            'duration_sec': duration_sec,
            'true_category': category # Зберігаємо справжню категорію для перевірки (алгоритм її не бачитиме)
        })
        
    df = pd.DataFrame(data)
    
    # Перевіряємо, чи існує папка, якщо ні - створюємо
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Успішно згенеровано {num_records} записів та збережено у '{output_path}'")
    print(df.head())

if __name__ == "__main__":
    # Запускаємо з кореня проєкту (base/)
    generate_mock_data(output_path="c:/универ/диплом + практика/диплом/project/base/data/raw/mock_videos.csv")
