import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Загружаем модель встраивания
model = SentenceTransformer("all-MiniLM-L6-v2")

# Исходные текстовые фрагменты
texts = [
    "Как работает RAG?",
    "Векторные базы данных хранят embeddings.",
    "Машинное обучение улучшает поиск информации.",
    "Глубокие нейросети используются в NLP."
]

# Преобразуем текст в векторы
embeddings = model.encode(texts)
dimension = embeddings.shape[1]  # Определяем размерность векторов

# Создаём индекс FAISS
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))  # Добавляем векторы в базу

# Запрос (тоже преобразуем в вектор)
query = "Как работают векторные базы?"
query_embedding = model.encode([query])

# Ищем 2 ближайших вектора
D, I = index.search(np.array(query_embedding), k=2)

# Выводим результаты
print("🔍 Найденные фрагменты:")
for idx in I[0]:
    print(f"- {texts[idx]}")
