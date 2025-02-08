from sentence_transformers import SentenceTransformer

# Загружаем модель
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Примеры фрагментов текста
texts = [
    "RAG использует внешнюю базу знаний для генерации ответов.",
    "Модели встраивания помогают искать релевантную информацию.",
    "Глубокие нейронные сети улучшают обработку естественного языка."
]

# Встраиваем текст в вектора
embeddings = model.encode(texts)

# Выводим размер вектора
print(f"Размер вектора: {len(embeddings[0])}")
print(f"Пример вектора: {embeddings[0][:5]}")  # Выведем первые 5 значений