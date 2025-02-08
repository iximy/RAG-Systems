import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# Загружаем модели
retrieval_model = SentenceTransformer("all-MiniLM-L6-v2")  # Для поиска
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # Для переранжирования


# Исходные текстовые фрагменты
documents = [
    "LLM могут генерировать ответы на основе найденных данных.",
    "Поиск информации возможен через BM25 и векторный поиск.",
    "Глубокие нейросети помогают анализировать текст.",
    "RAG использует векторные базы данных для поиска."
]

# Преобразуем документы в эмбеддинги
doc_embeddings = retrieval_model.encode(documents)
dimension = doc_embeddings.shape[1]

# Создаём FAISS индекс
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Запрос пользователя
query = "Как работает поиск в RAG?"
query_embedding = retrieval_model.encode([query])

# Поиск 3 ближайших фрагментов
D, I = index.search(np.array(query_embedding), k=3)
retrieved_docs = [documents[idx] for idx in I[0]]

# Переранжировка найденных документов
scores = reranker.predict([(query, doc) for doc in retrieved_docs])
ranked_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), reverse=True)]

# Вывод результатов
print("🔍 Найденные документы (после переранжировки):")
for doc in ranked_docs:
    print(f"- {doc}")

