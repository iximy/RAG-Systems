from transformers import pipeline

# Загружаем модель для резюмирования (BART)
summarizer = pipeline("summarization", model="sberbank-ai/rugpt3small_sum")

# Пример текста
text = """
В Retrieval-Augmented Generation (RAG) используется две ключевые технологии: извлечение информации и генерация ответа. 
Система сначала находит релевантные фрагменты из базы данных, а затем генерирует на основе этих фрагментов связанный ответ. 
Процесс резюмирования помогает сделать ответы более лаконичными, снижая объем ненужной информации и представляя только основные факты.
"""

# Генерация резюме
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

# Вывод резюме
print("Резюме:", summary[0]['summary_text'])
