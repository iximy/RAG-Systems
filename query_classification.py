

import torch
from transformers import pipeline

# Загружаем предобученную модель для zero-shot классификации
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Возможные классы запросов
labels = ["фактический поиск", "мнение", "рассуждение", "личный вопрос", "неопределённый"]

# Функция классификации запроса
def classify_query(query):
    result = classifier(query, candidate_labels=labels)
    return result["labels"][0], result["scores"][0]

# Примеры запросов
queries = [
    "Сколько стоит построить дом?",
    "Как ты думаешь, погрузчик лучше, чем экскаватор?",
    "Объясни принципы работы ДВС.",
    "Стоит ли мне переехать в другой город?",
    "Просто хочу поговорить."
]

# Классифицируем запросы
for query in queries:
    label, score = classify_query(query)
    print(f"Запрос: {query}\nКлассификация: {label} (уверенность: {score:.2f})\n")
