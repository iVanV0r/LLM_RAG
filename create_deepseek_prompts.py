# create_deepseek_prompts.py
import json
import pandas as pd
from pathlib import Path

# Загружаем результаты
df = pd.read_csv("data/output/detailed_results_all.csv")

prompts_path = "data/output/deepseek_prompts.txt"

with open(prompts_path, "w", encoding="utf-8") as f:
    f.write(f"ПРОМПТЫ ДЛЯ DEEPSEEK ОЦЕНКИ\n")
    f.write(f"{'=' * 60}\n\n")

    for i, row in df.iterrows():
        f.write(f"{'─' * 40}\n")
        f.write(f"Статья {i + 1} [{row.get('language', '?').upper()}]\n")
        f.write(f"Файл: {row['filename']}\n\n")
        f.write(f"Контекст: {row.get('reference', '')[:300]}\n\n")
        f.write(f"Вариант A (Base):\n{row['base_problem']}\n\n")
        f.write(f"Вариант B (RAG):\n{row['rag_problem']}\n\n")
        f.write("Оценка A: \nОценка B: \nЛучше: \n\n")

print(f"✓ Создано: {prompts_path}")