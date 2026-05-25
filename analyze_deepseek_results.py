#!/usr/bin/env python3
"""
Анализ результатов оценки DeepSeek
Читает deepseek_scores.json и выводит статистику
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel

# Пути
OUTPUT_DIR = Path("data/output")
SCORES_FILE = OUTPUT_DIR / "deepseek_scores.json"
ANALYSIS_FILE = OUTPUT_DIR / "deepseek_analysis.json"


def load_scores(filepath: str) -> list:
    """Загрузка оценок DeepSeek"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✓ Загружено {len(data)} оценок из {filepath}")
        return data
    except FileNotFoundError:
        print(f"❌ Файл не найден: {filepath}")
        print(f"   Создайте файл с оценками или запустите deepseek_batch_evaluate.py")
        return []
    except json.JSONDecodeError:
        print(f"❌ Ошибка формата JSON в {filepath}")
        return []


def analyze_scores(scores: list, language: str = "all") -> dict:
    """Анализ оценок DeepSeek"""

    # Фильтруем по языку
    if language != "all":
        filtered = [s for s in scores if s.get('source', '').lower() == language]
    else:
        filtered = scores

    if not filtered:
        return {}

    # Собираем оценки
    criteria = ['clarity', 'relevance', 'completeness']
    base_scores = {c: [] for c in criteria}
    rag_scores = {c: [] for c in criteria}
    winners = {'rag': 0, 'base': 0, 'tie': 0}

    for item in filtered:
        evaluation = item.get('evaluation', {})

        # Base оценки
        base = evaluation.get('base', {})
        for c in criteria:
            if c in base and isinstance(base[c], (int, float)):
                base_scores[c].append(float(base[c]))

        # RAG оценки
        rag = evaluation.get('rag', {})
        for c in criteria:
            if c in rag and isinstance(rag[c], (int, float)):
                rag_scores[c].append(float(rag[c]))

        # Победитель
        winner = evaluation.get('winner', 'tie')
        if winner in winners:
            winners[winner] += 1

    # Статистика по критериям
    results = {}

    for c in criteria:
        if base_scores[c] and rag_scores[c]:
            b_mean = np.mean(base_scores[c])
            r_mean = np.mean(rag_scores[c])

            # t-test
            if len(base_scores[c]) >= 2:
                t_stat, p_value = ttest_rel(rag_scores[c], base_scores[c])
            else:
                t_stat, p_value = 0, 1

            results[c] = {
                'base_mean': round(float(b_mean), 2),
                'base_std': round(float(np.std(base_scores[c])), 2),
                'rag_mean': round(float(r_mean), 2),
                'rag_std': round(float(np.std(rag_scores[c])), 2),
                'improvement': round(float(r_mean - b_mean), 2),
                'improvement_pct': round(float((r_mean - b_mean) / b_mean * 100), 1) if b_mean > 0 else 0,
                'p_value': round(float(p_value), 4),
                'significant': bool(p_value < 0.05),
                'count': len(base_scores[c])
            }

    # Overall
    if results:
        overall_base = np.mean([results[c]['base_mean'] for c in criteria if c in results])
        overall_rag = np.mean([results[c]['rag_mean'] for c in criteria if c in results])

        results['overall'] = {
            'base_mean': round(float(overall_base), 2),
            'rag_mean': round(float(overall_rag), 2),
            'improvement': round(float(overall_rag - overall_base), 2),
            'improvement_pct': round(float((overall_rag - overall_base) / overall_base * 100),
                                     1) if overall_base > 0 else 0,
        }

    # Победители
    total = sum(winners.values())
    results['winners'] = winners
    results['rag_win_rate'] = round(winners['rag'] / total * 100, 1) if total > 0 else 0
    results['total_evaluated'] = len(filtered)

    return results


def print_report(ru_results: dict, en_results: dict, all_results: dict):
    """Вывод отчета"""
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ DEEPSEEK")
    print("=" * 60)

    # RU результаты
    if ru_results:
        print(f"\n📊 RU ({ru_results.get('total_evaluated', 0)} статей):")
        print("-" * 50)

        for criterion in ['clarity', 'relevance', 'completeness', 'overall']:
            if criterion in ru_results:
                r = ru_results[criterion]
                if 'p_value' in r:
                    sig = "✓ ЗНАЧИМО" if r['significant'] else "✗ не значимо"
                    print(f"  {criterion.capitalize():12s}: "
                          f"Base={r['base_mean']:.2f}  RAG={r['rag_mean']:.2f}  "
                          f"Δ={r['improvement_pct']:+.1f}%  p={r['p_value']:.4f} {sig}")
                else:
                    print(f"  {criterion.capitalize():12s}: "
                          f"Base={r['base_mean']:.2f}  RAG={r['rag_mean']:.2f}  "
                          f"Δ={r['improvement_pct']:+.1f}%")

        if 'winners' in ru_results:
            w = ru_results['winners']
            total = sum(w.values())
            print(f"  Победители:    RAG={w['rag']}/{total} ({ru_results['rag_win_rate']}%)  "
                  f"Base={w['base']}  Ничья={w['tie']}")

    # EN результаты
    if en_results:
        print(f"\n📊 EN ({en_results.get('total_evaluated', 0)} статей):")
        print("-" * 50)

        for criterion in ['clarity', 'relevance', 'completeness', 'overall']:
            if criterion in en_results:
                r = en_results[criterion]
                if 'p_value' in r:
                    sig = "✓ ЗНАЧИМО" if r['significant'] else "✗ не значимо"
                    print(f"  {criterion.capitalize():12s}: "
                          f"Base={r['base_mean']:.2f}  RAG={r['rag_mean']:.2f}  "
                          f"Δ={r['improvement_pct']:+.1f}%  p={r['p_value']:.4f} {sig}")
                else:
                    print(f"  {criterion.capitalize():12s}: "
                          f"Base={r['base_mean']:.2f}  RAG={r['rag_mean']:.2f}  "
                          f"Δ={r['improvement_pct']:+.1f}%")

        if 'winners' in en_results:
            w = en_results['winners']
            total = sum(w.values())
            print(f"  Победители:    RAG={w['rag']}/{total} ({en_results['rag_win_rate']}%)  "
                  f"Base={w['base']}  Ничья={w['tie']}")

    # Общий итог
    if all_results:
        print(f"\n📊 ОБЩИЙ ИТОГ ({all_results.get('total_evaluated', 0)} статей):")
        print("-" * 50)

        for criterion in ['clarity', 'relevance', 'completeness', 'overall']:
            if criterion in all_results:
                r = all_results[criterion]
                if 'p_value' in r:
                    sig = "✓ ЗНАЧИМО" if r['significant'] else "✗ не значимо"
                    print(f"  {criterion.capitalize():12s}: "
                          f"Base={r['base_mean']:.2f}  RAG={r['rag_mean']:.2f}  "
                          f"Δ={r['improvement_pct']:+.1f}%  p={r['p_value']:.4f} {sig}")
                else:
                    print(f"  {criterion.capitalize():12s}: "
                          f"Base={r['base_mean']:.2f}  RAG={r['rag_mean']:.2f}  "
                          f"Δ={r['improvement_pct']:+.1f}%")

        if 'winners' in all_results:
            w = all_results['winners']
            total = sum(w.values())
            print(f"  Победители:    RAG={w['rag']}/{total} ({all_results['rag_win_rate']}%)  "
                  f"Base={w['base']}  Ничья={w['tie']}")

    print("\n" + "=" * 60)


def main():
    """Главная функция"""
    print("=" * 60)
    print("  АНАЛИЗ РЕЗУЛЬТАТОВ DEEPSEEK")
    print("=" * 60)

    # Загружаем оценки
    scores = load_scores(str(SCORES_FILE))

    if not scores:
        print("\n❌ Нет данных для анализа!")
        print(f"   Поместите оценки в {SCORES_FILE}")
        print("\n   Формат файла:")
        print('   [{"id": 1, "source": "ru", "evaluation": {...}}]')
        return

    # Анализируем
    ru_results = analyze_scores(scores, "ru")
    en_results = analyze_scores(scores, "en")
    all_results = analyze_scores(scores, "all")

    # Выводим отчет
    print_report(ru_results, en_results, all_results)

    # Сохраняем анализ
    analysis = {
        'total_scores': len(scores),
        'ru': ru_results,
        'en': en_results,
        'all': all_results
    }

    with open(ANALYSIS_FILE, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)

    print(f"✓ Анализ сохранен: {ANALYSIS_FILE}")

    # Простой вывод если нет разделения по языкам
    if not ru_results and not en_results and all_results:
        print(f"\n📊 Результаты ({all_results.get('total_evaluated', 0)} статей):")

        for criterion in ['clarity', 'relevance', 'completeness', 'overall']:
            if criterion in all_results:
                r = all_results[criterion]
                if 'p_value' in r:
                    sig = "✓" if r['significant'] else "✗"
                    print(f"  {criterion}: Base={r['base_mean']:.2f} RAG={r['rag_mean']:.2f} "
                          f"Δ={r['improvement_pct']:+.1f}% p={r['p_value']:.4f} {sig}")
                else:
                    print(f"  {criterion}: Base={r['base_mean']:.2f} RAG={r['rag_mean']:.2f} "
                          f"Δ={r['improvement_pct']:+.1f}%")

        if 'winners' in all_results:
            w = all_results['winners']
            total = sum(w.values())
            print(f"  RAG лучше: {w['rag']}/{total} ({all_results['rag_win_rate']}%)")
            print(f"  Base лучше: {w['base']}/{total}")
            print(f"  Ничья: {w['tie']}/{total}")


if __name__ == "__main__":
    main()