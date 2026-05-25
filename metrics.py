import numpy as np
from typing import List, Dict
from bert_score import score as bert_score
from scipy.stats import ttest_rel
import pandas as pd


class MetricsEvaluator:
    """Оценка качества — только BERTScore (работает с русским)"""

    def __init__(self, language: str = "ru"):
        self.language = language
        self.results = []

    def evaluate_pair(self, reference: str, base_pred: str, rag_pred: str,
                      article_info: Dict = None) -> Dict:
        """Оценка пары предсказаний через BERTScore"""
        result = {
            'article': article_info or {},
            'reference': reference[:200],
            'base_prediction': base_pred[:200],
            'rag_prediction': rag_pred[:200],
        }
        self.results.append(result)
        return result

    def compute_bertscore(self):
        """Вычисление BERTScore для всех результатов"""
        if not self.results:
            return None, None

        references = [r['reference'] for r in self.results]
        base_preds = [r['base_prediction'] for r in self.results]
        rag_preds = [r['rag_prediction'] for r in self.results]

        # Определяем язык
        bert_lang = "ru" if self.language in ["ru", "all"] else "en"

        print(f"  Вычисление BERTScore (lang={bert_lang})...")

        # BERTScore для Base
        P_base, R_base, F1_base = bert_score(
            base_preds, references,
            lang=bert_lang,
            model_type="distilbert-base-multilingual-cased",  # Мультиязычная модель
            verbose=False
        )

        # BERTScore для RAG
        P_rag, R_rag, F1_rag = bert_score(
            rag_preds, references,
            lang=bert_lang,
            model_type="distilbert-base-multilingual-cased",
            verbose=False
        )

        # Сохраняем в результаты
        for i, result in enumerate(self.results):
            result['bertscore_base'] = float(F1_base[i])
            result['bertscore_rag'] = float(F1_rag[i])

        return F1_base, F1_rag

    def get_summary(self) -> Dict:
        """Сводная статистика"""
        if not self.results:
            return {}

        # Вычисляем BERTScore
        self.compute_bertscore()

        df = pd.DataFrame(self.results)

        # BERTScore
        base_mean = df['bertscore_base'].mean()
        rag_mean = df['bertscore_rag'].mean()

        t_stat, p_value = ttest_rel(df['bertscore_rag'], df['bertscore_base'])

        summary = {
            'bertscore': {
                'base_mean': float(base_mean),
                'rag_mean': float(rag_mean),
                'improvement': float(rag_mean - base_mean),
                'improvement_pct': float((rag_mean - base_mean) / base_mean * 100) if base_mean > 0 else 0,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
        }

        # Постатейное сравнение
        df['rag_wins'] = df['bertscore_rag'] > df['bertscore_base']
        summary['win_rate'] = {
            'rag_wins': int(df['rag_wins'].sum()),
            'base_wins': int((~df['rag_wins']).sum()),
            'ties': int((df['bertscore_rag'] == df['bertscore_base']).sum()),
            'rag_win_rate': float(df['rag_wins'].mean())
        }

        return summary

    def print_report(self):
        """Вывод отчета"""
        summary = self.get_summary()

        print(f"\n{'=' * 60}")
        print(f"ОТЧЕТ (BERTScore, {self.language.upper()})")
        print(f"{'=' * 60}")
        print(f"Статей: {len(self.results)}")

        if 'bertscore' in summary:
            v = summary['bertscore']
            sig = "✓ ЗНАЧИМО" if v['significant'] else "✗ не значимо"
            print(f"\nBERTScore:")
            print(f"  Base: {v['base_mean']:.4f}")
            print(f"  RAG:  {v['rag_mean']:.4f}")
            print(f"  Δ:    {v['improvement_pct']:+.1f}%")
            print(f"  p={v['p_value']:.4f} {sig}")

        if 'win_rate' in summary:
            wr = summary['win_rate']
            total = wr['rag_wins'] + wr['base_wins'] + wr['ties']
            if total > 0:
                print(f"\nПостатейно:")
                print(f"  RAG лучше:  {wr['rag_wins']}/{total}")
                print(f"  Base лучше: {wr['base_wins']}/{total}")
                print(f"  Одинаково:  {wr['ties']}/{total}")