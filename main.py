#!/usr/bin/env python3
"""
RAG система для извлечения научных проблем из PDF
+ Сравнение моделей Ollama
+ Постатейный BERTScore
+ Раздельные выборки RU/EN
"""

import sys
import time
import json
import random
import re
import gc
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

from config import *
from pdf_loader import PDFLoader
from vector_db import ScientificVectorDB
from ollama_pipeline import OllamaPipeline
from metrics import MetricsEvaluator
from logger import ExperimentLogger
from cache_manager import cache

RESULTS_FILE = None


# ============================================
# УТИЛИТЫ
# ============================================

def log_to_file(text: str, also_print: bool = True):
    global RESULTS_FILE
    if also_print:
        print(text)
    if RESULTS_FILE:
        try:
            with open(RESULTS_FILE, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except:
            pass


def log_separator(title: str = "", char: str = "=", width: int = 60):
    if title:
        log_to_file(f"\n{char * width}\n  {title}\n{char * width}\n")
    else:
        log_to_file(f"{char * width}")


def print_banner():
    log_to_file("""
╔══════════════════════════════════════════════════════════╗
║     Multilingual RAG — Извлечение научных проблем       ║
║     RU/EN → PDF → Ollama → FAISS → LLM → BERTScore     ║
╚══════════════════════════════════════════════════════════╝
    """)


# ============================================
# ЭТАЛОН ДЛЯ BERTSCORE
# ============================================

def get_reference(article: Dict, language: str) -> str:
    """Умный выбор эталона: маркеры → аннотация → fallback"""
    text = article.get('text', '')

    markers = {
        "ru": ["проблема", "задача", "цель", "исследование посвящено",
               "статья посвящена", "рассматривается проблема",
               "решается задача", "предлагается метод", "предложен подход",
               "настоящая работа", "данная работа", "актуальность",
               "научная проблема", "целью работы", "в данной статье"],
        "en": ["problem", "challenge", "issue", "goal", "objective",
               "this paper addresses", "we address", "we tackle",
               "we propose", "this work", "research problem"]
    }

    text_lower = text.lower()
    found = []

    for marker in markers.get(language, markers["en"]):
        idx = text_lower.find(marker)
        if idx == -1:
            continue

        start = idx
        for i in range(idx - 1, max(0, idx - 200), -1):
            if text[i] in '.!?\n' and i < idx - 5:
                start = i + 1
                break

        end = min(len(text), idx + 400)
        for i in range(idx + len(marker), min(len(text), idx + 450)):
            if text[i] in '.!?' and i > idx + 20:
                end = i + 1
                break

        sentence = text[start:end].strip()
        sentence = re.sub(r'\s+', ' ', sentence)

        if len(sentence) >= 30:
            weight = len(sentence) + (100 - (start / max(len(text), 1) * 100))
            found.append({'text': sentence, 'weight': weight})

    if found:
        found.sort(key=lambda x: x['weight'], reverse=True)
        return found[0]['text']

    abstract = article.get('metadata', {}).get('abstract', '')
    if abstract and len(abstract.strip()) >= 50:
        return abstract.strip()

    start = min(200, len(text))
    end = min(start + 500, len(text))
    fallback = text[start:end].strip()
    return fallback if len(fallback) >= 30 else text[:500]


# ============================================
# ПРОВЕРКА ОКРУЖЕНИЯ
# ============================================

def check_environment() -> bool:
    log_separator("ПРОВЕРКА ОКРУЖЕНИЯ")

    log_to_file("1. Ollama...")
    try:
        import ollama
        result = ollama.list()

        if hasattr(result, 'models'):
            model_names = [m.model for m in result.models]
        else:
            model_names = [m.get('name', '') for m in result.get('models', [])]

        log_to_file(f"  Доступно: {len(model_names)} моделей")
        for name in model_names:
            log_to_file(f"    • {name}")

        # Проверяем необходимые модели
        required = set()
        if MODE_COMPARE_MODELS:
            required.update(LLM_MODELS_TO_COMPARE)
        else:
            required.update(LLM_MODELS.values())
        required.add(EMBED_MODEL)

        log_to_file(f"\n  Необходимые модели:")
        for model in required:
            base = model.split(':')[0]
            found = any(base in name for name in model_names)
            status = "✓" if found else "✗ (скачайте: ollama pull " + model + ")"
            log_to_file(f"    {status} {model}")

    except Exception as e:
        log_to_file(f"  ✗ Ollama не доступен: {e}")
        return False

    log_to_file("\n2. PDF...")
    ru = len(list(PDF_RU_DIR.glob("*.pdf"))) if PDF_RU_DIR.exists() else 0
    en = len(list(PDF_EN_DIR.glob("*.pdf"))) if PDF_EN_DIR.exists() else 0
    log_to_file(f"  RU: {ru} | EN: {en}")

    log_to_file(f"\n3. Режим: {'СРАВНЕНИЕ МОДЕЛЕЙ' if MODE_COMPARE_MODELS else 'ОДНА МОДЕЛЬ'}")
    log_to_file(f"  Кэш: {'включен' if ENABLE_CACHE else 'отключен'}")

    return True


# ============================================
# ШАГ 1: ЗАГРУЗКА PDF
# ============================================

def load_all_articles(logger: ExperimentLogger) -> Dict[str, List[Dict]]:
    log_separator("ШАГ 1: Загрузка PDF")

    loader = PDFLoader()
    ru_articles, en_articles = loader.load_directory()
    articles = {"ru": ru_articles, "en": en_articles}

    stats = loader.get_statistics()
    log_to_file(f"  RU={stats['russian_articles']}, EN={stats['english_articles']}, "
                f"из кэша={stats.get('cached', 0)}, ошибок={stats['failed']}")

    logger.log_phase('pdf_loading', {'ru': len(ru_articles), 'en': len(en_articles)})
    return articles


# ============================================
# ШАГ 2: ВЕКТОРНЫЕ БД
# ============================================

def create_or_load_dbs(articles, ollama_pipeline, logger):
    log_separator("ШАГ 2: Векторные БД")
    dbs = {}

    for lang in ['ru', 'en']:
        arts = articles.get(lang, [])
        if not arts:
            continue

        db_path = DATA_DIR / f"vector_db_{lang}"

        if (db_path / "index.faiss").exists():
            try:
                db = ScientificVectorDB(ollama_pipeline)
                db.load(str(db_path))

                folder = str(PDF_RU_DIR if lang == 'ru' else PDF_EN_DIR)
                comparison = db.compare_with_folder(folder)

                log_to_file(f"  [{lang.upper()}] в БД: {comparison['total_in_db']}, "
                            f"новых: {len(comparison['only_in_folder'])}")

                if comparison['only_in_folder'] and MODE_CREATE_DB:
                    new = [a for a in arts if a['filename'] in comparison['only_in_folder']]
                    db.add_articles(new, ollama_pipeline)
                    db.save(str(db_path))

                ollama_pipeline.set_vector_db(db, lang)
                dbs[lang] = db
            except Exception as e:
                log_to_file(f"  ✗ [{lang.upper()}] {e}")

        if lang not in dbs and MODE_CREATE_DB:
            log_to_file(f"  [{lang.upper()}] создание ({len(arts)} ст.)...")
            db = ScientificVectorDB(ollama_pipeline)
            db.build_from_articles(arts)
            db.save(str(db_path))
            ollama_pipeline.set_vector_db(db, lang)
            dbs[lang] = db
            log_to_file(f"  ✓ готово")

    return dbs


# ============================================
# ОЦЕНКА БАТЧА С BERTSCORE
# ============================================

def evaluate_batch(articles: List[Dict], language: str,
                   ollama_pipeline, model_name: str = "") -> Tuple[Dict, List]:
    """Оценка батча статей одного языка с постатейным BERTScore"""

    results = []
    base_times, rag_times = [], []
    refs, base_preds, rag_preds = [], [], []
    tag = f"[{model_name}] " if model_name else ""

    for i, article in enumerate(articles):
        lang = article['language']
        log_to_file(f"\n  {tag}[{i + 1}/{len(articles)}] {article['filename'][:50]}")

        try:
            base_res = ollama_pipeline.extract_problem_base(article['text'], lang)
            base_times.append(base_res['time'])

            rag_res = ollama_pipeline.extract_problem_rag(article['text'], lang)
            rag_times.append(rag_res['time'])

            reference = get_reference(article, lang)

            refs.append(reference)
            base_preds.append(base_res['text'])
            rag_preds.append(rag_res['text'])

            log_to_file(f"    Base: {base_res['text'][:80]}...")
            log_to_file(f"    RAG:  {rag_res['text'][:80]}...")
            log_to_file(f"    ⏱ {base_res['time']:.1f}s | {rag_res['time']:.1f}s")

            results.append({
                'filename': article['filename'],
                'language': lang,
                'model': model_name,
                'title': article.get('metadata', {}).get('title', '')[:100],
                'ref_length': len(reference.strip()),
                'reference': reference[:200],
                'base_problem': base_res['text'],
                'rag_problem': rag_res['text'],
                'base_time': base_res['time'],
                'rag_time': rag_res['time'],
                'base_tokens': base_res.get('tokens_generated', 0),
                'rag_tokens': rag_res.get('tokens_generated', 0),
                'base_from_cache': base_res.get('from_cache', False),
                'rag_from_cache': rag_res.get('from_cache', False),
                'rag_chunks': rag_res.get('chunks_used', 0),
                'rag_chunk_score': rag_res.get('avg_chunk_score', 0),
            })
        except Exception as e:
            log_to_file(f"    ❌ {str(e)[:100]}")
            refs.append("")
            base_preds.append("")
            rag_preds.append("")

    if not results:
        return None, []

    # BERTScore
    log_to_file(f"\n  📊 BERTScore...")
    try:
        from bert_score import score as bert_score
        from scipy.stats import ttest_rel

        _, _, F1_base = bert_score(base_preds, refs, lang=language,
                                   model_type="distilbert-base-multilingual-cased", verbose=False)
        _, _, F1_rag = bert_score(rag_preds, refs, lang=language,
                                  model_type="distilbert-base-multilingual-cased", verbose=False)

        for i in range(len(results)):
            results[i]['bertscore_base_f1'] = round(float(F1_base[i]), 4)
            results[i]['bertscore_rag_f1'] = round(float(F1_rag[i]), 4)
            results[i]['bertscore_diff'] = round(float(F1_rag[i] - F1_base[i]), 4)
            results[i]['bertscore_better'] = "RAG" if F1_rag[i] > F1_base[i] else (
                "Base" if F1_base[i] > F1_rag[i] else "Tie")

        base_mean = np.mean([float(f) for f in F1_base])
        rag_mean = np.mean([float(f) for f in F1_rag])
        base_std = np.std([float(f) for f in F1_base])
        rag_std = np.std([float(f) for f in F1_rag])

        t_stat, p_value = ttest_rel([float(f) for f in F1_rag], [float(f) for f in F1_base])

        rag_wins = sum(1 for r in results if r['bertscore_better'] == "RAG")
        base_wins = sum(1 for r in results if r['bertscore_better'] == "Base")
        ties = sum(1 for r in results if r['bertscore_better'] == "Tie")

        sorted_diff = sorted(results, key=lambda x: x['bertscore_diff'], reverse=True)

        # Постатейная таблица
        log_to_file(f"\n  {'Статья':<35} | {'Base':<8} | {'RAG':<8} | {'Δ':<10} | {'Лучше'}")
        log_to_file(f"  {'─' * 75}")
        for r in results:
            ind = "📈" if r['bertscore_diff'] > 0.01 else ("📉" if r['bertscore_diff'] < -0.01 else "➡️")
            log_to_file(f"  {r['filename'][:33]:<35} | {r['bertscore_base_f1']:.4f}   | "
                        f"{r['bertscore_rag_f1']:.4f}   | {r['bertscore_diff']:+.4f}     | {ind} {r['bertscore_better']}")

        # Сводка
        log_to_file(f"\n  {'=' * 50}")
        log_to_file(f"  СТАТИСТИКА BERTScore ({language.upper()})")
        log_to_file(f"  {'=' * 50}")
        log_to_file(f"  Base: {base_mean:.4f} ± {base_std:.4f}")
        log_to_file(f"  RAG:  {rag_mean:.4f} ± {rag_std:.4f}")
        log_to_file(f"  Δ: {(rag_mean - base_mean):+.4f} ({(rag_mean - base_mean) / base_mean * 100:+.1f}%)")
        log_to_file(f"  p={p_value:.4f} {'✓ ЗНАЧИМО' if p_value < 0.05 else '✗ не значимо'}")
        log_to_file(f"  RAG лучше: {rag_wins}/{len(results)} ({rag_wins / len(results) * 100:.0f}%)")
        log_to_file(f"  ⏱ Base: {np.mean(base_times):.1f}с | RAG: {np.mean(rag_times):.1f}с")

        summary = {
            'bertscore': {
                'base_mean': float(base_mean), 'base_std': float(base_std),
                'rag_mean': float(rag_mean), 'rag_std': float(rag_std),
                'improvement': float(rag_mean - base_mean),
                'improvement_pct': float((rag_mean - base_mean) / base_mean * 100) if base_mean > 0 else 0,
                'p_value': float(p_value), 'significant': bool(p_value < 0.05)
            },
            'win_rate': {'rag_wins': rag_wins, 'base_wins': base_wins, 'ties': ties,
                         'rag_win_rate': rag_wins / len(results) if results else 0},
            'timing': {'base_avg': float(np.mean(base_times)), 'rag_avg': float(np.mean(rag_times))}
        }

        return summary, results

    except Exception as e:
        log_to_file(f"  ❌ BERTScore: {e}")
        return None, results


# ============================================
# ОЦЕНКА RU + EN (ОДНА МОДЕЛЬ)
# ============================================

def evaluate_all_languages(articles, ollama_pipeline, logger):
    log_separator(f"ОЦЕНКА: RU + EN (по {EVAL_SAMPLE_SIZE} статей)")

    output_dir = OUTPUT_DIR
    charts_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Выборка
    def sample_articles(arts, n):
        if len(arts) <= n:
            return arts
        return random.sample(arts, n)

    random.seed(RANDOM_SEED)
    ru_sel = sample_articles(articles.get('ru', []), EVAL_SAMPLE_SIZE)
    random.seed(RANDOM_SEED + 1)
    en_sel = sample_articles(articles.get('en', []), EVAL_SAMPLE_SIZE)

    log_to_file(f"  Выбрано: RU={len(ru_sel)}, EN={len(en_sel)}")

    # Оценка
    ru_summary, ru_results = None, []
    en_summary, en_results = None, []

    if ru_sel:
        log_separator("RU", "─")
        ru_summary, ru_results = evaluate_batch(ru_sel, 'ru', ollama_pipeline)

    if en_sel:
        log_separator("EN", "─")
        en_summary, en_results = evaluate_batch(en_sel, 'en', ollama_pipeline)

    all_results = ru_results + en_results
    if not all_results:
        return None, None

    # Сохранение
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "detailed_results_all.csv", index=False, encoding='utf-8')
    if ru_results:
        pd.DataFrame(ru_results).to_csv(output_dir / "detailed_results_ru.csv", index=False, encoding='utf-8')
    if en_results:
        pd.DataFrame(en_results).to_csv(output_dir / "detailed_results_en.csv", index=False, encoding='utf-8')

    # Сравнение RU vs EN
    if ru_summary and en_summary:
        log_separator("RU vs EN", "═")
        ru_bs = ru_summary['bertscore']
        en_bs = en_summary['bertscore']
        log_to_file(f"  RU: Base={ru_bs['base_mean']:.4f} RAG={ru_bs['rag_mean']:.4f} "
                    f"({ru_bs['improvement_pct']:+.1f}%) {'✓' if ru_bs['significant'] else '✗'}")
        log_to_file(f"  EN: Base={en_bs['base_mean']:.4f} RAG={en_bs['rag_mean']:.4f} "
                    f"({en_bs['improvement_pct']:+.1f}%) {'✓' if en_bs['significant'] else '✗'}")

    # JSON
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump({'ru': ru_summary, 'en': en_summary}, f, ensure_ascii=False, indent=2, default=str)

    # График
    if ru_summary and en_summary:
        try:
            import matplotlib;
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 5))
            ru_imp = ru_summary['bertscore']['improvement_pct']
            en_imp = en_summary['bertscore']['improvement_pct']
            ax.bar(['RU', 'EN'], [ru_imp, en_imp], color=['#FF8C42', '#45B7D1'])
            ax.set_ylabel('BERTScore Improvement %')
            ax.set_title('RAG Improvement by Language')
            ax.axhline(y=0, color='black', linestyle='-')
            plt.tight_layout()
            plt.savefig(charts_dir / "language_comparison.png", dpi=150)
            plt.close()
            log_to_file(f"  ✓ График сохранен")
        except:
            pass

    # DeepSeek
    _export_deepseek(all_results, output_dir)

    log_to_file(f"\n  ✓ Результаты: {output_dir}")
    return ru_summary, en_summary


# ============================================
# СРАВНЕНИЕ МОДЕЛЕЙ
# ============================================

def evaluate_with_multiple_models(articles, ollama_pipeline, logger):
    log_separator(f"СРАВНЕНИЕ {len(LLM_MODELS_TO_COMPARE)} МОДЕЛЕЙ", "═")

    output_dir = OUTPUT_DIR
    charts_dir = output_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    log_to_file(f"\n  Модели: {', '.join(m.split(':')[0] for m in LLM_MODELS_TO_COMPARE)}")

    # Выборка
    def sample(arts, n):
        return random.sample(arts, min(n, len(arts))) if len(arts) > n else arts

    random.seed(RANDOM_SEED)
    ru_sel = sample(articles.get('ru', []), EVAL_SAMPLE_SIZE)
    random.seed(RANDOM_SEED + 1)
    en_sel = sample(articles.get('en', []), EVAL_SAMPLE_SIZE)

    all_sel = []
    for a in ru_sel:
        a['source_lang'] = 'ru'
        all_sel.append(a)
    for a in en_sel:
        a['source_lang'] = 'en'
        all_sel.append(a)

    log_to_file(f"  Статей: RU={len(ru_sel)}, EN={len(en_sel)} (всего {len(all_sel)})")

    # Прогон
    results = {}
    total_start = time.time()

    for idx, model_name in enumerate(LLM_MODELS_TO_COMPARE):
        log_separator(f"МОДЕЛЬ {idx + 1}/{len(LLM_MODELS_TO_COMPARE)}: {model_name}", "─")

        try:
            import ollama
            ollama.show(model_name)
        except:
            log_to_file(f"  ⚠ Не найдена, пропускаю")
            continue

        ollama_pipeline.models["ru"] = model_name
        ollama_pipeline.models["en"] = model_name

        start = time.time()
        summary, res = evaluate_batch(all_sel, 'ru', ollama_pipeline, model_name)
        elapsed = time.time() - start

        if summary:
            results[model_name] = {
                'base_mean': summary['bertscore']['base_mean'],
                'rag_mean': summary['bertscore']['rag_mean'],
                'improvement': summary['bertscore']['improvement'],
                'improvement_pct': summary['bertscore']['improvement_pct'],
                'p_value': summary['bertscore']['p_value'],
                'significant': summary['bertscore']['significant'],
                'rag_win_rate': summary['win_rate']['rag_win_rate'],
                'time_min': round(elapsed / 60, 1),
                'results': res
            }

        gc.collect()

    if not results:
        return {}

    # Таблица
    log_separator("РЕЗУЛЬТАТЫ", "═")
    log_to_file(f"\n  {'Модель':<22} | {'Base':<8} | {'RAG':<8} | {'Δ':<8} | {'Δ%':<8} | {'p':<8} | {'Win':<6}")
    log_to_file(f"  {'─' * 80}")

    best_model = max(results, key=lambda m: results[m]['improvement'])

    for model, data in results.items():
        star = " ⭐" if model == best_model else ""
        sig = "✓" if data['significant'] else "✗"
        log_to_file(f"  {model:<22} | {data['base_mean']:.4f}   | {data['rag_mean']:.4f}   | "
                    f"{data['improvement']:+.4f}  | {data['improvement_pct']:+6.1f}% | "
                    f"{data['p_value']:.4f} {sig} | {data['rag_win_rate']:.0%}{star}")

    log_to_file(f"\n  🏆 Лучшая: {best_model} ({results[best_model]['improvement_pct']:+.1f}%)")
    log_to_file(f"  Общее время: {(time.time() - total_start) / 60:.1f} мин")

    # Сохранение
    all_rows = []
    for model, data in results.items():
        for r in data['results']:
            r['model'] = model
            all_rows.append(r)

    pd.DataFrame(all_rows).to_csv(output_dir / "model_comparison_results.csv", index=False, encoding='utf-8')

    with open(output_dir / "model_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            'best_model': best_model,
            'results': {m: {k: v for k, v in d.items() if k != 'results'}
                        for m, d in results.items()}
        }, f, ensure_ascii=False, indent=2)

    # График
    try:
        import matplotlib;
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        models = list(results.keys())
        short = [m.split(':')[0][:12] for m in models]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Base vs RAG
        ax = axes[0]
        x = range(len(models))
        w = 0.35
        ax.bar([i - w / 2 for i in x], [results[m]['base_mean'] for m in models], w, label='Base', color='#FF6B6B')
        ax.bar([i + w / 2 for i in x], [results[m]['rag_mean'] for m in models], w, label='RAG', color='#4ECDC4')
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=9)
        ax.set_ylabel('BERTScore F1')
        ax.set_title('Base vs RAG')
        ax.legend()

        # Improvement
        ax = axes[1]
        imps = [results[m]['improvement'] for m in models]
        colors = ['green' if i > 0 else 'red' for i in imps]
        ax.bar(short, imps, color=colors)
        ax.set_ylabel('BERTScore Improvement')
        ax.set_title('RAG Improvement')
        ax.axhline(y=0, color='black', linestyle='-')

        plt.tight_layout()
        plt.savefig(charts_dir / "model_comparison.png", dpi=150)
        plt.close()
        log_to_file(f"  ✓ График сохранен")
    except:
        pass

    # DeepSeek
    _export_deepseek(all_rows, output_dir)

    return results


def _export_deepseek(results_list, output_dir):
    """Экспорт для DeepSeek оценки"""
    items = []
    for i, r in enumerate(results_list):
        items.append({
            'id': i + 1,
            'filename': r.get('filename', ''),
            'source': r.get('language', r.get('source', '')),
            'reference': r.get('reference', '')[:300],
            'base_problem': r.get('base_problem', ''),
            'rag_problem': r.get('rag_problem', ''),
        })

    with open(output_dir / "for_deepseek_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    with open(output_dir / "deepseek_prompts.txt", "w", encoding="utf-8") as f:
        f.write(f"ПРОМПТЫ ДЛЯ DEEPSEEK\n{'=' * 60}\n\n")
        for item in items:
            f.write(f"{'─' * 40}\nСтатья {item['id']} [{item['source'].upper()}]\n")
            f.write(f"Файл: {item['filename']}\n\n")
            f.write(f"Эталон: {item['reference'][:200]}...\n\n")
            f.write(f"A (Base): {item['base_problem']}\n\n")
            f.write(f"B (RAG):  {item['rag_problem']}\n\n")
            f.write("A: clarity= , relevance= , completeness= \n")
            f.write("B: clarity= , relevance= , completeness= \n")
            f.write("Лучше: \n\n")

    log_to_file(f"  ✓ DeepSeek: {output_dir / 'deepseek_prompts.txt'}")


# ============================================
# MAIN
# ============================================

def main():
    global RESULTS_FILE
    RESULTS_FILE = OUTPUT_DIR / "results.txt"

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("")

    log_to_file("=" * 60)
    log_to_file("  RAG — ИЗВЛЕЧЕНИЕ НАУЧНЫХ ПРОБЛЕМ")
    log_to_file("=" * 60)
    log_to_file(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")

    print_banner()
    check_environment()

    logger = ExperimentLogger("rag_experiment")
    ollama_pipeline = OllamaPipeline()

    # Шаг 1: PDF
    articles = load_all_articles(logger) if (MODE_CREATE_DB or MODE_EVALUATE) else {"ru": [], "en": []}

    # Шаг 2: БД
    if articles:
        create_or_load_dbs(articles, ollama_pipeline, logger)

    # Шаг 3: Оценка
    if MODE_EVALUATE and articles:
        if MODE_COMPARE_MODELS:
            evaluate_with_multiple_models(articles, ollama_pipeline, logger)
        else:
            evaluate_all_languages(articles, ollama_pipeline, logger)

    # Статистика
    log_separator("СТАТИСТИКА")
    cache.print_stats()
    ollama_pipeline.print_stats()
    logger.save()

    log_separator("ГОТОВО")
    log_to_file(f"  {RESULTS_FILE}")
    log_to_file(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()