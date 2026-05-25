"""
Менеджер кэширования для ускорения повторных запусков
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timedelta
from config import CACHE_DIR, ENABLE_CACHE


class CacheManager:
    """Управление кэшем проекта"""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.enabled = ENABLE_CACHE
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0
        }

    def _get_cache_path(self, key: str, category: str) -> Path:
        """Путь к файлу кэша"""
        # Создаем подпапку для категории
        cat_dir = self.cache_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Хэшируем ключ для имени файла
        hashed = hashlib.md5(key.encode()).hexdigest()
        return cat_dir / f"{hashed}.pkl"

    def get(self, key: str, category: str, max_age_hours: int = 168) -> Optional[Any]:
        """
        Получить из кэша

        Args:
            key: уникальный ключ
            category: категория (pdf_text, embeddings, llm_response, metrics)
            max_age_hours: максимальный возраст кэша в часах (по умолчанию 7 дней)
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(key, category)

        if not cache_path.exists():
            self.stats['misses'] += 1
            return None

        # Проверяем возраст
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if age > timedelta(hours=max_age_hours):
            # Кэш устарел
            cache_path.unlink()
            self.stats['misses'] += 1
            return None

        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)

            self.stats['hits'] += 1
            return data
        except Exception:
            self.stats['misses'] += 1
            return None

    def set(self, key: str, category: str, data: Any):
        """Сохранить в кэш"""
        if not self.enabled:
            return

        cache_path = self._get_cache_path(key, category)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self.stats['saves'] += 1
        except Exception as e:
            pass  # Не критично если не сохранилось

    def get_pdf_text(self, pdf_path: str) -> Optional[str]:
        """Кэш для текста PDF"""
        key = str(Path(pdf_path).absolute())
        return self.get(key, 'pdf_text', max_age_hours=720)  # 30 дней

    def set_pdf_text(self, pdf_path: str, text: str):
        """Сохранить текст PDF в кэш"""
        key = str(Path(pdf_path).absolute())
        self.set(key, 'pdf_text', text)

    def get_embeddings(self, text_hash: str) -> Optional[list]:
        """Кэш для эмбеддингов"""
        return self.get(text_hash, 'embeddings', max_age_hours=720)

    def set_embeddings(self, text_hash: str, embeddings: list):
        """Сохранить эмбеддинги"""
        self.set(text_hash, 'embeddings', embeddings)

    def get_llm_response(self, prompt: str, model: str) -> Optional[str]:
        """Кэш для ответов LLM"""
        key = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()
        return self.get(key, 'llm_responses', max_age_hours=168)

    def set_llm_response(self, prompt: str, model: str, response: str):
        """Сохранить ответ LLM"""
        key = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()
        self.set(key, 'llm_responses', response)

    def get_metrics(self, article_hash: str) -> Optional[Dict]:
        """Кэш для метрик"""
        return self.get(article_hash, 'metrics', max_age_hours=720)

    def set_metrics(self, article_hash: str, metrics: Dict):
        """Сохранить метрики"""
        self.set(article_hash, 'metrics', metrics)

    def clear_category(self, category: str):
        """Очистить категорию кэша"""
        cat_dir = self.cache_dir / category
        if cat_dir.exists():
            import shutil
            shutil.rmtree(cat_dir)
            cat_dir.mkdir(parents=True, exist_ok=True)

    def clear_all(self):
        """Очистить весь кэш"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {'hits': 0, 'misses': 0, 'saves': 0}

    def get_stats(self) -> Dict:
        """Статистика кэша"""
        total = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / max(total, 1) * 100

        # Размер кэша
        cache_size = 0
        if self.cache_dir.exists():
            for f in self.cache_dir.rglob('*'):
                if f.is_file():
                    cache_size += f.stat().st_size

        return {
            **self.stats,
            'hit_rate': f"{hit_rate:.1f}%",
            'cache_size_mb': cache_size / (1024 * 1024)
        }

    def print_stats(self):
        """Вывод статистики"""
        stats = self.get_stats()
        print(f"\n  📦 Статистика кэша:")
        print(f"    Попаданий: {stats['hits']}")
        print(f"    Промахов: {stats['misses']}")
        print(f"    Hit rate: {stats['hit_rate']}")
        print(f"    Сохранений: {stats['saves']}")
        print(f"    Размер: {stats['cache_size_mb']:.1f} MB")


# Глобальный экземпляр
cache = CacheManager()