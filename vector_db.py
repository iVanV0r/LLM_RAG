import os
import re
import pickle
import numpy as np
import faiss
from typing import List, Dict, Set, Optional
from tqdm import tqdm
from config import *


class ScientificVectorDB:
    """Векторная БД с использованием Ollama для эмбеддингов"""

    def __init__(self, ollama_pipeline=None):
        self.ollama = ollama_pipeline
        self.index = None
        self.chunks = []
        self.metadata = []
        self.dim = None
        self.fallback_embedder = None

        self.stats = {
            'total_articles': 0,
            'total_chunks': 0,
            'avg_chunk_size': 0,
            'embedding_method': 'ollama',
            'index_type': 'FlatIP',
            'db_size_mb': 0
        }

    # ============================================
    # ОЧИСТКА ТЕКСТА
    # ============================================

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Очистка текста от проблемных Unicode символов

        Удаляет:
        - Unicode surrogate символы (0xD800-0xDFFF)
        - Математические символы за пределами базовых плоскостей
        - Эмодзи и другие редко используемые символы
        """
        if not text:
            return ""

        # 1. Удаляем surrogate символы
        text = text.encode('utf-8', errors='replace').decode('utf-8')

        # 2. Оставляем только разрешенные символы:
        #    - ASCII (0x00-0x7F): базовый английский
        #    - Кириллица (0x400-0x4FF): русский алфавит
        #    - Доп. кириллица (0x500-0x52F): расширенный русский
        #    - Latin-1 Supplement (0x80-0xFF): буквы с диакритикой
        #    - Latin Extended-A (0x100-0x17F): европейские языки
        #    - Latin Extended-B (0x180-0x24F): дополнительные латинские
        #    - Пробелы, знаки препинания, цифры
        allowed_pattern = (
            r'[^\x00-\x7F'  # ASCII
            r'\u0400-\u04FF'  # Кириллица
            r'\u0500-\u052F'  # Доп. кириллица
            r'\u0080-\u00FF'  # Latin-1 Supplement
            r'\u0100-\u017F'  # Latin Extended-A
            r'\u0180-\u024F'  # Latin Extended-B
            r'\s\.\,\!\?\-\:\;\"\'\(\)\[\]\{\}\d\*\#\%\&\+\=\/\\\@\$\€\£\¥\°]'
        )
        text = re.sub(allowed_pattern, ' ', text)

        # 3. Заменяем множественные пробелы на один
        text = re.sub(r'\s+', ' ', text)

        # 4. Удаляем пробелы в начале и конце
        text = text.strip()

        return text

    # ============================================
    # ЭМБЕДДИНГИ
    # ============================================

    def _embed_batch(self, texts: List[str], batch_size: int = 10) -> np.ndarray:
        """
        Получение эмбеддингов через Ollama с fallback на sentence-transformers
        """
        # Очищаем все тексты
        cleaned_texts = [self._clean_text(t) for t in texts]

        # Фильтруем пустые
        valid_indices = [i for i, t in enumerate(cleaned_texts) if t and len(t) > 5]
        valid_texts = [cleaned_texts[i] for i in valid_indices]

        if not valid_texts:
            print("  ⚠ Все тексты пустые после очистки!")
            return np.zeros((len(texts), 384), dtype=np.float32)

        embeddings = []

        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]

            # Пробуем Ollama
            try:
                batch_embs = self.ollama.embed(batch)
                embeddings.extend(batch_embs)
                continue
            except Exception as e:
                print(f"  ⚠ Ошибка Ollama embeddings (batch {i // batch_size}): {str(e)[:100]}")

            # Fallback на sentence-transformers
            try:
                if self.fallback_embedder is None:
                    from sentence_transformers import SentenceTransformer
                    print("  Загрузка fallback модели (sentence-transformers)...")
                    self.fallback_embedder = SentenceTransformer('intfloat/multilingual-e5-large')
                    self.stats['embedding_method'] = 'fallback_sentence_transformers'

                fallback_embs = self.fallback_embedder.encode(batch, normalize_embeddings=True)
                embeddings.extend(fallback_embs.tolist())

            except Exception as e2:
                print(f"  ❌ Ошибка fallback: {str(e2)[:100]}")
                # Нулевые векторы как последнее средство
                for _ in batch:
                    embeddings.append([0.0] * (self.dim or 384))

        # Восстанавливаем полный массив (включая пустые тексты)
        if len(valid_indices) < len(texts):
            full_embeddings = np.zeros((len(texts), len(embeddings[0]) if embeddings else 384), dtype=np.float32)
            for j, idx in enumerate(valid_indices):
                if j < len(embeddings):
                    full_embeddings[idx] = embeddings[j]
            return full_embeddings

        return np.array(embeddings, dtype=np.float32)

    # ============================================
    # ЧАНКОВАНИЕ
    # ============================================

    def _chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Разбиение текста на чанки по параграфам и предложениям
        """
        # Очищаем текст перед чанкованием
        text = self._clean_text(text)

        chunks = []
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 10]

        for para_idx, paragraph in enumerate(paragraphs):
            # Если параграф короткий — оставляем как есть
            if len(paragraph) <= CHUNK_SIZE * 1.2:
                if len(paragraph) >= 50:
                    chunk_meta = {
                        'chunk_id': len(chunks),
                        'paragraph_id': para_idx,
                        'position_ratio': para_idx / max(len(paragraphs), 1),
                        'chunk_type': 'paragraph',
                        **(metadata or {})
                    }
                    chunks.append({'text': paragraph, 'metadata': chunk_meta})
            else:
                # Длинный параграф — разбиваем на предложения
                sentences = self._split_sentences(paragraph)

                i = 0
                while i < len(sentences):
                    chunk_text = ""
                    chunk_start = i

                    # Собираем предложения пока не достигнем CHUNK_SIZE
                    while i < len(sentences) and len(chunk_text) + len(sentences[i]) < CHUNK_SIZE:
                        chunk_text += sentences[i] + " "
                        i += 1

                    chunk_text = chunk_text.strip()
                    if len(chunk_text) >= 50:
                        chunk_meta = {
                            'chunk_id': len(chunks),
                            'paragraph_id': para_idx,
                            'sentences_range': (chunk_start, i),
                            'position_ratio': para_idx / max(len(paragraphs), 1),
                            'chunk_type': 'sentences',
                            **(metadata or {})
                        }
                        chunks.append({'text': chunk_text, 'metadata': chunk_meta})

                    # Перекрытие для сохранения контекста
                    if CHUNK_OVERLAP > 0 and i < len(sentences):
                        overlap_chars = 0
                        while i > chunk_start and overlap_chars < CHUNK_OVERLAP:
                            i -= 1
                            overlap_chars += len(sentences[i]) if i < len(sentences) else 0
                        i = max(i, chunk_start + 1)

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Разбиение текста на предложения"""
        sentences = []
        current = ""

        for char in text:
            current += char
            if char in '.!?' and len(current) > 20:
                sentences.append(current.strip())
                current = ""

        if current.strip() and len(current.strip()) > 10:
            sentences.append(current.strip())

        return sentences

    # ============================================
    # ПОСТРОЕНИЕ БД
    # ============================================

    def build_from_articles(self, articles: List[Dict]):
        """
        Полное построение БД с нуля
        """
        if self.ollama is None:
            raise ValueError("Ollama pipeline не передан")

        print(f"\nЧанкование {len(articles)} статей...")

        all_chunks = []
        for article_idx, article in enumerate(tqdm(articles, desc="Чанкование")):
            # Очищаем текст статьи
            cleaned_text = self._clean_text(article['text'])

            article_meta = {
                'article_id': article_idx,
                'filename': article.get('filename', f'unknown_{article_idx}'),
                'title': article.get('metadata', {}).get('title', '')[:100],
                'language': article.get('language', 'unknown')
            }

            chunks = self._chunk_text(cleaned_text, article_meta)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("Не создано ни одного чанка! Проверьте качество текста в PDF.")

        print(f"  Создано чанков: {len(all_chunks)}")

        chunk_sizes = [len(c['text']) for c in all_chunks]
        print(f"  Размеры чанков: мин={min(chunk_sizes)}, макс={max(chunk_sizes)}, "
              f"средний={np.mean(chunk_sizes):.0f}")

        # Эмбеддинги
        print(f"\nСоздание эмбеддингов ({len(all_chunks)} чанков)...")
        texts = [c['text'] for c in all_chunks]
        embeddings = self._embed_batch(texts, batch_size=10)

        # Проверка на NaN
        if np.isnan(embeddings).any():
            print("  ⚠ Обнаружены NaN в эмбеддингах, заменяю на нули...")
            embeddings = np.nan_to_num(embeddings, nan=0.0)

        # Создание FAISS индекса
        self.dim = embeddings.shape[1]

        if len(all_chunks) < 10000:
            self.index = faiss.IndexFlatIP(self.dim)
            self.stats['index_type'] = 'FlatIP (точный)'
        else:
            nlist = min(int(np.sqrt(len(all_chunks))), 4096)
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            self.index.train(embeddings.astype(np.float32))
            self.stats['index_type'] = f'IVFFlat (nlist={nlist})'

        self.index.add(embeddings.astype(np.float32))

        # Сохраняем данные
        self.chunks = [c['text'] for c in all_chunks]
        self.metadata = [c['metadata'] for c in all_chunks]

        # Статистика
        unique_files = set(m.get('filename', '') for m in self.metadata)
        self.stats.update({
            'total_articles': len(articles),
            'total_chunks': len(all_chunks),
            'avg_chunk_size': float(np.mean(chunk_sizes)),
            'embedding_dim': self.dim,
            'db_size_mb': embeddings.nbytes / (1024 * 1024),
            'unique_files': len(unique_files)
        })

        self._print_stats()

    # ============================================
    # ДОБАВЛЕНИЕ СТАТЕЙ
    # ============================================

    def add_articles(self, new_articles: List[Dict], ollama_pipeline=None) -> int:
        """
        Инкрементальное добавление новых статей
        """
        if ollama_pipeline:
            self.ollama = ollama_pipeline

        if self.ollama is None:
            raise ValueError("Ollama pipeline не передан")

        existing_files = self.get_article_filenames()

        truly_new = [
            a for a in new_articles
            if a.get('filename', '') not in existing_files
        ]

        if not truly_new:
            print(f"  ✓ Все {len(new_articles)} статей уже в БД")
            return 0

        skipped = len(new_articles) - len(truly_new)
        print(f"\n  Добавление статей:")
        print(f"    Новых: {len(truly_new)}")
        if skipped > 0:
            print(f"    Уже в БД: {skipped}")

        existing_ids = set(m.get('article_id', -1) for m in self.metadata)
        next_id = max(existing_ids) + 1 if existing_ids else 0

        new_chunks = []
        for i, article in enumerate(truly_new):
            cleaned_text = self._clean_text(article['text'])

            article_meta = {
                'article_id': next_id + i,
                'filename': article.get('filename', f'new_{next_id + i}'),
                'title': article.get('metadata', {}).get('title', '')[:100],
                'language': article.get('language', 'unknown')
            }
            chunks = self._chunk_text(cleaned_text, article_meta)
            new_chunks.extend(chunks)

        if not new_chunks:
            return 0

        print(f"    Новых чанков: {len(new_chunks)}")
        print(f"    Создание эмбеддингов...")

        new_texts = [c['text'] for c in new_chunks]
        new_embeddings = self._embed_batch(new_texts, batch_size=10)

        self.index.add(new_embeddings.astype(np.float32))

        self.chunks.extend([c['text'] for c in new_chunks])
        self.metadata.extend([c['metadata'] for c in new_chunks])

        unique_files = set(m.get('filename', '') for m in self.metadata)
        self.stats['total_articles'] += len(truly_new)
        self.stats['total_chunks'] += len(new_chunks)
        self.stats['unique_files'] = len(unique_files)

        print(f"    ✓ Добавлено: {len(truly_new)} статей, {len(new_chunks)} чанков")
        print(f"    Всего в БД: {self.stats['total_articles']} статей, {self.stats['total_chunks']} чанков")

        return len(truly_new)

    # ============================================
    # ПРОВЕРКА СТАТЕЙ
    # ============================================

    def get_article_filenames(self) -> Set[str]:
        """Множество имен файлов в БД"""
        if not self.metadata:
            return set()
        return set(m.get('filename', '') for m in self.metadata if m.get('filename'))

    def get_article_ids(self) -> Set[int]:
        """Множество ID статей"""
        if not self.metadata:
            return set()
        return set(m.get('article_id', -1) for m in self.metadata if 'article_id' in m)

    def has_article(self, filename: str) -> bool:
        """Проверка наличия статьи"""
        return filename in self.get_article_filenames()

    def get_articles_summary(self) -> List[Dict]:
        """Сводка по статьям"""
        summary = {}
        for meta in self.metadata:
            fid = meta.get('article_id', -1)
            if fid not in summary:
                summary[fid] = {
                    'article_id': fid,
                    'filename': meta.get('filename', 'unknown'),
                    'title': meta.get('title', ''),
                    'language': meta.get('language', 'unknown'),
                    'chunks_count': 0
                }
            summary[fid]['chunks_count'] += 1
        return list(summary.values())

    def compare_with_folder(self, folder_path: str) -> Dict:
        """Сравнение статей в БД с файлами в папке"""
        from pathlib import Path
        folder_files = set(f.name for f in Path(folder_path).glob("*.pdf"))
        db_files = self.get_article_filenames()

        return {
            'in_both': folder_files & db_files,
            'only_in_folder': folder_files - db_files,
            'only_in_db': db_files - folder_files,
            'total_in_folder': len(folder_files),
            'total_in_db': len(db_files)
        }

    # ============================================
    # ПОИСК
    # ============================================

    def search(self, query: str, k: int = TOP_K_CHUNKS) -> List[Dict]:
        """Поиск k ближайших чанков"""
        if self.index is None:
            return []

        # Очищаем запрос
        query = self._clean_text(query)

        if not query or len(query) < 5:
            return []

        query_emb = self._embed_batch([query])
        distances, indices = self.index.search(query_emb, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            results.append({
                'text': self.chunks[idx],
                'metadata': self.metadata[idx],
                'score': float(dist)
            })

        return results

    def search_by_article(self, article_text: str, k: int = TOP_K_CHUNKS) -> List[Dict]:
        """Поиск чанков внутри той же статьи"""
        query = ' '.join(article_text.split()[:100])
        return self.search(query, k=k)

    # ============================================
    # СОХРАНЕНИЕ И ЗАГРУЗКА
    # ============================================

    def save(self, path: str = None):
        """Сохранение БД"""
        if path is None:
            path = DATA_DIR / "vector_db"

        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        data = {
            'chunks': self.chunks,
            'metadata': self.metadata,
            'dim': self.dim,
            'stats': self.stats
        }
        with open(os.path.join(path, "data.pkl"), "wb") as f:
            pickle.dump(data, f)

        import json
        with open(os.path.join(path, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2, default=str)

        index_size = os.path.getsize(os.path.join(path, "index.faiss")) / 1024 / 1024
        data_size = os.path.getsize(os.path.join(path, "data.pkl")) / 1024 / 1024

        print(f"  ✓ БД сохранена в {path}")
        print(f"    index.faiss: {index_size:.1f} MB")
        print(f"    data.pkl: {data_size:.1f} MB")

    def load(self, path: str = None):
        """Загрузка БД"""
        if path is None:
            path = DATA_DIR / "vector_db"

        index_path = os.path.join(path, "index.faiss")
        data_path = os.path.join(path, "data.pkl")

        if not os.path.exists(index_path) or not os.path.exists(data_path):
            raise FileNotFoundError(f"БД не найдена в {path}")

        self.index = faiss.read_index(index_path)

        with open(data_path, "rb") as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']
            self.dim = data['dim']
            self.stats = data.get('stats', {})

        print(f"  ✓ БД загружена из {path}")
        print(f"    Статей: {self.stats.get('total_articles', '?')}")
        print(f"    Чанков: {len(self.chunks)}")

    # ============================================
    # СТАТИСТИКА
    # ============================================

    def _print_stats(self):
        """Вывод статистики"""
        print(f"\n  📊 Статистика БД:")
        print(f"    Статей: {self.stats.get('total_articles', 0)}")
        print(f"    Уникальных файлов: {self.stats.get('unique_files', 0)}")
        print(f"    Чанков: {self.stats['total_chunks']}")
        print(f"    Средний размер чанка: {self.stats['avg_chunk_size']:.0f} симв.")
        print(f"    Размерность: {self.stats.get('embedding_dim', '?')}")
        print(f"    Размер: {self.stats.get('db_size_mb', 0):.1f} MB")
        print(f"    Тип индекса: {self.stats.get('index_type', 'unknown')}")
        print(f"    Метод: {self.stats.get('embedding_method', 'ollama')}")

    def get_stats(self) -> Dict:
        """Полная статистика"""
        return {
            **self.stats,
            'unique_files': len(self.get_article_filenames()),
            'unique_article_ids': len(self.get_article_ids())
        }

    def clear(self):
        """Очистка БД"""
        self.index = None
        self.chunks = []
        self.metadata = []
        self.dim = None
        self.stats = {
            'total_articles': 0,
            'total_chunks': 0,
            'avg_chunk_size': 0,
            'embedding_method': 'ollama',
            'index_type': 'FlatIP',
            'db_size_mb': 0
        }
        print("  ✓ БД очищена")