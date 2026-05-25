import ollama
import time
import re
from typing import Dict, List, Tuple
from config import *
from cache_manager import cache


class PromptTemplates:
    """Шаблоны промптов для разных языков"""

    SYSTEM_PROMPTS = {
        "ru": """Ты - эксперт по анализу научных статей на русском языке.
Твоя задача - извлечь основную научную проблему из текста.
Отвечай кратко, только формулировкой проблемы (1-2 предложения).
Не используй вводные фразы типа "В данной статье рассматривается..." или "Авторы исследуют...".
Формулируй проблему как научный вопрос или утверждение о нерешенной задаче.""",

        "en": """You are an expert in analyzing scientific papers in English.
Your task is to extract the main scientific problem from the text.
Answer concisely, only with the problem statement (1-2 sentences).
Do not use introductory phrases like "This paper addresses..." or "The authors investigate...".
Formulate the problem as a scientific question or statement about an unsolved challenge."""
    }

    EXTRACTION_PROMPTS = {
        "ru": {
            "base": """Прочитай текст научной статьи и сформулируй научную проблему, 
которую она решает.

Текст статьи:
{article_text}

Научная проблема:""",

            "rag": """Используя контекст из статьи, определи научную проблему.

Контекст (ключевые фрагменты):
{context}

Дополнительный текст статьи:
{article_text}

Сформулируй научную проблему:"""
        },

        "en": {
            "base": """Read the scientific paper text and formulate the scientific problem 
it addresses.

Paper text:
{article_text}

Scientific problem:""",

            "rag": """Using the context from the paper, identify the scientific problem.

Context (key excerpts):
{context}

Additional paper text:
{article_text}

Formulate the scientific problem:"""
        }
    }

    @classmethod
    def get_prompt(cls, prompt_type: str, language: str, **kwargs) -> Tuple[str, str]:
        """Возвращает system_prompt и user_prompt"""
        if language not in cls.SYSTEM_PROMPTS:
            language = "en"

        system_prompt = cls.SYSTEM_PROMPTS[language]

        if prompt_type in cls.EXTRACTION_PROMPTS[language]:
            user_prompt = cls.EXTRACTION_PROMPTS[language][prompt_type].format(**kwargs)
        else:
            user_prompt = kwargs.get('text', '')

        return system_prompt, user_prompt


class OllamaPipeline:
    """Мультиязычный пайплайн с Ollama + кэшированием"""

    def __init__(self, vector_db_ru=None, vector_db_en=None):
        self.vector_dbs = {
            "ru": vector_db_ru,
            "en": vector_db_en
        }

        self.models = LLM_MODELS.copy()
        self.embed_model = EMBED_MODEL

        # Кэш в памяти для эмбеддингов
        self._memory_cache = {}
        self._fallback_embedder = None

        # Проверка и автоисправление моделей
        self._check_models()

        # Статистика
        self.stats = {
            "ru": self._init_stats(),
            "en": self._init_stats()
        }

    def _init_stats(self) -> Dict:
        """Начальная статистика"""
        return {
            'total_queries': 0,
            'total_time': 0,
            'avg_time': 0,
            'total_tokens': 0,
            'cache_hits': 0
        }

    # ============================================
    # ОЧИСТКА ТЕКСТА
    # ============================================

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Очистка текста от проблемных Unicode символов
        """
        if not text:
            return ""

        # Удаляем surrogate символы
        text = text.encode('utf-8', errors='replace').decode('utf-8')

        # Оставляем только разрешенные символы
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

        # Заменяем множественные пробелы
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    # ============================================
    # ПРОВЕРКА МОДЕЛЕЙ
    # ============================================

    def _check_models(self):
        """Проверка наличия моделей Ollama"""
        print("Проверка моделей Ollama...")

        try:
            result = ollama.list()

            # Извлекаем имена моделей
            model_names = []
            if hasattr(result, 'models'):
                for m in result.models:
                    name = m.model if hasattr(m, 'model') else str(m)
                    model_names.append(name)
            elif isinstance(result, dict):
                for m in result.get('models', []):
                    if isinstance(m, dict):
                        name = m.get('name', m.get('model', ''))
                    else:
                        name = str(m)
                    model_names.append(name)

            print(f"  Доступно моделей: {len(model_names)}")
            for name in model_names:
                print(f"    - {name}")

            # Проверяем LLM модели
            for lang, model in self.models.items():
                # Ищем точное совпадение или с суффиксом :latest
                found = None
                for available in model_names:
                    if available == model or available == f"{model}:latest" or available.startswith(
                            model.split(':')[0]):
                        found = available
                        break

                if found:
                    self.models[lang] = found  # Используем точное имя
                    print(f"  ✓ {lang.upper()}: {found}")
                else:
                    print(f"  ⚠ {lang.upper()}: {model} не найдена")

                    # Ищем любую похожую модель
                    base = model.split(':')[0].lower()
                    for available in model_names:
                        if base in available.lower():
                            self.models[lang] = available
                            print(f"    → использую {available}")
                            found = available
                            break

                    if not found:
                        print(f"    Скачайте: ollama pull {model}")

            # Проверяем эмбеддер
            embed_found = None
            for available in model_names:
                if self.embed_model in available or 'embed' in available.lower():
                    embed_found = available
                    break

            if embed_found:
                self.embed_model = embed_found
                print(f"  ✓ Эмбеддер: {embed_found}")
            else:
                print(f"  ⚠ Эмбеддер: {self.embed_model} не найден")
                print(f"    Скачайте: ollama pull {self.embed_model}")

        except Exception as e:
            print(f"  ⚠ Ошибка проверки моделей: {e}")
            print(f"  Использую текущие настройки:")
            for lang, model in self.models.items():
                print(f"    {lang.upper()}: {model}")
            print(f"    Эмбеддер: {self.embed_model}")

    # ============================================
    # УПРАВЛЕНИЕ БД
    # ============================================

    def set_vector_db(self, db, language: str):
        """Установка векторной БД для языка"""
        self.vector_dbs[language] = db

    def get_model_for_language(self, language: str) -> str:
        """Возвращает модель для указанного языка"""
        return self.models.get(language, self.models.get("en", "gemma2:9b"))

    # ============================================
    # ЭМБЕДДИНГИ
    # ============================================

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Получение эмбеддингов через Ollama с очисткой текста и кэшированием
        """
        embeddings = []

        for text in texts:
            # Очищаем текст
            text = self._clean_text(text)

            if not text or len(text) < 5:
                embeddings.append([0.0] * 384)
                continue

            # Проверяем кэш в памяти
            text_hash = str(hash(text))
            if text_hash in self._memory_cache:
                embeddings.append(self._memory_cache[text_hash])
                continue

            # Проверяем файловый кэш
            if CACHE_EMBEDDINGS:
                cached = cache.get_embeddings(text_hash)
                if cached is not None:
                    self._memory_cache[text_hash] = cached
                    embeddings.append(cached)
                    continue

            # Запрашиваем у Ollama
            try:
                response = ollama.embeddings(
                    model=self.embed_model,
                    prompt=text
                )
                emb = response['embedding']

                # Сохраняем в кэш
                self._memory_cache[text_hash] = emb
                if CACHE_EMBEDDINGS:
                    cache.set_embeddings(text_hash, emb)

                embeddings.append(emb)

            except Exception as e:
                # Fallback на sentence-transformers
                try:
                    if self._fallback_embedder is None:
                        from sentence_transformers import SentenceTransformer
                        self._fallback_embedder = SentenceTransformer('intfloat/multilingual-e5-large')

                    emb = self._fallback_embedder.encode([text], normalize_embeddings=True)[0].tolist()

                    self._memory_cache[text_hash] = emb
                    embeddings.append(emb)

                except Exception as e2:
                    embeddings.append([0.0] * 384)

        return embeddings

    # ============================================
    # ГЕНЕРАЦИЯ
    # ============================================

    def generate(self, prompt: str, language: str, system_prompt: str = None,
                 temperature: float = LLM_TEMPERATURE) -> Dict:
        """Генерация с правильным измерением времени"""
        model = self.get_model_for_language(language)
        full_prompt = f"{system_prompt or ''}\n{prompt}"

        # Кэш
        if CACHE_LLM_RESPONSES:
            cached = cache.get_llm_response(full_prompt, model)
            if cached is not None:
                if language in self.stats:
                    self.stats[language]['cache_hits'] += 1
                cached['from_cache'] = True
                cached['model_loading_time'] = 0
                return cached

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            # ============================================
            # ЗАМЕР ТОЛЬКО ВРЕМЕНИ ГЕНЕРАЦИИ
            # ============================================

            # Отправляем запрос (модель уже должна быть загружена)
            start_time = time.time()

            response = ollama.chat(
                model=model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": LLM_MAX_TOKENS,
                    "top_p": 0.9,
                    "num_gpu_layers": 5
                }
            )

            elapsed = time.time() - start_time

            # Из ответа Ollama берем метрики
            eval_count = response.get('eval_count', 0)
            eval_duration = response.get('eval_duration', 0)  # наносекунды
            load_duration = response.get('load_duration', 0)  # время загрузки модели

            # Реальное время генерации (без загрузки)
            if eval_duration > 0:
                generation_time = eval_duration / 1e9  # наносекунды → секунды
            else:
                generation_time = elapsed

            # Время загрузки модели
            loading_time = load_duration / 1e9 if load_duration > 0 else 0

            generated_text = response['message']['content'].strip()

            # Статистика
            if language in self.stats:
                self.stats[language]['total_queries'] += 1
                self.stats[language]['total_time'] += generation_time
                self.stats[language]['avg_time'] = (
                        self.stats[language]['total_time'] /
                        max(self.stats[language]['total_queries'], 1)
                )
                self.stats[language]['total_tokens'] += eval_count

            result = {
                'text': generated_text,
                'time': round(generation_time, 2),  # Только генерация
                'loading_time': round(loading_time, 2),  # Загрузка модели
                'total_time': round(elapsed, 2),  # Общее время
                'tokens_generated': eval_count,
                'tokens_per_second': eval_count / generation_time if generation_time > 0 else 0,
                'model_used': model,
                'language': language,
                'from_cache': False
            }

            # Кэш
            if CACHE_LLM_RESPONSES:
                cache.set_llm_response(full_prompt, model, result)

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            return {
                'text': f"ERROR: {str(e)[:100]}",
                'time': 0,
                'loading_time': 0,
                'total_time': elapsed,
                'tokens_generated': 0,
                'model_used': model,
                'language': language,
                'error': True,
                'from_cache': False
            }

    # ============================================
    # ИЗВЛЕЧЕНИЕ ПРОБЛЕМ
    # ============================================

    def extract_problem_base(self, article_text: str, language: str = None) -> Dict:
        """
        Извлечение проблемы БЕЗ RAG (базовый метод)
        """
        if language is None:
            language = self._detect_language(article_text)

        # Очищаем текст
        article_text = self._clean_text(article_text)

        system_prompt, user_prompt = PromptTemplates.get_prompt(
            "base", language, article_text=article_text[:4000]
        )

        result = self.generate(user_prompt, language, system_prompt)
        result['method'] = 'base'
        return result

    def extract_problem_rag(self, article_text: str, language: str = None) -> Dict:
        """
        Извлечение проблемы С RAG (с поиском по БД)
        """
        if language is None:
            language = self._detect_language(article_text)

        # Исправление mixed языка
        if language not in ["ru", "en"]:
            language = "ru" if self._detect_language(article_text) == "ru" else "en"

        # Очищаем текст
        article_text = self._clean_text(article_text)

        # Получаем векторную БД
        db = self.vector_dbs.get(language)
        if db is None:
            print(f"  ⚠ Нет БД для языка {language}, использую base метод")
            return self.extract_problem_base(article_text, language)

        # Формируем поисковый запрос (первые 100 слов)
        query = ' '.join(article_text.split()[:100])

        # Ищем релевантные чанки
        search_results = db.search(query, k=TOP_K_CHUNKS)

        if not search_results:
            print(f"  ⚠ Ничего не найдено в БД, использую base метод")
            return self.extract_problem_base(article_text, language)

        # Формируем контекст
        context_parts = []
        for i, res in enumerate(search_results):
            score = res.get('score', 0)
            context_parts.append(f"[Фрагмент {i + 1} (score: {score:.2f})]\n{res['text']}")
        context = '\n\n'.join(context_parts)

        # Промпт для RAG
        system_prompt, user_prompt = PromptTemplates.get_prompt(
            "rag", language,
            context=context,
            article_text=article_text[:2000]
        )

        result = self.generate(user_prompt, language, system_prompt)
        result['method'] = 'rag'
        result['chunks_used'] = len(search_results)

        scores = [r.get('score', 0) for r in search_results]
        result['avg_chunk_score'] = sum(scores) / len(scores) if scores else 0

        return result

    def extract_problem_hybrid(self, article_text: str, language: str = None) -> Dict:
        """
        Гибридный метод: RAG + агрегация нескольких чанков
        """
        if language is None:
            language = self._detect_language(article_text)

        if language not in ["ru", "en"]:
            language = "ru" if self._detect_language(article_text) == "ru" else "en"

        # Очищаем текст
        article_text = self._clean_text(article_text)

        db = self.vector_dbs.get(language)
        if db is None:
            return self.extract_problem_base(article_text, language)

        # Поиск чанков
        query = ' '.join(article_text.split()[:100])
        search_results = db.search(query, k=TOP_K_CHUNKS)

        if not search_results:
            return self.extract_problem_rag(article_text, language)

        # Извлекаем проблемы из топ-3 чанков
        chunk_problems = []
        for chunk in search_results[:3]:
            if language == "ru":
                prompt = f"""Прочитай фрагмент статьи. Если в нем описывается научная проблема,
извлеки её. Если нет - ответь "НЕТ ПРОБЛЕМЫ".

Фрагмент:
{chunk['text']}

Ответ:"""
            else:
                prompt = f"""Read this excerpt. If it describes a scientific problem,
extract it. If not, answer "NO PROBLEM".

Excerpt:
{chunk['text']}

Response:"""

            result = self.generate(prompt, language, temperature=0.0)
            problem = result['text'].strip()

            no_problem = ["НЕТ ПРОБЛЕМЫ", "нет проблемы", "NO PROBLEM", "no problem"]
            if problem and not any(np in problem for np in no_problem):
                chunk_problems.append(problem)

        if not chunk_problems:
            return self.extract_problem_rag(article_text, language)

        # Агрегируем
        problems_list = '\n'.join(f"{i + 1}. {p}" for i, p in enumerate(chunk_problems))

        if language == "ru":
            agg_prompt = f"""На основе анализа разных частей статьи найдены следующие 
формулировки проблемы:

{problems_list}

Объедини их в одну общую формулировку научной проблемы:"""
        else:
            agg_prompt = f"""Based on analysis of different parts of the paper,
the following problem formulations were found:

{problems_list}

Combine them into one overall scientific problem statement:"""

        result = self.generate(agg_prompt, language)
        result['method'] = 'hybrid_rag'
        result['sub_problems_found'] = len(chunk_problems)

        return result

    # ============================================
    # ОПРЕДЕЛЕНИЕ ЯЗЫКА
    # ============================================

    def _detect_language(self, text: str) -> str:
        """
        Быстрое определение языка текста
        """
        if not text:
            return "en"

        cyrillic = len(re.findall(r'[а-яёА-ЯЁ]', text))
        latin = len(re.findall(r'[a-zA-Z]', text))
        total = cyrillic + latin

        if total == 0:
            return "en"

        return "ru" if cyrillic / total > RU_THRESHOLD else "en"

    # ============================================
    # СТАТИСТИКА
    # ============================================

    def get_stats(self) -> Dict:
        """Полная статистика"""
        return {
            "ru": self.stats["ru"],
            "en": self.stats["en"],
            "models_used": self.models,
            "memory_cache_size": len(self._memory_cache)
        }

    def get_comparative_stats(self) -> Dict:
        """Статистика для сравнения языков"""
        return {
            "ru": self.stats["ru"],
            "en": self.stats["en"],
            "models_used": self.models
        }

    def clear_cache(self):
        """Очистка кэша в памяти"""
        self._memory_cache.clear()
        print("  ✓ Кэш в памяти очищен")

    def print_stats(self):
        """Вывод статистики"""
        stats = self.get_stats()

        print(f"\n  📊 Статистика Ollama:")
        for lang in ['ru', 'en']:
            s = stats[lang]
            cache_hits = s.get('cache_hits', 0)
            total = s['total_queries'] + cache_hits

            if total > 0:
                print(f"    {lang.upper()}:")
                print(f"      Запросов: {s['total_queries']}")
                print(f"      Из кэша: {cache_hits}")
                print(f"      Среднее время: {s['avg_time']:.2f}с")
                print(f"      Всего токенов: {s['total_tokens']}")

        print(f"    Модели: {self.models}")
        print(f"    Кэш в памяти: {len(self._memory_cache)} записей")