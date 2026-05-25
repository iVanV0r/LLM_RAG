import re
import pdfplumber
import PyPDF2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from config import PDF_RU_DIR, PDF_EN_DIR, CACHE_PDF_TEXT, RU_THRESHOLD
from cache_manager import cache


class LanguageDetector:
    """Определение языка текста"""

    @staticmethod
    def detect(text: str) -> str:
        """Быстрое определение языка (ru/en)"""
        if not text or len(text) < 100:
            return "en"

        cyrillic = len(re.findall(r'[а-яёА-ЯЁ]', text))
        latin = len(re.findall(r'[a-zA-Z]', text))
        total = cyrillic + latin

        if total == 0:
            return "en"

        return "ru" if cyrillic / total > RU_THRESHOLD else "en"

    @staticmethod
    def detect_with_confidence(text: str) -> Dict:
        """Определение языка с уверенностью"""
        cyrillic = len(re.findall(r'[а-яёА-ЯЁ]', text))
        latin = len(re.findall(r'[a-zA-Z]', text))
        total = cyrillic + latin

        if total == 0:
            return {"language": "en", "confidence": 0.5}

        cyrillic_ratio = cyrillic / total

        if cyrillic_ratio > 0.7:
            return {"language": "ru", "confidence": min(cyrillic_ratio, 0.95)}
        elif cyrillic_ratio < 0.3:
            return {"language": "en", "confidence": min(1 - cyrillic_ratio, 0.95)}
        else:
            return {"language": "mixed", "confidence": 0.5}


class PDFLoader:
    """Загрузчик PDF с автоопределением языка, очисткой текста и кэшированием"""

    def __init__(self):
        self.lang_detector = LanguageDetector()
        self.stats = {"ru": 0, "en": 0, "failed": 0, "cached": 0}

    # ============================================
    # ОЧИСТКА ТЕКСТА
    # ============================================

    @staticmethod
    def _clean_unicode(text: str) -> str:
        """
        Удаление проблемных Unicode символов (surrogates, математические символы и т.д.)
        """
        if not text:
            return ""

        # 1. Удаляем Unicode surrogate символы (0xD800-0xDFFF)
        text = text.encode('utf-8', errors='replace').decode('utf-8')

        # 2. Заменяем проблемные символы на пробелы
        # Оставляем только:
        # - ASCII (0x00-0x7F): базовый английский
        # - Кириллица (0x400-0x4FF): русский алфавит
        # - Доп. кириллица (0x500-0x52F)
        # - Latin-1 Supplement (0x80-0xFF): буквы с диакритикой
        # - Latin Extended-A (0x100-0x17F)
        # - Latin Extended-B (0x180-0x24F)
        # - Греческий (0x370-0x3FF): для формул α, β, γ
        # - Пробелы, знаки препинания, цифры, математические операторы
        allowed = (
            r'[^\x00-\x7F'  # ASCII
            r'\u0400-\u04FF'  # Кириллица
            r'\u0500-\u052F'  # Доп. кириллица
            r'\u0080-\u00FF'  # Latin-1
            r'\u0100-\u017F'  # Latin Extended-A
            r'\u0180-\u024F'  # Latin Extended-B
            r'\u0370-\u03FF'  # Греческий
            r'\s\.\,\!\?\-\:\;\"\'\(\)\[\]\{\}\d'  # Базовые знаки
            r'\*\#\%\&\+\=\/\\\@\$\€\£\¥\°\№'  # Спецсимволы
            r'\u2200-\u22FF'  # Математические операторы
            r'\u2202\u2207\u221E\u222B'  # ∂ ∇ ∞ ∫
            r']'
        )
        text = re.sub(allowed, ' ', text)

        # 3. Заменяем множественные пробелы на один
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    @staticmethod
    def clean_text(text: str, language: str) -> str:
        """
        Полная очистка текста научной статьи
        """
        # Сначала чистим Unicode
        text = PDFLoader._clean_unicode(text)

        if not text:
            return ""

        # Удаляем номера страниц
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        # Удаляем ссылки в квадратных скобках
        text = re.sub(r'\[\d+(?:[,-]\d+)*\]', '', text)

        # Удаляем колонтитулы
        if language == "ru":
            text = re.sub(r'(?:РИС\.|ТАБЛИЦА|Рис\.|Таблица)\s*\d+[\.\:]?\s*', '', text, flags=re.IGNORECASE)
        else:
            text = re.sub(r'(?:FIG\.|TABLE|Fig\.|Table)\s*\d+[\.\:]?\s*', '', text, flags=re.IGNORECASE)

        # Удаляем URL
        text = re.sub(r'https?://\S+', '', text)

        # Удаляем email
        text = re.sub(r'\S+@\S+', '', text)

        # Удаляем пустые строки и строки из одних цифр
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            # Пропускаем пустые строки
            if not line:
                continue
            # Пропускаем строки из одних цифр/спецсимволов
            if re.match(r'^[\d\s\.\,\-\+\=*#@]+$', line):
                continue
            # Пропускаем слишком короткие строки (кроме заголовков)
            if len(line) < 5 and not line.isupper():
                continue
            lines.append(line)

        text = '\n'.join(lines)

        # Финальная очистка пробелов
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    # ============================================
    # ИЗВЛЕЧЕНИЕ ТЕКСТА ИЗ PDF
    # ============================================

    @staticmethod
    def extract_text_pdfplumber(pdf_path: str) -> str:
        """Извлечение текста через pdfplumber"""
        import warnings
        warnings.filterwarnings('ignore')

        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text and len(page_text.strip()) > 10:
                            # Убираем переносы слов
                            page_text = re.sub(r'-\n', '', page_text)
                            page_text = re.sub(r'\s+', ' ', page_text)
                            text_parts.append(page_text)
                    except Exception:
                        # Пробуем извлечь по словам
                        try:
                            words = page.extract_words()
                            if words:
                                page_text = ' '.join(w.get('text', '') for w in words)
                                if page_text.strip():
                                    text_parts.append(page_text)
                        except:
                            continue

                return '\n\n'.join(text_parts) if text_parts else ""
        except Exception as e:
            return ""

    @staticmethod
    def extract_text_pypdf2(pdf_path: str) -> str:
        """Извлечение текста через PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text_parts = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 10:
                        page_text = re.sub(r'\s+', ' ', page_text)
                        text_parts.append(page_text)
                return '\n\n'.join(text_parts) if text_parts else ""
        except Exception:
            return ""

    @staticmethod
    def extract_text_pymupdf(pdf_path: str) -> str:
        """Извлечение текста через PyMuPDF (самый надежный)"""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            text_parts = []

            for page in doc:
                text = page.get_text()
                if text and len(text.strip()) > 10:
                    text = re.sub(r'-\n', '', text)
                    text = re.sub(r'\s+', ' ', text)
                    text_parts.append(text)

            doc.close()
            return '\n\n'.join(text_parts) if text_parts else ""
        except ImportError:
            return ""
        except Exception:
            return ""

    # ============================================
    # ЗАГРУЗКА PDF
    # ============================================

    def load_pdf(self, pdf_path: str) -> Dict:
        """
        Загружает PDF с определением языка, очисткой и кэшированием

        Returns:
            Dict с ключами: text, filename, language, length, metadata
        """
        # Проверка кэша
        if CACHE_PDF_TEXT:
            cached = cache.get_pdf_text(pdf_path)
            if cached:
                self.stats['cached'] += 1
                print(f"  📦 Из кэша: {Path(pdf_path).name}")
                return cached

        # Извлечение текста (пробуем 3 метода)
        text = ""

        # Метод 1: PyMuPDF (самый надежный)
        text = self.extract_text_pymupdf(pdf_path)

        # Метод 2: PyPDF2
        if not text or len(text) < 100:
            text = self.extract_text_pypdf2(pdf_path)

        # Метод 3: pdfplumber
        if not text or len(text) < 100:
            text = self.extract_text_pdfplumber(pdf_path)

        if not text or len(text) < 100:
            raise ValueError(f"Не удалось извлечь текст из {pdf_path}")

        # Определение языка
        lang_info = self.lang_detector.detect_with_confidence(text)
        language = lang_info['language']

        # Исправление mixed языка
        if language == "mixed":
            cyrillic = len(re.findall(r'[а-яёА-ЯЁ]', text))
            latin = len(re.findall(r'[a-zA-Z]', text))
            language = "ru" if cyrillic >= latin else "en"

        # Очистка текста
        text = self.clean_text(text, language)

        # Проверка после очистки
        if len(text) < 100:
            raise ValueError(f"Текст слишком короткий после очистки: {len(text)} симв.")

        # Извлечение метаданных
        metadata = self._extract_metadata(text, language, pdf_path)
        metadata['language'] = language
        metadata['language_confidence'] = lang_info['confidence']
        metadata['original_detected'] = lang_info['language']

        result = {
            'text': text,
            'filename': Path(pdf_path).name,
            'language': language,
            'length': len(text),
            'metadata': metadata
        }

        # Сохранение в кэш
        if CACHE_PDF_TEXT:
            cache.set_pdf_text(pdf_path, result)

        # Обновление статистики
        if language in self.stats:
            self.stats[language] += 1

        return result

    # ============================================
    # МЕТАДАННЫЕ
    # ============================================

    def _extract_metadata(self, text: str, language: str, pdf_path: str) -> Dict:
        """Извлечение метаданных: заголовок, аннотация, ключевые слова, секции"""
        metadata = {
            'filename': Path(pdf_path).name,
            'title': '',
            'abstract': '',
            'keywords': [],
            'sections': [],
            'language': language
        }

        lines = text.split('\n')

        # Заголовок
        for i, line in enumerate(lines[:20]):
            line = line.strip()
            if 20 < len(line) < 400 and not line.startswith('http'):
                if not re.match(r'^\d+$', line):
                    metadata['title'] = line
                    break

        # Аннотация
        abstract_markers = {
            'ru': ['аннотация', 'abstract', 'аннотация —', 'реферат'],
            'en': ['abstract', 'abstract —', 'summary', 'abstract.']
        }
        markers = abstract_markers.get(language, abstract_markers['en'])
        abstract_text = self._extract_section(text, markers, max_lines=30)
        metadata['abstract'] = abstract_text if abstract_text else text[:500]

        # Ключевые слова
        kw_patterns = {
            'ru': [
                r'(?:ключевые\s*слова|keywords)[\s:]*([^\n]+)',
                r'(?:ключевые\s*слова|keywords)[\s—:]+([^\n]+)'
            ],
            'en': [
                r'(?:keywords|key\s*words|index\s*terms)[\s:]*([^\n]+)',
                r'(?:keywords|key\s*words)[\s—:]+([^\n]+)'
            ]
        }
        patterns = kw_patterns.get(language, kw_patterns['en'])
        for pattern in patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE)
            if match:
                kw_text = match.group(1)
                keywords = re.split(r'[,;.]', kw_text)
                metadata['keywords'] = [k.strip() for k in keywords if len(k.strip()) > 2]
                if metadata['keywords']:
                    break

        # Секции
        section_markers = {
            'ru': [
                'введение', 'обзор литературы', 'метод', 'методология',
                'результаты', 'обсуждение', 'заключение', 'выводы',
                'постановка задачи', 'актуальность'
            ],
            'en': [
                'introduction', 'related work', 'method', 'methodology',
                'results', 'discussion', 'conclusion', 'references',
                'problem statement', 'background'
            ]
        }
        markers = section_markers.get(language, section_markers['en'])
        for marker in markers:
            pattern = re.compile(
                rf'(?:^|\n)\s*(?:\d+[\.\)]\s*)?{re.escape(marker)}',
                re.IGNORECASE
            )
            for match in pattern.finditer(text):
                metadata['sections'].append({
                    'name': marker,
                    'position': match.start()
                })

        return metadata

    def _extract_section(self, text: str, markers: List[str], max_lines: int = 30) -> Optional[str]:
        """Извлекает текст секции по маркерам"""
        lines = text.split('\n')

        stop_markers = ['introduction', 'введение', 'keywords', 'ключевые слова']

        for marker in markers:
            for i, line in enumerate(lines[:50]):
                if marker.lower() in line.lower():
                    section_lines = []
                    for j in range(i + 1, min(i + max_lines, len(lines))):
                        next_line = lines[j].strip()
                        if next_line and not any(m.lower() in next_line.lower() for m in stop_markers):
                            section_lines.append(next_line)
                        else:
                            break

                    if section_lines:
                        return ' '.join(section_lines)
        return None

    # ============================================
    # ЗАГРУЗКА ДИРЕКТОРИИ
    # ============================================

    def load_directory(self) -> Tuple[List[Dict], List[Dict]]:
        """Загружает все PDF из папок ru/ и en/"""
        ru_articles = []
        en_articles = []

        # Русские статьи
        if PDF_RU_DIR.exists():
            ru_files = list(PDF_RU_DIR.glob("*.pdf"))
            print(f"\n📄 Русские PDF: {len(ru_files)} файлов")

            for pdf_path in tqdm(ru_files, desc="Загрузка RU"):
                try:
                    article = self.load_pdf(str(pdf_path))
                    article['expected_language'] = 'ru'
                    article['language'] = 'ru'  # Доверяем папке
                    ru_articles.append(article)
                except Exception as e:
                    print(f"  ✗ {pdf_path.name}: {str(e)[:100]}")
                    self.stats['failed'] += 1

        # Английские статьи
        if PDF_EN_DIR.exists():
            en_files = list(PDF_EN_DIR.glob("*.pdf"))
            print(f"\n📄 Английские PDF: {len(en_files)} файлов")

            for pdf_path in tqdm(en_files, desc="Загрузка EN"):
                try:
                    article = self.load_pdf(str(pdf_path))
                    article['expected_language'] = 'en'
                    article['language'] = 'en'  # Доверяем папке
                    en_articles.append(article)
                except Exception as e:
                    print(f"  ✗ {pdf_path.name}: {str(e)[:100]}")
                    self.stats['failed'] += 1

        # Статистика
        print(f"\n📊 Статистика загрузки:")
        print(f"  Русских: {len(ru_articles)}")
        print(f"  Английских: {len(en_articles)}")
        print(f"  Из кэша: {self.stats.get('cached', 0)}")
        print(f"  Ошибок: {self.stats.get('failed', 0)}")

        return ru_articles, en_articles

    def _validate_languages(self, articles: List[Dict], expected_lang: str):
        """Проверяет соответствие языков"""
        if not articles:
            return

        mismatches = 0
        for article in articles:
            if article['language'] != expected_lang:
                mismatches += 1
                print(f"  ⚠ {article['filename']}: {article['language']} (ожидался {expected_lang})")

        if mismatches > 0:
            print(f"  ⚠ {mismatches} статей с несоответствием языка")

    def get_statistics(self) -> Dict:
        """Полная статистика загрузки"""
        return {
            'russian_articles': self.stats.get('ru', 0),
            'english_articles': self.stats.get('en', 0),
            'failed': self.stats.get('failed', 0),
            'cached': self.stats.get('cached', 0),
            'total': self.stats.get('ru', 0) + self.stats.get('en', 0)
        }

    def load_single(self, pdf_path: str) -> Dict:
        """Загрузка одного PDF (для интерактивного режима)"""
        path = Path(pdf_path)

        if not path.exists():
            for lang_dir in [PDF_RU_DIR, PDF_EN_DIR]:
                candidate = lang_dir / path.name
                if candidate.exists():
                    path = candidate
                    break

        if not path.exists():
            raise FileNotFoundError(f"PDF не найден: {pdf_path}")

        return self.load_pdf(str(path))