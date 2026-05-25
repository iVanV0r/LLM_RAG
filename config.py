import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Базовые пути
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
PDF_RU_DIR = PDF_DIR / "ru"
PDF_EN_DIR = PDF_DIR / "en"
OUTPUT_DIR = DATA_DIR / "output"
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = DATA_DIR / "cache"  # НОВОЕ: директория для кэша

for dir_path in [
    DATA_DIR, PDF_DIR, PDF_RU_DIR, PDF_EN_DIR,
    OUTPUT_DIR, LOG_DIR, CACHE_DIR
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# Ollama настройки
# ============================================
OLLAMA_BASE_URL = "http://localhost:11434"

# Список моделей для сравнения
LLM_MODELS_TO_COMPARE = [
    "gemma-3-12b-it-iq2xs:latest",
    "mistral:7b",
    "qwen2.5:7b"

]

# Текущая модель (для обратной совместимости)
LLM_MODELS = {
    "ru": LLM_MODELS_TO_COMPARE[0] if LLM_MODELS_TO_COMPARE else "gemma-3-12b-it-iq2xs:latest",
    "en": LLM_MODELS_TO_COMPARE[0] if LLM_MODELS_TO_COMPARE else "gemma-3-12b-it-iq2xs:latest",
}

# Режим сравнения моделей
MODE_COMPARE_MODELS = True  # True = прогон на всех моделях, False = только одна модель

EMBED_MODEL = "nomic-embed-text:latest"

# ============================================
# Кэширование
# ============================================
ENABLE_CACHE = True           # Включить кэширование
CACHE_PDF_TEXT = True         # Кэшировать извлеченный текст PDF
CACHE_EMBEDDINGS = True       # Кэшировать эмбеддинги
CACHE_LLM_RESPONSES = True    # Кэшировать ответы LLM
CACHE_METRICS = True          # Кэшировать метрики

# ============================================
# Параметры чанкования
# ============================================
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100

# ============================================
# Параметры поиска
# ============================================
TOP_K_CHUNKS = 3

# ============================================
# Параметры генерации
# ============================================
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 300

# ============================================
# Режимы работы
# ============================================
MODE_CREATE_DB = True
MODE_EVALUATE = True
MODE_VISUALIZE = True
MODE_INTERACTIVE = False
MODE_COMPARE_LANGUAGES = True

# ============================================
# Пропуск уже выполненных шагов
# ============================================
SKIP_IF_DB_EXISTS = True      # Пропустить создание БД если уже есть
SKIP_IF_EVALUATED = True      # Пропустить оценку если уже есть результаты
SKIP_IF_CHARTS_EXIST = True   # Пропустить графики если уже есть

# ============================================
# Тестовый сплит
# ============================================
TEST_SPLIT = 0.2
RANDOM_SEED = 42

# ============================================
# Определение языка
# ============================================
RU_THRESHOLD = 0.3

EVAL_SAMPLE_SIZE = 15