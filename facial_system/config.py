from pathlib import Path
import logging

# Diretório base do projeto (pasta onde está este arquivo)
BASE_DIR = Path(__file__).resolve().parent

# Diretórios principais
DATABASE_DIR = BASE_DIR / "database"
IMAGES_DIR = BASE_DIR / "images"
UI_DIR = BASE_DIR / "ui"

# Arquivos de banco de dados
USERS_JSON = DATABASE_DIR / "users.json"
EMBEDDINGS_PKL = DATABASE_DIR / "embeddings.pkl"
SQLITE_DB_PATH = DATABASE_DIR / "facepro.db"
CLASSIFIER_PKL = DATABASE_DIR / "face_classifier.pkl"
# Histórico de acessos reconhecidos
ACCESS_LOG_CSV = DATABASE_DIR / "access_log.csv"

# Arquivo de log
LOG_FILE = BASE_DIR / "facial_system.log"


def ensure_directories() -> None:
    """Garante que a estrutura mínima de diretórios exista."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    (UI_DIR / "assets").mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    """Configura o logging básico da aplicação."""
    ensure_directories()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def init_environment() -> None:
    """Inicializa diretórios e logging. Deve ser chamado no início da aplicação."""
    ensure_directories()
    setup_logging()



