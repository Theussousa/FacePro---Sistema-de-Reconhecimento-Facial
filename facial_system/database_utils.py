import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any

from config import USERS_JSON, IMAGES_DIR, ensure_directories

logger = logging.getLogger(__name__)


def _safe_read_json(path: Path) -> Any:
    """Lê JSON de forma segura, retornando uma estrutura vazia em caso de erro."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error("Falha ao decodificar JSON em %s. Arquivo será ignorado.", path)
        return []
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro inesperado ao ler JSON em %s: %s", path, exc)
        return []


def _safe_write_json(path: Path, data: Any) -> None:
    """Escreve JSON de forma segura, criando diretórios se necessário."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro ao escrever JSON em %s: %s", path, exc)


def load_users() -> List[Dict[str, Any]]:
    """Carrega a lista de usuários do arquivo users.json."""
    ensure_directories()
    data = _safe_read_json(USERS_JSON)
    if isinstance(data, list):
        return data
    logger.warning("Estrutura inesperada em users.json; esperado list, obtido %s", type(data))
    return []


def save_users(users: List[Dict[str, Any]]) -> None:
    """Salva a lista de usuários em users.json."""
    ensure_directories()
    _safe_write_json(USERS_JSON, users)


def get_next_user_id(users: List[Dict[str, Any]]) -> int:
    """Retorna o próximo ID inteiro disponível baseado nos usuários existentes."""
    if not users:
        return 1
    try:
        max_id = max(int(u.get("id", 0)) for u in users)
    except ValueError:
        max_id = 0
    return max_id + 1


def register_user(name: str, cpf: str = "") -> int:
    """
    Registra um novo usuário no banco (sem capturar imagens).

    Retorna:
        int: ID do usuário criado.
    """
    users = load_users()
    user_id = get_next_user_id(users)
    user: Dict[str, Any] = {"id": user_id, "name": name}
    if cpf:
        user["cpf"] = cpf
    users.append(user)
    save_users(users)
    logger.info("Usuário registrado: id=%s, nome=%s, cpf=%s", user_id, name, cpf or "-")
    return user_id


def delete_user(user_id: int, delete_images: bool = True) -> bool:
    """
    Remove um usuário do banco e, opcionalmente, apaga suas imagens.

    Args:
        user_id: ID do usuário a ser removido.
        delete_images: Se True, apaga também a pasta images/<user_id>/.

    Returns:
        bool: True se algum usuário foi removido, False caso contrário.
    """
    users = load_users()
    remaining = [u for u in users if int(u.get("id", -1)) != int(user_id)]
    if len(remaining) == len(users):
        logger.warning("Nenhum usuário encontrado com id=%s para remoção.", user_id)
        return False

    save_users(remaining)
    logger.info("Usuário removido: id=%s", user_id)

    if delete_images:
        user_dir = IMAGES_DIR / str(user_id)
        if user_dir.exists():
            try:
                shutil.rmtree(user_dir)
                logger.info("Pasta de imagens removida: %s", user_dir)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Erro ao remover pasta de imagens %s: %s", user_dir, exc)

    return True



