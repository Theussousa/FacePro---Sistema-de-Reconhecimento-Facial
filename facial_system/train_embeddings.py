"""
Módulo responsável por gerar embeddings faciais a partir das imagens cadastradas.

Fluxo:
- Percorre todas as pastas em images/<user_id>/.
- Para cada imagem, gera um embedding usando DeepFace (modelo Facenet512).
- Salva a lista de embeddings em embeddings.pkl.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np  # type: ignore[import-untyped]
from deepface import DeepFace  # type: ignore[import-untyped]

from config import IMAGES_DIR, EMBEDDINGS_PKL, init_environment
from database_utils import load_users

logger = logging.getLogger(__name__)


def _collect_image_paths() -> List[Dict[str, Any]]:
    """Coleta caminhos de imagens para todos os usuários com diretórios válidos."""
    users = load_users()
    mapping: List[Dict[str, Any]] = []
    for user in users:
        uid = user.get("id")
        name = user.get("name")
        cpf = user.get("cpf")
        if uid is None:
            continue
        user_dir = IMAGES_DIR / str(uid)
        if not user_dir.exists() or not user_dir.is_dir():
            logger.warning("Diretório de imagens não encontrado para usuário id=%s", uid)
            continue
        for img_path in sorted(user_dir.glob("*.jpg")):
            mapping.append({"id": uid, "name": name, "cpf": cpf, "path": img_path})
    return mapping


def generate_embeddings(model_name: str = "Facenet512") -> int:
    """
    Gera embeddings faciais para todas as imagens cadastradas.

    Args:
        model_name: Nome do modelo DeepFace a ser utilizado (default: Facenet512).

    Returns:
        int: Quantidade de embeddings gerados.
    """
    init_environment()
    logger.info("Iniciando geração de embeddings com modelo '%s'.", model_name)

    images_info = _collect_image_paths()
    if not images_info:
        logger.warning("Nenhuma imagem encontrada para geração de embeddings.")
        return 0

    embeddings: List[Dict[str, Any]] = []

    for item in images_info:
        img_path: Path = item["path"]
        try:
            logger.info("Gerando embedding para imagem %s", img_path)
            reps = DeepFace.represent(
                img_path=str(img_path),
                model_name=model_name,
                enforce_detection=False,
            )
            if not reps:
                logger.warning("Nenhum embedding retornado para %s", img_path)
                continue

            embedding_vector = np.array(reps[0]["embedding"], dtype="float32")
            embeddings.append(
                {
                    "id": int(item["id"]),
                    "name": item["name"],
                    "cpf": item.get("cpf"),
                    "image": str(img_path),
                    "embedding": embedding_vector,
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro ao processar imagem %s: %s", img_path, exc)

    if not embeddings:
        logger.warning("Nenhum embedding foi gerado.")
        return 0

    try:
        EMBEDDINGS_PKL.parent.mkdir(parents=True, exist_ok=True)
        with EMBEDDINGS_PKL.open("wb") as f:
            pickle.dump(embeddings, f)
        logger.info("Embeddings salvos em %s (total=%d).", EMBEDDINGS_PKL, len(embeddings))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro ao salvar embeddings em %s: %s", EMBEDDINGS_PKL, exc)
        return 0

    return len(embeddings)


if __name__ == "__main__":
    total = generate_embeddings()
    print(f"Treinamento concluído. Total de embeddings gerados: {total}")


