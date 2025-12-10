from __future__ import annotations

"""
Script para importar automaticamente o dataset LFW para a estrutura do projeto.

O que ele faz:
- Percorre as pastas de pessoas em C:\\Users\\mathe\\Downloads\\lfw-deepfunneled.
- Para cada pessoa (pasta), cria/usa um usuário em users.json.
- Copia um conjunto de imagens dessa pessoa para images/<user_id>/.

Depois de rodar este script, você pode executar train_classifier.py para
gerar embeddings, treinar o classificador e ver a precisão.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict

from config import IMAGES_DIR, init_environment
from database_utils import load_users, register_user

logger = logging.getLogger(__name__)


# Caminho raiz do LFW baixado pelo usuário
LFW_ROOT = Path(r"C:\Users\mathe\Downloads\lfw-deepfunneled")

# Limite de imagens por pessoa.
# Se for 0, usa TODAS as imagens disponíveis para cada pessoa.
MAX_IMAGES_PER_PERSON = 0


def _find_people_root() -> Path:
    """
    Descobre a pasta onde estão as subpastas de pessoas do LFW.

    Alguns downloads vêm como lfw-deepfunneled/lfw/<pessoa>/...
    Outros já vêm direto como lfw-deepfunneled/<pessoa>/...
    """
    if not LFW_ROOT.exists():
        raise FileNotFoundError(f"Pasta LFW não encontrada em {LFW_ROOT}")

    # 1) Se existir uma subpasta "lfw", usamos ela.
    candidate = LFW_ROOT / "lfw"
    if candidate.is_dir():
        return candidate

    # 2) Alguns zips criam lfw-deepfunneled/lfw-deepfunneled/<pessoa>/...
    #    Se existir apenas UMA subpasta dentro do root, usamos ela como raiz.
    subdirs = [d for d in LFW_ROOT.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]

    # 3) Caso contrário, assumimos que o root já contém as pastas de pessoas.
    return LFW_ROOT


def ingest_lfw_dataset() -> None:
    init_environment()
    logging.basicConfig(level=logging.INFO)

    people_root = _find_people_root()
    logger.info("Usando pasta LFW em: %s", people_root)

    users = load_users()
    # Mapa de nome -> id já existente
    existing_by_name: Dict[str, int] = {
        str(u.get("name")): int(u.get("id")) for u in users if "id" in u and "name" in u
    }

    person_dirs = [d for d in people_root.iterdir() if d.is_dir()]
    if not person_dirs:
        logger.warning("Nenhuma subpasta de pessoa encontrada em %s", people_root)
        return

    logger.info("Encontradas %d pessoas no dataset LFW.", len(person_dirs))

    imported_people = 0

    for person_dir in sorted(person_dirs):
        name = person_dir.name

        # Ignora se não houver nenhuma imagem .jpg/.jpeg/.png
        image_files = sorted(
            [
                p
                for p in person_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
        if not image_files:
            continue

        # Reaproveita usuário se já existir com o mesmo nome
        if name in existing_by_name:
            user_id = existing_by_name[name]
            logger.info("Reutilizando usuário existente id=%s para '%s'.", user_id, name)
        else:
            user_id = register_user(name)
            existing_by_name[name] = user_id
            logger.info("Criado usuário id=%s para '%s'.", user_id, name)

        user_dir = IMAGES_DIR / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)

        # Se já existem imagens na pasta, não sobrescrevemos; apenas pulamos
        existing_imgs = list(user_dir.glob("*.jpg"))
        if existing_imgs:
            logger.info(
                "Pasta images/%s já contém %d imagens; pulando cópia para evitar misturar.",
                user_id,
                len(existing_imgs),
            )
            continue

        # Copia as imagens (todas, ou até o limite configurado)
        count = 0
        for src in image_files:
            dst = user_dir / f"{count:03d}.jpg"
            shutil.copy2(src, dst)
            count += 1
            if MAX_IMAGES_PER_PERSON and count >= MAX_IMAGES_PER_PERSON:
                break

        logger.info(
            "Importadas %d imagens de '%s' para images/%s.",
            count,
            name,
            user_id,
        )
        imported_people += 1

    logger.info("Importação concluída. Pessoas importadas: %d.", imported_people)


if __name__ == "__main__":
    ingest_lfw_dataset()


