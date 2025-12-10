from __future__ import annotations

"""
Script para treinar e avaliar um classificador facial usando embeddings.

Fluxo:
- Gera embeddings para todas as imagens em `images/<user_id>/` (reusa lógica de train_embeddings).
- Salva esses embeddings no banco SQLite.
- Carrega o dataset do SQLite.
- Separa em treino e teste, treina um classificador (SVM) e calcula acurácia.
- Salva o classificador treinado em CLASSIFIER_PKL.
"""

import logging
import pickle
from datetime import datetime

import numpy as np  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from config import CLASSIFIER_PKL, init_environment
from sql_database import init_db, replace_embeddings, load_embeddings_dataset, save_metric
from train_embeddings import generate_embeddings

logger = logging.getLogger(__name__)


def _ensure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)


def generate_and_store_embeddings() -> int:
    """
    Gera embeddings usando o script existente e salva todos no banco SQLite.

    Returns:
        int: quantidade de embeddings gerados.
    """
    from pathlib import Path
    import pickle as _pickle

    from config import EMBEDDINGS_PKL

    total = generate_embeddings()
    if total <= 0:
        logger.warning("Nenhum embedding gerado; treino do classificador será abortado.")
        return 0

    # Reabre o arquivo embeddings.pkl para carregar a lista gerada
    with EMBEDDINGS_PKL.open("rb") as f:
        embeddings = _pickle.load(f)

    if not isinstance(embeddings, list):
        logger.error("Estrutura inesperada em embeddings.pkl, esperado list.")
        return 0

    replace_embeddings(embeddings)
    return len(embeddings)


def train_classifier(test_size: float = 0.2, random_state: int = 42) -> None:
    """Treina um classificador SVM usando embeddings armazenados no SQLite."""
    init_environment()
    init_db()

    _ensure_logging()

    logger.info("Gerando embeddings e salvando no banco SQLite...")
    total_embeddings = generate_and_store_embeddings()
    if total_embeddings <= 0:
        logger.warning("Nenhum embedding disponível para treino do classificador.")
        return

    logger.info("Carregando dataset de embeddings do banco...")
    X, y, meta = load_embeddings_dataset()
    if X.size == 0 or y.size == 0:
        logger.warning("Dataset vazio após leitura do banco; abortando treino.")
        return

    logger.info("Total de amostras: %d | Dimensão do embedding: %d", X.shape[0], X.shape[1])

    # Divide em treino e teste preservando a proporção de classes
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    logger.info("Treinando classificador SVM (linear)...")
    clf = SVC(kernel="linear", probability=True, random_state=random_state)
    clf.fit(X_train, y_train)

    logger.info("Avaliando no conjunto de teste...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("Acurácia no conjunto de teste: %.4f", acc)

    report = classification_report(
        y_test,
        y_pred,
        target_names=[meta[int(sid)]["name"] for sid in sorted(np.unique(y)) if int(sid) in meta],
        zero_division=0,
    )
    logger.info("Relatório de classificação:\n%s", report)

    # Salva métrica no banco
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_metric(created_at=created_at, dataset_size=int(X.shape[0]), accuracy=float(acc), notes="SVM linear")

    # Salva o classificador e o metadata juntos para uso futuro
    bundle = {
        "model": clf,
        "meta": meta,
    }
    CLASSIFIER_PKL.parent.mkdir(parents=True, exist_ok=True)
    with CLASSIFIER_PKL.open("wb") as f:
        pickle.dump(bundle, f)

    logger.info("Classificador salvo em %s", CLASSIFIER_PKL)


if __name__ == "__main__":
    train_classifier()


