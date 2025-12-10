from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np  # type: ignore[import-untyped]

from config import SQLITE_DB_PATH, ensure_directories

logger = logging.getLogger(__name__)


def _get_connection() -> sqlite3.Connection:
    """Abre uma conexão com o banco SQLite, garantindo diretórios."""
    ensure_directories()
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Cria as tabelas necessárias para armazenar embeddings e métricas."""
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            cpf TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id INTEGER NOT NULL,
            image_path TEXT,
            embedding_json TEXT NOT NULL,
            FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            dataset_size INTEGER NOT NULL,
            accuracy REAL NOT NULL,
            notes TEXT
        );
        """
    )

    conn.commit()
    conn.close()
    logger.info("Banco SQLite inicializado em %s", SQLITE_DB_PATH)


def upsert_subject(subject: Dict[str, Any], conn: sqlite3.Connection | None = None) -> None:
    """
    Garante que o sujeito (usuário) exista na tabela subjects.

    Se um `conn` for fornecido, reutiliza essa conexão (sem fechar/commit).
    Caso contrário, abre e fecha uma conexão própria.
    """
    owns_connection = conn is None
    if conn is None:
        conn = _get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO subjects (id, name, cpf)
        VALUES (:id, :name, :cpf)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            cpf = excluded.cpf;
        """,
        {
            "id": int(subject.get("id")),
            "name": str(subject.get("name", "")),
            "cpf": subject.get("cpf"),
        },
    )

    if owns_connection:
        conn.commit()
        conn.close()


def replace_embeddings(embeddings: List[Dict[str, Any]]) -> None:
    """
    Substitui completamente a tabela de embeddings pelos dados fornecidos.

    Espera uma lista de dicionários com chaves:
      - id (int)
      - name (str)
      - cpf (opcional)
      - image (str)
      - embedding (np.ndarray ou lista)
    """
    if not embeddings:
        logger.warning("Nenhum embedding fornecido para armazenamento no banco.")
        return

    init_db()

    conn = _get_connection()
    try:
        cur = conn.cursor()

        # Limpa embeddings antigos
        cur.execute("DELETE FROM embeddings;")

        for item in embeddings:
            subject_id = int(item["id"])
            name = item.get("name", "")
            cpf = item.get("cpf")
            image_path = item.get("image")
            vector = item.get("embedding")

            # Garante que o sujeito exista reutilizando a mesma conexão
            upsert_subject({"id": subject_id, "name": name, "cpf": cpf}, conn=conn)

            if isinstance(vector, np.ndarray):
                emb_list = vector.astype("float32").tolist()
            else:
                emb_list = list(vector)

            cur.execute(
                """
                INSERT INTO embeddings (subject_id, image_path, embedding_json)
                VALUES (?, ?, ?);
                """,
                (subject_id, image_path, json.dumps(emb_list)),
            )

        conn.commit()
    finally:
        conn.close()
    logger.info("Embeddings salvos no banco SQLite (total=%d).", len(embeddings))


def load_embeddings_dataset() -> Tuple[np.ndarray, np.ndarray, Dict[int, Dict[str, Any]]]:
    """
    Carrega todos os embeddings do banco para treino/avaliação.

    Returns:
        X: np.ndarray [n_samples, n_features]
        y: np.ndarray [n_samples] (ids dos sujeitos)
        metadata: dict subject_id -> {name, cpf}
    """
    init_db()
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT e.subject_id, e.embedding_json, s.name, s.cpf
        FROM embeddings e
        JOIN subjects s ON s.id = e.subject_id;
        """
    )

    rows = cur.fetchall()
    conn.close()

    if not rows:
        return np.empty((0,)), np.empty((0,)), {}

    X_list: List[List[float]] = []
    y_list: List[int] = []
    meta: Dict[int, Dict[str, Any]] = {}

    for r in rows:
        emb = json.loads(r["embedding_json"])
        X_list.append(emb)
        sid = int(r["subject_id"])
        y_list.append(sid)
        if sid not in meta:
            meta[sid] = {"name": r["name"], "cpf": r["cpf"]}

    X = np.array(X_list, dtype="float32")
    y = np.array(y_list, dtype="int32")
    return X, y, meta


def save_metric(created_at: str, dataset_size: int, accuracy: float, notes: str | None = None) -> None:
    """Salva um registro de métrica de avaliação do classificador."""
    init_db()
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO metrics (created_at, dataset_size, accuracy, notes)
        VALUES (?, ?, ?, ?);
        """,
        (created_at, int(dataset_size), float(accuracy), notes),
    )

    conn.commit()
    conn.close()


