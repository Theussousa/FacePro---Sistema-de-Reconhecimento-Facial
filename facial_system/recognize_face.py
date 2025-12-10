"""
Módulo responsável pelo reconhecimento facial em tempo real.

Fluxo:
- Carrega embeddings de embeddings.pkl.
- Abre a webcam.
- Para cada frame, detecta/representa o rosto com DeepFace.
- Compara com a base utilizando distância L2.
"""

from __future__ import annotations

import csv
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
from deepface import DeepFace  # type: ignore[import-untyped]

from config import EMBEDDINGS_PKL, ACCESS_LOG_CSV, CLASSIFIER_PKL, init_environment
from database_utils import load_users

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Parâmetros de reconhecimento
# --------------------------------------------------------------------------- #
# Usamos distância L2 em embeddings normalizados; para facilitar separar
# rostos diferentes, usamos um limiar mais rígido (quanto menor, menos
# falsos positivos). Ajuste se necessário após testes no seu ambiente.
DEFAULT_THRESHOLD = 0.6

# Heurística simples de "liveness" (detecção de vivacidade):
# - Compara a diferença média entre regiões da face em frames consecutivos.
# - Se a mudança for muito pequena por vários frames seguidos, assume-se que
#   pode ser uma foto estática.
# Deixamos o limiar bem baixo para ser mais rígido com imagens estáticas (fotos).
LIVENESS_DIFF_THRESHOLD = 1.0  # quanto menor, mais sensível a mudanças sutis
LIVENESS_STATIC_FRAMES = 10    # nº de frames estáticos para marcar como "possível foto"

# Probabilidade mínima do classificador para aceitar um match.
# Quanto maior, mais rígido (menos falsos positivos).
CLASSIFIER_MIN_PROBA = 0.8

_classifier_bundle: Optional[Dict[str, Any]] = None


def _load_classifier_bundle() -> Optional[Dict[str, Any]]:
    """Carrega o classificador treinado (SVM) salvo em CLASSIFIER_PKL, se existir."""
    global _classifier_bundle
    if _classifier_bundle is not None:
        return _classifier_bundle

    if not CLASSIFIER_PKL.exists():
        logger.warning(
            "Arquivo de classificador não encontrado em %s. "
            "Execute train_classifier.py para treinar o modelo.",
            CLASSIFIER_PKL,
        )
        return None
    try:
        with CLASSIFIER_PKL.open("rb") as f:
            _classifier_bundle = pickle.load(f)
        logger.info("Classificador carregado de %s.", CLASSIFIER_PKL)
        return _classifier_bundle
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro ao carregar classificador de %s: %s", CLASSIFIER_PKL, exc)
        return None


def _load_embeddings() -> List[Dict[str, Any]]:
    """Carrega a lista de embeddings salvos em embeddings.pkl."""
    if not EMBEDDINGS_PKL.exists():
        logger.warning("Arquivo de embeddings não encontrado em %s.", EMBEDDINGS_PKL)
        return []
    try:
        with EMBEDDINGS_PKL.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, list):
            logger.warning("Estrutura inesperada em embeddings.pkl; esperado list.")
            return []
        return data
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro ao carregar embeddings de %s: %s", EMBEDDINGS_PKL, exc)
        return []


def _log_access_event(user: Dict[str, Any], timestamp: str) -> None:
    """Registra um acesso reconhecido em ACCESS_LOG_CSV."""
    try:
        ACCESS_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

        file_exists = ACCESS_LOG_CSV.exists()
        with ACCESS_LOG_CSV.open("a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            if not file_exists:
                writer.writerow(["timestamp", "user_id", "name", "cpf"])

            user_id = user.get("id")
            name = user.get("name")
            cpf = user.get("cpf", "")
            writer.writerow([timestamp, user_id, name, cpf])
        logger.info(
            "Acesso registrado: user_id=%s, name=%s, cpf=%s, timestamp=%s",
            user_id,
            name,
            cpf or "-",
            timestamp,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro ao registrar acesso em %s: %s", ACCESS_LOG_CSV, exc)


def _find_best_match(
    query_embedding: np.ndarray,
    embeddings: List[Dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """Encontra o melhor match usando distância L2 em embeddings normalizados."""
    if not embeddings:
        return None, None

    db_vectors = np.stack([np.array(e["embedding"], dtype="float32") for e in embeddings])

    # Normaliza embeddings da base
    db_norms = np.linalg.norm(db_vectors, axis=1, keepdims=True) + 1e-8
    db_vectors_norm = db_vectors / db_norms

    # Normaliza embedding de consulta
    query_embedding = query_embedding.astype("float32")
    q_norm = np.linalg.norm(query_embedding) + 1e-8
    query_norm = query_embedding / q_norm

    diff = db_vectors_norm - query_norm
    dists = np.linalg.norm(diff, axis=1)
    idx = int(np.argmin(dists))
    best_dist = float(dists[idx])

    if best_dist < threshold:
        return embeddings[idx], best_dist
    return None, best_dist


def recognize_from_camera(
    threshold: float = DEFAULT_THRESHOLD,
    camera_index: int = 0,
    model_name: str = "Facenet512",
) -> None:
    """
    Executa o reconhecimento facial em tempo real usando a webcam.

    Args:
        threshold: Limiar de distância L2 para considerar um match.
        camera_index: Índice da webcam (padrão 0).
        model_name: Modelo DeepFace a ser utilizado.
    """
    init_environment()
    logger.info(
        "Iniciando reconhecimento facial (threshold=%.3f, camera_index=%d, modelo=%s)",
        threshold,
        camera_index,
        model_name,
    )

    embeddings = _load_embeddings()

    # Carrega usuários atualmente cadastrados; somente estes serão
    # considerados "válidos" no reconhecimento (mesmo que o classificador
    # conheça outros IDs antigos).
    users = load_users()
    active_user_ids = {
        int(u.get("id"))
        for u in users
        if u.get("id") is not None
    }

    # Embeddings filtrados apenas para usuários ativos
    filtered_embeddings: List[Dict[str, Any]] = []
    for e in embeddings:
        try:
            sid = int(e.get("id"))  # type: ignore[arg-type]
        except Exception:
            continue
        if sid in active_user_ids:
            filtered_embeddings.append(e)

    classifier_bundle = _load_classifier_bundle()
    classifier = None
    classifier_meta: Dict[int, Dict[str, Any]] = {}
    if classifier_bundle is not None:
        classifier = classifier_bundle.get("model")
        classifier_meta = classifier_bundle.get("meta", {})

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Não foi possível acessar a webcam (índice %d).", camera_index)
        return

    window_name = "Reconhecimento Facial - Pressione Q para sair"

    # Estado para liveness detection simples
    prev_face_roi_gray: Optional[np.ndarray] = None
    static_frames = 0

    # Guarda último usuário já processado nesta sessão
    last_recognized_user_id: Optional[int] = None

    # Controle de estabilidade: exige que o mesmo usuário seja reconhecido
    # continuamente por um período antes de liberar o acesso.
    pending_user_id: Optional[int] = None
    pending_start_time: Optional[datetime] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Falha ao ler frame da webcam.")
                break

            text = ""
            color = (255, 255, 255)
            box_color = None
            box_coords = None
            is_live: Optional[bool] = None

            reps = None
            try:
                reps = DeepFace.represent(
                    img_path=frame,
                    model_name=model_name,
                    enforce_detection=True,
                )
            except Exception as exc:  # noqa: BLE001
                # Quando não há rosto, o DeepFace lança um erro típico:
                # "Face could not be detected. Please confirm that the picture is a face photo."
                msg = str(exc)
                if "Face could not be detected" in msg:
                    text = "Nenhum rosto detectado"
                    color = (255, 255, 0)
                else:
                    logger.exception("Erro durante a representação facial: %s", exc)
                    text = "Erro no reconhecimento"
                    color = (0, 0, 255)

            if reps:
                query_emb = np.array(reps[0]["embedding"], dtype="float32")

                match: Optional[Dict[str, Any]] = None
                dist: Optional[float] = None
                best_proba: Optional[float] = None

                # Primeiro tentamos usar o classificador treinado, se disponível
                if classifier is not None:
                    try:
                        probs = classifier.predict_proba(query_emb.reshape(1, -1))[0]
                        best_idx = int(np.argmax(probs))
                        best_proba = float(probs[best_idx])
                        predicted_id = int(classifier.classes_[best_idx])

                        # Só aceita o resultado do classificador se:
                        # - probabilidade suficiente
                        # - ID ainda estiver entre os usuários ativos
                        if best_proba >= CLASSIFIER_MIN_PROBA and predicted_id in active_user_ids:
                            subject_meta = classifier_meta.get(predicted_id, {})
                            match = {
                                "id": predicted_id,
                                "name": subject_meta.get("name", str(predicted_id)),
                                "cpf": subject_meta.get("cpf"),
                            }
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Erro ao usar classificador treinado: %s", exc)
                        match = None

                # Se não houver classificador, confiança baixa ou ID inativo,
                # caímos no match por distância usando apenas embeddings ativos.
                if match is None:
                    if not filtered_embeddings:
                        logger.warning(
                            "Nenhum embedding carregado para comparação por distância; "
                            "treine o modelo ou gere embeddings."
                        )
                    else:
                        match, dist = _find_best_match(
                            query_emb,
                            filtered_embeddings,
                            threshold=threshold,
                        )

                facial_area = reps[0].get("facial_area")
                if isinstance(facial_area, dict):
                    x = int(facial_area.get("x", 0))
                    y = int(facial_area.get("y", 0))
                    w = int(facial_area.get("w", 0))
                    h = int(facial_area.get("h", 0))
                    # Garante que a ROI está dentro do frame
                    x = max(0, x)
                    y = max(0, y)
                    w = max(0, w)
                    h = max(0, h)
                    x2 = min(frame.shape[1], x + w)
                    y2 = min(frame.shape[0], y + h)
                    if x2 > x and y2 > y:
                        box_coords = (x, y, x2 - x, y2 - y)

                        # --- Liveness detection simples (comparação de frames) ---
                        face_roi = frame[y:y2, x:x2]
                        if face_roi.size > 0:
                            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                            gray_roi = cv2.resize(gray_roi, (64, 64))

                            if prev_face_roi_gray is not None and prev_face_roi_gray.shape == gray_roi.shape:
                                diff = cv2.absdiff(gray_roi, prev_face_roi_gray)
                                mean_diff = float(diff.mean())
                                if mean_diff < LIVENESS_DIFF_THRESHOLD:
                                    static_frames += 1
                                else:
                                    static_frames = 0
                            else:
                                static_frames = 0

                            prev_face_roi_gray = gray_roi

                            if static_frames >= LIVENESS_STATIC_FRAMES:
                                is_live = False
                            else:
                                is_live = True

                        # --- Tela de acesso bonita com rosto e informações ---
                        if match is not None and is_live is not False:
                            try:
                                current_user_id = int(match.get("id"))  # type: ignore[arg-type]
                            except Exception:
                                current_user_id = None

                            # Exige estabilidade de ~3 segundos com o mesmo usuário
                            now = datetime.now()
                            if current_user_id is not None:
                                if pending_user_id != current_user_id:
                                    pending_user_id = current_user_id
                                    pending_start_time = now
                                else:
                                    # Mesmo usuário que já estava sendo visto
                                    if (
                                        pending_start_time is not None
                                        and now - pending_start_time >= timedelta(seconds=3)
                                        and current_user_id != last_recognized_user_id
                                    ):
                                        last_recognized_user_id = current_user_id
                                        entry_time = now.strftime("%d/%m/%Y %H:%M:%S")

                                        # Registra no histórico de acessos
                                        _log_access_event(match, entry_time)

                                        face_preview = frame[y:y2, x:x2].copy()
                                        if face_preview.size > 0:
                                            try:
                                                face_preview = cv2.resize(face_preview, (320, 320))
                                            except Exception:
                                                pass

                                            cpf = match.get("cpf")
                                            name = str(match.get("name", ""))

                                            # Cria um "card" moderno com fundo no mesmo estilo do app
                                            # Fundo aproximado de #0F172A, com detalhes em azul (#38BDF8)
                                            card_h, card_w = 400, 640
                                            card = np.zeros((card_h, card_w, 3), dtype=np.uint8)
                                            card[:] = (15, 23, 42)  # #0F172A

                                            # Posição e tamanho da foto
                                            try:
                                                face_preview = cv2.resize(face_preview, (260, 260))
                                            except Exception:
                                                pass
                                            fh, fw, _ = face_preview.shape
                                            face_x, face_y = 40, 60
                                            card[face_y:face_y + fh, face_x:face_x + fw] = face_preview

                                            # Borda em volta da foto (azul do tema)
                                            cv2.rectangle(
                                                card,
                                                (face_x - 4, face_y - 4),
                                                (face_x + fw + 4, face_y + fh + 4),
                                                (37, 99, 235),  # azul mais forte
                                                2,
                                            )

                                            # Painel de informações à direita
                                            info_x = face_x + fw + 40
                                            info_y = face_y
                                            line_h = 34

                                            # Título
                                            cv2.putText(
                                                card,
                                                "Acesso liberado",
                                                (info_x, info_y),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.9,
                                                (56, 189, 248),  # azul claro
                                                2,
                                                cv2.LINE_AA,
                                            )

                                            info_y += int(line_h * 1.8)

                                            # Nome
                                            cv2.putText(
                                                card,
                                                f"Nome: {name}",
                                                (info_x, info_y),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.7,
                                                (229, 231, 235),  # texto claro
                                                2,
                                                cv2.LINE_AA,
                                            )

                                            info_y += line_h

                                            # CPF
                                            if cpf:
                                                cv2.putText(
                                                    card,
                                                    f"CPF: {cpf}",
                                                    (info_x, info_y),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.7,
                                                    (229, 231, 235),
                                                    2,
                                                    cv2.LINE_AA,
                                                )
                                                info_y += line_h

                                            # Data e horário (quebrados em duas linhas para caber melhor)
                                            date_part, time_part = (
                                                entry_time.split(" ", 1)
                                                if " " in entry_time
                                                else (entry_time, "")
                                            )
                                            cv2.putText(
                                                card,
                                                f"Data: {date_part}",
                                                (info_x, info_y),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.65,
                                                (156, 163, 175),  # cinza claro
                                                2,
                                                cv2.LINE_AA,
                                            )
                                            info_y += line_h
                                            if time_part:
                                                # OpenCV pode ter problemas com acentos, então evitamos o "á"
                                                cv2.putText(
                                                    card,
                                                    f"Horario: {time_part}",
                                                    (info_x, info_y),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.65,
                                                    (156, 163, 175),
                                                    2,
                                                    cv2.LINE_AA,
                                                )

                                            # Rodapé com dica para fechar
                                            footer_text = "Pressione Q ou ESC para fechar"
                                            cv2.putText(
                                                card,
                                                footer_text,
                                                (face_x, card_h - 20),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.55,
                                                (148, 163, 184),
                                                1,
                                                cv2.LINE_AA,
                                            )

                                            # Mostra a tela de acesso em uma janela separada,
                                            # mantendo a câmera aberta e o loop principal rodando.
                                            summary_window = "Acesso registrado"
                                            cv2.imshow(summary_window, card)
                            else:
                                # Reset se não conseguirmos extrair um id válido
                                pending_user_id = None
                                pending_start_time = None

                # --- Lógica de cores / mensagens considerando vivacidade ---
                if match is not None:
                    # Garante que o ID previsto pertence a um usuário ativo.
                    try:
                        mid = int(match.get("id"))  # type: ignore[arg-type]
                    except Exception:
                        mid = None
                    if mid is None or mid not in active_user_ids:
                        match = None

                if match is not None:
                    if is_live is False:
                        text = f"{match['name']} - rosto estático (possível foto)"
                        color = (0, 255, 255)  # texto amarelo
                        box_color = (0, 255, 255)  # caixa amarela
                    else:
                        if best_proba is not None:
                            text = f"{match['name']} (proba={best_proba:.2f})"
                        elif dist is not None:
                            text = f"{match['name']} (dist={dist:.3f})"
                        else:
                            text = str(match["name"])
                        color = (0, 255, 0)  # texto verde
                        box_color = (0, 255, 0)  # caixa verde
                else:
                    # Reset de estabilidade se não houver match
                    pending_user_id = None
                    pending_start_time = None
                    if is_live is False:
                        text = "Rosto estático (possível foto)"
                        color = (0, 255, 255)  # amarelo
                        box_color = (0, 255, 255)
                    else:
                        text = "Usuário não encontrado"
                        color = (0, 0, 255)  # texto vermelho
                        box_color = (0, 0, 255)  # caixa vermelha

            # Desenhar caixa ao redor do rosto, se disponível
            if box_coords is not None and box_color is not None:
                x, y, w, h = box_coords
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                text_pos = (x, max(20, y - 10))
            else:
                # Fallback: texto no topo da tela
                text_pos = (10, 30)

            if text:
                cv2.putText(
                    frame,
                    text,
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                logger.info("Reconhecimento interrompido pelo usuário (tecla Q).")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_from_camera()


