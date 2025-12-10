"""
Módulo responsável por capturar imagens da webcam para cadastro de usuários.

Fluxo:
- Recebe um nome de usuário.
- Registra o usuário em users.json e gera um ID.
- Cria uma pasta images/<user_id>/.
- Captura N imagens da webcam e salva nessa pasta.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]
from deepface import DeepFace  # type: ignore[import-untyped]

from config import IMAGES_DIR, init_environment
from database_utils import register_user, delete_user

logger = logging.getLogger(__name__)


def _create_user_image_dir(user_id: int) -> Path:
    """Cria o diretório de imagens para o usuário, se necessário."""
    user_dir = IMAGES_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def capture_user_faces(
    name: str,
    cpf: str = "",
    num_images: int = 5,
    camera_index: int = 0,
) -> Optional[int]:
    """
    Captura imagens da webcam para um novo usuário.

    A captura é automática, mas somente quando:
    - Um rosto é detectado pela DeepFace.
    - Há alguma variação entre frames (liveness simples para evitar fotos estáticas).
    """
    init_environment()
    logger.info(
        "Iniciando captura de faces para o usuário '%s' (cpf=%s, num_imagens=%d, camera_index=%d)",
        name,
        cpf or "-",
        num_images,
        camera_index,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Não foi possível acessar a webcam (índice %d).", camera_index)
        return None

    # Somente registra o usuário após garantir que a webcam abriu
    user_id = register_user(name, cpf)
    user_dir = _create_user_image_dir(user_id)

    captured = 0
    frame_count = 0
    window_name = "Cadastro - Captura automática (Q para sair)"

    # Estado para verificação simples de vivacidade
    prev_face_roi_gray: Optional[np.ndarray] = None
    static_frames = 0
    live_stable_frames = 0

    # Parâmetros de liveness (iguais/similares ao reconhecimento, porém mais rígidos)
    LIVENESS_DIFF_THRESHOLD = 1.0  # quanto menor, mais sensível a fotos estáticas
    LIVENESS_STATIC_FRAMES = 10    # nº de frames muito parecidos para marcar como estático
    STABLE_LIVE_FRAMES = 5         # nº de frames "vivos" antes de começar a capturar
    CAPTURE_INTERVAL = 5           # captura a cada N frames enquanto está vivo

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Falha ao ler frame da webcam.")
                break

            # Mensagem padrão
            text = "Olhe para a câmera por alguns segundos para registrar seu rosto."
            color = (0, 255, 0)

            has_live_face = False

            # Tenta detectar rosto e usar isso também como liveness
            reps = None
            try:
                reps = DeepFace.represent(
                    img_path=frame,
                    model_name="Facenet512",
                    enforce_detection=True,
                )
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if "Face could not be detected" in msg:
                    text = "Nenhum rosto detectado. Aproxime-se da câmera."
                    color = (0, 255, 255)
                else:
                    logger.exception("Erro durante a detecção de rosto no cadastro: %s", exc)
                    text = "Erro ao detectar rosto."
                    color = (0, 0, 255)

            if reps:
                facial_area = reps[0].get("facial_area")
                if isinstance(facial_area, dict):
                    x = int(facial_area.get("x", 0))
                    y = int(facial_area.get("y", 0))
                    w = int(facial_area.get("w", 0))
                    h = int(facial_area.get("h", 0))
                    x = max(0, x)
                    y = max(0, y)
                    w = max(0, w)
                    h = max(0, h)
                    x2 = min(frame.shape[1], x + w)
                    y2 = min(frame.shape[0], y + h)
                    if x2 > x and y2 > y:
                        face_roi = frame[y:y2, x:x2]
                        if face_roi.size > 0:
                            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                            gray_roi = cv2.resize(gray_roi, (64, 64))

                            if (
                                prev_face_roi_gray is not None
                                and prev_face_roi_gray.shape == gray_roi.shape
                            ):
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
                                text = "Rosto estático (possível foto). Mova-se um pouco."
                                color = (0, 255, 255)
                                live_stable_frames = 0
                                # Caixa vermelha indicando possível foto
                                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
                            else:
                                has_live_face = True
                                live_stable_frames += 1
                                text = "Rosto detectado. Mantenha-se olhando para a câmera."
                                color = (0, 255, 0)
                                # Caixa verde indicando rosto vivo
                                cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            # Captura automática somente se tivermos um rosto "vivo" estável
            if has_live_face and live_stable_frames >= STABLE_LIVE_FRAMES:
                frame_count += 1
                if frame_count % CAPTURE_INTERVAL == 0 and captured < num_images:
                    img_path = user_dir / f"{captured:03d}.jpg"
                    cv2.imwrite(str(img_path), frame)
                    captured += 1
                    logger.info("Imagem capturada automaticamente: %s", img_path)

                    if captured >= num_images:
                        logger.info("Número desejado de imagens atingido (%d).", num_images)
                        break

            if key in (ord("q"), ord("Q")):
                logger.info("Captura interrompida pelo usuário (tecla Q).")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured == 0:
        logger.warning("Nenhuma imagem foi capturada para o usuário id=%d.", user_id)
        # Remove o usuário e quaisquer pastas/imagens criadas
        try:
            delete_user(user_id, delete_images=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Erro ao remover usuário sem imagens (id=%d): %s", user_id, exc)
        return None

    logger.info("Captura concluída para usuário id=%d, imagens=%d.", user_id, captured)
    return user_id


if __name__ == "__main__":
    # Execução direta para testes manuais:
    nome = input("Digite o nome do usuário para cadastro: ").strip()
    cpf = input("Digite o CPF (opcional): ").strip()
    if nome:
        uid = capture_user_faces(nome, cpf=cpf)
        print(f"Cadastro finalizado. ID do usuário: {uid}")
    else:
        print("Nome inválido.")


