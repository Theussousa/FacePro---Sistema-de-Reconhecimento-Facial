"""
Aplicação principal com interface gráfica em PyQt5 para o sistema de
cadastro e reconhecimento facial.

Telas/ações principais:
- Cadastrar usuário (captura de imagens)
- Treinar modelo / Atualizar base
- Reconhecer usuário
- Sair
"""

from __future__ import annotations

import sys
import traceback
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore[import-untyped]

from config import init_environment
from capture_faces import capture_user_faces
from train_embeddings import generate_embeddings
from recognize_face import recognize_from_camera
from database_utils import load_users, delete_user


class UserManagementDialog(QtWidgets.QDialog):
    """Janela para listar usuários cadastrados e permitir exclusão."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gerenciar Usuários")
        self.resize(520, 420)
        self.setModal(True)
        # Estilo claro para melhor leitura de textos e campos
        self.setStyleSheet(
            """
            QDialog {
                background-color: #F9FAFB;
            }
            QTableWidget {
                background-color: #FFFFFF;
                color: #111827;
                gridline-color: #E5E7EB;
                alternate-background-color: #F3F4F6;
            }
            QTableWidget::item {
                background-color: #FFFFFF;
                color: #111827;
            }
            QTableWidget::item:alternate {
                background-color: #F3F4F6;
                color: #111827;
            }
            QHeaderView::section {
                background-color: #E5E7EB;
                color: #111827;
                font-weight: 600;
            }
            QPushButton {
                background-color: #111827;
                color: #F9FAFB;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #1F2937;
            }
            """
        )
        self._build_ui()
        self._load_users()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Usuários cadastrados")
        title.setStyleSheet(
            """
            QLabel {
                font-size: 18px;
                font-weight: 600;
            }
            """
        )

        # Agora exibimos também o CPF na tabela
        self.table = QtWidgets.QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderLabels(["ID", "Nome", "CPF"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)

        btn_refresh = QtWidgets.QPushButton("Atualizar lista")
        btn_delete = QtWidgets.QPushButton("Excluir selecionado")
        btn_close = QtWidgets.QPushButton("Fechar")

        btn_bar = QtWidgets.QHBoxLayout()
        btn_bar.addWidget(btn_refresh)
        btn_bar.addStretch(1)
        btn_bar.addWidget(btn_delete)
        btn_bar.addWidget(btn_close)

        layout.addWidget(title)
        layout.addWidget(self.table)
        layout.addLayout(btn_bar)

        btn_refresh.clicked.connect(self._load_users)
        btn_delete.clicked.connect(self._delete_selected_user)
        btn_close.clicked.connect(self.accept)

    def _load_users(self) -> None:
        users = load_users()
        self.table.setRowCount(0)
        for user in users:
            row = self.table.rowCount()
            self.table.insertRow(row)
            id_item = QtWidgets.QTableWidgetItem(str(user.get("id", "")))
            name_item = QtWidgets.QTableWidgetItem(str(user.get("name", "")))
            cpf_item = QtWidgets.QTableWidgetItem(str(user.get("cpf", "")))
            self.table.setItem(row, 0, id_item)
            self.table.setItem(row, 1, name_item)
            self.table.setItem(row, 2, cpf_item)

    def _delete_selected_user(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(self, "Informação", "Selecione um usuário para excluir.")
            return

        id_item = self.table.item(row, 0)
        name_item = self.table.item(row, 1)
        if id_item is None:
            return

        user_id = int(id_item.text())
        user_name = name_item.text() if name_item is not None else ""

        resp = QtWidgets.QMessageBox.question(
            self,
            "Confirmar exclusão",
            f"Deseja realmente excluir o usuário '{user_name}' (ID {user_id})?\n"
            "As imagens associadas também serão removidas.",
        )
        if resp != QtWidgets.QMessageBox.Yes:
            return

        if delete_user(user_id, delete_images=True):
            # Após excluir o usuário, re-treina automaticamente a base de embeddings
            try:
                total = generate_embeddings()
                if total == 0:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Aviso",
                        "Usuário excluído com sucesso, porém nenhum embedding foi gerado ao atualizar a base.",
                    )
                else:
                    QtWidgets.QMessageBox.information(
                        self,
                        "Sucesso",
                        "Usuário excluído com sucesso.\nBase de embeddings atualizada.",
                    )
            except Exception:
                # Se der erro no re-treinamento, pelo menos o usuário já foi removido
                QtWidgets.QMessageBox.warning(
                    self,
                    "Aviso",
                    "Usuário excluído, mas ocorreu um erro ao atualizar a base de embeddings.\n"
                    "Tente rodar o treinamento manualmente na tela principal.",
                )
            self._load_users()
        else:
            QtWidgets.QMessageBox.warning(self, "Aviso", "Não foi possível excluir o usuário selecionado.")


class RegisterUserDialog(QtWidgets.QDialog):
    """Caixa moderna para cadastro de novo usuário."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Novo usuário")
        self.setModal(True)
        # Janela com tamanho inicial maior e redimensionável para evitar cortes
        self.resize(480, 340)
        self.setMinimumSize(440, 320)

        self.setStyleSheet(
            """
            QDialog {
                background-color: #0B1120;
            }
            QLabel#TitleLabel {
                color: #E5E7EB;
                font-size: 20px;
                font-weight: 600;
            }
            QLabel#SubtitleLabel {
                color: #9CA3AF;
                font-size: 12px;
            }
            QLabel#FieldLabel {
                color: #D1D5DB;
                font-size: 12px;
                font-weight: 500;
            }
            QLineEdit {
                background-color: #020617;
                color: #F9FAFB;
                border-radius: 8px;
                border: 1px solid #1F2937;
                padding: 6px 10px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: #2563EB;
            }
            QPushButton {
                border-radius: 10px;
                padding: 8px 14px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton#PrimaryButton {
                background-color: #2563EB;
                color: #F9FAFB;
                border: 1px solid #2563EB;
            }
            QPushButton#PrimaryButton:hover {
                background-color: #1D4ED8;
            }
            QPushButton#SecondaryButton {
                background-color: #020617;
                color: #E5E7EB;
                border: 1px solid #374151;
            }
            QPushButton#SecondaryButton:hover {
                background-color: #111827;
            }
            """
        )

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title = QtWidgets.QLabel("Cadastrar novo usuário")
        title.setObjectName("TitleLabel")

        subtitle = QtWidgets.QLabel(
            "Informe os dados da pessoa e, em seguida, olhe para a câmera.\n"
            "A captura de imagens será feita automaticamente."
        )
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setWordWrap(True)

        name_label = QtWidgets.QLabel("Nome completo")
        name_label.setObjectName("FieldLabel")

        self.name_edit = QtWidgets.QLineEdit(self)
        self.name_edit.setPlaceholderText("Ex.: Maria Silva")

        cpf_label = QtWidgets.QLabel("CPF")
        cpf_label.setObjectName("FieldLabel")

        self.cpf_edit = QtWidgets.QLineEdit(self)
        self.cpf_edit.setPlaceholderText("Ex.: 000.000.000-00")
        # Limita o CPF a 11 dígitos numéricos (quantidade de números de um CPF)
        self.cpf_edit.setMaxLength(11)

        helper = QtWidgets.QLabel(
            "Dica: use o nome completo e o CPF real para facilitar os registros de acesso."
        )
        helper.setObjectName("SubtitleLabel")
        helper.setWordWrap(True)

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addStretch(1)

        btn_cancel = QtWidgets.QPushButton("Cancelar")
        btn_cancel.setObjectName("SecondaryButton")
        btn_ok = QtWidgets.QPushButton("Iniciar captura")
        btn_ok.setObjectName("PrimaryButton")

        buttons_layout.addWidget(btn_cancel)
        buttons_layout.addWidget(btn_ok)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(4)
        layout.addWidget(name_label)
        layout.addWidget(self.name_edit)
        layout.addWidget(cpf_label)
        layout.addWidget(self.cpf_edit)
        layout.addWidget(helper)
        layout.addLayout(buttons_layout)

        btn_cancel.clicked.connect(self.reject)
        btn_ok.clicked.connect(self._on_accept_clicked)

    def _on_accept_clicked(self) -> None:
        name = self.name_edit.text().strip()
        cpf = self.cpf_edit.text().strip()

        if not name:
            QtWidgets.QMessageBox.information(
                self,
                "Informação",
                "Digite um nome para continuar com o cadastro.",
            )
            return

        if not cpf or len(cpf) != 11 or not cpf.isdigit():
            QtWidgets.QMessageBox.information(
                self,
                "Informação",
                "Digite um CPF com 11 dígitos numéricos (apenas números).",
            )
            return
        self.accept()

    def get_name(self) -> str:
        return self.name_edit.text().strip()

    def get_cpf(self) -> str:
        return self.cpf_edit.text().strip()


class RoundedMainWindow(QtWidgets.QMainWindow):
    """Janela principal com conteúdo central estilizado (bordas arredondadas)."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sistema de Reconhecimento Facial")
        self.setFixedSize(900, 600)
        self._center_on_screen()
        self._apply_window_icon()
        self._build_ui()

    def _center_on_screen(self) -> None:
        """Centraliza a janela na tela principal."""
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        screen_geometry = screen.availableGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def _apply_window_icon(self) -> None:
        """Define um ícone minimalista (fallback para ícone padrão se não houver asset)."""
        self.setWindowIcon(self.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon))

    def _build_ui(self) -> None:
        """Constroi toda a interface da janela principal."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root_layout = QtWidgets.QHBoxLayout(central)
        root_layout.setContentsMargins(40, 40, 40, 40)
        root_layout.setSpacing(30)

        # Painel lateral (branding / título)
        side_panel = QtWidgets.QFrame()
        side_panel.setObjectName("sidePanel")
        side_panel.setFixedWidth(260)
        side_panel.setStyleSheet(
            """
            QFrame#sidePanel {
                background-color: #020617;
                border-radius: 24px;
            }
            """
        )

        side_layout = QtWidgets.QVBoxLayout(side_panel)
        side_layout.setContentsMargins(24, 24, 24, 24)
        side_layout.setSpacing(16)

        logo_label = QtWidgets.QLabel("FacePro")
        logo_label.setStyleSheet(
            """
            QLabel {
                color: #38BDF8;
                font-size: 22px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            """
        )

        tagline = QtWidgets.QLabel("Segurança e reconhecimento\nfacial em tempo real.")
        tagline.setStyleSheet(
            """
            QLabel {
                color: #9CA3AF;
                font-size: 12px;
            }
            """
        )
        tagline.setWordWrap(True)

        side_layout.addWidget(logo_label)
        side_layout.addWidget(tagline)
        side_layout.addStretch(1)

        # Informações de status (simples)
        status_title = QtWidgets.QLabel("Status")
        status_title.setStyleSheet(
            """
            QLabel {
                color: #E5E7EB;
                font-size: 13px;
                font-weight: 600;
            }
            """
        )
        self.status_label = QtWidgets.QLabel("Aguardando ação do usuário")
        self.status_label.setStyleSheet(
            """
            QLabel {
                color: #9CA3AF;
                font-size: 12px;
            }
            """
        )
        self.status_label.setWordWrap(True)

        side_layout.addWidget(status_title)
        side_layout.addWidget(self.status_label)

        side_layout.addStretch(2)

        root_layout.addWidget(side_panel)

        # Container principal com bordas arredondadas e sombra
        container = QtWidgets.QFrame()
        container.setObjectName("mainContainer")
        container.setFrameShape(QtWidgets.QFrame.NoFrame)
        container.setStyleSheet(
            """
            QFrame#mainContainer {
                background-color: #0F172A;
                border-radius: 24px;
            }
            """
        )

        shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 16)
        shadow.setColor(QtGui.QColor(0, 0, 0, 140))
        container.setGraphicsEffect(shadow)

        root_layout.addWidget(container)

        main_layout = QtWidgets.QVBoxLayout(container)
        main_layout.setContentsMargins(32, 32, 32, 32)
        main_layout.setSpacing(24)

        # Cabeçalho
        title = QtWidgets.QLabel("Sistema de Reconhecimento Facial")
        title.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        title.setStyleSheet(
            """
            QLabel {
                color: #E5E7EB;
                font-size: 24px;
                font-weight: 600;
            }
            """
        )

        subtitle = QtWidgets.QLabel("Cadastro, treinamento e reconhecimento em um só lugar.")
        subtitle.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        subtitle.setStyleSheet(
            """
            QLabel {
                color: #9CA3AF;
                font-size: 13px;
            }
            """
        )

        header_layout = QtWidgets.QVBoxLayout()
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.setSpacing(4)

        main_layout.addLayout(header_layout)

        # Área de botões em grade
        buttons_layout = QtWidgets.QGridLayout()
        buttons_layout.setHorizontalSpacing(20)
        buttons_layout.setVerticalSpacing(16)

        self.btn_register = self._create_primary_button("Cadastrar Usuário")
        self.btn_train = self._create_secondary_button("Treinar Modelo / Atualizar Base")
        self.btn_recognize = self._create_primary_button("Reconhecer Usuário")
        self.btn_manage_users = self._create_secondary_button("Gerenciar Usuários")
        self.btn_exit = self._create_danger_button("Sair")

        buttons_layout.addWidget(self.btn_register, 0, 0)
        buttons_layout.addWidget(self.btn_train, 0, 1)
        buttons_layout.addWidget(self.btn_recognize, 1, 0)
        buttons_layout.addWidget(self.btn_manage_users, 1, 1)
        buttons_layout.addWidget(self.btn_exit, 2, 0, 1, 2)

        main_layout.addLayout(buttons_layout)

        main_layout.addStretch(1)

        # Conexão dos sinais
        self.btn_register.clicked.connect(self._on_register_user_clicked)
        self.btn_train.clicked.connect(self._on_train_model_clicked)
        self.btn_recognize.clicked.connect(self._on_recognize_user_clicked)
        self.btn_manage_users.clicked.connect(self._on_manage_users_clicked)
        self.btn_exit.clicked.connect(self.close)

    # --------------------------------------------------------------------- #
    # Criação de botões estilizados
    # --------------------------------------------------------------------- #
    def _base_button(self) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton()
        btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        btn.setMinimumHeight(70)
        btn.setStyleSheet(
            """
            QPushButton {
                border-radius: 16px;
                font-size: 18px;
                font-weight: 500;
            }
            QPushButton:pressed {
                transform: scale(0.98);
            }
            """
        )
        return btn

    def _create_primary_button(self, text: str) -> QtWidgets.QPushButton:
        btn = self._base_button()
        btn.setText(text)
        btn.setStyleSheet(
            btn.styleSheet()
            + """
            QPushButton {
                background-color: #1D4ED8;
                color: #F9FAFB;
                border: 1px solid #1D4ED8;
            }
            QPushButton:hover {
                background-color: #2563EB;
            }
            QPushButton:disabled {
                background-color: #1F2937;
                color: #6B7280;
                border-color: #374151;
            }
            """
        )
        return btn

    def _create_secondary_button(self, text: str) -> QtWidgets.QPushButton:
        btn = self._base_button()
        btn.setText(text)
        btn.setStyleSheet(
            btn.styleSheet()
            + """
            QPushButton {
                background-color: #111827;
                color: #E5E7EB;
                border: 1px solid #374151;
            }
            QPushButton:hover {
                background-color: #1F2937;
            }
            """
        )
        return btn

    def _create_ghost_button(self, text: str) -> QtWidgets.QPushButton:
        btn = self._base_button()
        btn.setText(text)
        btn.setEnabled(False)
        btn.setStyleSheet(
            btn.styleSheet()
            + """
            QPushButton {
                background-color: transparent;
                color: #6B7280;
                border: 1px dashed #4B5563;
            }
            """
        )
        return btn

    def _create_danger_button(self, text: str) -> QtWidgets.QPushButton:
        btn = self._base_button()
        btn.setText(text)
        btn.setStyleSheet(
            btn.styleSheet()
            + """
            QPushButton {
                background-color: #B91C1C;
                color: #F9FAFB;
                border: 1px solid #B91C1C;
            }
            QPushButton:hover {
                background-color: #DC2626;
            }
            """
        )
        return btn

    # --------------------------------------------------------------------- #
    # Ações dos botões
    # --------------------------------------------------------------------- #
    def _on_register_user_clicked(self) -> None:
        dlg = RegisterUserDialog(self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        name = dlg.get_name()
        cpf = dlg.get_cpf()
        if not name:
            return

        self._run_with_feedback(
            operation=lambda: self._register_and_train(name.strip(), cpf.strip()),
            success_message="Cadastro realizado e modelo atualizado com sucesso.",
            error_context="Erro ao capturar imagens do usuário.",
        )

    def _on_train_model_clicked(self) -> None:
        self._run_with_feedback(
            operation=lambda: generate_embeddings(),
            success_message="Treinamento concluído com sucesso.",
            error_context="Erro durante o treinamento dos embeddings.",
        )

    def _on_recognize_user_clicked(self) -> None:
        self._run_with_feedback(
            operation=lambda: recognize_from_camera(),
            success_message="Sessão de reconhecimento encerrada.",
            error_context="Erro durante o reconhecimento facial.",
        )

    def _on_manage_users_clicked(self) -> None:
        dlg = UserManagementDialog(self)
        dlg.exec_()

    # --------------------------------------------------------------------- #
    # Utilitários de feedback e tratamento de erros
    # --------------------------------------------------------------------- #
    def _run_with_feedback(
        self,
        operation,
        success_message: str,
        error_context: str,
    ) -> None:
        """
        Executa uma operação potencialmente longa com feedback básico.

        Observação: As operações são chamadas de forma síncrona; a janela pode
        ficar temporariamente bloqueada durante o uso da webcam/DeepFace.
        """
        try:
            self._set_buttons_enabled(False)
            # Atualiza status visual
            self.status_label.setText("Executando operação, aguarde...")
            result = operation()
            if result is None or (isinstance(result, int) and result == 0):
                # Resultado "vazio" é tratado como aviso, não erro fatal.
                QtWidgets.QMessageBox.warning(self, "Aviso", success_message)
            else:
                QtWidgets.QMessageBox.information(self, "Sucesso", success_message)
            self.status_label.setText("Pronto. Escolha a próxima ação.")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self,
                "Erro",
                f"{error_context}\n\nDetalhes:\n{traceback.format_exc()}",
            )
            self.status_label.setText("Ocorreu um erro. Verifique os detalhes e tente novamente.")
        finally:
            self._set_buttons_enabled(True)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self.btn_register.setEnabled(enabled)
        self.btn_train.setEnabled(enabled)
        self.btn_recognize.setEnabled(enabled)
        self.btn_manage_users.setEnabled(enabled)
        self.btn_exit.setEnabled(enabled)

    # --------------------------------------------------------------------- #
    # Operações compostas
    # --------------------------------------------------------------------- #
    def _register_and_train(self, name: str, cpf: str) -> Optional[int]:
        """
        Realiza o cadastro do usuário e atualiza a base de embeddings.

        Retorna o ID do usuário cadastrado, ou None em caso de falha.
        """
        user_id = capture_user_faces(name, cpf=cpf)
        if user_id is None:
            return None
        # Atualiza/treina automaticamente a base após novo cadastro.
        generate_embeddings()
        return user_id


def main() -> int:
    init_environment()
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Sistema de Reconhecimento Facial")

    # Paleta geral: fundo escuro, mas mantendo textos em caixas/inputs com cor padrão
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#020617"))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#E5E7EB"))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#020617"))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#020617"))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#E5E7EB"))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#111827"))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#E5E7EB"))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#2563EB"))
    app.setPalette(palette)

    # Estilo geral para campos de texto e message boxes com texto preto
    app.setStyleSheet(
        """
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #FFFFFF;
            color: #111827;
            border-radius: 6px;
            padding: 4px 6px;
        }
        QMessageBox {
            background-color: #F9FAFB;
        }
        QMessageBox QLabel {
            color: #111827;
        }
        """
    )

    font = QtGui.QFont("Segoe UI", 10)
    app.setFont(font)

    window = RoundedMainWindow()
    window.show()

    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())


