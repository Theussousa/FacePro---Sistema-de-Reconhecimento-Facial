## Sistema de Reconhecimento Facial

Aplicação completa em Python 3.10.11 para **cadastro, treinamento e reconhecimento facial** com interface gráfica moderna em **PyQt5** e registro de acessos.

Funcionalidades principais:

- **Cadastro de usuário** com nome + CPF e captura automática de imagens pela webcam, com checagem simples de vivacidade (caixa verde/vermelha).
- **Treinamento / atualização da base** de embeddings faciais usando **DeepFace (Facenet512)**.
- **Reconhecimento facial em tempo real**, usando classificador treinado (SVM) e/ou distância L2, apenas para usuários cadastrados.
- **Tela de acesso liberado** com foto, nome, CPF, data e hora.
- **Histórico de acessos** em arquivo CSV e banco SQLite.

---

### 1. Estrutura do projeto

```text
facial_system/
│── app.py                     # Aplicação principal com interface gráfica (PyQt5)
│── capture_faces.py           # Captura imagens para cadastro
│── train_embeddings.py        # Gera embeddings faciais e salva no banco
│── recognize_face.py          # Executa reconhecimento facial em tempo real
│── config.py                  # Configurações de caminhos, logging, etc.
│── database_utils.py          # Utilitários de acesso ao "banco" JSON
│
│── database/
│     ├── users.json           # Banco simples com nomes e IDs
│     └── embeddings.pkl       # Lista de embeddings salvas (gerado após treinamento)
│
│── images/
│     └── <user_id>/           # Pastas com fotos de cada usuário (criadas em tempo de execução)
│
│── ui/
│     └── assets/              # Ícones e imagens da interface (opcional)
│
│── requirements.txt
│── README.md
```

---

### 2. Pré-requisitos

- **Python 3.10.11** (ou compatível com as versões de dependências).
- Webcam funcional conectada ao computador.
- Acesso à internet na primeira execução para download de pesos dos modelos do DeepFace (se necessário).

---

### 3. Instalação

Recomenda-se usar um ambiente virtual:

```bash
cd facial_system
python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac
```

Instale as dependências:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Execução

Na raiz do projeto (`facial_system/`), execute:

```bash
python app.py
```

A janela principal será aberta com os botões:

- **Cadastrar Usuário**
- **Treinar Modelo / Atualizar Base**
- **Reconhecer Usuário**
- **Sair**

---

### 5. Fluxos principais (passo a passo de uso)

#### 5.1. Cadastro de Usuário

1. Na janela principal, clique em **"Cadastrar Usuário"**.
2. Na caixa de diálogo:
   - Preencha o **Nome completo**.
   - Preencha o **CPF** com **11 dígitos numéricos** (apenas números).
   - Clique em **"Iniciar captura"**.
3. A webcam será aberta com a mensagem:
   - **"Olhe para a câmera por alguns segundos para registrar seu rosto."**
4. O sistema vai:
   - Detectar automaticamente o rosto com o DeepFace.
   - Verificar se há movimento (vivacidade simples):
     - Se detectar **possível foto** (rosto muito parado), mostrará:
       - Mensagem: *"Rosto estático (possível foto). Mova-se um pouco."*
       - Caixa **vermelha** ao redor do rosto.
       - **Não salva imagens**.
     - Se detectar um rosto **vivo** (movimento natural):
       - Mensagem: *"Rosto detectado. Mantenha-se olhando para a câmera."*
       - Caixa **verde** ao redor do rosto.
       - Captura automaticamente **5 imagens** enquanto você mantém o rosto na câmera.
5. Pressione **Q** para encerrar a captura caso queira cancelar.
6. Se nenhuma imagem for capturada (por exemplo, se você sair sem olhar para a câmera):
   - O usuário é removido automaticamente e o cadastro **não** é concluído.

Onde os dados ficam salvos:

- Imagens: 

  ```text
  images/<user_id>/
  ```

- Informações básicas (ID, nome, CPF):

- `database/users.json`

---

#### 5.2. Treinamento / Atualização da Base (embeddings)

1. Clique em **"Treinar Modelo / Atualizar Base"** na tela principal.
2. O sistema irá:
   - Ler todos os usuários ativos em `database/users.json`.
   - Percorrer as pastas de imagens em `images/<user_id>/`.
   - Gerar embeddings com **DeepFace (Facenet512)**.
   - Salvar tudo em `database/embeddings.pkl`.
3. Esse passo é suficiente para o reconhecimento funcionar para novos usuários cadastrados.

---

#### 5.3. Reconhecimento Facial (tempo real)

1. Clique em **"Reconhecer Usuário"**.
2. A webcam será aberta com o texto de status na parte superior da janela.
3. O sistema vai:
   - Detectar o rosto no frame (DeepFace).
   - Calcular o embedding do rosto capturado.
   - Tentar reconhecer o usuário:
     - Primeiro usando um **classificador SVM** (se `database/face_classifier.pkl` existir).
     - Caso não exista classificador treinado ou confiança seja baixa, usa o **match por distância L2** com os embeddings de `embeddings.pkl`.
   - Aplicar checagem de vivacidade (liveness) simples:
     - Se o rosto ficar quase idêntico em muitos frames, exibirá:
       - *"Rosto estático (possível foto)"* em **amarelo** e **não libera acesso**.
4. Quando um usuário cadastrado for reconhecido com vivacidade aceitável:
   - O nome será exibido em **verde** sobre o vídeo, por exemplo:

     ```text
     <Nome do usuário> (proba=0.85)
     ```

   - Após alguns segundos com o mesmo rosto estável, será aberta uma janela adicional:
     - **"Acesso liberado"** com:
       - Foto do rosto recortado.
       - Nome e CPF.
       - Data e horário do acesso.
5. Todos os acessos reconhecidos são registrados em:

   - `database/access_log.csv`

6. Pressione **Q** para encerrar a sessão de reconhecimento.

---

#### 5.4. Treinar e avaliar o classificador (opcional, via script)

Além do botão da interface, é possível treinar um **classificador SVM** e ver a acurácia usando o script `train_classifier.py`:

1. Ative o ambiente virtual e, na raiz do projeto, execute:

   ```bash
   .venv\Scripts\python.exe facial_system\train_classifier.py
   ```

2. O script irá:
   - Gerar/atualizar `database/embeddings.pkl`.
   - Enviar os embeddings para o banco SQLite `database/facepro.db`.
   - Dividir em treino/teste, treinar um SVM e imprimir a **acurácia** e o relatório de classificação no terminal.
   - Salvar o modelo em `database/face_classifier.pkl`.
   - Registrar as métricas na tabela `metrics` dentro do `facepro.db`.

---

### 6. Observações de implementação

- Embeddings são gerados com `DeepFace.represent(..., model_name="Facenet512")`.
- A base de embeddings é um `list` de dicionários serializado em `embeddings.pkl` via `pickle`.
- O sistema utiliza **logging** para registrar eventos em `facial_system.log`.
- A interface gráfica é construída com **PyQt5**, em uma janela 900x600, centralizada, com container principal estilizado com **bordas arredondadas** e botões grandes.
- Há uma checagem simples de **vivacidade** tanto no cadastro quanto no reconhecimento, baseada na diferença média entre frames consecutivos da região do rosto (para tentar diferenciar rosto real de foto estática).

---

### 7. Comandos rápidos

- **Instalar dependências**

  ```bash
  pip install -r requirements.txt
  ```

- **Executar a aplicação**

  ```bash
  python app.py
  ```



