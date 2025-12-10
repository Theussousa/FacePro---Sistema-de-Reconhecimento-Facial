# FacePro – Sistema de Reconhecimento Facial com Controle de Acesso

Aplicação completa em **Python 3.10** para **cadastro, treinamento e reconhecimento facial** com interface gráfica moderna em **PyQt5**, usando **DeepFace** para embeddings e **scikit-learn (SVM)** para classificação.  
O sistema foi pensado como um controle de acesso simples, registrando **nome, CPF, data e horário** de cada reconhecimento bem-sucedido.

---

## 1. Visão geral do projeto

O FacePro é um sistema de reconhecimento facial com foco em:

- **Cadastro guiado** de usuários (nome + CPF).
- **Captura automática de fotos** pela webcam com verificação simples de vivacidade (liveness).
- **Treinamento** de uma base de embeddings faciais.
- **Reconhecimento em tempo real** apenas de usuários cadastrados.
- Exibição de uma **tela de acesso liberado** com dados da pessoa.
- **Registro de acessos** em CSV e em um banco SQLite.

Ele foi desenvolvido como projeto acadêmico, mas com uma arquitetura próxima a aplicações reais (separação de módulos, logs, banco, etc.).

---

## 2. Funcionalidades principais

- **Interface gráfica (PyQt5)**
  - Tela principal com botões:
    - `Cadastrar Usuário`
    - `Treinar Modelo / Atualizar Base`
    - `Reconhecer Usuário`
    - `Gerenciar Usuários`
    - `Sair`
  - Layout escuro, container central com bordas arredondadas, botões grandes e legíveis.

- **Cadastro de usuário**
  - Caixa de diálogo moderna solicitando:
    - **Nome completo** (obrigatório).
    - **CPF** com 11 dígitos numéricos (obrigatório).
  - Abre a webcam e exibe mensagens:
    - “**Olhe para a câmera por alguns segundos para registrar seu rosto.**”
    - Caixa **verde** em volta do rosto quando é detectado um **rosto vivo** (movimento).
    - Caixa **vermelha** e mensagem:
      - “**Rosto estático (possível foto). Mova-se um pouco.**”
      - quando parece ser uma foto ou rosto parado demais.
  - Captura **automaticamente 5 imagens** apenas quando:
    - Há rosto detectado.
    - Há movimento entre frames (heurística de vivacidade).
  - Se nenhuma imagem for capturada (por exemplo, o usuário sai da frente da câmera):
    - O cadastro é **cancelado** e o usuário removido de `users.json`.

- **Treinamento / atualização da base**
  - Botão **“Treinar Modelo / Atualizar Base”**:
    - Lê todos os usuários ativos da base (`database/users.json`).
    - Percorre `images/<user_id>/`.
    - Gera embeddings com **DeepFace (Facenet512)**.
    - Salva a lista em `database/embeddings.pkl`.
  - Script opcional `train_classifier.py`:
    - Lê `embeddings.pkl`, grava no SQLite (`facepro.db`).
    - Separa treino/teste, treina um **SVM linear**.
    - Imprime **acurácia** e relatório de classificação no terminal.
    - Salva o modelo em `database/face_classifier.pkl`.
    - Registra métricas na tabela `metrics` do SQLite.

- **Reconhecimento em tempo real**
  - Botão **“Reconhecer Usuário”**:
    - Abre a webcam.
    - Para cada frame:
      - Detecta o rosto (DeepFace).
      - Calcula o embedding.
      - Tenta reconhecer:
        - Primeiro usando o **classificador SVM** (se `face_classifier.pkl` existir).
        - Se a confiança for baixa ou o ID não estiver mais em `users.json`, usa **distância L2** com `embeddings.pkl` (apenas embeddings dos usuários ativos).
      - Aplica heurística de vivacidade:
        - Se o rosto ficar praticamente igual em vários frames:
          - Mostra “**Rosto estático (possível foto)**” em **amarelo**.
          - **Não libera acesso**.
      - Se um usuário ativo for reconhecido com vivacidade aceitável:
        - Exibe o nome em **verde**, por exemplo:
          - `João Silva (proba=0.85)`
  - Após alguns segundos com o mesmo rosto vivo:
    - Abre uma janela “**Acesso liberado**” com:
      - Foto recortada do rosto.
      - Nome + CPF.
      - Data e horário do acesso.

- **Gerenciamento de usuários**
  - Tela **“Gerenciar Usuários”**:
    - Lista usuários em uma tabela com **ID, Nome, CPF**.
    - Permite excluir um usuário.
    - Ao excluir:
      - Remove o registro de `users.json`.
      - Apaga a pasta `images/<id>/`.
      - Regera embeddings automaticamente (atualiza `embeddings.pkl`).

- **Histórico de acessos**
  - Cada reconhecimento bem-sucedido gera uma linha em:
    - `database/access_log.csv`
  - Campos:
    - `timestamp;user_id;name;cpf`
  - É possível abrir esse arquivo em Excel, LibreOffice, etc., para auditoria.

---

## 3. Estrutura do projeto

facial_system/
│── app.py                     # Aplicação principal com interface (PyQt5)
│── capture_faces.py           # Captura automática de rostos para cadastro
│── recognize_face.py          # Reconhecimento facial em tempo real + liveness
│── train_embeddings.py        # Geração de embeddings faciais (DeepFace)
│── train_classifier.py        # Treina SVM e avalia acurácia (opcional)
│── ingest_lfw.py              # Script opcional para importar o dataset LFW
│── sql_database.py            # Acesso ao banco SQLite (embeddings, métricas)
│── database_utils.py          # Utilitários para users.json e pastas de imagens
│── config.py                  # Configurações de caminhos, logging, etc.
│
│── database/
│     ├── users.json           # Banco simples com usuários (id, nome, cpf)
│     ├── embeddings.pkl       # Embeddings gerados pelo DeepFace
│     ├── facepro.db           # Banco SQLite com subjects, embeddings, metrics
│     ├── face_classifier.pkl  # Classificador SVM treinado (opcional)
│     └── access_log.csv       # Histórico de acessos reconhecidos
│
│── images/
│     └── <user_id>/           # Pastas com fotos de cada usuário (criado em runtime)
│
│── ui/
│     └── assets/              # Ícones e imagens da interface (opcional)
│
│── requirements.txt           # Dependências Python
│── README.md                  # Este arquivo---

## 4. Pré-requisitos

- **Python 3.10** (recomendado 3.10.11, usado no desenvolvimento).
- Webcam funcional conectada ao computador.
- Acesso à internet na primeira execução, para download automático dos pesos dos modelos DeepFace (se ainda não estiverem em cache).

---

## 5. Como executar o projeto

### 5.1. Criar e ativar o ambiente virtual

Na raiz do projeto (`facepro/` ou `facial_system/` dependendo de como você organizou):

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate### 5.2. Instalar dependências

pip install --upgrade pip
pip install -r facial_system/requirements.txt### 5.3. Rodar a aplicação

Na raiz do projeto:

python facial_system/app.pyA interface gráfica será aberta.

---

## 6. Passo a passo de uso

### 6.1. Cadastrar usuário

1. Clique em **“Cadastrar Usuário”**.
2. Na janela de cadastro:
   - Digite o **Nome completo**.
   - Digite o **CPF** com **11 dígitos numéricos** (sem ponto ou traço).
   - Clique em **“Iniciar captura”**.
3. A janela da webcam será aberta:
   - Mensagem: “**Olhe para a câmera por alguns segundos para registrar seu rosto.**”
   - Quando um rosto real for detectado:
     - Caixa **verde** ao redor do rosto.
     - Mensagem: “**Rosto detectado. Mantenha-se olhando para a câmera.**”
   - Se o sistema detectar um rosto muito parado (possível foto):
     - Caixa **vermelha** ao redor do rosto.
     - Mensagem: “**Rosto estático (possível foto). Mova-se um pouco.**”
4. O sistema captura automaticamente **5 imagens** enquanto o rosto real está estável.
5. Se você apertar **Q** antes de capturar imagens:
   - Nenhuma imagem é salva.
   - O cadastro é cancelado e o usuário é removido automaticamente.

### 6.2. Treinar / atualizar os embeddings

1. Após cadastrar ou remover usuários, clique em **“Treinar Modelo / Atualizar Base”**.
2. O sistema:
   - Recalcula embeddings para todas as imagens em `images/<id>/`.
   - Atualiza `database/embeddings.pkl`.

Isso já é suficiente para o reconhecimento ao vivo funcionar com esses usuários.

### 6.3. Reconhecer usuário

1. Clique em **“Reconhecer Usuário”**.
2. A webcam será aberta.
3. O sistema:
   - Detecta o rosto.
   - Calcula o embedding.
   - Tenta reconhecer:
     - Com o **classificador SVM** (se existir `face_classifier.pkl`).
     - Caso contrário, ou se a confiança for baixa, usa **distância L2** com `embeddings.pkl`.
   - Aplica liveness:
     - Se for **possível foto**, mostra aviso em amarelo e **não libera acesso**.
4. Quando um usuário **cadastrado e ativo** é reconhecido:
   - O nome aparece em verde (com probabilidade ou distância).
   - Após alguns segundos de rosto estável:
     - Abre a tela **“Acesso liberado”** com:
       - Foto do rosto.
       - Nome + CPF.
       - Data e horário.
5. Cada acesso reconhecido é registrado em:

facial_system/database/access_log.csv6. Aperte **Q** para encerrar o reconhecimento.

### 6.4. Gerenciar usuários

1. Clique em **“Gerenciar Usuários”**.
2. Na tabela são exibidos:
   - ID, Nome, CPF.
3. Selecione um usuário e clique em **“Excluir selecionado”**:
   - Remove o usuário de `users.json`.
   - Apaga `images/<id>/`.
   - Regenera `embeddings.pkl`.

---

## 7. Treinar e avaliar o classificador (opcional)

Se quiser treinar e avaliar um **classificador SVM** com acurácia no console:

1. Com o ambiente virtual ativo, rode:

python facial_system/train_classifier.py2. O script irá:
   - Rodar `train_embeddings.py` internamente.
   - Gravar embeddings no SQLite (`database/facepro.db`).
   - Dividir em treino/teste e treinar um SVM linear.
   - Imprimir no terminal:
     - **Acurácia no conjunto de teste**.
     - Um `classification_report` por usuário.
   - Salvar o modelo pronto em:
     - `database/face_classifier.pkl`.
   - Registrar métricas na tabela `metrics` do `facepro.db`.

Esse classificador é utilizado automaticamente por `recognize_face.py` quando presente.

---

## 8. Tecnologias utilizadas

- **Linguagem**: Python 3.10
- **Interface gráfica**: PyQt5
- **Visão computacional / IA**:
  - OpenCV
  - DeepFace (modelo Facenet512)
  - NumPy
  - scikit-learn (SVM, métricas)
- **Persistência**:
  - JSON (`users.json`)
  - Pickle (`embeddings.pkl`, `face_classifier.pkl`)
  - SQLite (`facepro.db`)
  - CSV (`access_log.csv`)
- **Outros**:
  - Logging (`facial_system.log`)
  - Scripts de ingestão de datasets (ex.: `ingest_lfw.py` para LFW)

---

## 9. Limitações e próximos passos

- A detecção de vivacidade é **heurística**, baseada apenas em movimento entre frames 2D:
  - Bloqueia bem fotos paradas, mas não substitui um modelo profissional de anti-spoofing.
- Para uso em cenários de alta segurança (banco, acesso físico crítico) seria desejável:
  - Integrar um modelo de anti-spoofing específico (rede treinada para `real` x `fake`).
  - Ou usar sensores adicionais (profundidade/IR) ou desafios ativos mais robustos.

Mesmo assim, para fins de estudo e demonstração, o sistema mostra um pipeline completo de:

> captura → cadastro → treinamento → reconhecimento → logging de acessos

com uma arquitetura organizada e extensível.

---
