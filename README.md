# API de Detecção de Emoções em Frases

Esta API classifica emoções predominantes em frases em português, utilizando técnicas de Processamento de Linguagem Natural (PLN) com TF-IDF e Regressão Logística. As emoções previstas incluem: **alegria, tristeza, medo, raiva, surpresa e nojo**.

---

## 📁 Estrutura do Projeto

```
Prova-LDDM-II/
├── app/
│   ├── data/
│   │    └── dataset_emocoes_sintetico.csv  # Base de dados
│   ├── main.py               # API FastAPI
│   ├── model.py              # Treinamento e carregamento do modelo
│   ├── preprocessing.py      # Limpeza e vetorizacao de texto
│   └── modelos/              # Modelos salvos (após treino)
├── requirements.txt
```

---

## ✅ Instalação

```bash
# 1. Crie o ambiente virtual (opcional)
python -m venv venv
venv\Scripts\activate   # Windows

# 2. Instale as dependências
pip install -r requirements.txt
```

---

## ⚙️ Treinamento do Modelo

Antes de rodar a API, treine o modelo com:

```bash
python -m app.model
```

Isso irá gerar os arquivos em `app/modelos/`.

---

## 🚀 Rodando a API

```bash
uvicorn app.main:app --reload
```

Acesse a documentação automática da API em:

[http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔎 Endpoints

### POST `/api/detectar-emocao`

Recebe uma frase e retorna a emoção detectada.

**Exemplo de requisição:**

```json
{
  "frase": "Estou muito feliz hoje!"
}
```

**Exemplo de resposta:**

```json
{
  "emocao_predita": "alegria",
  "probabilidades": {
    "alegria": 0.987,
    "medo": 0.003,
    ...
  }
}
```

---

### GET `/api/resultados`

Retorna métricas de avaliação do modelo (acurácia e precisão por classe).

---

## 📌 Requisitos

* Python 3.8+
* FastAPI
* Scikit-learn
* NLTK
