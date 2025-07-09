import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

from .preprocessing import limpar_texto, vetorizar_textos

MODELO_PATH = "app/modelos"
os.makedirs(MODELO_PATH, exist_ok=True)

def treinar_modelo(salvar_resultado_json=True):
    df = pd.read_csv(
        "app/data/dataset_emocoes_sintetico.csv",
        usecols=[0, 1],
        names=["texto", "emocao"],
        skiprows=1
    )

    # Limpeza da coluna de rótulos de emoção (remove ;;; e espaços extras)
    df["emocao"] = df["emocao"].astype(str).str.strip().str.replace(";", "", regex=False)

    df['texto'] = df['texto'].astype(str)
    df['clean_text'] = df['texto'].apply(limpar_texto)
    y = df['emocao']

    X_tfidf, vetor = vetorizar_textos(df['clean_text'])

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)

    joblib.dump(modelo, os.path.join(MODELO_PATH, "modelo_emocao.pkl"))
    joblib.dump(vetor, os.path.join(MODELO_PATH, "vetor_tfidf.pkl"))

    y_pred = modelo.predict(X_test)
    relatorio = classification_report(y_test, y_pred, output_dict=True)

    if salvar_resultado_json:
        with open(os.path.join(MODELO_PATH, "relatorio_avaliacao.json"), "w") as f:
            json.dump(relatorio, f, indent=4)

    return relatorio

def carregar_modelo():
    modelo_path = os.path.join(MODELO_PATH, "modelo_emocao.pkl")
    vetor_path = os.path.join(MODELO_PATH, "vetor_tfidf.pkl")
    if not os.path.exists(modelo_path) or not os.path.exists(vetor_path):
        raise FileNotFoundError("Modelo ou vetor TF-IDF não encontrado. Treine o modelo primeiro.")
    modelo = joblib.load(modelo_path)
    vetor = joblib.load(vetor_path)
    return modelo, vetor

def carregar_resultados():
    with open(os.path.join(MODELO_PATH, "relatorio_avaliacao.json"), "r") as f:
        return json.load(f)

if __name__ == "__main__":
    relatorio = treinar_modelo()
    print("Modelo treinado com sucesso!")
    print("Acurácia:", relatorio["accuracy"])
