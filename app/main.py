from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.model import carregar_modelo, carregar_resultados
from app.preprocessing import limpar_texto

modelo, vetor = carregar_modelo()

class EntradaTexto(BaseModel):
    frase: str

app = FastAPI(title="API de Detecção de Emoções")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/detectar-emocao")
def detectar_emocao(entrada: EntradaTexto):
    texto_limpo = limpar_texto(entrada.frase)
    vetor_frase = vetor.transform([texto_limpo])
    predicao = modelo.predict(vetor_frase)[0]
    probabilidades = modelo.predict_proba(vetor_frase)[0]
    labels = modelo.classes_
    return {
        "emocao_predita": predicao,
        "probabilidades": dict(zip(labels, probabilidades.round(3)))
    }

@app.get("/api/resultados")
def resultados():
    relatorio = carregar_resultados()
    return {
        "accuracy": relatorio["accuracy"],
        "precision_por_classe": {
            k: v["precision"] for k, v in relatorio.items() if isinstance(v, dict)
        }
    }