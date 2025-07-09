import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    stop_words = set(stopwords.words("portuguese"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("portuguese"))

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"\d+", "", texto)
    texto = re.sub(f"[{string.punctuation}]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    tokens = texto.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

def vetorizar_textos(textos):
    vetor = TfidfVectorizer()
    X = vetor.fit_transform(textos)
    return X, vetor