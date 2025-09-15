# src/utils/transformers.py

import re
import unicodedata
from typing import List
import emoji

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.base import BaseEstimator, TransformerMixin

# Télécharger les ressources si non présentes
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Stopwords anglais avec maintien de la négation
neg_keep = {"not", "no", "nor", "n't"}
stop_en = {w for w in set(stopwords.words("english")) if w not in neg_keep}

# Initialisations
stemmer = PorterStemmer()

# Regex & contractions
CONTRACTIONS = {
    "won't": "will not", "can't": "can not", "ain't": "is not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "shouldn't": "should not", "wouldn't": "would not", "couldn't": "could not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "mustn't": "must not", "needn't": "need not", "n't": " not"
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
HTML_RE = re.compile(r"&\w+;")
NONALPHA_RE = re.compile(r"[^a-zA-Z\s:_]")
WS_RE = re.compile(r"\s+")
REPEAT_RE = re.compile(r"(.)\1{2,}")  # aaaa -> aa

def expand_contractions(text: str) -> str:
    for c, rep in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(c)}\b", rep, text)
    return text

class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn compatible pour nettoyage de texte + stemming.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self._clean_and_stem(t) for t in X]

    def _clean_and_stem(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        t = unicodedata.normalize("NFKC", text)
        t = t.lower()
        t = expand_contractions(t)
        t = URL_RE.sub(" ", t)
        t = MENTION_RE.sub(" ", t)
        t = HASHTAG_RE.sub(" ", t)
        t = HTML_RE.sub(" ", t)
        t = REPEAT_RE.sub(r"\1\1", t)
        t = NONALPHA_RE.sub(" ", t)
        t = WS_RE.sub(" ", t).strip()
        tokens = t.split()
        tokens = [w for w in tokens if (len(w) >= 3) and (w in neg_keep or w not in stop_en)]
        stems = [stemmer.stem(t) for t in tokens]
        return " ".join(stems)
    