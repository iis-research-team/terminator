import functools

from pymystem3 import Mystem

morph = Mystem()


@functools.lru_cache(maxsize=10000)
def normalize_mystem(term: str) -> str:
    lemmas = morph.lemmatize(term)
    return ''.join(lemmas[:-1])
