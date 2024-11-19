import re
import functools
from typing import List, Dict, Any
from collections import OrderedDict

import numpy as np
import fasttext as ft
from sklearn.metrics.pairwise import cosine_similarity

from utils.normalize import normalize_mystem
from entity_linker.entity_linking_pipeline.candidates_ranger import BaseCandidatesRanger
from utils.paths import FASTTEXT_MODEL_PATH


class CosineSimRanger(BaseCandidatesRanger):

    def __init__(self):
        fasttext_model_path = FASTTEXT_MODEL_PATH
        self._ft_model = ft.load_model(FASTTEXT_MODEL_PATH)

    def range_candidates_set(self, candidates: List[Dict[str, Any]], context: List[str], **kwargs) -> Dict[str, float]:
        _context_vector = self._get_vector_for_phrase(' '.join(context))
        sorted_candidates = self._range_candidates_by_cosine_similarity(candidates, _context_vector)
        return sorted_candidates

    def _range_candidates_by_cosine_similarity(self, candidates: List[Dict[str, Any]], vector) -> Dict[str, int]:
        distances = dict()
        for candidate_dict in candidates:
            full_desc = candidate_dict['names']
            full_desc.extend([candidate_dict['desc']])
            candidate_vector = self._get_vector_for_phrase(' '.join(full_desc))
            distances[candidate_dict['id']] = cosine_similarity([vector], [candidate_vector])[0]
        sorted_candidates = OrderedDict(sorted(distances.items(), key=lambda x: x[1], reverse=True))

        return sorted_candidates

    @functools.lru_cache(maxsize=10000)
    def _get_vector_for_phrase(self, phrase: str) -> np.array:
        """ Считает вектор для фразы/строки
        :param phrase: строка
        :return: усредненный вектор
        """
        phrase = re.sub('[.,!?:;]', ' ', phrase)
        phrase = re.sub(' {2}', ' ', phrase)
        wordlist = normalize_mystem(phrase).split()
        sentence_vec = np.zeros((300,))
        number_of_words = len(wordlist)
        for word in wordlist:
            wordvec = self._ft_model.get_word_vector(str(word))
            if wordvec.any():
                sentence_vec += wordvec
            else:
                number_of_words -= 1
        if number_of_words == 0:
            return sentence_vec
        else:
            return sentence_vec / number_of_words
