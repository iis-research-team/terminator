import re
import functools
from typing import List, Dict, Set
from collections import OrderedDict

import orjson
import numpy as np
import fasttext as ft
from pymystem3 import Mystem
from sklearn.metrics.pairwise import cosine_similarity

from utils.paths import WIKIDATA_DUMP_PATH, FASTTEXT_MODEL_PATH


class RussianEntityLinker:

    def __init__(self):
        self._morph = Mystem()
        self._ft_model = ft.load_model(FASTTEXT_MODEL_PATH)

    def get_linked_mention(self, term: str, context: List[str]):
        """ Получение сущностей из викидаты
        :param term: слово или фраза, для которой нужно найти сущности в базе знаний
        :param context: список слов, которые окружают входную сущность
        :return: отранжированный список идентификаторов сущностей
        """
        normalized_term = self._normalize_phrase_by_word(term).lower()
        candidates, weights = self._get_candidates(normalized_term)
        context_vector = self._get_fasttext_vectors_for_phrase(' '.join(context))
        ranged_candidates = self._range_candidates_by_cosine_similarity(candidates, weights, context_vector)
        return ranged_candidates

    @functools.lru_cache(maxsize=10000)
    def _get_candidates(self, normalized_term: str) -> Dict[str, int]:
        """ Получение кандидатов сущностей (по строковому совпадению)
        :param normalized_term: лемматизированное слово или фраза
        :return: множество идентификаторов сущностей
        """
        candidates = dict()
        weights = dict()
        print(f'iter dump for entity [{normalized_term}]...')
        with open(WIKIDATA_DUMP_PATH, 'r') as f:
            _normalized_terms = self._generate_bi_tri_grams(normalized_term)
            for line in f:
                if line == '\n':
                    continue
                try:
                    entity = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                if entity['type'] != 'item':
                    continue
                entity_names = set()
                full_desc = list()
                if 'label' in entity:
                    entity_names.add(entity['label']['value'].lower())
                if 'alias' in entity:
                    for alias in entity['alias']:
                        entity_names.add(alias.lower())
                full_desc.extend(entity_names)
                if 'description' in entity:
                    desc = entity['description']['value']
                    if 'страница значений' in desc.lower():
                        continue
                    full_desc.append(desc)
                if normalized_term in entity_names:
                    vector = self._get_fasttext_vectors_for_phrase(' '.join(full_desc))
                    # здесь вес равен 1, т.к. нашли полный матчинг строк
                    candidates[entity['id']] = vector
                    weights[entity['id']] = 1.0
                else:
                    for t in _normalized_terms:
                        if t in entity_names:
                            vector = self._get_fasttext_vectors_for_phrase(' '.join(full_desc))

                            unique_normalized_tokens = set(normalized_term.split())
                            # считаем пересечение токенов между найденной фразой и входной
                            n_shared_tokens = len(set(t.split()) & unique_normalized_tokens)
                            weight = n_shared_tokens / len(unique_normalized_tokens)

                            candidates[entity['id']] = vector
                            weights[entity['id']] = weight
                            break
        return candidates, weights

    def _generate_bi_tri_grams(self, term: str) -> Set[str]:
        input_terms = term.split()
        result_grams = list()
        result_grams.extend(input_terms)
        for i, t in enumerate(input_terms):
            if t != input_terms[-1]:
                result_grams.append(' '.join(input_terms[i:i + 2]))
            try:
                if t != input_terms[-2]:
                    result_grams.append(' '.join(input_terms[i:i + 3]))
            except IndexError:
                continue
        return set(result_grams)

    def _range_candidates_by_cosine_similarity(self, candidates, weights, vector):
        distances = dict()
        for candidate, candidate_vector in candidates.items():
            distances[candidate] = cosine_similarity([vector], [candidate_vector])[0] * weights[candidate]
        sorted_candidates = OrderedDict(sorted(distances.items(), key=lambda x: x[1], reverse=True))

        # sorted_candidates = sorted(candidates.items(), key=lambda x: cosine_similarity([x[1]], [vector]), reverse=True)
        ranged_candidates = [candidate[0] for candidate in sorted_candidates]
        # return ranged_candidates
        return sorted_candidates

    @functools.lru_cache(maxsize=10000)
    def _normalize_phrase_by_word(self, phrase: str) -> str:
        """Нормализует фразу по отдельному слову слово: обработка естественного языка -> "обработка естественный язык"""
        lemmas = self._morph.lemmatize(phrase)
        return ''.join(lemmas[:-1])

    @functools.lru_cache(maxsize=10000)
    def _get_fasttext_vectors_for_phrase(self, phrase: str):
        """ Считает вектор для фразы/строки
        :param phrase: строка
        :return: усредненный вектор
        """
        phrase = re.sub('[.,!?:;]', ' ', phrase)
        phrase = re.sub(' {2}', ' ', phrase)
        wordlist = self._normalize_phrase_by_word(phrase).split()
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
