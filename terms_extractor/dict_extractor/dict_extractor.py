import os
from typing import List, Tuple, Set, Union

import pymorphy2
from ahocorapy.keywordtree import KeywordTree

from utils.constants import Tags
from utils.utilities import tokenize
from utils.paths import DICT_EXTRACTOR_PATH
from terms_extractor.base_extractor import BaseExtractor

TERMS_DIR_NAME = 'ngramm_lemma_terms'


class DictExtractor(BaseExtractor):
    """ Класс для извлечения терминов с помощью словаря """

    def __init__(self):
        self._kw_trees = self._load_kw_trees()
        self._morph = pymorphy2.MorphAnalyzer()

    def extract(self, text: Union[str, List[str]]) -> List[Tuple[str, str]]:
        """ Извлекает термины из входного текста

        :param text: входной текст
        :return: Результат в виде списка кортежей  (Термин, Тэг)
        """
        resulted_tokens = []
        unique_indices = set()
        tokens, normalized_tokens = self._normalize_tokens(text)
        for kw_tree in self._kw_trees:
            self._search_for_key_words(kw_tree, normalized_tokens, resulted_tokens, unique_indices)
        result = self._aggregate_result(tokens, resulted_tokens)
        return result

    def _load_kw_trees(self) -> List[KeywordTree]:
        """ Загружает префиксные деревья для терминов из словарей (название каждого файла соответствует количеству
        токенов в терминах этого файла

        :return: Список префиксных деревьев
        """
        fnames = ['1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '6.txt', '7.txt', '8.txt', '9.txt', '10.txt', '11.txt',
                  '12.txt', '13.txt', '14.txt', '20.txt']
        files_dir_path = os.path.join(DICT_EXTRACTOR_PATH, TERMS_DIR_NAME)
        kw_trees = []
        for fname in fnames[::-1]:
            kwtree = KeywordTree()
            with open(os.path.join(files_dir_path, fname), 'r') as f:
                for ngramm in f.read().split('\n'):
                    if ngramm != '':
                        kwtree.add(ngramm.split())
                kwtree.finalize()
                kw_trees.append(kwtree)
        return kw_trees

    def _aggregate_result(self, tokens: List[str], resulted_tokens: List[List[int]]) -> List[Tuple[str, str]]:
        """ Аггрегация результатов: для каждого токена определяется его тэг

        :param tokens: Список токенов
        :param resulted_tokens: Список id токенов, которые являются терминами
        :return:
        """
        result = []
        is_previous_term = False
        for i, token in enumerate(tokens):
            tag = ''
            is_term = False
            for ids in resulted_tokens:
                if i in ids:
                    if ids.index(i) > 0:
                        tag = Tags.I_TERM.value
                        is_term = True
                        is_previous_term = True
                    else:
                        if is_previous_term:
                            tag = Tags.I_TERM.value
                        else:
                            tag = Tags.B_TERM.value
                        is_term = True
                        is_previous_term = True
            if not is_term:
                tag = Tags.NOT_TERM.value
                is_previous_term = False
            result.append((token, tag))
        return result

    def _normalize_tokens(self, sentence_string: Union[str, List[str]]) -> Tuple[List[str], List[str]]:
        """ Токенизирует входной текст и нормализует полученные токены

        :param sentence_string: входной текст
        :return: Список токенов и список лемматизированных токенов
        """
        if isinstance(sentence_string, str):
            tokens = tokenize(sentence_string)
        else:
            tokens = sentence_string
        normalized_tokens = []
        for token in tokens:
            normalized_token = self._morph.parse(token)[0].normal_form
            normalized_tokens.append(normalized_token)
        return tokens, normalized_tokens

    def _search_for_key_words(
            self, kw_tree: KeywordTree, tokens: List[str], resulted_tokens: List[int], unique_indices: Set[int]):
        """ Ищет термины в тексте

        :param kw_tree: Префиксное дерево
        :param tokens: Список токенов, среди которых ищутся термины
        :param resulted_tokens: Список токенов, которые являются терминами
        :param unique_indices: Уникальные id токенов, которые являются терминами (нужно, чтобы избежать пересечения
        терминов)
        """
        for result in kw_tree.search_all(tokens):
            indexes = []
            tokens_r = result[0]
            start_index = result[1]
            for k in range(len(tokens_r)):
                indexes.append(start_index + k)
            is_in_indices = False
            for i in indexes:
                if i in unique_indices:
                    is_in_indices = True
            if not is_in_indices:
                resulted_tokens.append(indexes)
            for i in indexes:
                unique_indices.add(i)
