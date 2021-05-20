from typing import List, Tuple

import pymorphy2
from pymorphy2.analyzer import Parse

from utils.constants import Tags, TERM_SET

ADJF = 'ADJF'
CONJ = 'CONJ'
GRND = 'GRND'
NOUN = 'NOUN'
PREP = 'PREP'
PRTF = 'PRTF'
PRTS = 'PRTS'
VERB = 'VERB'

GENT = 'gent'


class HeuristicValidator:

    def __init__(self):
        self._morph = pymorphy2.MorphAnalyzer()

    def validate(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        self._heuristic_1(result)
        self._heuristic_2(result)
        self._heuristic_3(result)
        self._heuristic_4(result)
        self._heuristic_5(result)
        self._heuristic_6(result)
        self._heuristic_7(result)
        return result

    def _heuristic_1(self, result: List[Tuple[str, str]]):
        """ Валидация цепочек, которые представляют собой СУЩ + СУЩ в род.п., например: методы сжатия данных"""

        for result_pair_1, result_pair_2 in zip(result, result[1:]):
            id_1 = result.index(result_pair_1)
            id_2 = id_1 + 1
            # если последовательность не содержит терминов, то пропускаем
            if result_pair_1[1] == Tags.NOT_TERM.value and result_pair_2[1] == Tags.NOT_TERM.value:
                continue
            token_1, token_2 = result_pair_1[0], result_pair_2[0]
            pos_1 = self._morph.parse(token_1)[0].tag.POS
            case_2 = self._morph.parse(token_2)[0].tag.case
            if pos_1 in [NOUN, ADJF] and case_2 == GENT:
                if result_pair_1[1] not in TERM_SET:
                    result_pair_1 = (token_1, Tags.B_TERM.value)
                    result[id_1] = result_pair_1
                result_pair_2 = (token_2, Tags.I_TERM.value)
                result[id_2] = result_pair_2

    def _heuristic_2(self, result: List[Tuple[str, str]]):
        """
        Если токены представляют собой последовательность ПРИЛ + СУЩ и оба помечены B-TERM, то приводим к
        последовательности B-TERM I-TERM
        """

        for result_pair_1, result_pair_2 in zip(result, result[1:]):
            id_1 = result.index(result_pair_1)
            id_2 = id_1 + 1
            # если последовательность не содержит терминов, то пропускаем
            if result_pair_1[1] == Tags.B_TERM.value and result_pair_2[1] == Tags.B_TERM.value:
                token_1, token_2 = result_pair_1[0], result_pair_2[0]
                parse_1 = self._morph.parse(token_1)
                parse_2 = self._morph.parse(token_2)
                is_adj_1 = self.__check_pos(ADJF, parse_1)
                is_noun_2 = self.__check_pos(NOUN, parse_2)
                if is_adj_1 and is_noun_2:
                    result[id_1] = (token_1, Tags.B_TERM.value)
                    result[id_2] = (token_2, Tags.I_TERM.value)

    def _heuristic_3(self, result: List[Tuple[str, str]]):
        """ Удаление тэга B-TERM или I-TERM, если он был присовен токену знака пунктуации """
        for i, result_pair in enumerate(result):
            if result_pair[0] in ['.', ',', ':', ';'] and result_pair[1] in [Tags.B_TERM.value, Tags.I_TERM.value]:
                result[i] = (result_pair[0], Tags.NOT_TERM.value)

    def _heuristic_4(self, result: List[Tuple[str, str]]):
        """ Если последний токен в термине имеет часть речи ПРИЛ, а следующий токен - СУЩ, но либо не входит в термин,
        либо имеет тэг "B-TERM", то второй токен включаем в состав термина
        """
        for result_pair_1, result_pair_2 in zip(result, result[1:]):
            id_1 = result.index(result_pair_1)
            id_2 = id_1 + 1
            if result_pair_1[1] in TERM_SET and result_pair_2[1] != Tags.I_TERM.value:
                token_1, token_2 = result_pair_1[0], result_pair_2[0]
                parse_1 = self._morph.parse(token_1)
                parse_2 = self._morph.parse(token_2)
                is_adj_1 = self.__check_pos(ADJF, parse_1)
                is_prtf_1 = self.__check_pos(PRTF, parse_1)
                is_noun_2 = self.__check_pos(NOUN, parse_2)
                if is_noun_2:
                    if is_adj_1 or is_prtf_1:
                        result[id_2] = (token_2, Tags.I_TERM.value)

    def _heuristic_5(self, result: List[Tuple[str, str]]):
        """ Удаление тэга B-TERM у предлога и союза (допускаем, что предлог может входить в состав термина, но не может
        начинать его """
        for i, result_pair in enumerate(result):
            if result_pair[1] == Tags.B_TERM.value:
                parse = self._morph.parse(result_pair[0])
                is_prep = self.__check_pos(PREP, parse)
                is_conj = self.__check_pos(CONJ, parse)
                if is_prep or is_conj:
                    result[i] = (result_pair[0], Tags.NOT_TERM.value)

    def _heuristic_6(self, result: List[Tuple[str, str]]):
        """ Удаление тэга Термин у однозначного глагола или деепричастия """
        for i, result_pair in enumerate(result):
            if result_pair[1] in [Tags.B_TERM.value, Tags.I_TERM.value]:
                parse = self._morph.parse(result_pair[0])
                is_verb = self.__check_pos(VERB, parse)
                is_grnd = self.__check_pos(GRND, parse)
                is_prts = self.__check_pos(PRTS, parse)
                if len(parse) == 1:
                    if is_verb or is_grnd or is_prts:
                        result[i] = (result_pair[0], Tags.NOT_TERM.value)

    def _heuristic_7(self, result: List[Tuple[str, str]]):
        """ Если следующий за термином токен состоит только из латинских символов, то включаем его в состав термина """
        for result_pair_1, result_pair_2 in zip(result, result[1:]):
            id_1 = result.index(result_pair_1)
            id_2 = id_1 + 1
            if result_pair_1[1] in TERM_SET:
                token_2 = result_pair_2[0]
                is_latin = self._is_latin(token_2)
                if is_latin:
                    result[id_2] = (token_2, Tags.I_TERM.value)

    def __check_pos(self, pos: str, parses: List[Parse]) -> bool:
            for parse in parses:
                if pos in parse.tag:
                    return True
            return False

    def _is_latin(self, token: str) -> bool:
        latin_symbols = 'qwertyuiopasdfghjklzxcvbnm'
        is_latin = True
        for char in token.lower():
            if char not in latin_symbols:
                is_latin = False
                return is_latin
        return is_latin
