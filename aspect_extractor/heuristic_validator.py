from typing import List, Tuple

import pymorphy2
from pymorphy2.analyzer import Parse


class HeuristicValidator:

    def __init__(self):
        self._morph = pymorphy2.MorphAnalyzer()
        self._long_aspects = ['Contrib', 'Conc', 'Goal']
        self._paired_punct = ['"', '\'', '(', ')', '«', '»']

    def validate(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        result = self._heuristic_1(result)  # точка
        result = self._heuristic_2(result)  # причастия
        result = self._heuristic_3(result)  # частица не
        result = self._heuristic_4(result)  # начало аспекта
        result = self._heuristic_5(result)  # конец аспекта
        result = self._heuristic_6(result)  # служебная часть речи одна
        result = self._heuristic_7(result)  # предлог в начале
        result = self._heuristic_8(result)  # разрывы
        result = self._heuristic_9(result)  # однословные аспекты

        return result

    def _heuristic_1(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Удаление аспекта, присвоенного точке"""

        updated_result = {i: res for i, res in enumerate(result)}

        for i, result_pair in enumerate(result):
            if result_pair[0] in ['.']:
                updated_result[i] = (result_pair[0], 'O')

        res = [updated_result[i] for i in range(len(result))]

        return res

    def _heuristic_2(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Включение страдательного причастия или возвратного глагола 3 лица настоящего времени (кроме форм глагола "являться") в идущий следом за ним тэг Contrib"""

        updated_result = {i: res for i, res in enumerate(result)}

        for i, result_pair in list(enumerate(result))[:-1]:
            if 'Contrib' not in result_pair[1] and 'Contrib' in result[i + 1][1]:
                parse = self._morph.parse(result_pair[0])
                if self.__check_pos('PRTS', parse) and self.__check_pos('pssv', parse):
                    contrib_added = self.__add_aspect(result_pair[1], 'Contrib')
                    updated_result[i] = (result_pair[0], contrib_added)
                elif self.__check_pos('VERB', parse) and self.__check_pos('3per', parse) and result_pair[0].endswith('ся') and 'являться' not in self.__check_normal_form(parse):
                    contrib_added = self.__add_aspect(result_pair[1], 'Contrib')
                    updated_result[i] = (result_pair[0], contrib_added)

        res = [updated_result[i] for i in range(len(result))]

        return res

    def _heuristic_3(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Если перед аспектом стоит частица "не", то её нужно включить в аспект"""

        updated_result = {i: res for i, res in enumerate(result)}

        for i, result_pair in list(enumerate(result))[:-1]:
            if result_pair[1] != result[i + 1][1] and result_pair[0] == 'не':
                updated_result[i] = (result_pair[0], result[i + 1][1])

        res = [updated_result[i] for i in range(len(result))]

        return res

    def _heuristic_4(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Аспект не может начинаться на знак препинания или союза"""

        updated_result = {i: res for i, res in enumerate(result)}

        for i, result_pair in list(enumerate(result))[1:-1]:
            if result_pair[1] != 'O':
                aspects_for_token = result_pair[1].split('|')
                auxilary = 0
                aspects_deleted = result_pair[1]
                for aspects_id, aspect, in enumerate(aspects_for_token):
                    if aspect not in result[i - 1][1]:
                        if auxilary == 0:
                            auxilary = -1
                            parse = self._morph.parse(result_pair[0])
                            if (self.__check_pos('PNCT', parse) and result_pair[
                                0] not in self._paired_punct) or self.__check_pos('CONJ', parse):
                                auxilary = 1
                        if auxilary == 1:
                            aspects_deleted = self.__delete_aspect(aspect, aspects_deleted)
                updated_result[i] = (result_pair[0], aspects_deleted)
        res = [updated_result[i] for i in range(len(result))]

        return res

    def _heuristic_5(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Аспект не может заканчиваться на знак препинания, союз или предлог. Чтобы удостовериться, что токен - не аббревиатура, проверяем, что он в нижнем регистре"""

        updated_result = {i: res for i, res in enumerate(result)}

        for i, result_pair in list(enumerate(result))[1:-1]:
            if result_pair[1] != 'O':
                aspects_for_token = result_pair[1].split('|')
                auxilary = 0
                aspects_deleted = result_pair[1]
                for aspects_id, aspect, in enumerate(aspects_for_token):
                    if aspect not in result[i + 1][1] and result_pair[0].islower():
                        if auxilary == 0:
                            auxilary = -1
                            parse = self._morph.parse(result_pair[0])
                            if (self.__check_pos('PNCT', parse) and result_pair[
                                0] not in self._paired_punct) or self.__check_multiple_pos(['CONJ', 'PREP'], parse):
                                auxilary = 1
                        if auxilary == 1:
                            aspects_deleted = self.__delete_aspect(aspect, aspects_deleted)
                updated_result[i] = (result_pair[0], aspects_deleted)
        res = [updated_result[i] for i in range(len(result))]

        return res

    def _heuristic_6(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Аспект не может состоять только из служебной части речи или занка препинания. """

        updated_result = {i: res for i, res in enumerate(result)}

        for i, result_pair in list(enumerate(result))[1:-1]:
            if result_pair[1] != 'O':
                aspects_for_token = result_pair[1].split('|')
                auxilary = 0
                aspects_deleted = result_pair[1]
                for aspects_id, aspect, in enumerate(aspects_for_token):
                    if aspect not in result[i - 1][1] and aspect not in result[i + 1][1]:
                        if auxilary == 0:
                            auxilary = -1
                            parse = self._morph.parse(result_pair[0])
                            if (self.__check_pos('PNCT', parse) and result_pair[
                                0] not in self._paired_punct) or self.__check_multiple_pos(['CONJ', 'PREP', 'PRCL'],parse):
                                auxilary = 1
                        if auxilary == 1:
                            aspects_deleted = self.__delete_aspect(aspect, aspects_deleted)
                updated_result[i] = (result_pair[0], aspects_deleted)
        res = [updated_result[i] for i in range(len(result))]

        return res

    def _heuristic_7(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Аспект не может начинаться на предлог (кроме аспектов Contrib и Conc) """

        updated_result = {i: res for i, res in enumerate(result)}

        for i, result_pair in list(enumerate(result))[1:-1]:
            if result_pair[1] != 'O':
                aspects_for_token = result_pair[1].split('|')
                auxilary = 0
                aspects_deleted = result_pair[1]
                for aspects_id, aspect, in enumerate(aspects_for_token):
                    if aspect not in result[i - 1][1] and aspect not in ['Contrib', 'Conc', 'Use', 'Adv']:
                        if auxilary == 0:
                            auxilary = -1
                            parse = self._morph.parse(result_pair[0])
                            if self.__check_pos('PREP', parse):
                                auxilary = 1
                        if auxilary == 1:
                            aspects_deleted = self.__delete_aspect(aspect, aspects_deleted)
                        updated_result[i] = (result_pair[0], aspects_deleted)
        res = [updated_result[i] for i in range(len(result))]

        return res

    def _heuristic_8(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Удаление разрывов между одинаковыми тэгами"""

        updated_result = {i: res for i, res in enumerate(result)}

        for i, result_pair in list(enumerate(result))[1:-1]:
            if result[i - 1][1] != 'O' and result_pair[1] != result[i - 1][1] and '|' not in result_pair[1] and result[i - 1][1] == result[i + 1][1] and result_pair[0] not in ['.', ',', 'и']:
                updated_result[i] = (result_pair[0], result[i - 1][1])

        res = [updated_result[i] for i in range(len(result))]

        return res

    def _heuristic_9(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Удаление однословных тегов"""

        updated_result = {i: res for i, res in enumerate(result)}

        for i, result_pair in list(enumerate(result))[1:-1]:
            if result_pair[1] != 'O':
                aspects_for_token = result_pair[1].split('|')
                aspects_deleted = result_pair[1]
                for aspects_id, aspect, in enumerate(aspects_for_token):
                    if aspect not in result[i - 1][1] and aspect not in result[i + 1][1] and aspect in self._long_aspects:
                        aspects_deleted = self.__delete_aspect(aspect, aspects_deleted)
                updated_result[i] = (result_pair[0], aspects_deleted)
        res = [updated_result[i] for i in range(len(result))]

        return res

    def __check_multiple_pos(self, pos_list: List[str], parses: List[Parse]) -> bool:
        for pos in pos_list:
            if self.__check_pos(pos, parses):
                return True
        return False

    def __check_pos(self, pos: str, parses: List[Parse]) -> bool:
        for parse in parses:
            if pos in parse.tag:
                return True
        return False

    def __check_normal_form(self, parses: List[Parse]) -> List[str]:
        normal_forms = []
        for parse in parses:
            normal_forms.append(parse.normal_form)
        return normal_forms

    def __add_aspect(self, tag:str, aspect:str) -> str:
        if tag == 'O':
            aspect_added = aspect
        elif '|' not in tag:
            aspect_added = tag + '|' + aspect
        else:
            aspect_added = tag
        return aspect_added

    def __delete_aspect(self, tag:str, aspect:str) -> str:
        if tag == aspect:
            aspect_deleted = 'O'
        else:
            aspect_deleted = tag.replace(aspect, '').replace('|', '')
        return aspect_deleted

