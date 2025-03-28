from typing import List, Tuple
import pymorphy2
from pymorphy2.analyzer import Parse


class HeuristicValidator:

    def __init__(self):
        self._morph = pymorphy2.MorphAnalyzer()

        self._isaspect = lambda label: label != 'O'
        self._isnested = lambda label: '|' in label
        self._islong = lambda aspect: aspect in ['Contrib', 'Conc', 'Task']
        self._canstartwithprep = lambda aspect: aspect in ['Contrib', 'Conc']

        self.R_PAIRED_PUNCT = ')»]}'
        self.L_PAIRED_PUNCT = '(«[{'
        self._ispunct = lambda parse, token: self.__check_pos('PNCT', parse)
        self._isconj_punct = lambda token: token in ['.', ',', 'и']

        # чтобы не спутать служебные части с аббревиатурами, проверяем, что они не в верхнем регистре.
        # при этом если слово состоит из одной буквы и стоит в начале предложения, считаем его служебной частью речи.
        self._isabbr = lambda cur_token, prev_token: cur_token.isupper() and len(cur_token) > 1 and prev_token != '.'
        # функция для проверки, является ли слово предлогом
        self._isprep = lambda parse, cur_token, prev_token: self.__check_pos('PREP', parse) and not self._isabbr(
            cur_token, prev_token)
        # функция для проверки, является ли слово союзом или частицей
        self._isconj_prcl = lambda parse, cur_token, prev_token: self.__check_multiple_pos(['CONJ', 'PRCL'],
                                                                                           parse) and not self._isabbr(
            cur_token, prev_token)
        # функция для проверки, является ли слово служебной частью речи
        self._isfunc_pos = lambda parse, cur_token, prev_token: self._isprep(parse, cur_token,
                                                                             prev_token) or self._isconj_prcl(parse,
                                                                                                              cur_token,
                                                                                                              prev_token)
        # функция для проверки, является ли слово самостоятельной частью речи
        self._istrue_pos = lambda parse, cur_token, prev_token: not self._ispunct(parse,
                                                                                  cur_token) and not self._isfunc_pos(
            parse, cur_token, prev_token)
        # функция для проверки, является ли слово страдательным причастием
        self._isprts_pssv = lambda parse: self.__check_pos('PRTS', parse) and self.__check_pos('pssv', parse)
        # функция для проверки, является ли слово глаголом 3 лица (кроме форм глагола являться)
        self._isverb_3per = lambda parse: self.__check_pos('VERB', parse) and self.__check_pos('3per',
                                                                                               parse) and not self.__check_normal_form(
            'являться', parse)

    def __check_multiple_pos(self, pos_list: List[str], parses: List[Parse]) -> bool:
        """
        Проверяем, может ли слово быть формой хотя бы одной из заданных частей речи
        :param pos_list: Список частей речи
        :param parses: Список возможных морфологических разборов слова
        :return: Может ли слово быть формой хотя бы одной из заданных частей речи
        """
        for pos in pos_list:
            if self.__check_pos(pos, parses):
                return True
        return False

    def __check_pos(self, pos: str, parses: List[Parse]) -> bool:
        """
        Проверяем, может ли слово быть формой заданной части речи
        :param pos_list: Список частей речи
        :param parses: Список возможных морфологических разборов слова
        :return: Может ли слово быть формой заданной части речи
        """
        for parse in parses:
            if pos in parse.tag:
                return True
        return False

    def __check_normal_form(self, word: str, parses: List[Parse]) -> bool:
        """
        Проверяем, является ли слово падежной формой заданного слова
        :param word: Заданное слово (в начальной форме)
        :param parses: Список возможных морфологических разборов слова
        :return: Является ли слово падежной формой заданного слова
        """
        for parse in parses:
            if word in parse.normal_form:
                return True
        return False

    def _add_aspect(self, aspect: str, label: str) -> str:
        """
        Добавление аспекта в тэг. Если тэг уже состоит из двух аспектов, то третий просто не добавляется
        :param aspect: Аспект, который нужно добавить
        :param label: Тэг, в который нужно добавить аспект
        :return: Тэг с добавленым аспектом
        """
        if label == 'O':
            aspect_added = aspect
        elif '|' not in label:
            aspect_added = label + '|' + aspect
        else:
            aspect_added = label
        return aspect_added

    def _delete_aspect(self, aspect: str, label: str) -> str:
        """
        Удаление аспекта из тэга
        :param aspect: Аспект, который нужно удалить
        :param label: Тэг, из которого нужно удалить аспект
        :return: Тэг с удаленным аспектом
        """
        if label == aspect:
            aspect_deleted = 'O'
        else:
            aspect_deleted = label.replace(aspect, '').replace('|', '')
        return aspect_deleted

    def _heuristic_dot(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Удаление аспекта, присвоенного точке
        :param result: Список кортежей, в которых первый элемент - токен, второй - тэг
        :return: Список кортежей, в которых первый элемент - токен, второй - тэг
        """
        updated_result = {i: res for i, res in enumerate(result)}
        for i, (cur_token, cur_label) in enumerate(result):
            if i + 1 < len(result):
                next_token, _ = result[i + 1]
            else:
                next_token = 'A'  # для последнего токена нет следующего, но проверить его нужно, поэтому присвоеим next_label капитализированную строку
            if cur_token == '.' and self._isaspect(cur_label) and next_token[0].isupper():
                updated_result[i] = (cur_token, 'O')
        return [updated_result[i] for i in range(len(result))]

    def _heuristic_verb(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Включение страдательного причастия или глагола 3 лица настоящего времени (кроме форм глагола "являться") в идущий следом за ним тэг Contrib
        :param result: Список кортежей, в которых первый элемент - токен, второй - тэг
        :return: Список кортежей, в которых первый элемент - токен, второй - тэг
        """
        updated_result = {i: res for i, res in enumerate(result)}
        for i, (cur_token, cur_label) in list(enumerate(result))[:-1]:
            next_token, next_label = result[i + 1]
            if 'Contrib' not in cur_label and 'Contrib' in next_label:
                cur_parse = self._morph.parse(cur_token)
                if self._isprts_pssv(cur_parse) or self._isverb_3per(cur_parse):
                    contrib_added = self._add_aspect('Contrib', cur_label)
                    updated_result[i] = (cur_token, contrib_added)
        return [updated_result[i] for i in range(len(result))]

    def _heuristic_end(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Аспект не может заканчиваться на знак препинания, союз, частицу или предлог.
        :param result: Список кортежей, в которых первый элемент - токен, второй - тэг
        :return: Список кортежей, в которых первый элемент - токен, второй - тэг
        """
        updated_result = {i: res for i, res in enumerate(result)}
        for i, (cur_token, cur_label) in list(enumerate(result))[1:-1]:
            prev_token, prev_label = result[i - 1]
            next_token, next_label = result[i + 1]
            if self._isaspect(cur_label):
                aspects_for_token = cur_label.split('|')
                aspects_deleted = cur_label
                for aspect in aspects_for_token:
                    if aspect not in next_label:
                        cur_parse = self._morph.parse(cur_token)
                        if not self._istrue_pos(cur_parse, cur_token,
                                                prev_token) and cur_token not in self.R_PAIRED_PUNCT:
                            aspects_deleted = self._delete_aspect(aspect, aspects_deleted)
                updated_result[i] = (cur_token, aspects_deleted)
        return [updated_result[i] for i in range(len(result))]

    def _heuristic_begining(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Аспект не может начинаться на знак препинания, союз или частицу. Аспект Method е может начинаться на предлог
        :param result: Список кортежей, в которых первый элемент - токен, второй - тэг
        :return: Список кортежей, в которых первый элемент - токен, второй - тэг
        """
        updated_result = {i: res for i, res in enumerate(result)}
        for i, (cur_token, cur_label) in list(enumerate(result))[1:]:
            prev_token, prev_label = result[i - 1]
            if self._isaspect(cur_label):
                aspects_for_token = cur_label.split('|')
                aspects_deleted = cur_label
                for aspect in aspects_for_token:
                    if aspect not in prev_label:
                        cur_parse = self._morph.parse(cur_token)
                        if (self._ispunct(cur_parse,
                                          cur_token) and cur_token not in self.L_PAIRED_PUNCT) or self._isconj_prcl(
                            cur_parse, cur_token, prev_token) or (
                                self._isprep(cur_parse, cur_token, prev_token) and not self._canstartwithprep(aspect)):
                            aspects_deleted = self._delete_aspect(aspect, aspects_deleted)
                updated_result[i] = (cur_token, aspects_deleted)
        return [updated_result[i] for i in range(len(result))]

    def _heuristic_single_word(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Удаление однословных тэгов
        :param result: Список кортежей, в которых первый элемент - токен, второй - тэг
        :return: Список кортежей, в которых первый элемент - токен, второй - тэг
        """
        updated_result = {i: res for i, res in enumerate(result)}
        for i, (cur_token, cur_label) in list(enumerate(result)):
            prev_token, prev_label = ('', '')
            next_token, next_label = ('', '')
            if i > 0:
                prev_token, prev_label = result[i - 1]
            if i + 1 < len(result):
                next_token, next_label = result[i + 1]
            if self._isaspect(cur_label):
                aspects_for_token = cur_label.split('|')
                aspects_deleted = cur_label
                for aspect in aspects_for_token:
                    # если аспекта нет в соседних тэгах
                    if aspect not in prev_label and aspect not in next_label:
                        cur_parse = self._morph.parse(cur_token)
                        # если аспект не может быть выражен одним словом или токен не является самостоятельной частью речи
                        if self._islong(aspect) or not self._istrue_pos(cur_parse, cur_token, prev_token):
                            aspects_deleted = self._delete_aspect(aspect, aspects_deleted)
                            updated_result[i] = (cur_token, aspects_deleted)
        return [updated_result[i] for i in range(len(result))]

    def _heuristic_gap(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Удаление разрывов между одинаковыми тэгами
        :param result: Список кортежей, в которых первый элемент - токен, второй - тэг
        :return: Список кортежей, в которых первый элемент - токен, второй - тэг
        """
        updated_result = {i: res for i, res in enumerate(result)}
        for i, (cur_token, cur_label) in list(enumerate(result))[1:-1]:
            prev_token, prev_label = result[i - 1]
            next_token, next_label = result[i + 1]
            if cur_label != prev_label and not self._isnested(
                    cur_label) and prev_label == next_label and not self._isconj_punct(cur_token):
                updated_result[i] = (cur_token, prev_label)
        return [updated_result[i] for i in range(len(result))]

    def _heuristic_order(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Вложенный аспект должен стоять на втором месте. Это нужно для правильной разметки тэгами.
        :param result: Список кортежей, в которых первый элемент - токен, второй - тэг
        :return: Список кортежей, в которых первый элемент - токен, второй - тэг
        """
        updated_result = {i: res for i, res in enumerate(result)}
        for i, (cur_token, cur_label) in list(enumerate(result))[1:]:
            prev_token, prev_label = updated_result[i - 1]
            cur_tags = cur_label.split('|')
            prev_tags = prev_label.split('|')
            # если наборы аспектов в текущем и предыдущем тэгах совпадают, порядок аспектов тоже должен совпадать
            # Contrib|Task, Task|Contrib -> Contrib|Task, Contrib|Task
            if set(cur_tags) == set(prev_tags):
                updated_result[i] = (cur_token, prev_label)
            # если начинается новый выложенный аспект, он должен стоять на втором месте
            # Contrib, Task|Contrib -> Contrib, Contrib|Task
            # на невложенные аспекты это не влияет: Contrib, Task -> Contrib, Task
            elif cur_tags[0] not in prev_tags:
                updated_result[i] = (cur_token, '|'.join(cur_tags[::-1]))
        return [updated_result[i] for i in range(len(result))]

    def validate(self, result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Применение эвристик к результату извлечения аспектов
        :param result: Список кортежей, в которых первый элемент - токен, второй - тэг
        :return: Список кортежей, в которых первый элемент - токен, второй - тэг
        """
        previous_result = result
        while True:
            result = previous_result
            result = self._heuristic_dot(result)
            result = self._heuristic_verb(result)
            result = self._heuristic_begining(result)
            result = self._heuristic_end(result)
            result = self._heuristic_single_word(result)
            result = self._heuristic_gap(result)
            if result == previous_result:
                return self._heuristic_order(result)
            previous_result = result
