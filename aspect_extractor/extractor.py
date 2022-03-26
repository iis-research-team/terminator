from typing import List, Union, Tuple
from collections import Counter, defaultdict

import numpy as np

from utils.constants import aspects_class2label
from utils.paths import ASPECT_EXTRACTOR_WEIGHTS_PATH
from utils.utilities import tokenize
from aspect_extractor.model import get_model
from aspect_extractor.vectorizer import Vectorizer
from aspect_extractor.heuristic_validator import HeuristicValidator


class AspectExtractor():
    """ Класс для извлечения аспектов из текста с помощью модели """

    def __init__(self):
        weights_path = ASPECT_EXTRACTOR_WEIGHTS_PATH
        self._model = get_model()
        self._model.load_weights(weights_path)
        self._vectorizer = Vectorizer()
        self._heuristic_validator = HeuristicValidator()
        self._class2label = aspects_class2label
        self._THRESHOLD = 0.1

    def extract(self, text: Union[str, List[str]]) -> List[Tuple[str, str]]:
        """ Извлечение аспектов из входного текста
        :param text: входной текст, может быть строкой либо уже токенизированным (тогда списком строк)
        :return: Список кортежей, в которых первый элемент - токен, второй элемент - тэги
        """
        if isinstance(text, str):
            tokens = tokenize(text)
        else:
            tokens = text
        labels = ['O' for i in range(len(tokens))]
        all_bpe_tokens = []
        all_predictions = []

        # делим список токенов на батчи, которые будут последовательно обрабатываться
        n_batches = int(len(tokens) / 50) + 1
        for i in range(n_batches):
            start = 50 * i
            end = 50 * i + 50
            if end > len(tokens):
                end = len(tokens)

            if start == end:
                break

            bpe_tokens, input_ids, input_masks, tags = self._vectorizer.vectorize(tokens[start: end], labels[start: end])
            preds = self._model.predict([np.array([input_ids]), np.array([input_masks])])[0][0]
            all_bpe_tokens.extend(bpe_tokens)
            all_predictions.extend(preds[:len(bpe_tokens)])

        result = self._get_preds_with_tokens(all_bpe_tokens, all_predictions)
        if isinstance(text, list):
            result = self._align_tokens(text, result)
        result = self._heuristic_validator.validate(result)
        return result

    def _align_tokens(self, input_tokens: List[str], result: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """ Выравнивание токенов
        В случаях, когда на вход пришёл уже токенизированный текст, токены в результирующем списке могут отличаться от
        тех, что в исходном из-за bpe-токенизации. Поэтому нужно выровнить результирующий список относительно входного,
        т.е. список токенов в обоих списках должен совпадать
        :param input_tokens: список токенов во входном списке
        :param result: результирующий список кортежей, в которых первый элемент - токен, второй - тэг
        :return: список кортежей, в которых первый элемент - токен, второй - тэг
        """
        # если списки токенов изначально совпадают, то сразу возвращаем результат
        resulted_tokens = [res[0] for res in result]
        if resulted_tokens == input_tokens:
            return result

        updated_result = []

        # фиксируем позицию токена в результирующем списке
        res_cursor = 0
        for i, token in enumerate(input_tokens):

            # токенизируем токен из входного списка. Если длина получившихся токенов == 1, то это не составной токен
            tokenized = tokenize(token)
            if len(tokenized) == 1:
                updated_result.append(result[res_cursor])
                res_cursor += 1
                continue

            full_resulted = []
            tags = Counter()
            # собираем все токены в результирующем списке, которые лежат в промежутке от res_cursor до
            # res_cursor + количество токенов в tokenized
            for j in range(res_cursor, res_cursor + len(tokenized)):
                full_resulted.append(result[j][0])
                tags[result[j][1]] += 1

            # на случай, если составным токенам были присвоены разные тэги, то выберем тэг с максимальной частотой
            tag = tags.most_common()[0][0]
            updated_result.append((''.join(full_resulted), tag))

            # переведём позицию курсора на количество составных частей исходного токена
            res_cursor += len(tokenized)

        assert len(input_tokens) == len(updated_result), 'Alignment worked incorrect'

        return updated_result

    def _get_preds_with_tokens(self, bpe_tokens: List[str], preds: List[float]) -> List[Tuple[str, str]]:
        """ Из предсказаний для bpe-токенов получаем предскания для целых токенов
        :param bpe_tokens: список bpe-токенов
        :param preds: список предиктов от модели
        :return: Список кортежей, в которых первый элемент - полноценный токен, второй элемент - тэги
        """
        result = []
        token = []
        tags = []
        for bpe_token, pred in zip(bpe_tokens, preds):
            if bpe_token == '[UNK]':
                bpe_token = '–'

            # если bpe-токен не является началом целого токена, то он начинается с "##"
            if bpe_token.startswith('##'):
                token.append(bpe_token[2:])
                tags.extend(self._select_tags(pred))
            else:
                # если уже собрали токен до этого, то обработаем его и положим в результирущий список
                if len(token) > 0:
                    self._process_token(result, tags, token)
                token = [bpe_token]
                tags = self._select_tags(pred)

        # обработаем последний токен и положим его в результирующий список
        self._process_token(result, tags, token)

        return result

    def _select_tags(self, pred: List[float]) -> List[Tuple[int, float]]:
        """ Выбор тэгов, вероятности для которых больше порогового значения
        :param pred : предиктов от модели для токена
        :return: Список кортежей, в которых первый элемент - номер тэга, второй элемент - вероятность этого тэга
        """
        selected_tags = [p for p in enumerate(pred) if p[1] > self._THRESHOLD]
        return selected_tags

    def _process_token(self, result: List[Tuple[str, str]], tags: List[Tuple[int, float]], token: List[str]):
        """ Обработка токена: собираем его из bpe-токенов, выбираем нужные тэги
        :param result: результирующий список с токенами и тэгами
        :param tags: список тэгов, который был получен для составных bpe-токенов
        :param token: список bpe-токенов для данного токена
        """
        # объединяем составные bpe-токены в единую строку
        token_str = ''.join(token)

        # складываем вероятности тэгов для каждого из составных bpe-токенов
        probability_sums = defaultdict(float)
        for tag, probability in tags:
            probability_sums[tag] += probability
        best_tags = [item[0] for item in sorted(probability_sums.items(), key=lambda item: item[1])]
        best_labels = [self._class2label[tag] for tag in best_tags]

        # если в списке кандидатов больше двух элементов, выбираются два тэга с наибольшими вероятностями
        if len(best_labels) > 2:
            best_labels = best_labels[:2]

        # если в список кандидатов пуст, выбираются тэг "O"
        if len(best_labels) == 0:
            best_labels = ['O']

        # если самый вероятный тэг — "O", остается только он
        elif best_labels[0] == 'O':
            best_labels = ['O']
        if len(best_labels) > 1:
            # если "O" стоит на втором месте, остается только тэг, стоящий на первом месте
            if best_labels[1] == 'O':
                best_labels = best_labels[:1]
        tag_union = '|'.join(best_labels)
        result.append((token_str, tag_union))


if __name__ == '__main__':
    extractor = AspectExtractor()
    text = "Определена модель для визуализации связей между объектами и их атрибутами в различных процессах. " \
           "На основании модели разработан универсальный абстрактный компонент графического пользовательского интерфейса и приведены примеры его программной реализации. " \
           "Также проведена апробация компонента для решения прикладной задачи по извлечению информации из документов."
    result = extractor.extract(text)
    for token, tag in result:
        print(f'{token} -> {tag}')
