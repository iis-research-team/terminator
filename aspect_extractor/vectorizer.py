import numpy as np
from typing import List, Tuple, Any
from transformers import BertTokenizer

from utils.constants import aspects_label2class, ASPECTS_PRETRAINED_MODEL_NAME, ASPECTS_NUM_LABELS


class Vectorizer:

    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained(ASPECTS_PRETRAINED_MODEL_NAME, do_lower_case=False)
        self._label2class = aspects_label2class
        self._NUM_LABELS = ASPECTS_NUM_LABELS
        self._max_length = 200

    def vectorize(self, text: List[str], token_labels: List[str] = None) -> Tuple[
        List[str], List[int], List[int], List[int]]:
        """
        Векторизация текста и тэгов токенов в тексте
        :param text: Текст (список токенов)
        :param token_labels: Список тэгов для токенов из текста
        :param max_length: Максимальное число bpe-токенов в векторизованном тексте (остальные обрезается)
        :return:   ``tokenized_text``: Текст, разделенный на bpe-токены,
                   ``input_ids``: Вектор текста,
                   ``input_masks``: Маска для текста,
                   ``tags``: One-hot-encoded тэги для токенов в тексте
        """
        if not token_labels:
            token_labels = ['O'] * len(text)
        tokenized_text, input_masks, labels = self._tokenize(text, token_labels)
        input_ids = self._tokenizer.convert_tokens_to_ids(tokenized_text)
        tags = []
        for label in labels:
            tags.append(self.vectorize_label(label))
        input_ids = self._pad(input_ids, 0)
        input_masks = self._pad(input_masks, 0)
        tags = self._pad(tags, np.zeros(self._NUM_LABELS))

        return tokenized_text, input_ids, input_masks, tags

    def vectorize_label(self, label: str) -> np.array:
        """
        Преобразует тэг в one-hot вектор
        :param label: Тэг
        :return: One-hot вектор для тэга
        """
        vector = np.zeros(self._NUM_LABELS)
        classes = [aspect in label.split('|') for aspect in aspects_label2class]
        vector[classes] = 1
        return vector

    def _pad(self, input: List[Any], padding: Any) -> List[Any]:
        if len(input) >= self._max_length:
            return input[:self._max_length]
        while len(input) < self._max_length:
            input.append(padding)
        return input

    def _tokenize(self, text: List[str], token_labels: List[str]) -> Tuple[List[str], List[int], List[str]]:
        """
        Денение текста на bpe-токены и векторизация
        :param text: Текст (список токенов)
        :param token_labels: Тэги для токенов в тексте
        :param max_length: Максимальное число bpe-токенов в векторизованном тексте (остальные обрезается)
        :return: ``tokenized_text``: Текст, разделенный на bpe-токены,
                 ``input_masks``: Маска для текста,
                 ``labels``:  Тэги для bpe-токенов в тексте (one-hot encoded)
        """
        tokenized_text = []
        labels = []

        for token, label in zip(text, token_labels):
            tokenized_word = self._tokenizer.tokenize(token)
            n_subwords = len(tokenized_word)
            tokenized_text.extend(tokenized_word)
            labels.extend([label] * n_subwords)

        try:
            inputs = self._tokenizer.encode_plus(
                tokenized_text,
                is_pretokenized=True,
                return_attention_mask=True,
                max_length=self._max_length,
                truncation=True
            )

        except:
            print(text)
            inputs = dict()
            inputs['attention_mask'] = np.zeros(self._max_length)

        return tokenized_text, inputs['attention_mask'], labels
