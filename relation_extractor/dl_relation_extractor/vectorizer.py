from typing import List, Tuple, Any

import numpy as np
from transformers import BertTokenizer

from utils.constants import re_label2class


class Vectorizer:

    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',
                                                        do_lower_case=False)

        self._label2class = re_label2class
        self._max_length = 128

    def vectorize(self, text: List[str], label: str) -> Tuple[List[str], List[int], List[int], np.ndarray]:
        tokenized_text, input_masks = self._tokenize(text)

        input_ids = self._tokenizer.convert_tokens_to_ids(tokenized_text)

        tag = np.zeros(len(re_label2class))
        tag[re_label2class[label]] = 1.0

        input_ids = self._pad(input_ids)
        input_masks = self._pad(input_masks)

        return tokenized_text, input_ids, input_masks, tag

    def _pad(self, input: List[Any]) -> List[Any]:
        if len(input) >= self._max_length:
            return input[:self._max_length]
        while len(input) < self._max_length:
            input.append(0)
        return input

    def _tokenize(self, text: List[str]) -> Tuple[List[str], List[int]]:
        tokenized_text = []

        for token in text:
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = self._tokenizer.tokenize(token)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_text.extend(tokenized_word)

        inputs = self._tokenizer.encode_plus(
            tokenized_text,
            is_pretokenized=True,
            return_attention_mask=True,
            max_length=self._max_length,
            truncation=True
        )

        return tokenized_text, inputs['attention_mask']
