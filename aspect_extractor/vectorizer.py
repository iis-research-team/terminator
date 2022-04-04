from typing import List, Tuple, Any

import numpy as np

from transformers import BertTokenizer

from utils.constants import aspects_label2class


class Vectorizer:

    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased',
                                                        do_lower_case=False)

        self._label2class = aspects_label2class
        self._NUM_LABELS = 10
        self._max_length = 150

    def vectorize(self, text: List[str], token_labels: List[str]) -> Tuple[List[str], List[int], List[int], List[int]]:
        tokenized_text, input_masks, labels = self._tokenize(text, token_labels)
        input_ids = self._tokenizer.convert_tokens_to_ids(tokenized_text)
        tags = []
        for label in labels:
            tag = np.zeros(self._NUM_LABELS)
            label = label.split('|')
            for t in label:
                tag[self._label2class[t]] = 1.0
            tags.append(tag)
        input_ids = self._pad(input_ids, 0)
        input_masks = self._pad(input_masks, 0)
        tags = self._pad(tags, np.zeros(self._NUM_LABELS))

        return tokenized_text, input_ids, input_masks, tags

    def _pad(self, input: List[Any], padding: Any) -> List[Any]:
        if len(input) >= self._max_length:
            return input[:self._max_length]
        while len(input) < self._max_length:
            input.append(padding)
        return input

    def _tokenize(self, text: List[str], token_labels: List[str]) -> Tuple[List[str], List[int], List[str]]:
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
