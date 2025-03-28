from typing import List, Tuple, Any

from transformers import BertTokenizer

from utils.constants import label2class


class Vectorizer:

    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased',
                                                        do_lower_case=False)

        self._label2class = label2class
        self._max_length = 128

    def vectorize(self, text: List[str], token_labels: List[str]) -> Tuple[List[str], List[int], List[int], List[int]]:
        tokenized_text, input_masks, labels = self._tokenize(text, token_labels)

        input_ids = self._tokenizer.convert_tokens_to_ids(tokenized_text)

        tags = []
        for label in labels:
            tags.append(self._label2class[label])

        input_ids = self._pad(input_ids)
        input_masks = self._pad(input_masks)
        tags = self._pad(tags)

        return tokenized_text, input_ids, input_masks, tags

    def _pad(self, input: List[Any]) -> List[Any]:
        if len(input) >= self._max_length:
            return input[:self._max_length]
        while len(input) < self._max_length:
            input.append(0)
        return input

    def _tokenize(self, text: List[str], token_labels: List[str]) -> Tuple[List[str], List[int], List[str]]:
        tokenized_text = []
        labels = []

        for token, label in zip(text, token_labels):
            # Tokenize the word and count # of subwords the word is broken into
            tokenized_word = self._tokenizer.tokenize(token)
            n_subwords = len(tokenized_word)

            # Add the tokenized word to the final tokenized word list
            tokenized_text.extend(tokenized_word)

            # Add the same label to the new list of labels `n_subwords` times
            labels.extend([label] * n_subwords)

        inputs = self._tokenizer.encode_plus(
            tokenized_text,
            is_pretokenized=True,
            return_attention_mask=True,
            max_length=self._max_length,
            truncation=True
        )

        return tokenized_text, inputs['attention_mask'], labels
