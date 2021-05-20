from typing import List, Union

import numpy as np

from utils.constants import Tags, class2label
from utils.paths import TERMS_EXTRACTOR_WEIGHTS_PATH
from utils.utilities import validate_sequence, tokenize
from terms_extractor.base_extractor import BaseExtractor
from terms_extractor.dl_extractor.model import get_model
from terms_extractor.dl_extractor.vectorizer import Vectorizer
from terms_extractor.dl_extractor.heuristic_validator import HeuristicValidator


class DLExtractor(BaseExtractor):

    def __init__(self):
        self._model = get_model()
        self._model.load_weights(TERMS_EXTRACTOR_WEIGHTS_PATH)
        self._vectorizer = Vectorizer()
        self._heuristic_validator = HeuristicValidator()
        self._class2label = class2label

    def extract(self, text: Union[str, List[str]]):
        if isinstance(text, str):
            tokens = tokenize(text)
        else:
            tokens = text
        labels = [Tags.NOT_TERM.value for i in range(len(tokens))]

        all_bpe_tokens = []
        all_predictions = []

        n_batches = int(len(tokens) / 50) + 1
        for i in range(n_batches):
            start = 50 * i
            end = 50 * i + 50
            if end > len(tokens):
                end = len(tokens)

            if start == end:
                break

            bpe_tokens, input_ids, input_masks, tags = self._vectorizer.vectorize(
                tokens[start: end], labels[start: end]
            )
            preds = self._model.predict([np.array([input_ids]), np.array([input_masks])])[0]
            all_bpe_tokens.extend(bpe_tokens)
            all_predictions.extend(preds)

        result = self._get_preds_with_tokens(all_bpe_tokens, all_predictions)
        result = self._heuristic_validator.validate(result)
        result = validate_sequence(result)
        return result

    def _get_preds_with_tokens(self, bpe_tokens, preds):
        result = []
        token = []
        tags = []
        for bpe_token, pred in zip(bpe_tokens, preds):
            if bpe_token.startswith('##'):
                token.append(bpe_token[2:])
                tags.append(self._class2label[np.argmax(pred)])
            else:
                if len(token) > 0:
                    self._process_token(result, tags, token)
                token = [bpe_token]
                tags = [self._class2label[np.argmax(pred)]]
        self._process_token(result, tags, token)
        return result

    def _process_token(self, result, tags, token):
        token_str = ''.join(token)
        tag = Tags.NOT_TERM.value
        if Tags.B_TERM.value in tags:
            tag = Tags.B_TERM.value
        elif Tags.I_TERM.value in tags:
            tag = Tags.I_TERM.value
        result.append((token_str, tag))
