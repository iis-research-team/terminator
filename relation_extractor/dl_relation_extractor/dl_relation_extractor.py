import torch
import os
import numpy as np

from transformers import BertConfig
from utils.paths import RELATION_EXTRACTOR_WEIGHTS_PATH
from utils.constants import RE_LABELS
from relation_extractor.dl_relation_extractor.model import get_model
from relation_extractor.dl_relation_extractor.vectorizer import Vectorizer


class DLRelationExtractor:

    def __init__(self):
        self._vectorizer = Vectorizer()
        self._model_dir = RELATION_EXTRACTOR_WEIGHTS_PATH
        self._label_lst = RE_LABELS
        self._num_labels = len(self._label_lst)
        self._config = BertConfig.from_pretrained(
            self._model_dir,
            num_labels=self._num_labels,
            id2label={str(i): label for i, label in enumerate(self._label_lst)},
            label2id={label: i for i, label in enumerate(self._label_lst)},
        )
        self._model_args = torch.load(os.path.join(RELATION_EXTRACTOR_WEIGHTS_PATH, 'training_args.bin'))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = get_model(self._model_dir, self._config, self._model_args, self._device)

    def extract(self, text):

        # Convert text into features
        input_ids, attention_mask, token_type_ids, e1_mask, e2_mask = self._vectorizer.vectorize(text, args=self._model_args, add_sep_token=['add_sep_token'])

        # Predict
        with torch.no_grad():
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": None,
                "e1_mask": e1_mask,
                "e2_mask": e2_mask,
            }
            outputs = self._model(**inputs)
            logits = outputs[0]
            pred = logits.detach().cpu().numpy()

        pred = np.argmax(pred, axis=1)
        pred = self._label_lst[int(pred)]

        return pred
