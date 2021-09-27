import numpy as np

from utils.constants import re_class2label
from utils.paths import RELATION_EXTRACTOR_WEIGHTS_PATH
from relation_extractor.dl_relation_extractor.model import get_model
from relation_extractor.dl_relation_extractor.vectorizer import Vectorizer


class DLRelationExtractor:

    def __init__(self):

        self._vectorizer = Vectorizer()

        self._model = get_model('bert_for_sequence_classification', num_labels=len(re_class2label))
        self._model.load_weights(RELATION_EXTRACTOR_WEIGHTS_PATH)

    def extract(self, sample: dict) -> str:
        text_sample = sample['token']

        X_ids = []
        X_masks = []

        _, input_ids, input_masks, tag = self._vectorizer.vectorize(text_sample, 'NO-RELATION')
        X_ids.append(np.array(input_ids))
        X_masks.append(np.array(input_masks))

        pred = self._model.predict_on_batch(
            [np.asarray(X_ids, dtype='int32'), np.asarray(X_masks, dtype='int32')]
        )[0][0]
        ml_pred = re_class2label[np.argmax(pred)]

        return ml_pred
