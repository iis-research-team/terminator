import pickle
import unittest
import numpy as np

from utils.normalize import normalize_mystem
from entity_linker.entity_linking_pipeline.candidates_ranger import CosineSimRanger


class TestCosineSimRanger(unittest.TestCase):

    def setUp(self):
        self._normalized_term = normalize_mystem('язык программирования python')
        self._text = 'Для разработки системы использовался язык программирования python'
        self._candidates_ranger = CosineSimRanger()

    def test_vector_length(self):
        # посмотреть, что возвращается вектор нужной длины
        self.assertEqual((300,), np.shape(self._candidates_ranger._get_vector_for_phrase(self._normalized_term)))

    def test_vector_without_tf_idf(self):
        with open('02.pickle', 'rb') as f:
            true_vector_second = pickle.load(f)
        # попробовать без tf-idf
        self.assertEqual(true_vector_second.all(), self._candidates_ranger._get_vector_for_phrase(self._text).all())

    def test_vector_with_tf_idf(self):
        with open('01.pickle', 'rb') as f:
            true_vector_first = pickle.load(f)
        candidates_ranger = CosineSimRanger(is_use_tf_idf=True)
        # для определённой фразы посмотреть, что возвращается вектор с нужными значениями
        self.assertEqual(true_vector_first.all(), candidates_ranger._get_vector_for_phrase(self._text).all())
