import unittest

from entity_linker.entity_linking_pipeline.candidates_ranger import CosineSimRangerWeights


class TestCosineSimRangerWeights(unittest.TestCase):

    def setUp(self):
        self._term = 'язык программирования python'

    def test_get_weights(self):
        true_weights = [2/3, 1.0, 1.0, 1.0]
        test_first = ['язык программирования питон']
        test_second = ['язык программирования python']
        test_third = [test_second[0], test_first[0]]
        test_forth = []
        generated_weights = list()
        generated_weights.append(CosineSimRangerWeights._get_weight(test_first, self._term))
        generated_weights.append(CosineSimRangerWeights._get_weight(test_second, self._term))
        generated_weights.append(CosineSimRangerWeights._get_weight(test_third, self._term))
        generated_weights.append(CosineSimRangerWeights._get_weight(test_forth, self._term))
        self.assertEqual(true_weights, generated_weights)
