import unittest

from entity_linker.entity_linking_pipeline.query_creator.n_gram_query_creator import NGramQueryCreator


class TestNGramQueryCreator(unittest.TestCase):

    def setUp(self):
        self._query_creator = NGramQueryCreator()

    def test_query_creator(self):
        normalized_term = 'язык программирование python'
        true_query_set = {
            'язык',
            'программирование',
            'python',
            'язык программирование',
            'программирование python',
            'язык программирование python'
        }

        generated_query_set = self._query_creator.create_queries_set(normalized_term)

        self.assertEqual(sorted(true_query_set), sorted(generated_query_set))
