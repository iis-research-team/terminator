import os
import unittest

from utils.paths import ENTITY_LINKER
from utils.normalize import normalize_mystem
from utils.config_utilities import load_config

from entity_linker.entity_linking_pipeline.query_creator.n_gram_query_creator import NGramQueryCreator
from entity_linker.entity_linking_pipeline.candidates_generator import StringMatchCandidatesGenerator


class TestStringMatch(unittest.TestCase):

    def setUp(self):
        config = load_config()
        config['entity_linker']['path_to_json_file'] = os.path.join(ENTITY_LINKER, 'test_data/test_dump.json')

        self._normalized_term = normalize_mystem('нарушение слуха')
        self._queries = NGramQueryCreator().create_queries_set(self._normalized_term)
        self._candidates_generator = StringMatchCandidatesGenerator(config=config)

    def test_candidates_generator(self):
        true_candidate_set = [{'id': 'Q12133', 'desc': "снижение способности обнаруживать и понимать звуки",
                               'names': {self._normalized_term, "глухота", "тугоухость", "потеря слух"}}]

        generated_candidates_set = self._candidates_generator.create_candidates_set(self._normalized_term, self._queries)

        self.assertEqual(true_candidate_set, generated_candidates_set)
