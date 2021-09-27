import os
import json
from typing import Dict, List

import pymorphy2

from utils.utilities import tokenize
from utils.paths import RELATION_EXTRACTOR_PATH


class RuleBasedExtractor:

    NO_RELATION = 'NO-RELATION'
    MAX_CONTEXT_LENGTH = 10
    MIN_CONTEXT_LENGTH = 2

    RULE_BASED_PATH = os.path.join(RELATION_EXTRACTOR_PATH, 'rule_based_extractor')

    def __init__(self):
        self._morph = pymorphy2.MorphAnalyzer()
        self._pattern2relation = self._load_patterns(os.path.join(self.RULE_BASED_PATH, 'patterns.json'))
        self._one_word_pattern2relation = self._load_patterns(os.path.join(
            self.RULE_BASED_PATH, 'one_word_patterns.json')
        )

    def extract_relations(self, context: str) -> str:
        """ Extract relation based on context between two entities

        :param context: text between two entities
        :return: relation type
        """
        relation = self.NO_RELATION
        tokens = tokenize(context.lower())
        context = ' '.join(tokens)
        if len(tokens) > self.MAX_CONTEXT_LENGTH:
            return relation
        if len(tokens) <= self.MIN_CONTEXT_LENGTH:
            relation = self._search_one_word_patterns(tokens)
        if relation == self.NO_RELATION:
            relation = self._search(self._pattern2relation, context)
        return relation

    def _search(self, pattern2relation: Dict[str, str], text: str) -> str:
        relation = self.NO_RELATION
        for pattern, rel in pattern2relation.items():
            if pattern in text:
                relation = rel
        return relation

    def _search_one_word_patterns(self, tokens: List[str]) -> str:
        relation = self.NO_RELATION
        for pattern, rel in self._one_word_pattern2relation.items():
            if pattern in tokens:
                relation = rel
        return relation

    def _load_patterns(self, path: str) -> Dict[str, str]:
        with open(path, 'r') as f:
            patterns = json.load(f)
            for relation, pats in patterns.items():
                print(sorted(pats))
        pattern2relation = dict()
        for relation, phrases in patterns.items():
            for phrase in phrases:
                pattern2relation[phrase] = relation
        return pattern2relation

    def _lemmatize(self, text: str) -> str:
        tokens = tokenize(text.lower())
        lemmas = []
        for token in tokens:
            lemmas.append(self._morph.normal_forms(token)[0])
        return ' '.join(lemmas)
