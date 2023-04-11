import os
import json
from typing import Dict, List

import pymorphy2

from utils.utilities import tokenize
from utils.paths import RELATION_EXTRACTOR_PATH


class RuleBasedExtractor:

    NO_RELATION = 'NO-RELATION'
    MAX_CONTEXT_LENGTH = 6
    MIN_CONTEXT_LENGTH = 2

    RULE_BASED_PATH = os.path.join(RELATION_EXTRACTOR_PATH, 'rule_based_extractor')

    def __init__(self):
        self._morph = pymorphy2.MorphAnalyzer()
        self._pattern2relation = self._load_patterns(os.path.join(self.RULE_BASED_PATH, 'patterns.json'))
        self._one_word_pattern2relation = self._load_patterns(os.path.join(
            self.RULE_BASED_PATH, 'one_word_patterns.json')
        )

    def _extract_relations(self, context: str) -> str:
        """ Extract relation based on context between two entities

        :param context: text between two entities
        :return: relation type
        """
        relation = self.NO_RELATION
        context = self._lemmatize(context)
        tokens = context.split(' ')
        if len(tokens) > self.MAX_CONTEXT_LENGTH:
            return relation
        if len(tokens) <= self.MIN_CONTEXT_LENGTH:
            relation = self._search_one_word_patterns(tokens)
        if relation == self.NO_RELATION:
            relation = self._search(self._pattern2relation, context)
        return relation

    def _load_patterns(self, path: str) -> Dict[str, str]:
        with open(path, 'r') as f:
            patterns = json.load(f)
        pattern2relation = dict()
        for relation, phrases in patterns.items():
            for phrase in phrases:
                phrase = self._lemmatize(phrase)
                pattern2relation[phrase] = relation
        return pattern2relation

    def _search(self, pattern2relation: Dict[str, str], text: str) -> str:
        relation = self.NO_RELATION
        for pattern, rel in pattern2relation.items():
            if rel == "TOOL":
                if pattern in text:
                    relation = rel
                    break
            else:
                if pattern in text:
                    relation = rel
        return relation

    def _search_one_word_patterns(self, tokens: List[str]) -> str:
        relation = self.NO_RELATION
        for pattern, rel in self._one_word_pattern2relation.items():
            if pattern in tokens:
                relation = rel
        return relation

    def _lemmatize(self, text: str) -> str:
        tokens = tokenize(text.lower())
        lemmas = []
        for token in tokens:
            lemmas.append(self._morph.normal_forms(token)[0])
        return ' '.join(lemmas)

    def predict_rule_based(self, sample: str) -> str:
        sample = sample.replace('<e1>', '<e1> ')
        sample = sample.replace('<e2>', '<e2> ')
        sample = sample.replace('</e1>', ' </e1>')
        sample = sample.replace('</e2>', ' </e2>')
        tokens = sample.split(' ')
        subj_start = tokens.index("<e1>") + 1  # the start position of entity1
        subj_end = tokens.index("</e1>") - 1  # the end position of entity1
        obj_start = tokens.index("<e2>") + 1  # the start position of entity2
        obj_end = tokens.index("</e2>") - 1  # the end position of entity2

        if subj_end < obj_start:
            context = tokens[subj_end + 2: obj_start - 1]
        else:
            context = tokens[obj_end + 2: subj_start - 1]

        rule_based_pred = self._extract_relations(' '.join(context))

        return rule_based_pred
