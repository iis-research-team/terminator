from utils.constants import RUSSERC_TO_RC_SCIENCE_LABELS
from relation_extractor.rule_based_extractor.rule_based_extractor import RuleBasedExtractor
from relation_extractor.dl_relation_extractor.dl_relation_extractor import DLRelationExtractor


class CombinedRelationExtractor:

    def __init__(self):
        self._rule_based_extractor = RuleBasedExtractor()
        self._dl_relation_extractor = DLRelationExtractor()

    def extract(self, sample: dict) -> str:
        dl_pred = self._dl_relation_extractor.extract(sample)
        rule_based_pred = self._predict_rule_based(sample)
        predicted_relation = 'NO-RELATION'
        if dl_pred != 'NO-RELATION':
            predicted_relation = dl_pred
        elif rule_based_pred != 'NO-RELATION':
            predicted_relation = rule_based_pred

        return predicted_relation

    def _predict_rule_based(self, sample: dict) -> str:
        filtered_tokens = []
        for token in sample['token']:
            if token not in {'$', '#'}:
                filtered_tokens.append(token)

        if sample['subj_end'] < sample['obj_start']:
            context = filtered_tokens[sample['subj_end'] + 1: sample['obj_start']]
        else:
            context = filtered_tokens[sample['obj_end'] + 1: sample['subj_start']]

        rule_based_pred = self._rule_based_extractor.extract_relations(' '.join(context))
        if rule_based_pred in RUSSERC_TO_RC_SCIENCE_LABELS:
            rule_based_pred = RUSSERC_TO_RC_SCIENCE_LABELS[rule_based_pred]
        else:
            rule_based_pred = 'NO-RELATION'
        return rule_based_pred
