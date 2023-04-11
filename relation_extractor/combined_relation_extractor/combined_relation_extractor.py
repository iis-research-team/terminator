from relation_extractor.rule_based_extractor.rule_based_extractor import RuleBasedExtractor
from relation_extractor.dl_relation_extractor.dl_relation_extractor import DLRelationExtractor


class CombinedRelationExtractor:

    def __init__(self):
        self._rule_based_extractor = RuleBasedExtractor()
        self._dl_relation_extractor = DLRelationExtractor()

    def extract(self, sample: str) -> str:
        dl_pred = self._dl_relation_extractor.extract(sample)
        rule_based_pred = self._rule_based_extractor.predict_rule_based(sample)
        predicted_relation = 'NO-RELATION'
        if dl_pred != 'NO-RELATION':
            predicted_relation = dl_pred
        elif rule_based_pred != 'NO-RELATION':
            predicted_relation = rule_based_pred

        return predicted_relation
