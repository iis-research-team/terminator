from typing import List, Dict, Any
from collections import OrderedDict

from sklearn.metrics.pairwise import cosine_similarity

from entity_linker.entity_linking_pipeline.candidates_ranger import CosineSimRanger


class CosineSimRangerWeights(CosineSimRanger):

    def range_candidates_set(self, candidates: List[Dict[str, Any]], context: List[str], term: str = None)\
            -> Dict[str, float]:
        _context_vector = self._get_vector_for_phrase(' '.join(context))
        sorted_candidates = self._range_candidates_by_cosine_similarity_weights(candidates, _context_vector, term)
        return sorted_candidates

    def _range_candidates_by_cosine_similarity_weights(self, candidates: List[Dict[str, Any]], vector: int, term: str = None) -> Dict[str, float]:
        distances = dict()
        for candidate_dict in candidates:
            full_desc = list(candidate_dict['names'])
            full_desc.extend([candidate_dict['desc']])
            candidate_vector = self._get_vector_for_phrase(' '.join(full_desc))
            weight = self._get_weight(candidate_dict['names'], term)
            distances[candidate_dict['id']] = cosine_similarity([vector], [candidate_vector])[0] * weight
        sorted_candidates = OrderedDict(sorted(distances.items(), key=lambda x: x[1], reverse=True))

        return sorted_candidates

    @staticmethod
    def _get_weight(names: List[str], term: str) -> float:
        weight = 1.0
        if term not in names:
            weight = 0.0
            # полностью термин не входит, значит перебираем все возможные имена сущности
            for t in names:
                unique_normalized_tokens = set(str(term).split())
                # считаем пересечение токенов между найденной фразой и входной
                n_shared_tokens = len(set(t.split()) & unique_normalized_tokens)
                weight_upd = n_shared_tokens / len(unique_normalized_tokens)
                if weight_upd > weight:
                    weight = weight_upd
        if weight != 0.0:
            return weight
        else:
            return 1.0
