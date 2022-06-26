from typing import Set

from entity_linker.entity_linking_pipeline.query_creator import BaseQueryCreator


class NGramQueryCreator(BaseQueryCreator):
    """ Генерация поисковых запросов, которая включает 1-, 2-, 3-граммы входного термина """

    def create_queries_set(self, term: str) -> Set[str]:
        n_grams = self._generate_bi_tri_grams(term)
        return n_grams

    def _generate_bi_tri_grams(self, term: str) -> Set[str]:
        input_terms = term.split()
        result_grams = list()
        result_grams.extend(input_terms)
        for i, t in enumerate(input_terms):
            if t != input_terms[-1]:
                result_grams.append(' '.join(input_terms[i:i + 2]))
            try:
                if t != input_terms[-2]:
                    result_grams.append(' '.join(input_terms[i:i + 3]))
            except IndexError:
                continue
        return set(result_grams)
