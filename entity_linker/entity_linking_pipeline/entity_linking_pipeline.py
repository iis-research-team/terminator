from typing import List

from utils.normalize import normalize_mystem
from entity_linker.entity_linking_pipeline.query_creator import BaseQueryCreator
from entity_linker.entity_linking_pipeline.candidates_ranger import BaseCandidatesRanger
from entity_linker.entity_linking_pipeline.candidates_generator import BaseCandidatesGenerator


class EntityLinkingPipeline:

    def __init__(self, query_creator: BaseQueryCreator, candidates_generator: BaseCandidatesGenerator,
                 candidates_ranger: BaseCandidatesRanger):
        self._query_creator = query_creator
        self._candidates_generator = candidates_generator
        self._candidates_ranger = candidates_ranger

    def get_linked_mention(self, term: str, context: List[str]):
        normalized_term = normalize_mystem(term)
        queries = self._query_creator.create_queries_set(normalized_term)
        candidates = self._candidates_generator.create_candidates_set(normalized_term, queries)
        ranged_candidates = self._candidates_ranger.range_candidates_set(candidates, context, term=normalized_term)
        return ranged_candidates
