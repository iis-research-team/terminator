from entity_linker.entity_linking_pipeline.entity_linking_pipeline import EntityLinkingPipeline
from entity_linker.entity_linking_pipeline.query_creator.n_gram_query_creator import NGramQueryCreator
from entity_linker.entity_linking_pipeline.candidates_generator.string_match_candidates_generator import StringMatchCandidatesGenerator
from entity_linker.entity_linking_pipeline.candidates_ranger import CosineSimRangerWeights

class RussianEntityLinker(EntityLinkingPipeline):

    def __init__(self):
        super().__init__(
            query_creator=NGramQueryCreator(),
            candidates_generator=StringMatchCandidatesGenerator(),
            candidates_ranger=CosineSimRangerWeights()
        )

