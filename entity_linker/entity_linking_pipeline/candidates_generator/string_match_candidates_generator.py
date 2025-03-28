from typing import List, Dict, Set, Any

import orjson

from entity_linker.entity_linking_pipeline.candidates_generator import BaseCandidatesGenerator


class StringMatchCandidatesGenerator(BaseCandidatesGenerator):
    """ Генерация кандидатов по построковому совпадению """

    def __init__(self):
        super().__init__()

    def create_candidates_set(self, normalized_term: str, queries: Set[str]):
        return self._get_string_match_candidates(normalized_term, queries)

    def _get_string_match_candidates(self, normalized_term: str, queries: Set[str]) -> List[Dict[str, Any]]:
        result = list()
        print(f'iter dump for entity [{normalized_term}]...')
        with open(self._dump_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == '\n':
                    continue
                try:
                    entity = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue
                if entity['type'] != 'item':
                    continue
                entity_names = set()
                if 'label' in entity:
                    entity_names.add(entity['label']['value'].lower())
                if 'alias' in entity:
                    for alias in entity['alias']:
                        entity_names.add(alias['value'].lower())
                desc = ''
                if 'description' in entity:
                    desc = entity['description']['value']
                    if 'страница значений' in desc.lower():
                        continue
                if normalized_term in entity_names:
                    result.append({'id': entity['id'], 'desc': desc, 'names': list(entity_names)})
                else:
                    for t in queries:
                        if t in entity_names:
                            result.append({'id': entity['id'], 'desc': desc, 'names': list(entity_names)})
                            break
        return result
