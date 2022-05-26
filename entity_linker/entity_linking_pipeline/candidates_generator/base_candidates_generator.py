from typing import Dict, List, Set, Any


class BaseCandidatesGenerator:

    def __init__(self):
        self._dump_path = 'entity_linker/wikidata_dump'

    def create_candidates_set(self, normalized_term: str, queries: Set[str]) -> List[Dict[str, Any]]:
        """ Создание множества кандидатов сущностей 

        :param queries: набор запросов
        normalized_term: термин
        :return: список словарей (идентификатор, описания, название)
        описания и названия - для упрощения тестирования и разметки
        """
        raise NotImplementedError

