from typing import Dict, List, Set, Any
from utils.paths import WIKIDATA_DUMP_PATH


class BaseCandidatesGenerator:

    def __init__(self):
        wikidata_dump_path = WIKIDATA_DUMP_PATH
        self._dump_path = wikidata_dump_path

    def create_candidates_set(self, normalized_term: str, queries: Set[str]) -> List[Dict[str, Any]]:
        """ Создание множества кандидатов сущностей 

        :param queries: набор запросов
        normalized_term: термин
        :return: список словарей (идентификатор, описания, название)
        описания и названия - для упрощения тестирования и разметки
        """
        raise NotImplementedError

