from typing import Dict, List, Any


class BaseCandidatesRanger:

    def range_candidates_set(self, candidates: List[Dict[str, Any]], context: List[str], term: str = False) -> Dict[str, float]:
        """ Ранжирование множества кандидатов сущностей

        :param candidates: список из словарей (идентификатор, описания, название),
        context: список слов контекста
        :return: Dict [id, расстояние]
        """
        raise NotImplementedError
