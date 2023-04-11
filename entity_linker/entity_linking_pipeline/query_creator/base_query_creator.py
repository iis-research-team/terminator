from typing import Set


class BaseQueryCreator:

    def create_queries_set(self, term: str) -> Set[str]:
        """ Создание множества запросов для поиска кандидатов

        :param term: термин, для которого будет осуществляться поиск кандидатов
        :return: множество запросов для входного термина
        """
        raise NotImplementedError
