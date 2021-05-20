from typing import List, Tuple


class BaseExtractor:

    def extract(self, text: str) -> List[Tuple[str, str]]:
        raise NotImplementedError
