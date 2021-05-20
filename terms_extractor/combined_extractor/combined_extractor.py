from typing import List, Tuple, Union

from utils.constants import Tags
from utils.utilities import validate_sequence
from terms_extractor.base_extractor import BaseExtractor
from terms_extractor.dl_extractor.dl_extractor import DLExtractor
from terms_extractor.dict_extractor.dict_extractor import DictExtractor


class CombinedExtractor(BaseExtractor):

    def __init__(self):
        self._dict_extractor = DictExtractor()
        self._dl_extractor = DLExtractor()

    def extract(self, text: Union[str, List[str]]) -> List[Tuple[str, str]]:
        dict_results = self._dict_extractor.extract(text)
        dl_results = self._dl_extractor.extract(text)
        merged_results = self._merge_results(dict_results, dl_results)
        merged_results = validate_sequence(merged_results)
        return merged_results

    def _merge_results(
            self, dict_results: List[Tuple[str, str]], dl_results: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        merged_result = []
        for dict_result, dl_result in zip(dict_results, dl_results):
            if dl_result[1] == Tags.NOT_TERM.value:
                merged_result.append(dict_result)
            else:
                merged_result.append(dl_result)
        return merged_result
