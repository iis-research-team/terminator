from typing import List, Tuple

from nltk.tokenize import wordpunct_tokenize

from utils.constants import Tags


def validate_sequence(seq: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """ Валидация последовательности тэгов: убеждаемся, что первый токен каждого термина имеет тэг "B-TERM"

    :param seq: входная последовательность
    :return: провалидированная последовательность
    """
    is_previous_token = False
    validated_seq = []
    for token, tag in seq:
        if tag == Tags.I_TERM.value:
            if not is_previous_token:
                validated_seq.append((token, Tags.B_TERM.value))
            else:
                validated_seq.append((token, tag))
            is_previous_token = True
        elif tag == Tags.B_TERM.value:
            validated_seq.append((token, tag))
            is_previous_token = True
        else:
            validated_seq.append((token, tag))
            is_previous_token = False
    return validated_seq


def tokenize(text: str) -> List[str]:
    puncts = {'(', ')', ':', ';', ',', '.', '"', '»', '«', '[', ']', '{', '}', '%'}

    tokens = wordpunct_tokenize(text)
    validated_tokens = []
    for token in tokens:
        is_all_puncts = True
        for char in token:
            if char not in puncts:
                is_all_puncts = False
        if is_all_puncts:
            validated_tokens.extend(list(token))
        else:
            validated_tokens.append(token)
    return validated_tokens
