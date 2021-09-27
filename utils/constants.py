from enum import Enum


class Tags(Enum):
    B_TERM = 'B-TERM'
    I_TERM = 'I-TERM'
    NOT_TERM = 'O'


TERM_SET = {Tags.B_TERM.value, Tags.I_TERM.value}

label2class = {
            Tags.NOT_TERM.value: 0,
            Tags.B_TERM.value: 1,
            Tags.I_TERM.value: 2
        }

class2label = {
            0: Tags.NOT_TERM.value,
            1: Tags.B_TERM.value,
            2: Tags.I_TERM.value
        }

re_class2label = {
    0: 'USED-FOR',
    1: 'FEATURE-OF',
    2: 'HYPONYM-OF',
    3: 'PART-OF',
    4: 'COMPARE',
    5: 'CONJUNCTION',
    6: 'NO-RELATION'
}

RUSSERC_TO_RC_SCIENCE_LABELS = {
    'ISA': 'HYPONYM-OF',
    'PART_OF': 'PART-OF',
    'USAGE': 'USED-FOR',
    'TOOL': 'USED-FOR',
    'COMPARE': 'COMPARE',
    'NO-RELATION': 'NO-RELATION'
}
