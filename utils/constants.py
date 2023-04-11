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

aspects_class2label = {
    0: 'Task',
    1: 'Contrib',
    2: 'Method',
    3: 'Conc'
}

aspects_label2class = {
    'Task': 0,
    'Contrib': 1,
    'Method': 2,
    'Conc': 3
}


ASPECTS_PRETRAINED_MODEL_NAME = 'bert-base-multilingual-cased'
ASPECTS_NUM_LABELS = 4

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
RE_LABELS = ["PART_OF", "ISA", "USAGE", "TOOL", "SYNONYMS", "COMPARE", "CAUSE", "NO-RELATION"]
