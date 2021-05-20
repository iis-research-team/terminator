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
