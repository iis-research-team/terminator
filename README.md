# Terminator
_Tool for Information extraction from Russian texts_

This tool includes the following modules:
* Terms extraction
* Relation extraction
* Entity linking

## Installation and preparation

To install:

`git clone https://github.com/iis-research-team/Terminator.git`

To use this tool one should download the files:
1. For terms extraction download weights file from here and put it to `terms_extractor/dl_extractor/weights`
2. For relation extraction ...
3. For entity linking ...

## How to use

### Terms extraction

This module extracts terms from the raw text.

```python
from terms_extractor.combined_extractor.combined_extractor import CombinedExtractor   

combined_extractor = CombinedExtractor()
text = 'Научные вычисления включают прикладную математику (особенно численный анализ), вычислительную технику ' \
       '(особенно высокопроизводительные вычисления) и математическое моделирование объектов изучаемых научной ' \
       'дисциплиной.'
result = combined_extractor.extract(text)
for token, tag in result:
    print(f'{token} -> {tag}')
```

## Citation

If you use this project, please cite this paper:
...
