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
1. For terms extraction download weights file from [here](https://drive.google.com/file/d/1d-p1kJ391wTG8t0WkYWBZ5l8Ph2hNkxd/view?usp=sharing) 
and put it to `terms_extractor/dl_extractor/weights`
2. For relation extraction ...
3. For entity linking:

 3.1. Download prepocessed wikidata dump from [here](https://drive.google.com/file/d/1pkVAsjqsUlJBWvU1322jm9fDvWHfsXoQ/view?usp=sharing),
  unzip and put it to `entity_linker/wikidata_dump`;
 
 3.2. Download fasttext model from [here](http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_remstopwords/ft_native_300_ru_wiki_lenta_remstopwords.bin)
 and put it to `entity_linker/fasttext_model`.

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

### Relation extraction

_in progress_

### Entity linking

This module links terms with entities in Wikidata. 
It requires extracted terms and their context as input.

```python
from entity_linker.entity_linker import RussianEntityLinker

ru_el = RussianEntityLinker()
term = 'язык программирования Python'
context = ['язык программирования Python', 'использовался', 'в']
print(ru_el.get_linked_mention(term, context))
```

## Citation

If you use this project, please cite this paper:
...
