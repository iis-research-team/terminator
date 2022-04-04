# Terminator
_Tool for Information extraction from Russian texts_

This tool includes the following modules:
* Terms extraction
* Relation extraction
* Entity linking
* Aspect extraction

## Installation and preparation

To install:

`git clone https://github.com/iis-research-team/Terminator.git`

To use this tool one should download the files:
1. For terms extraction download weights file from [here](https://drive.google.com/file/d/1d-p1kJ391wTG8t0WkYWBZ5l8Ph2hNkxd/view?usp=sharing) 
and put it to `terms_extractor/dl_extractor/weights`
2. For relation extraction download weights file from [here](https://drive.google.com/file/d/11LMTNf-u7BY6hzeFAR5jWW7x7jGaRef3/view?usp=sharing)
and put it to `relation_extractor/dl_relation_extractor/weights`
3. For entity linking:

 3.1. Download prepocessed wikidata dump from [here](https://drive.google.com/file/d/1pkVAsjqsUlJBWvU1322jm9fDvWHfsXoQ/view?usp=sharing),
  unzip and put it to `entity_linker/wikidata_dump`;
 
 3.2. Download fasttext model from [here](http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_remstopwords/ft_native_300_ru_wiki_lenta_remstopwords.bin)
 and put it to `entity_linker/fasttext_model`.

4. For aspect extraction download weights file from [here](https://drive.google.com/file/d/1uHjHWm4CC19TPCzVr1Jy-f_XAWr7hyA6/view?usp=sharing)
and put it to `aspect_extractor/weights`
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

This module extracts relations between two terms. To extract relations it requires dict with the following keys:
```json
{
  "token": "list of tokens for a given sentence",
  "subj_start": "int, position of the first token for the subject term",
  "subj_end": "int, position of the last token for the subject term",
  "obj_start": "int, position of the first token for the object term",
  "obj_end": "int, position of the last token for the object term"
}
```

Example of relation extraction:
```python
from relation_extractor.combined_relation_extractor.combined_relation_extractor import CombinedRelationExtractor

combined_extractor = CombinedRelationExtractor()
sample = {
    'token': ['извлечение', 'отношений', '-', 'это', 'задача', 'NLP'],
    'subj_start': 0,
    'subj_end': 1,
    'obj_start': 5,
    'obj_end': 5 
}

relation = combined_extractor.extract(sample)
```

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

### Aspect extraction

This module extracts aspects from the raw text.

```python
from aspect_extractor import AspectExtractor   

extractor = AspectExtractor()
text = "Определена модель для визуализации связей между объектами и их атрибутами в различных процессах. " \
           "На основании модели разработан универсальный абстрактный компонент графического пользовательского интерфейса и приведены примеры его программной реализации. " \
           "Также проведена апробация компонента для решения прикладной задачи по извлечению информации из документов."
result = extractor.extract(text)
for token, tag in result:
    print(f'{token} -> {tag}')
```

## Citation

If you use this project, please cite this paper:

Elena Bruches, Anastasia Mezentseva, Tatiana Batura. 
A system for information extraction from scientific texts in Russian. 2021.

Link: https://arxiv.org/abs/2109.06703
