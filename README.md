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
1. For terms extraction download weights file from [here](https://drive.google.com/file/d/1IO-eVNLbGqeR6loMwxBMNxBHOZq3sg0f/view?usp=sharing) 
and put it to `terms_extractor/dl_extractor/weights`

2. For relation extraction: 

 2.1. Download config file from [here](https://drive.google.com/file/d/1JtD3-GAs58xqrKiquFtcSsrV42DeGE0r/view?usp=sharing)
 
 2.2. Download model file from [here](https://drive.google.com/file/d/1ksg-ZXDa8Fd10w3wPNxU8j-bk8B2YhTb/view?usp=sharing)
 
 2.3. Download model arguments file from [here](https://drive.google.com/file/d/1IvCCwj7-68MFx71bFX9kkUm_A1RQzxs-/view?usp=sharing)

 and put it all to `relation_extractor/dl_relation_extractor/weights`

3. For entity linking:

 3.1. Download prepocessed wikidata dump from [here](https://drive.google.com/file/d/1cSWLrbpq3f4PtRkAgIKiw_UNhshTgQOx/view?usp=sharing),
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

This module extracts relations between two terms. 
To extract relations it requires text with terms highlighted by special tokens.

Example of relation extraction:

```python
from relation_extractor.combined_relation_extractor.combined_relation_extractor import CombinedRelationExtractor

combined_extractor = CombinedRelationExtractor()
sample = '<e1>Модель</e1> используется в методе генерации и определения форм слов для решения ' \ 
         '<e2>задач морфологического синтеза</e2> и анализа текстов.'

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

## Data

[RuSERRC](https://github.com/iis-research-team/ruserrc-dataset) is the dataset of scientific texts in Russian, which is annotated with terms, aspects, linked entities, and relations. 

## Citation

If you find this repository useful, feel free to cite our papers:

Bruches E., Tikhobaeva O., Dementyeva Y., Batura T. [TERMinator: A System for Scientific Texts Processing](https://aclanthology.org/2022.coling-1.302). In Proceedings of the 29th International Conference on Computational Linguistics (COLING 2022). International Committee on Computational Linguistics. 2022. pp. 3420–3426.
```
@inproceedings{terminator2022,
    title={{TERM}inator: A System for Scientific Texts Processing},
    author={Bruches, Elena and Tikhobaeva, Olga and Dementyeva, Yana and Batura, Tatiana},
    booktitle={Proceedings of the 29th International Conference on Computational Linguistics},
    year={2022},
    pages={3420--3426}
}
```
Bruches E., Mezentseva A., Batura T. [A system for information extraction from scientific texts in Russian](https://arxiv.org/pdf/2109.06703.pdf). Data Analytics and Management in Data Intensive Domains. DAMDID/RCDL 2021. Communications in Computer and Information Science. Springer, Cham, 2022. vol. 1620. pp. 234–245.
```
@inproceedings{ruserrc,
  title={A system for information extraction from scientific texts in Russian},
  author={Bruches, Elena and Mezentseva, Anastasia and Batura, Tatiana},
  booktitle={Data Analytics and Management in Data Intensive Domains. DAMDID/RCDL 2021. Communications in Computer and Information Science},
  volume={1620}
  pages={234--245},
  year={2022}
}
```
