import os

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
CONFIG_PATH = os.path.join(PROJECT_PATH, 'config', 'config.json')
TERMS_EXTRACTOR_PATH = os.path.join(PROJECT_PATH, 'terms_extractor')
DICT_EXTRACTOR_PATH = os.path.join(TERMS_EXTRACTOR_PATH, 'dict_extractor')
ENTITY_LINKER_PATH = os.path.join(PROJECT_PATH, 'entity_linker')
RELATION_EXTRACTOR_PATH = os.path.join(PROJECT_PATH, 'relation_extractor')
ASPECT_EXTRACTOR_PATH = os.path.join(PROJECT_PATH, 'aspect_extractor')

TERMS_EXTRACTOR_WEIGHTS_PATH = os.path.join(TERMS_EXTRACTOR_PATH, 'dl_extractor/weights', 'weights.h5')
RELATION_EXTRACTOR_WEIGHTS_PATH = os.path.join(
    RELATION_EXTRACTOR_PATH, 'dl_relation_extractor/weights'
)
ASPECT_EXTRACTOR_WEIGHTS_PATH = os.path.join(ASPECT_EXTRACTOR_PATH, 'weights', 'weights.h5')

WIKIDATA_DUMP_PATH = os.path.join(ENTITY_LINKER_PATH, 'wikidata_dump', 'dump.json')
FASTTEXT_MODEL_PATH = os.path.join(ENTITY_LINKER_PATH, 'fasttext_model', 'ft_native_300_ru_wiki_lenta_remstopwords.bin')