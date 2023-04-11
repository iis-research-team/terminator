from tensorflow.keras.activations import sigmoid
from transformers import BertConfig, TFBertForTokenClassification
from utils.constants import ASPECTS_PRETRAINED_MODEL_NAME, ASPECTS_NUM_LABELS

def get_model():
    """
    Создание модели
    :return: Модель
    """

    config = BertConfig.from_pretrained(ASPECTS_PRETRAINED_MODEL_NAME, num_labels=ASPECTS_NUM_LABELS)
    model = TFBertForTokenClassification.from_pretrained(ASPECTS_PRETRAINED_MODEL_NAME, config=config)
    model.layers[-1].activation = sigmoid
    print(model.summary())
    return model