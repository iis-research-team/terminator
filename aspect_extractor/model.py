import tensorflow as tf
from transformers import TFBertModel, BertConfig, TFBertForTokenClassification

NUM_LABELS = 10


def get_model():
    config = BertConfig.from_pretrained('bert-base-multilingual-cased', num_labels=NUM_LABELS)
    model = TFBertForTokenClassification.from_pretrained(
        "bert-base-multilingual-cased",
        config=config
    )
    model.layers[-1].activation = tf.keras.activations.sigmoid
    print(model.summary())
    return model
