import tensorflow as tf
from transformers import TFBertModel, BertConfig, TFBertForTokenClassification


def get_model():
    config = BertConfig.from_pretrained('bert-base-multilingual-cased', num_labels=3)
    model = TFBertForTokenClassification.from_pretrained(
        "bert-base-multilingual-cased",
        config=config
    )
    model.layers[-1].activation = tf.keras.activations.softmax
    print(model.summary())
    return model
