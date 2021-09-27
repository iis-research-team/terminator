import tensorflow as tf
from transformers import BertConfig, TFBertForSequenceClassification

MAX_LENGTH = 128


def get_model(model_name, num_labels):
    if model_name == 'bert_for_sequence_classification':
        config = BertConfig.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels)
        model = TFBertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased",
            config=config
        )
        model.layers[-1].activation = tf.keras.activations.softmax

        for layer in model.layers:
            layer.trainable = False

        model.layers[-1].trainable = True

        print(model.summary())
        return model
