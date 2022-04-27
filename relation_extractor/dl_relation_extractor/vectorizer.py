import torch
from transformers import BertTokenizer
from utils.constants import ADDITIONAL_SPECIAL_TOKENS


class Vectorizer:

    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self._tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

    def vectorize(self, text, args, cls_token="[CLS]", cls_token_segment_id=0, sep_token="[SEP]", pad_token_id=0,
                  pad_token_segment_id=0, sequence_a_segment_id=0, add_sep_token=False, mask_padding_with_zero=True):
        # Setting based on the current model type

        tokens = self._tokenizer.tokenize(text)

        e11_p = tokens.index("<e1>")  # the start position of entity1
        e12_p = tokens.index("</e1>")  # the end position of entity1
        e21_p = tokens.index("<e2>")  # the start position of entity2
        e22_p = tokens.index("</e2>")  # the end position of entity2

        # Replace the token
        tokens[e11_p] = "$"
        tokens[e12_p] = "$"
        tokens[e21_p] = "#"
        tokens[e22_p] = "#"

        # Add 1 because of the [CLS] token
        e11_p += 1
        e12_p += 1
        e21_p += 1
        e22_p += 1

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        if add_sep_token:
            special_tokens_count = 2
        else:
            special_tokens_count = 1
        if len(tokens) > args['max_seq_len'] - special_tokens_count:
            tokens = tokens[: (args['max_seq_len'] - special_tokens_count)]

        # Add [SEP] token
        if add_sep_token:
            tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args['max_seq_len'] - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # e1 mask, e2 mask
        e1_mask = [0] * len(attention_mask)
        e2_mask = [0] * len(attention_mask)

        for i in range(e11_p, e12_p + 1):
            e1_mask[i] = 1
        for i in range(e21_p, e22_p + 1):
            e2_mask[i] = 1

        # Convert to Tensors
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)
        token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)
        e1_mask = torch.tensor([e1_mask], dtype=torch.long)
        e2_mask = torch.tensor([e2_mask], dtype=torch.long)

        return input_ids, attention_mask, token_type_ids, e1_mask, e2_mask
