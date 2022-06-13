import os

from pprint import pprint
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from .BertForMultiLabelSequenceClassification import BertForMultiLabelSequenceClassification
from ..helper.TrainHelper import AverageMeter, LoggerPather

def get_tokenizer(pretrained_model_name_or_path, config):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        do_lower_case = config.MODEL.DO_LOWER_CASE
    )

    special_tokens = dict(config.MODEL.SPECIAL_TOKENS)

    # add the special tokens with the well-known and common attribute names
    if tokenizer.pad_token is None:
        print('set pad_token...')
        tokenizer.add_special_tokens({ 'pad_token': special_tokens.pop('PAD_TOKEN') })
    if tokenizer.cls_token is None:
        print('set cls_token...')
        tokenizer.add_special_tokens({ 'cls_token': special_tokens.pop('CLS_TOKEN') })
    if tokenizer.sep_token is None:
        print('set sep_token...')
        tokenizer.add_special_tokens({ 'sep_token': special_tokens.pop('SEP_TOKEN') })
    if tokenizer.bos_token is None:
        print('set bos_token...')
        tokenizer.add_special_tokens({ 'bos_token': special_tokens.pop('BOS_TOKEN') })
    if tokenizer.eos_token is None:
        print('set eos_token...')
        tokenizer.add_special_tokens({ 'eos_token': special_tokens.pop('EOS_TOKEN') })

    # add other task-specific or architecture-specific special tokens
    tokenizer.add_tokens(list(special_tokens.values()))

    return tokenizer


def get_model(config):
    pretrained_model_name_or_path = config.MODEL.PRETRAINED_MODEL_NAME_OR_PATH

    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    tokenizer = get_tokenizer(pretrained_model_name_or_path, config)
    LM = eval(config.MODEL.LM_TYPE)
    model = LM.from_pretrained(
        pretrained_model_name_or_path,
        config = model_config
    )
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
