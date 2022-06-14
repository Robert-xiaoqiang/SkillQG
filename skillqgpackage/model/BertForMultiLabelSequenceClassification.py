import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


# # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
# # so we need to apply the function recursively.
# def load(module: nn.Module, prefix=""):
#     local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
#     module._load_from_state_dict(
#         state_dict,
#         prefix,
#         local_metadata,
#         True,
#         missing_keys,
#         unexpected_keys,
#         error_msgs,
#     )
#     for name, child in module._modules.items():
#         if child is not None:
#             load(child, prefix + name + ".")

'''
    Huggingface Transformers v4.4.2 only
'''
class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.num_labels = config.num_labels
        self.num_labels = 5

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) # B x num_labels

        loss = None
        if labels is not None:
            assert logits.shape == labels.shape, 'Error in the labels of multilabel classification: {} != {}'.format(logits.shape, labels.shape)
            # Multi-label classification == num_labels times binary classification
            independent_normalized_logits = F.sigmoid(logits)
            # B(optional=1) x num_labels x (optional shape HxW, DxHxW, or L)
            loss_fct = nn.BCELoss(reduction = 'sum')
            '''
            `nn.BCELoss` is typically applied to binary classification (pipeline shape: B1, B1xHxW, or B1xDxHxW).
            Hereon, we binarily classify a label of a sample and its shape (BxC) seems vague and confusing because this shape is more common in a multi-class classification setting. However, according to the official implementation, `nn.BCELoss` will work very well and exactly calculate what we want.
            The final loss is computed as the average of losses of samples in the batch, while the loss of a sample is the summation of all its labels.
            '''
            raw_loss = loss_fct(independent_normalized_logits, labels.float())
            loss = raw_loss / labels.size(0)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
