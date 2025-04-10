import torch
from transformers import BertPreTrainedModel, BertModel
from transformers.utils import ModelOutput
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Optional,Tuple, Union
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification

class SequenceClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class CustomBertForSequenceClassification(BertPreTrainedModel):
  def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.config = config

        self.bert = BertModel.from_pretrained("bert-base-uncased",attn_implementation="eager")
        self.distilbert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english",attn_implementation="eager").distilbert
        self.t_layer = nn.TransformerEncoderLayer(d_model=768, nhead = 4)
        self.t_layer_weight = nn.TransformerEncoderLayer(d_model=768, nhead = 1)
        self.token_classifier = nn.Linear(768, self.num_labels)
        self.attention_reduction = nn.Linear(768,1)

        self.post_init()

  def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        with torch.no_grad():
          outputs_bert = self.bert(
              input_ids,
              attention_mask=attention_mask,
              output_attentions=False,
              output_hidden_states=output_hidden_states,
          )

        last_hidden_size = outputs_bert[0]

        t_output = self.t_layer(last_hidden_size)

        token_classification = self.token_classifier(t_output)
        token_classification = F.softmax(token_classification, dim=2)

        with torch.no_grad():
          output_distilbert = self.distilbert(input_ids,attention_mask=attention_mask,output_attentions=True)
        attentions = output_distilbert.attentions
        sep_index = (input_ids[0] == 102).nonzero(as_tuple=True)[0].item()
        token_classification = token_classification[:,1:sep_index,:]


        attention_scores = []
        for layer_attention in attentions[:]:
            layer_attention_mean = layer_attention.mean(dim=1)
            layer_attention_sum = layer_attention_mean.sum(dim=1)
            attention_scores.append(layer_attention_sum)

        attention_scores = torch.stack(attention_scores, dim=0).mean(dim=0)
        token_attention_scores = attention_scores[0]
        score = token_attention_scores[1:sep_index]
        attention_cls = score / score.sum()
        weighted_outputs = token_classification * attention_cls.unsqueeze(0).unsqueeze(2)
        logits = torch.sum(weighted_outputs, dim=1)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss,logits