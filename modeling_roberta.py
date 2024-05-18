from typing import Optional, Tuple, Union
import dgl
import dgl.function as fn
from dgl import DGLGraph
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

gcn_msg = fn.copy_u(u="h", out="m")
gcn_reduce = fn.sum(msg="m", out="h")

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata["h"] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata["h"]
            return self.linear(h)
        

class GCNModel(nn.Module):
    def __init__(self):
        super(GCNModel, self).__init__()
        self.layer1 = GCNLayer(768, 768)
        self.layer2 = GCNLayer(768, 768)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, with_graph=False):
        super().__init__()
        if (with_graph):
            self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# baseline
class RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
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

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
# GCN global nodes
class RobertaForSequenceClassification_GCN(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.GCN = GCNModel()
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config, with_graph=True)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        nodes: Optional[list] = None,
        edges: Optional[list] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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
        self.the_device = input_ids.device
        sequence_output = outputs[0]
        batch_size = sequence_output.size(0)
        seq_len = sequence_output.size(1)
        hidden_size = sequence_output.size(2)
        graph_reps = torch.ones(batch_size, hidden_size).to(self.the_device)
        
        # Iterate over batches
        for batchi in range(batch_size):
            node_features = []
            # num_entities = torch.sum((nodes_batch[:, 0] != 0) & (nodes_batch[:, 1] != 0))
            num_entities = 0
            for node in nodes[batchi]:
                if node[0] == 0 and node[1] == 0:
                    break
                node_features.append(sequence_output[batchi, node[0]+1:node[1]+1, :].mean(dim=0))
                num_entities += 1
            if num_entities > 0:
                # 文本有对应的语义图
                G = dgl.graph(([], [])).to(self.the_device)
                # 添加节点
                G.add_nodes(num_entities + 1)
                # 添加实体关系边
                for edge in edges[batchi]:
                    if edge[0] == 0 and edge[1] == 0:
                        break
                    G.add_edges(edge[0], edge[1])
                    G.add_edges(edge[1], edge[0])
                # 添加自环关系边
                for x in range(num_entities + 1):
                    G.add_edges(x,x)
                # 添加全局关系边
                for x in range(num_entities):
                    G.add_edges(num_entities, x)
                # add node feature
                for i in range(num_entities + 1):
                    if i < num_entities:
                        G.nodes[[i]].data['h'] = node_features[i].unsqueeze(0)
                    elif i == num_entities:
                        # 全局节点的初始特征为[CLS]特征
                        G.nodes[[i]].data['h'] = sequence_output[batchi, 0, :].unsqueeze(0)
                
                X = self.GCN(G, G.ndata['h'])
                graph_reps[batchi, :] = X[num_entities]
                if torch.isnan(X[num_entities]).any():
                    print("GCN output contains NaN values!")
                # if torch.all(X[num_entities] == 0):
                #     print("GCN output are all zero!")
            else:
                # 如果没有图表示，添加全0特征
                graph_reps[batchi, :] = torch.zeros(1, hidden_size, dtype=graph_reps.dtype, device=graph_reps.device)

        whole_rep = torch.cat([sequence_output[:, 0, :], graph_reps], dim=-1)
        logits = self.classifier(whole_rep)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
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

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
# GCN Mean
class RobertaForSequenceClassification_GCN_Mean(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.GCN = GCNModel()
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config, with_graph=True)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        nodes: Optional[list] = None,
        edges: Optional[list] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
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
        self.the_device = input_ids.device
        sequence_output = outputs[0]
        batch_size = sequence_output.size(0)
        hidden_size = sequence_output.size(2)
        graph_reps = torch.ones(batch_size, hidden_size).to(self.the_device)
        
        # Iterate over batches
        for batchi in range(batch_size):
            node_features = []
            num_entities = 0
            for node in nodes[batchi]:
                if node[0] == 0 and node[1] == 0:
                    break
                node_features.append(sequence_output[batchi, node[0]+1:node[1]+1, :].mean(dim=0))
                num_entities += 1
            if num_entities > 0:
                # 文本有对应的语义图
                G = dgl.graph(([], [])).to(self.the_device)
                # 添加节点
                G.add_nodes(num_entities)
                # 添加实体关系边
                for edge in edges[batchi]:
                    if edge[0] == 0 and edge[1] == 0:
                        break
                    G.add_edges(edge[0], edge[1])
                    G.add_edges(edge[1], edge[0])
                # 添加自环关系边
                for x in range(num_entities + 1):
                    G.add_edges(x,x)
                # 添加节点特征
                for i in range(num_entities):
                    G.nodes[[i]].data['h'] = node_features[i].unsqueeze(0)
                
                X = self.GCN(G, G.ndata['h'])
                graph_reps[batchi, :] = torch.mean(X, dim=0)
            else:
                # 如果没有图表示，添加全0特征
                graph_reps[batchi, :] = torch.zeros(1, hidden_size, dtype=graph_reps.dtype, device=graph_reps.device)

        whole_rep = torch.cat([sequence_output[:, 0, :], graph_reps], dim=-1)
        logits = self.classifier(whole_rep)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
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

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )