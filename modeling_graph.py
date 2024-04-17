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

class RGCNLayer(nn.Module):
    def __init__(self, feat_size, num_rels, activation=None, gated = True):
        
        super(RGCNLayer, self).__init__()
        self.feat_size = feat_size
        self.num_rels = num_rels
        self.activation = activation
        self.gated = gated

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.feat_size, self.feat_size))
        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,gain=nn.init.calculate_gain('relu'))
        
        if self.gated:
            self.gate_weight = nn.Parameter(torch.Tensor(self.num_rels, self.feat_size, 1))
            nn.init.xavier_uniform_(self.gate_weight,gain=nn.init.calculate_gain('sigmoid'))
        
    def forward(self, g):
        
        weight = self.weight
        gate_weight = self.gate_weight
        
        def message_func(edges):
            w = weight[edges.data['rel_type']]
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            msg = msg * edges.data['norm']
            
            if self.gated:
                gate_w = gate_weight[edges.data['rel_type']]
                gate = torch.bmm(edges.src['h'].unsqueeze(1), gate_w).squeeze().reshape(-1,1)
                gate = torch.sigmoid(gate)
                msg = msg * gate    
            return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)


class RGCNModel(nn.Module):
    def __init__(self, h_dim, num_rels, num_hidden_layers=1, gated = True):
        super(RGCNModel, self).__init__()

        self.h_dim = h_dim
        self.num_rels = num_rels
        self.num_hidden_layers = num_hidden_layers
        self.gated = gated
        
        # create rgcn layers
        self.build_model()
       
    def build_model(self):        
        self.layers = nn.ModuleList() 
        for _ in range(self.num_hidden_layers):
            rgcn_layer = RGCNLayer(self.h_dim, self.num_rels, activation=F.relu, gated = self.gated)
            self.layers.append(rgcn_layer)
    
    def forward(self, g):
        for layer in self.layers:
            layer(g)
        
        rst_hidden = []
        if isinstance(g, DGLGraph):
            rst_hidden.append(g.ndata['h'])
            # print("dglgraph")
        else:
            print("batcheddglgraph")
            for sub_g in dgl.unbatch(g):
                rst_hidden.append(sub_g.ndata['h'])
        return rst_hidden

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

class RobertaForSequenceClassification_RGCN(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.GCN = RGCNModel(768, 3, 1, True)
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
        for batchi in range(batch_size):
            # every sample in a batch
            node_features = []
            edge_type = []    # in total 3 type of edges
            edge_list = []
            num_entities = 0
            for node in nodes[batchi]:
                if node[0] == 0 and node[1] == 0:
                    break
                node_features.append(sequence_output[batchi, node[0]+1:node[1]+1, :].mean(dim=0))
                num_entities += 1
            if num_entities > 0:
                # 文本有对应的语义图
                G = dgl.DGLGraph().to(self.the_device)
                # 添加节点
                G.add_nodes(num_entities + 1)
                # 添加实体关系边
                for edge in edges[batchi]:
                    if edge[0] == 0 and edge[1] == 0:
                        break
                    G.add_edges(edge[0], edge[1])
                    edge_type.append(0)
                    edge_list.append([edge[0],edge[1]])
                    G.add_edges(edge[1], edge[0])
                    edge_type.append(0)
                    edge_list.append([edge[1],edge[0]])
                # 添加自环关系边
                for x in range(num_entities + 1):
                    G.add_edges(x,x)
                    edge_type.append(1)
                    edge_list.append([x,x])
                # 添加全局关系边
                for x in range(num_entities):
                    G.add_edges(num_entities, x)
                    edge_type.append(2)
                    edge_list.append([num_entities, x])
                # add node feature
                for i in range(num_entities + 1):
                    if i < num_entities:
                        G.nodes[[i]].data['h'] = node_features[i].unsqueeze(0)
                    elif i == num_entities:
                        G.nodes[[i]].data['h'] = torch.randn(hidden_size).unsqueeze(0).to(self.the_device)
                # add edge feature
                edge_type = torch.from_numpy(np.array(edge_type)).to(self.the_device)
                edge_norm = torch.ones(len(edge_list), device=self.the_device, dtype=torch.float32)
                for i, (e1, e2) in enumerate(edge_list):
                    if e1 != e2:
                        edge_norm[i] = 1 / (G.in_degrees(e2) - 1)
                edge_norm = edge_norm.unsqueeze(1)
                G.edata.update({'rel_type': edge_type})
                G.edata.update({'norm': edge_norm})
                X = self.GCN(G)[0]   # [bz, hdim]
                graph_reps[batchi, :] = X[num_entities]
                if torch.isnan(X[num_entities]).any():
                    print("GCN output contains NaN values!")
                if torch.all(X[num_entities] == 0):
                    print("GCN output are all zero!")
                # print("0, X[num_entities]", X[num_entities])
            else:
                # 如果没有图表示，添加随机特征
                graph_reps[batchi, :] = torch.randn(hidden_size).to(self.the_device)
        whole_rep = torch.cat([sequence_output[:, 0, :], graph_reps], dim=-1)
        if torch.all(graph_reps == 0):
            print("GCN output are all zero!")
        if torch.isnan(whole_rep).any():
            print("Concatenated representation contains NaN values!")
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