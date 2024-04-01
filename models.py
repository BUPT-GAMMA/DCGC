import torch
import torch_geometric.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GINConv, SGConv, TAGConv, AGNNConv, MLP, GCN2Conv
import torch.nn.functional as F
from conv import SAGEConv, CalibAttentionLayer, GATConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_edge, dropout):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.edge_weight = torch.ones(num_edge).cuda()

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(len(edge_index[0])).cuda()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_edge, dropout):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

        self.edge_weight = torch.ones(num_edge).cuda()

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = self.edge_weight
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_weight).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return x


class GATS(torch.nn.Module):
    def __init__(self, model, edge_index, num_nodes, train_mask, num_class, dist_to_train=None):
        super().__init__()
        self.model = model
        self.num_nodes = num_nodes
        self.cagat = CalibAttentionLayer(in_channels=num_class,
                                         out_channels=1,
                                         edge_index=edge_index,
                                         num_nodes=num_nodes,
                                         train_mask=train_mask,
                                         dist_to_train=dist_to_train,
                                         heads=8,
                                         bias=1).cuda()
        for para in self.model.parameters():
            para.requires_grad = False

    def forward(self, x, edge_index, edge_weight=None):
        logits = self.model(x, edge_index, edge_weight)
        temperature = self.graph_temperature_scale(logits)
        return logits / temperature

    def graph_temperature_scale(self, logits):
        """
        Perform graph temperature scaling on logits
        """
        temperature = self.cagat(logits).view(self.num_nodes, -1)
        return temperature.expand(self.num_nodes, logits.size(1))


class Edge_Weight(torch.nn.Module):
    def __init__(self, out_channels, base_model, dropout):
        super(Edge_Weight, self).__init__()
        self.base_model = base_model
        self.extractor = nn.MLP([out_channels*2, out_channels*4, 1], dropout=dropout)

        for para in self.base_model.parameters():
            para.requires_grad = False

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = self.get_weight(x, edge_index)
        logist = self.base_model(x, edge_index, edge_weight)
        return logist

    def get_weight(self, x, edge_index):

        emb = self.base_model(x, edge_index)
        col, row = edge_index
        f1, f2 = emb[col], emb[row]
        f12 = torch.cat([f1, f2], dim=-1)
        edge_weight = self.extractor(f12)
        return edge_weight.relu()


class Temperature_Scalling(torch.nn.Module):
    def __init__(self, base_model):
        super(Temperature_Scalling, self).__init__()
        self.base_model = base_model
        self.temperature = torch.nn.Parameter(torch.ones(1))

        for para in self.base_model.parameters():
            para.requires_grad = False

    def forward(self, x, edge_index, edge_weight=None):
        logist = self.base_model(x, edge_index, edge_weight)
        temperature = self.temperature.expand(logist.size(0), logist.size(1))
        return logist * temperature

    def reset_parameters(self):
        self.temperature.data.fill_(1)


class CaGCN(torch.nn.Module):
    def __init__(self, base_model, out_channels, hidden_channels):
        super(CaGCN, self).__init__()
        self.base_model = base_model
        self.conv1 = GCNConv(out_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

        for para in self.base_model.parameters():
            para.requires_grad = False

    def forward(self, x, edge_index, edge_weight=None):
        logist = self.base_model(x, edge_index, edge_weight)
        x = F.dropout(logist, p=0.5, training=self.training)
        x = self.conv1(x=x, edge_index=edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        temperature= self.conv2(x=x, edge_index=edge_index)
        temperature = torch.log(torch.exp(temperature) + torch.tensor(1.1))
        return logist * temperature


class VS(torch.nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.temperature = torch.nn.Parameter(torch.ones(num_classes))
        self.bias = torch.nn.Parameter(torch.ones(num_classes))

        for para in self.base_model.parameters():
            para.requires_grad = False

    def forward(self, x, edge_index, edge_weight=None):
        logits = self.base_model(x, edge_index, edge_weight)
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), logits.size(1))
        return logits * temperature + self.bias

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, num_edge):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)
        self.edge_weight = torch.ones(num_edge).cuda()


    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = self.edge_weight
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

class TAGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = TAGConv(in_channels, hidden_channels)
        self.conv2 = TAGConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class SGC(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_edge):
        super().__init__()
        self.conv1 = SGConv(in_channels, hidden_channels, K=2)
        self.conv2 = SGConv(hidden_channels, out_channels, K=2)
        self.edge_weight = torch.ones(num_edge).cuda()


    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = self.edge_weight
        x = F.dropout(x, 0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        x = F.dropout(x, 0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

