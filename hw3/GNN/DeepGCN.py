import os
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
import visdom
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GENConv,DeepGCNLayer

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class DeepGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.node_encoder = torch.nn.Linear(in_channels, 32)
        self.layers = torch.nn.ModuleList()
        for i in range(1, 2 + 1):
            conv = GENConv(32, 32, learn_t=True, norm='layer')
            norm = torch.nn.LayerNorm(32)
            act = torch.nn.ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.6, ckpt_grad=i % 3)
            self.layers.append(layer)
        self.out = torch.nn.Linear(32, out_channels)

    def forward(self, x, edge_index):
        x = self.node_encoder(x)
        x = self.layers[0].conv(x, edge_index)
        for layer in self.layers[1:]:
            x = layer(x, edge_index)
        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.out(x)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
model = DeepGCN(dataset.num_features, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(data):
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

vis = visdom.Visdom(port=8099)
for epoch in range(1, 501):
    train(data)
    train_acc, val_acc, test_acc = test(data)
    vis.line(
        X=[epoch],
        Y=[train_acc],
        win='DeepGcn',
        name='train_acc',
        opts=dict(title='acc', showlegend=True),
        update='append')
    vis.line(
        X=[epoch],
        Y=[val_acc],
        win='DeepGcn',
        name='val_acc',
        opts=dict(title='acc', showlegend=True),
        update='append')
    vis.line(
        X=[epoch],
        Y=[test_acc],
        win='DeepGcn',
        name='test_acc',
        opts=dict(title='acc', showlegend=True),
        update='append')
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
          f'Test: {test_acc:.4f}')
