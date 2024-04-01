import os
import os.path as osp
import argparse
import random
import statistics

import numpy as np
import scipy.stats as stats

import torch
import torch.optim as optim
from torch_geometric.datasets import Planetoid, Amazon, CitationFull
from torch.utils.data import random_split
import torch.nn.functional as F
from torch_geometric.utils import mask_to_index, index_to_mask, homophily
import torch_geometric.transforms as T

from util import ECELoss, plot_acc_calibration
from models import GCN, GraphSAGE, Edge_Weight, GATS, Temperature_Scalling, CaGCN, VS

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="Pubmed",
                    help='dataset for training')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--cal_epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--lr_for_cal', type=float, default=0.01)
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.7,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--l2_for_cal', type=float, default=5e-3,
                    help='Weight decay (L2 loss on parameters) for calibration.')
parser.add_argument('--n_bins', type=int, default=20)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--num1', type=int, default=1)
parser.add_argument('--num2', type=int, default=5)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=10)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
criterion = torch.nn.CrossEntropyLoss().cuda()
weight_loss = torch.nn.L1Loss()
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'CitationFull')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset = CitationFull(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)
nclass = data.y.max().item() + 1
nfeat = data.x.shape[1]
def get_dataset():
    torch.manual_seed(np.random.randint(0, 10000))
    train_num = int(data.num_nodes * 0.1)
    val_num = int(data.num_nodes * 0.05)
    test_num = data.num_nodes - train_num - val_num
    idx = range(data.num_nodes)
    train_idx, test_idx = random_split(dataset=idx, lengths=[train_num, val_num + test_num])
    val_idx, test_idx = random_split(dataset=test_idx, lengths=[val_num, test_num])

    return list(train_idx), list(val_idx), list(test_idx)


def get_homogeneous_edge(label):
    col, row = data.edge_index
    diff = label[col] - label[row]
    homogeneous, heterogeneous = torch.where(diff == 0)[0], torch.where(diff != 0)[0]
    return homogeneous, heterogeneous


def graph_attribution(model, idx_test):
    model.eval()
    edge_weight = torch.ones(data.num_edges).cuda()
    edge_weight.requires_grad = True
    logits = model(data.x, data.edge_index, edge_weight)
    loss = criterion(logits[idx_test], data.y[idx_test])
    loss.backward()
    return edge_weight.grad.detach().data.cpu()


def train_base(model, optimizer, data, idx_train, idx_val, idx_test, edge_weight=None):
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index, edge_weight)
    loss = criterion(output[idx_train], data.y[idx_train])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test_base(model, data, idx_train, idx_val, idx_test, edge_weight=None):
    model.eval()
    output = model(data.x, data.edge_index, edge_weight)
    pred = output.argmax(dim=-1)
    acc_train = int((pred[idx_train] == data.y[idx_train]).sum()) / len(idx_train)
    acc_val = int((pred[idx_val] == data.y[idx_val]).sum()) / len(idx_val)
    acc_test = int((pred[idx_test] == data.y[idx_test]).sum()) / len(idx_test)
    loss_train = criterion(output[idx_train], data.y[idx_train])
    loss_val = criterion(output[idx_val], data.y[idx_val])
    loss_test = criterion(output[idx_test], data.y[idx_test])
    ece_train = ECELoss(output[idx_train], data.y[idx_train], args.n_bins)
    ece_val = ECELoss(output[idx_val], data.y[idx_val], args.n_bins)
    ece_test = ECELoss(output[idx_test], data.y[idx_test], args.n_bins)
    return acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, ece_train, ece_val, ece_test

def train_cali(model, optimizer, data, idx_train, idx_val, idx_test, edge_weight=None):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, edge_weight)
    loss = criterion(out[idx_train], data.y[idx_train])
    loss.backward()
    optimizer.step()

@torch.no_grad()
def test_cali(model, data, idx_train, idx_val, idx_test, edge_weight=None):
    model.eval()
    output = model(data.x, data.edge_index, edge_weight)
    pred = output.argmax(dim=-1)
    acc_train = int((pred[idx_train] == data.y[idx_train]).sum()) / len(idx_train)
    acc_val = int((pred[idx_val] == data.y[idx_val]).sum()) / len(idx_val)
    acc_test = int((pred[idx_test] == data.y[idx_test]).sum()) / len(idx_test)
    loss_train = criterion(output[idx_train], data.y[idx_train])
    loss_val = criterion(output[idx_val], data.y[idx_val])
    loss_test = criterion(output[idx_test], data.y[idx_test])
    ece_train = ECELoss(output[idx_train], data.y[idx_train], args.n_bins)
    ece_val = ECELoss(output[idx_val], data.y[idx_val], args.n_bins)
    ece_test = ECELoss(output[idx_test], data.y[idx_test], args.n_bins)

    return acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, ece_train, ece_val, ece_test

def mian_base(model, optimizer_t, data, idx_train, idx_val, idx_test, edge_weight=None):

    best = 0
    for epoch in range(args.epochs):
        train_base(model, optimizer_t, data, idx_train, idx_val, idx_test, edge_weight)
        acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, ece_train, ece_val, ece_test = \
            test_base(model, data, idx_train, idx_val, idx_test, edge_weight)

        if acc_val > best:
            torch.save(model.state_dict(), 'base_model.pth')
            best = acc_val

    state_dict = torch.load('base_model.pth')
    model.load_state_dict(state_dict)
    with torch.no_grad():
        model.eval()
        output = model(data.x, data.edge_index, edge_weight)
        pred = output.argmax(dim=-1)
        acc_train = int((pred[idx_train] == data.y[idx_train]).sum()) / len(idx_train)
        acc_val = int((pred[idx_val] == data.y[idx_val]).sum()) / len(idx_val)
        acc_test = int((pred[idx_test] == data.y[idx_test]).sum()) / len(idx_test)
        ece_train = ECELoss(output[idx_train], data.y[idx_train], args.n_bins)
        ece_val = ECELoss(output[idx_val], data.y[idx_val], args.n_bins)
        ece_test = ECELoss(output[idx_test], data.y[idx_test], args.n_bins)

    print(f'acc_train: {acc_train:.4f}',
          f'acc_val: {acc_val:.4f}',
          f'acc_test: {acc_test:.4f}',
          f'ece_train: {ece_train:.4f}',
          f'ece_val: {ece_val:.4f}',
          f'ece_test: {ece_test:.4f}',
          )

    return acc_train, acc_val, acc_test, ece_train, ece_val, ece_test


def mian_cali(model, optimizer_s, data, idx_train, idx_val, idx_test, edge_weight=None):

    best = 100
    bad_counter = 0
    for epoch in range(args.cal_epochs):
        train_cali(model, optimizer_s, data, idx_train, idx_val, idx_test, edge_weight)
        acc_train, acc_val, acc_test, loss_train, loss_val, loss_test, ece_train, ece_val, ece_test = \
            test_cali(model, data, idx_train, idx_val, idx_test, edge_weight)
        if loss_val < best:
            torch.save(model.state_dict(), 'calibration.pth')
            best = loss_val
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == args.patience:
            break

    state_dict = torch.load('calibration.pth')
    model.load_state_dict(state_dict)
    with torch.no_grad():
        model.eval()
        output = model(data.x, data.edge_index, edge_weight)
        pred = output.argmax(dim=-1)
        acc_train = int((pred[idx_train] == data.y[idx_train]).sum()) / len(idx_train)
        acc_val = int((pred[idx_val] == data.y[idx_val]).sum()) / len(idx_val)
        acc_test = int((pred[idx_test] == data.y[idx_test]).sum()) / len(idx_test)
        ece_train = ECELoss(output[idx_train], data.y[idx_train], args.n_bins)
        ece_val = ECELoss(output[idx_val], data.y[idx_val], args.n_bins)
        ece_test = ECELoss(output[idx_test], data.y[idx_test], args.n_bins)

    print(f'acc_train: {acc_train:.4f}',
          f'acc_val: {acc_val:.4f}',
          f'acc_test: {acc_test:.4f}',
          f'ece_train: {ece_train:.4f}',
          f'ece_val: {ece_val:.4f}',
          f'ece_test: {ece_test:.4f}',
          )

    return acc_train, acc_val, acc_test, ece_train, ece_val, ece_test


def main():
    old_acc_train, old_acc_val, old_acc_test, old_ece_train, old_ece_val, old_ece_test = [], [], [], [], [], []
    new_acc_test0, new_ece_test0, new_acc_test1, new_ece_test1, new_acc_test2, new_ece_test2 = [], [], [], [], [], []
    new_acc_test3, new_ece_test3, new_acc_test4, new_ece_test4, new_acc_test5, new_ece_test5 = [], [], [], [], [], []
    new_acc_test6, new_ece_test6, new_acc_test7, new_ece_test7 = [], [], [], []
    for i in range(args.num1):
        print('---------------------------------')
        idx_train, idx_val, idx_test = get_dataset()
        base_model = GCN(nfeat, args.hidden, nclass, data.num_edges, args.dropout).to(device)

        optimizer_base = torch.optim.Adam([
            dict(params=base_model.conv1.parameters(), weight_decay=5e-4),
            dict(params=base_model.conv2.parameters(), weight_decay=0)
        ], lr=args.lr)

        acc_train, acc_val, acc_test, ece_train, ece_val, ece_test = \
            mian_base(base_model, optimizer_base, data, idx_train, idx_val, idx_test)

        old_acc_train.append(acc_train * 100), old_acc_val.append(acc_val * 100), old_acc_test.append(acc_test * 100)
        old_ece_train.append(ece_train * 100), old_ece_val.append(ece_val * 100), old_ece_test.append(ece_test * 100)

        state_dict = torch.load('base_model.pth')
        base_model.load_state_dict(state_dict)

        for j in range(args.num2):
            print('---------------------------------')
            ew = Edge_Weight(nclass, base_model, args.dropout).to(device)
            optimizer_ew = optim.Adam(filter(lambda p: p.requires_grad, ew.parameters()),
                                      lr=args.lr_for_cal, weight_decay=args.l2_for_cal)
            acc_train, acc_val, acc_test, ece_train, ece_val, ece_test = \
                mian_cali(ew, optimizer_ew, data, idx_train, idx_val, idx_test)

            with torch.no_grad():
                ew.eval()
                output = ew(data.x, data.edge_index)
                pred = F.softmax(output, dim=1)
                edge_weight = ew.get_weight(data.x, data.edge_index)

            pred = torch.exp(args.beta * pred)
            pred /= torch.sum(pred, dim=1, keepdim=True)

            col, row = data.edge_index
            coefficient = torch.norm(pred[col] - pred[row], dim=1)
            coefficient = 1 / (coefficient + args.alpha)

            edge_weight = edge_weight.reshape(-1)
            edge_weight = edge_weight * coefficient
            edge_weight = edge_weight.reshape([data.num_edges, 1])

            ts = Temperature_Scalling(base_model).to(device)
            optimizer_ts = optim.Adam(filter(lambda p: p.requires_grad, ts.parameters()),
                                      lr=args.lr_for_cal, weight_decay=args.l2_for_cal)

            acc_train, acc_val, acc_test, ece_train, ece_val, ece_test = \
                mian_cali(ts, optimizer_ts, data, idx_val, idx_val, idx_test)
            new_acc_test0.append(acc_test * 100), new_ece_test0.append(ece_test * 100)

            acc_train, acc_val, acc_test, ece_train, ece_val, ece_test = \
                mian_cali(ts, optimizer_ts, data, idx_val, idx_val, idx_test, edge_weight)
            new_acc_test1.append(acc_test * 100), new_ece_test1.append(ece_test * 100)

            cagcn = CaGCN(base_model, nclass, args.hidden).to(device)
            optimizer_cagcn = optim.Adam(filter(lambda p: p.requires_grad, cagcn.parameters()),
                                         lr=args.lr_for_cal, weight_decay=args.l2_for_cal)
            acc_train, acc_val, acc_test, ece_train, ece_val, ece_test = \
                mian_cali(cagcn, optimizer_cagcn, data, idx_val, idx_val, idx_test)
            new_acc_test2.append(acc_test * 100), new_ece_test2.append(ece_test * 100)

            acc_train, acc_val, acc_test, ece_train, ece_val, ece_test = \
                mian_cali(cagcn, optimizer_cagcn, data, idx_val, idx_val, idx_test, edge_weight)
            new_acc_test3.append(acc_test * 100), new_ece_test3.append(ece_test * 100)

            train_mask = index_to_mask(torch.LongTensor(idx_train), data.num_nodes).cuda()
            gats = GATS(base_model, data.edge_index, data.num_nodes, train_mask, dataset.num_classes)
            optimizer_gats = optim.Adam(filter(lambda p: p.requires_grad, gats.parameters()),
                                        lr=args.lr_for_cal, weight_decay=args.l2_for_cal)
            acc_train, acc_val, acc_test, ece_train, ece_val, ece_test = \
                mian_cali(gats, optimizer_gats, data, idx_val, idx_val, idx_test)
            new_acc_test4.append(acc_test * 100), new_ece_test4.append(ece_test * 100)

            acc_train, acc_val, acc_test, ece_train, ece_val, ece_test = \
                mian_cali(gats, optimizer_gats, data, idx_val, idx_val, idx_test, edge_weight)
            new_acc_test5.append(acc_test * 100), new_ece_test5.append(ece_test * 100)

            vs = VS(base_model, nclass).to(device)
            optimizer_vs = optim.Adam(filter(lambda p: p.requires_grad, vs.parameters()),
                                      lr=args.lr_for_cal, weight_decay=args.l2_for_cal)
            acc_train, acc_val, acc_test, ece_train, ece_val, ece_test = \
                mian_cali(vs, optimizer_vs, data, idx_val, idx_val, idx_test)
            new_acc_test6.append(acc_test * 100), new_ece_test6.append(ece_test * 100)

            acc_train, acc_val, acc_test, ece_train, ece_val, ece_test = \
                mian_cali(vs, optimizer_vs, data, idx_val, idx_val, idx_test, edge_weight)
            new_acc_test7.append(acc_test * 100), new_ece_test7.append(ece_test * 100)

    print(f'old_acc_test: {statistics.mean(old_acc_test):2f}±{statistics.stdev(old_acc_test):2f}',
          f'old_ece_test: {statistics.mean(old_ece_test):2f}±{statistics.stdev(old_ece_test):2f}',
          )

    print(f'new_acc_test0: {statistics.mean(new_acc_test0):2f}±{statistics.stdev(new_acc_test0):2f}',
          f'new_ece_test0: {statistics.mean(new_ece_test0):2f}±{statistics.stdev(new_ece_test0):2f}',
          )

    print(f'new_acc_test1: {statistics.mean(new_acc_test1):2f}±{statistics.stdev(new_acc_test1):2f}',
          f'new_ece_test1: {statistics.mean(new_ece_test1):2f}±{statistics.stdev(new_ece_test1):2f}',
          )

    print(f'new_acc_test2: {statistics.mean(new_acc_test2):2f}±{statistics.stdev(new_acc_test2):2f}',
          f'new_ece_test2: {statistics.mean(new_ece_test2):2f}±{statistics.stdev(new_ece_test2):2f}',
          )

    print(f'new_acc_test3: {statistics.mean(new_acc_test3):2f}±{statistics.stdev(new_acc_test3):2f}',
          f'new_ece_test3: {statistics.mean(new_ece_test3):2f}±{statistics.stdev(new_ece_test3):2f}',
          )

    print(f'new_acc_test4: {statistics.mean(new_acc_test4):2f}±{statistics.stdev(new_acc_test4):2f}',
          f'new_ece_test4: {statistics.mean(new_ece_test4):2f}±{statistics.stdev(new_ece_test4):2f}',
          )

    print(f'new_acc_test5: {statistics.mean(new_acc_test5):2f}±{statistics.stdev(new_acc_test5):2f}',
          f'new_ece_test5: {statistics.mean(new_ece_test5):2f}±{statistics.stdev(new_ece_test5):2f}',
          )

    print(f'new_acc_test6: {statistics.mean(new_acc_test6):2f}±{statistics.stdev(new_acc_test6):2f}',
          f'new_ece_test6: {statistics.mean(new_ece_test6):2f}±{statistics.stdev(new_ece_test6):2f}',
          )

    print(f'new_acc_test7: {statistics.mean(new_acc_test7):2f}±{statistics.stdev(new_acc_test7):2f}',
          f'new_ece_test7: {statistics.mean(new_ece_test7):2f}±{statistics.stdev(new_ece_test7):2f}',
          )



if __name__ == '__main__':
    main()
