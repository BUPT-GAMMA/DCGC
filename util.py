import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d

def ECELoss(logits, labels, n_bins=15):
    """
    Calculate Expected Calibration Error (ECE).
    :param logits: Output logits from the model
    :param labels: True labels
    :param n_bins: Number of bins for ECE
    :return: float value of ECE
    """
    confidences = F.softmax(logits, dim=1).max(dim=1)[0]
    predictions = torch.argmax(logits, dim=1)
    errors = predictions.eq(labels)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = errors[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()


def Brier(logits, label, idx_test):
    nodeprobs = torch.softmax(logits[idx_test], -1)
    nodeconfs = torch.gather(nodeprobs, -1, label[idx_test].unsqueeze(-1)).squeeze(-1)
    return (nodeprobs.square().sum(dim=-1) - 2.0 * nodeconfs).mean().add(1.0).item()

def NLLLoss(logits, label, idx_test):
    return F.cross_entropy(logits[idx_test], label[idx_test]).item()

def plot_acc_calibration(idx_test, output, labels, n_bins, title):
    output = torch.softmax(output, dim=1)
    pred_label = torch.max(output[idx_test], 1)[1]
    p_value = torch.max(output[idx_test], 1)[0]
    ground_truth = labels[idx_test]
    confidence_all, confidence_acc = np.zeros(n_bins), np.zeros(n_bins)
    for index, value in enumerate(p_value):
        #value -= suboptimal_prob[index]
        interval = int(value / (1 / n_bins) -0.0001)
        confidence_all[interval] += 1
        if pred_label[index] == ground_truth[index]:
            confidence_acc[interval] += 1
    for index, value in enumerate(confidence_acc):
        if confidence_all[index] == 0:
            confidence_acc[index] = 0
        else:
            confidence_acc[index] /= confidence_all[index]

    start = np.around(1/n_bins/2, 3)
    step = np.around(1/n_bins, 3)
    plt.figure(figsize=(6, 4))
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams["font.weight"] = "bold"
    plt.bar(np.around(np.arange(start, 1.0, step), 3), confidence_acc,
            alpha=0.7, width=0.03, color='dodgerblue', label='Outputs')
    plt.bar(np.around(np.arange(start, 1.0, step), 3),
            np.around(np.arange(start, 1.0, step), 3), alpha=0.7, width=0.03, color='lightcoral', label='Expected')
    plt.plot([0,1], [0,1], ls='--',c='k')
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.tick_params(labelsize=13)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    #title = 'Uncal. - Cora - 20 - GCN'
    plt.title(title, fontsize=16, fontweight="bold")
    plt.legend(fontsize=14)
    plt.show()

def shortest_path_length(edge_index, mask, max_hop):
    """
    Return the shortest path length to the mask for every node
    """
    dist_to_train = torch.ones_like(mask, dtype=torch.long, device=mask.device) * torch.iinfo(torch.long).max
    seen_mask = torch.clone(mask)
    for hop in range(max_hop):
        current_hop = torch.nonzero(mask)
        dist_to_train[mask] = hop
        next_hop = torch.zeros_like(mask, dtype=torch.bool, device=mask.device)
        for node in current_hop:
            node_mask = edge_index[0,:]==node
            nbrs = edge_index[1,node_mask]
            next_hop[nbrs] = True
        hop += 1
        # mask for the next hop shouldn't be seen before
        mask = torch.logical_and(next_hop, ~seen_mask)
        seen_mask[next_hop] = True
    return dist_to_train

def plot_3d():

    x = np.array([0.1, 0.3, 0.5, 0.7, 1])
    y = np.array([0.1, 0.5, 1, 3, 5, 7, 10])
    #Cora GCN
    z1 = np.array([[7.63743818, 7.28611276, 8.2213223 , 8.8864468 , 9.15831253, 8.18180367, 8.16115737],
       [6.43319637, 6.6688247 , 7.42787421, 7.00131878, 7.06016421, 6.66477382, 6.34589121],
       [6.38016239, 6.51505142, 6.63078129, 6.4131245 , 6.03959523, 5.72511293, 5.49931079],
       [5.91055974, 5.99644221, 5.92552722, 5.71918227, 5.374901  , 5.23672029, 4.99945097],
       [5.50005399, 5.47220744, 5.45441732, 5.18073998, 4.89550047, 4.77641784, 4.65484224]])

    #Citeseer GCN
    z2 = np.array([[4.93256226, 5.05924635, 5.27248457, 5.33418544, 5.49542792, 4.99785282, 4.49730866],
       [4.94130217, 4.8436217 , 4.94880341, 4.72629964, 4.41172533, 4.21517268, 3.77004072],
       [4.51496206, 4.58772741, 4.55574952, 4.33735885, 4.32361636, 4.05759747, 3.64837322],
       [4.13862579, 4.03528549, 3.92724425, 4.30073291, 3.98161821, 3.64109017, 3.49432203],
       [4.09745052, 3.93030755, 3.89870852, 3.59439924, 3.36580537, 3.81042399, 3.62159684]])

    # Pubmed GCN
    z3 = np.array([[10.42651683, 10.82411408, 10.99247634, 10.55762619, 10.08930132, 9.62714106,  9.3300581 ],
       [ 9.23975557,  9.23907459,  9.24941376,  8.83052498,  8.49127844, 8.30334127,  8.16566274],
       [ 8.63174498,  8.54092985,  8.43881518,  7.95373023,  7.68635795, 7.55601451,  7.38092437],
       [ 8.018969  ,  7.90881962,  7.76867419,  7.40827397,  7.15937614,7.06456006,  6.9507055 ],
       [ 7.38311857,  7.24182278,  7.12938011,  6.79285079,  6.64495379, 6.53554127,  6.46993741]])
    # CoraFull GCN
    z4 = np.array([[7.87762105, 7.87166134, 7.77015015, 7.46664405, 7.71772489, 7.67832398, 7.18512684],
                   [7.82792717, 7.80108199, 7.79492781, 7.59890229, 7.57778659, 7.31910989, 6.21935539],
                   [7.71785155, 7.68566206, 7.66975209, 7.4341014, 7.46731386, 7.05278218, 5.96963651],
                   [7.66028315, 7.65022114, 7.65099451, 7.39086345, 7.27474988, 6.86990246, 5.91567457],
                   [7.33681992, 7.3231779, 7.30818883, 7.18737021, 7.19186589, 6.76196292, 5.90262413]])
    # Photo GCN
    z5 = np.array([[2.10086871, 1.93996467, 1.866851  , 2.21821573, 2.10575145, 1.60811059, 1.68811306],
       [1.60954036, 1.66347567, 1.77331828, 1.72216389, 1.80896707, 1.53884059, 1.35593479],
       [1.66823603, 1.80893987, 1.62451323, 1.61955208, 1.52678983, 1.56568941, 1.25532504],
       [1.82018764, 1.58319026, 1.77547839, 1.37607027, 1.39761167, 1.20448368, 1.20684896],
       [1.65746715, 1.80497263, 1.8624343 , 1.5150521 , 1.21828616, 1.33177917, 0.91358339]])

    #Computers GCN
    z6 = np.array([[2.48192791, 2.18344685, 2.08526552, 2.86503807, 2.73145661, 2.60810778, 2.31288541],
       [2.31170207, 2.10312381, 1.98783465, 2.04607379, 2.25169696, 2.11282037, 2.03706045],
       [2.06944365, 1.97788477, 1.85364466, 1.91836134, 1.99852213, 1.83817223, 1.82566941],
       [2.00291928, 1.88733898, 1.76759381, 1.85947269, 2.04563774, 1.92078874, 1.82880107],
       [1.92526933, 1.85848363, 1.75803937, 1.80778857, 1.87522061, 1.76999271, 1.66599248]])
    # Arxiv GCN
    z7 = np.array([[7.55860731, 7.38122314, 7.16272071, 7.07861409, 7.43525624, 7.62259588, 7.47840106],
                    [6.78838789, 6.68058097, 6.38858899, 6.71472251, 7.2234191, 6.96563125, 6.59488887],
                    [6.16916008, 5.98162487, 6.12109974, 6.21947497, 6.43383861, 6.37161359, 6.18388131],
                    [5.89585379, 6.00466616, 5.92428669, 5.93052693, 6.168212, 5.97485788, 5.9566278],
                    [5.82433641, 5.70771247, 5.74734956, 5.73785976, 6.22465014, 6.17933497, 5.7080701]])
    # Cora GraphSAGE
    z8 = np.array([[5.9351854, 6.20917492, 6.40772134, 7.0375219, 7.52259493, 7.17011988, 6.48094863],
                   [5.65611459, 5.92760816, 5.3143315, 5.99400699, 6.51351213, 5.45099899, 5.74411526],
                   [4.9341511, 4.65982594, 4.06736694, 5.64788282, 5.3883221, 4.88490835, 4.6265278],
                   [4.53448072, 4.33355346, 4.37068567, 5.32287955, 5.36082014, 4.552022, 4.50098962],
                   [5.72946966, 5.96969351, 5.38737401, 6.12347983, 6.1701633, 5.63616827, 5.13143018]])
    # Citeseer GraphSAGE
    z9 = np.array([[12.18913421, 13.25724274, 13.36328685, 11.43135279, 7.76865408, 7.12778568, 6.50658086],
                   [9.04734954, 9.15007144, 8.50909874, 7.40212947, 6.19395971, 6.49884641, 6.10591173],
                   [8.0573827, 7.59865493, 7.65646771, 7.07272291, 6.54165447, 6.96623549, 6.88653141],
                   [7.90593475, 7.39459023, 6.84691146, 6.78104311, 6.4847827, 6.49891719, 6.88661039],
                   [6.05188385, 6.080769, 6.50780722, 5.82795218, 5.97904921, 5.85470125, 5.66597134]])
    #Pubmed GraphSAGE
    z10 = np.array([[4.81014997, 5.30550517, 6.04729652, 8.05056617, 8.7815471 , 8.66091475, 8.3875373 ],
       [3.44067253, 3.54449004, 3.77797969, 4.79340181, 5.2096799 , 5.29785007, 5.24239168],
       [2.10368913, 2.27472894, 2.49865633, 3.04850489, 3.30553725, 3.25215943, 3.16365622],
       [1.64894294, 1.66081786, 1.96067411, 2.0805154 , 2.06079315, 1.92877911, 1.76361408],
       [3.8714081 , 4.09623124, 4.37646434, 4.81480025, 4.51578423, 3.85929719, 3.9603278 ]])
    #Photo GraphSAGE
    z11 = np.array([[2.4536971 , 2.7019104 , 3.91190797, 7.00243115, 7.75698423, 6.78535998, 5.74049577],
       [2.14968175, 2.03613415, 2.51308866, 4.10982631, 4.52402495, 4.09892984, 3.41465138],
       [1.99221056, 1.88727155, 2.36512832, 3.02554704, 2.9789757 , 2.54868902, 2.01845113],
       [2.12677233, 2.01512352, 2.10626461, 1.99831482, 1.78410988, 1.59574673, 1.38373664],
       [1.6835168 , 1.66468564, 1.46794524, 0.97909151, 1.20417038, 1.44232363, 1.27880694]])
    #Computers GraphSAGE
    z12 = np.array([[ 5.74695207,  6.91046417,  7.78182054, 10.65588605, 11.3958807 , 9.99540997, 8.77738532],
       [ 6.32434413,  5.87095097,  5.91134802,  4.93559688,  5.07320575,5.57187721,  5.76099455],
       [ 2.95409691,  2.89352108,  2.83027627,  2.8458314 ,  2.45933589,2.45687086,  2.46047154],
       [ 2.28879806,  2.21746378,  2.12350748,  1.99286249,  1.70073994,1.7943617 ,  2.15862095],
       [ 1.75705012,  1.78028941,  1.83117036,  1.71662625,  1.76223908,1.99721754,  2.22761855]])
    #CoraFull GraphSAGE
    z13 = np.array([[ 5.04273623,  5.01994118,  5.12741394,  5.71895912,  9.66781899,13.38862479, 13.42803658],
       [ 6.25222325,  6.53840527,  6.43625557,  5.72136566,  6.99439347,8.63109902, 11.68893874],
       [ 5.94791435,  5.87349646,  5.94456457,  5.73746897,  5.70256524,6.85163662,  8.50551128],
       [ 4.69369181,  4.53197174,  4.62193005,  4.39073779,  4.53212708,5.26397899,  6.55674636],
       [ 4.876329  ,  4.92806137,  5.02763316,  6.04126155,  8.48009139,10.34570858, 10.43115705]])
    #Arxiv GraphSAGE
    z14 = np.array([[ 4.22779322,  4.03049886,  5.80853745,  4.70802411,  8.14010203,
         9.02038068, 14.63917196],
       [ 5.63906729,  5.76621294,  6.09147102,  6.13349937,  5.71469404,
         8.0010578 , 10.57532355],
       [ 4.65226844,  4.80792485,  4.72006127,  4.78818752,  5.05726375,
         6.64834231,  9.30966958],
       [ 5.22023365,  5.17916642,  5.20626865,  4.86373752,  4.64196242,
         6.11989722,  8.20395648],
       [ 5.74063212,  5.81261702,  5.92629351,  7.22015798,  8.65774655,
        9.1269491 ,  8.7332336 ]])
    z15 = np.array([[4.79040779, 4.87015173, 4.83423322, 4.81965691, 4.80517223,
        4.81852144, 4.8151508 ],
       [4.7961127 , 4.81815934, 4.83430438, 4.82313707, 4.87133637,
        4.83110771, 4.79058921],
       [4.79192398, 4.87275608, 4.74980921, 4.79348078, 4.77424376,
        4.80015688, 4.81613837],
       [4.86161597, 4.82506268, 4.79180217, 4.77983654, 4.77414131,
        4.82942834, 4.83913869],
       [4.76419553, 4.76315767, 4.84796166, 4.84513827, 4.84023131,
        4.81920838, 4.81149293]])
    xi, yi = np.meshgrid(np.arange(0.1, 1, 0.1), np.arange(0.1, 10, 0.3))
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    f = interp2d(x, y, np.transpose(z15), kind='cubic')
    zi = f(xi[0], yi[:, 0])
    zi_min = np.min(zi)
    zi_max = np.max(zi)
    face_color = plt.cm.viridis((np.array(zi) - zi_min) / (zi_max - zi_min))
    ax.plot_surface(xi, yi, zi, cmap='viridis', rstride=1, cstride=1, linewidth=0.5, facecolors=face_color,
                        alpha=0)
    ax.w_xaxis.set_pane_color('white')
    ax.w_yaxis.set_pane_color('white')
    ax.w_zaxis.set_pane_color('white')
    ax.set_xlabel('alpha', fontsize=12)
    ax.set_ylabel('1 / beta', fontsize=12)
    ax.set_zlabel('ECE(%)', fontsize=12)
    ax.view_init(elev=30, azim=-240)
    ax.set_box_aspect([2, 2, 1])
    ax.set_zlim(zmin=zi_max-3, zmax=zi_max+3)

    plt.show()

def plot_ece():
    categories = ['GCN', 'GAT', 'GraphSAGE', 'SGC', 'TAGCN']
    data1 = [15.62, 18.02, 10.25, 12.82, 11.37]
    data2 = [7.49, 4.59, 8.27, 5.00, 4.38]

    bar_width = 2

    x = np.array([0, 8, 16, 24, 32])

    plt.bar(x, data1, width=bar_width, label='Cora')
    plt.bar(x + bar_width, data2, width=bar_width, label='Photo')

    plt.xlabel('GNN')
    plt.ylabel('ECE')

    plt.xticks(x + bar_width / 2, categories)
    plt.legend()

    plt.show()

def plot_bar():
    categories = ['GCN', 'CaGCN', 'GATS', 'Ours']
    data1 = [416.44, 197.15, 139.4, 66.87]
    data2 = [139.54, 23.52, 58.73, 20.05]
    bar_width = 0.6
    x = np.array([1, 2, 3, 4])
    y1 = np.array(range(0, 420, 50))
    y2 = np.array(range(0, 200, 40))

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(x, data1, width=bar_width, color='blueviolet')
    ax1.set_xticks(x, categories, fontsize=20)
    ax1.set_xlabel('(a) Time resuming on Arxiv', fontsize=20)
    ax1.set_ylabel('Time', fontsize=20)
    ax1.set_yticks(y1, y1, fontsize=15)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(x, data2, width=bar_width, color='lightseagreen')
    ax2.set_xticks(x, categories, fontsize=20)
    ax2.set_xlabel('(b) Time resuming on CoraFull', fontsize=20)
    ax2.set_ylabel('Time', fontsize=20)
    ax2.set_yticks(y2, y2, fontsize=15)


    plt.show()

if __name__ == '__main__':
    #plot_bar()
    #plot_ece()
    plot_3d()