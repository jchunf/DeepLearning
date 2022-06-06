# coding: utf-8
import argparse
import copy
import time
import math

import numpy as np
import torch
import torch.nn as nn
import visdom
import torch.optim as optim

import data
import model
import model_Transformer
import os
import os.path as osp
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='GPU device id used')

args = parser.parse_args()
vis_show = 0
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
if use_gpu:
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size, 'valid': eval_batch_size}
data_loader = data.Corpus("../data/ptb", batch_size, args.max_sql)
max_sql = args.max_sql
# WRITE CODE HERE within two '#' bar
########################################
# Build LMModel model (bulid your language model here)
# visdom
vis = visdom.Visdom(port=8099)
novc = len(data_loader.vocabulary)
transformer = model_Transformer.Transformer(novc, 200, 2, 200, 2).to(device)
print(transformer)
########################################

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(transformer.parameters(), 0.001)
steps = [20, 40, 60, 80]
s = optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1)


# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.


def evaluate():
    transformer.train(False)
    total_loss = 0.0
    data_loader.set_valid()
    data, target, isend = data_loader.get_batch()
    data, target = data.to(device), target.to(device)
    src_mask = transformer.generate_square_subsequent_mask(max_sql).to(device)
    flag = 0
    while not isend:
        if data.size(0) != max_sql:
            src_mask = transformer.generate_square_subsequent_mask(data.size(0)).to(device)
        output, hidden = transformer(data, src_mask)
        if flag == 0:
            data_first_batch = copy.copy(data)
            target_first_batch = copy.copy(target)
            output_first_batch = copy.copy(output)
            attn_batch = copy.copy(hidden)
            flag = 1
        output = output.view(output.size(0) * output.size(1), output.size(2))
        loss = criterion(output, target)
        total_loss += loss.item() * data.size(0)
        data, target, isend = data_loader.get_batch()
        data, target = data.to(device), target.to(device)

    epoch_loss = total_loss / data_loader.valid.shape[0]
    return epoch_loss, data_first_batch, target_first_batch, output_first_batch, attn_batch


########################################


# WRITE CODE HERE within two '#' bar
########################################
# Train Function
def train():
    transformer.train(True)
    total_loss = 0.0
    data_loader.set_train()
    data, target, isend = data_loader.get_batch()
    data, target = data.to(device), target.to(device)
    src_mask = transformer.generate_square_subsequent_mask(max_sql).to(device)
    while not isend:
        transformer.zero_grad()
        if data.size(0) != max_sql:
            src_mask = transformer.generate_square_subsequent_mask(data.size(0)).to(device)
        output, hidden = transformer(data, src_mask)
        output = output.view(output.size(0) * output.size(1), output.size(2))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        data, target, isend = data_loader.get_batch()
        data, target = data.to(device), target.to(device)
    epoch_loss = total_loss / data_loader.train.shape[0]
    return epoch_loss


########################################
best_data_first_batch, best_target_first_batch, best_output_first_batch, \
best_attn_batch = None, None, None, None
train_avg_loss = []
valid_avg_loss = []
best_loss = 9999
best_model = None
# Loop over epochs.
for epoch in range(1, args.epochs + 1):
    print('epoch:{:d}/{:d}'.format(epoch, args.epochs + 1))
    print('*' * 100)
    train_loss = train()
    print("training: {:.4f}".format(train_loss))
    valid_loss, data_first_batch, target_first_batch, output_first_batch, \
    attn_batch = evaluate()
    if best_loss > valid_loss:
        best_loss = valid_loss
        best_data_first_batch = copy.copy(data_first_batch)
        best_target_first_batch = copy.copy(target_first_batch)
        best_output_first_batch = copy.copy(output_first_batch)
        best_attn_batch = copy.copy(attn_batch)

    print("validation: {:.4f}".format(valid_loss))
    train_avg_loss.append(train_loss)
    valid_avg_loss.append(valid_loss)
    # Transformer_show_validloss
    # Transformer_show_tranloss
    vis.line(
        X=[epoch],
        Y=[train_loss],
        win='Transformer_show_1',
        name='train_loss',
        opts=dict(title='Loss_Transformer', showlegend=True),
        update='append')
    vis.line(
        X=[epoch],
        Y=[valid_loss],
        win='Transformer_show_1',
        name='valid_loss',
        opts=dict(title='Loss_Transformer', showlegend=True),
        update='append')
    # s.step()
    if epoch >= 10 and vis_show == 1:
        break
print('*' * 100)
perplexity_train = torch.exp(torch.Tensor(train_avg_loss))
perplexity_valid = torch.exp(torch.Tensor(valid_avg_loss))
print("PPL Train: ", perplexity_train)
print("PPL Valid: ", perplexity_valid)


# vis
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=6)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=6)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar



print(best_output_first_batch)
for batch_num in range(eval_batch_size):
    input = []
    input_id = best_data_first_batch[:, batch_num]
    for i in range(input_id.size(0)):
        input.append(data_loader.vocabulary[input_id[i].item()])
    output = []
    output_id = torch.argmax(best_output_first_batch, dim=2)[:, batch_num]
    for i in range(output_id.size(0)):
        output.append(data_loader.vocabulary[output_id[i].item()])
    attention = best_attn_batch[batch_num].cpu().detach().numpy()

    fig, ax = plt.subplots()
    im, cbar = heatmap(attention, output, input, ax=ax,
                       cmap="magma_r", cbarlabel="attention score")
    # texts = annotate_heatmap(im, valfmt="{x:.1f}")
    fig.tight_layout()
    plt.savefig('attention_visualization_{}.png'.format(batch_num), dpi=400)
