#Visualize the features before the lastfully-connectedlayer using t-SNE
import torch
import torch.nn as nn
import data
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    # # about model
    num_classes = 10

    # # about data
    data_dir = "~/hw2/hw2_dataset/"  # # You need to specify the data_dir first
    input_size = 224
    batch_size = 500

    # # model initialization
    model = torch.load('best_model.pt')
    device = "cpu"
    model = model.to(device)

    # # data preparation
    train_loader, valid_loader = data.load_data(data_dir=data_dir, test_dir="tsne", input_size=input_size, batch_size=batch_size)
    model.train(False)

    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        out = model(inputs)
        tsne = TSNE(n_components=2)
        y = tsne.fit_transform(out.detach().cpu().numpy())
        fig = plot_embedding(y, labels.numpy(), 't-SNE Visualization')
        break
    plt.savefig(os.getcwd()+'tsne_modelA.png')