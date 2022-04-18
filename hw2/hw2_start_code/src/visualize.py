#Visualize the features before the lastfully-connectedlayer using t-SNE
import torch
import torch.nn as nn
import data
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
from visualize_fuctions import plot_embedding

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