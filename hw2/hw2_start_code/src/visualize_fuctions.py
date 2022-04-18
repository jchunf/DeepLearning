import numpy as np
import os
import matplotlib.pyplot as plt
import torch

#for tsne
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

#for conv
def conv_viz(_, input):
    global count
    x = input[0][0].detach().cpu().numpy()
    # Display up to 4 pictures
    min_num = np.minimum(4, x.shape[0])
    plt.figure()
    for i in range(min_num):
        plt.subplot(1, 4, i + 1)
        plt.imshow(x[i])
    save_path = os.getcwd() + '/result_png/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + '{}.jpg'.format(count))
    count += 1

#matrix
def confusion_matrix(model, valid_loader, device, num_classes):
    mat = np.zeros((num_classes, num_classes))
    model.train(False)
    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        for label, prediction in zip(labels, predictions):
            mat[label.item()][prediction.item()] += 1
    return mat
