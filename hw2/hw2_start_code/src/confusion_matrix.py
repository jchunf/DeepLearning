#plot the confusion matrix of your model C ontestset
import os
import torch
import data
import numpy as np
import matplotlib.pyplot as plt
from visualize_fuctions import confusion_matrix

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    # # about model
    num_classes = 10

    # # about data
    data_dir = "~/hw2/hw2_dataset/"  # # You need to specify the data_dir first
    input_size = 224
    batch_size = 36

    # # model initialization
    model = torch.load('~/hw2/hw2_start_code/src/best_model_c.pt')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # # data preparation
    _, valid_loader = data.load_data(data_dir=data_dir, input_size=input_size, batch_size=batch_size)
    con_mat = confusion_matrix(model, valid_loader, device, num_classes)

    plt.imshow(con_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    classes = range(0, 10)
    tick_marks = np.arange(len(classes))
    thresh = con_mat.max() / 2.
    for i in range(con_mat.shape[0]):
        for j in range(con_mat.shape[1]):
            num = con_mat[i, j]
            plt.text(j, i, int(num),
                     verticalalignment='center',
                     horizontalalignment="center",
                     color="white" if num > thresh else "black")

    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    save_path = os.getcwd() + '/result_png/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + 'confusion_matrix.png')