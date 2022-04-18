import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
import torch.nn.functional as F

import data
import models
import os
import visdom


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")
    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
                                                                           torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class CB_loss(nn.Module):
    def __init__(self, beta, gamma, epsilon=0.1):
        super(CB_loss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, logits, labels, loss_type='softmax'):
        """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
          labels: A int tensor of size [batch].
          logits: A float tensor of size [batch, no_of_classes].
          samples_per_cls: A python list of size [no_of_classes].
          no_of_classes: total number of classes. int
          loss_type: string. One of "sigmoid", "focal", "softmax".
          beta: float. Hyperparameter for Class balanced loss.
          gamma: float. Hyperparameter for Focal loss.
        Returns:
          cb_loss: A float tensor representing class balanced loss
        """
        # self.epsilon = 0.1 #labelsmooth
        beta = self.beta
        gamma = self.gamma

        no_of_classes = logits.shape[1]
        samples_per_cls = torch.Tensor([sum(labels == i) for i in range(logits.shape[1])])
        samples_per_cls = samples_per_cls.to(device)

        effective_num = 1.0 - torch.pow(beta, samples_per_cls)
        weights = (1.0 - beta) / ((effective_num) + 1e-8)
        # print(weights)
        weights = weights / torch.sum(weights) * no_of_classes
        labels = labels.reshape(-1, 1)

        weights = torch.tensor(weights).float()
        weights = weights.to(device)
        labels_one_hot = torch.zeros(len(labels), no_of_classes).to(device).scatter_(1, labels, 1).to(device)

        labels_one_hot = (1 - self.epsilon) * labels_one_hot + self.epsilon / no_of_classes
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, no_of_classes)

        if loss_type == "focal":
            cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
        elif loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, pos_weight=weights)
        elif loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss


## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=20):
    def train(model, train_loader, optimizer, criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def valid(model, valid_loader, criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()

    vis = visdom.Visdom(port=8099)
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = valid(model, valid_loader, criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))
        vis.line(
            X=[epoch],
            Y=[train_loss],
            win='train_loss_modelC',
            name='train_loss_Resample',
            opts=dict(title='train_loss_modelC', showlegend=True),
            update='append')
        vis.line(
            X=[epoch],
            Y=[valid_loss],
            win='valid_loss_modelC',
            name='valid_loss_Resample',
            opts=dict(title='valid_loss_modelC', showlegend=True),
            update='append')
        vis.line(
            X=[epoch],
            Y=[train_acc],
            win='train_acc_modelC',
            name='train_acc_Resample',
            opts=dict(title='train_acc_modelC', showlegend=True),
            update='append')
        vis.line(
            X=[epoch],
            Y=[valid_acc],
            win='valid_acc_modelC',
            name='valid_acc_Resample',
            opts=dict(title='valid_acc_modelC', showlegend=True),
            update='append')
        s.step()
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, 'best_model_c.pt')
        print("best acc:", best_acc)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    ## about model
    num_classes = 10

    ## about data
    data_dir = "~/hw2/hw2_dataset/"  ## You need to specify the data_dir first
    input_size = 224
    batch_size = 36

    ## about training
    num_epochs = 150
    lr = 0.001
    steps = [40, 80, 120]

    ## model initialization
    model = models.model_C(num_classes=num_classes)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = data.load_data_c(data_dir=data_dir,
                                                  train_dir='4-Long-Tailed',
                                                  input_size=input_size,
                                                  batch_size=batch_size)

    ## optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=4e-5, amsgrad=False)
    s = optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1)
    ## loss function
    # criterion = nn.CrossEntropyLoss()
    criterion = CB_loss(0.9999, 2.0)
    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=num_epochs)
