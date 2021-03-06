import torch
import torch.nn as nn
import torch.optim as optim
import visdom

import data
import models
import os
import visdom


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
            win='train_loss_modelA',
            name='train_loss_SGD',
            opts=dict(title='train_loss_modelA', showlegend=True),
            update='append')
        vis.line(
            X=[epoch],
            Y=[valid_loss],
            win='valid_loss_modelA',
            name='valid_loss_SGD',
            opts=dict(title='valid_loss_modelA', showlegend=True),
            update='append')
        vis.line(
            X=[epoch],
            Y=[train_acc],
            win='train_acc_modelA',
            name='train_acc_SGD',
            opts=dict(title='train_acc_modelA', showlegend=True),
            update='append')
        vis.line(
            X=[epoch],
            Y=[valid_acc],
            win='valid_acc_modelA',
            name='valid_acc_SGD',
            opts=dict(title='valid_acc_modelA', showlegend=True),
            update='append')
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, 'best_model_A.pt')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

    ## about model
    num_classes = 10

    ## about data
    data_dir = "~/hw2/hw2_dataset/"  ## You need to specify the data_dir first
    input_size = 224
    batch_size = 36

    ## about training
    num_epochs = 100
    lr = 0.001

    ## model initialization
    model = models.model_A(num_classes=num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = data.load_data(data_dir=data_dir, input_size=input_size, batch_size=batch_size)

    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9)
    #Adam
    #RMSprop

    ## loss function
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=num_epochs)
