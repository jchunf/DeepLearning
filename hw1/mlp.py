import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils

# input and output dimensions
input_dim = 784
output_dim = 10


def relu(z1):
    h1 = np.maximum(0, z1)
    return h1


def softmax(z2, y):
    data_size = y.shape[0]
    z_max = np.max(z2)
    z_exp = np.exp(z2 - z_max)
    z_sum = (np.sum(z_exp, axis=1)).reshape(data_size, 1)
    y_hat = z_exp / z_sum
    loss = -np.sum(np.log(y_hat) * y) / data_size
    return y_hat, loss


def calc_loss_and_grad(x, y, w1, b1, w2, b2, eval_only=False):
    """Forward Propagation and Backward Propagation.

    Given a mini-batch of images x, associated labels y, and a set of parameters, compute the
    cross-entropy loss and gradients corresponding to these parameters.

    :param x: images of one mini-batch.
    :param y: labels of one mini-batch.
    :param w1: weight parameters of layer 1.
    :param b1: bias parameters of layer 1.
    :param w2: weight parameters of layer 2.
    :param b2: bias parameters of layer 2.
    :param eval_only: if True, only return the loss and predictions of the MLP.
    :return: a tuple of (loss, db2, dw2, db1, dw1)
    """

    # TODO
    # forward pass
    data_size=x.shape[0]
    z1 = np.dot(x, w1) + b1
    h1 = relu(z1)
    z2 = np.dot(h1, w2) + b2
    y_hat, loss = softmax(z2, y)
    # loss, y_hat = None, None
    if eval_only:
        return loss, y_hat

    # TODO
    # backward pass
    # db2, dw2, db1, dw1 = None, None, None, None
    dy = y_hat - y
    dw2 = np.dot(np.transpose(h1), dy) / data_size
    db2 = np.sum(dy, axis=0) / data_size
    z_sgn = np.maximum(0, np.sign(z1))
    w2t = np.transpose(w2)
    dw1 = np.dot(np.transpose(x),np.dot(dy, w2t) * z_sgn) / data_size
    db1 = np.sum(np.dot(dy, w2t) * z_sgn, axis=0) / data_size
    return loss, db2, dw2, db1, dw1


def train(train_x, train_y, test_x, test_y, args: argparse.Namespace):
    """Train the network.

    :param train_x: images of the training set.
    :param train_y: labels of the training set.
    :param test_x: images of the test set.
    :param test_y: labels of the test set.
    :param args: a dict of hyper-parameters.
    """

    # TODO
    #  randomly initialize the parameters (weights and biases)
    # w1, b1, w2, b2 = None, None, None, None
    global batch_size
    batch_size = args.batch_size
    w1 = np.random.randn(input_dim, args.hidden_dim) * np.sqrt(2 / input_dim)
    b1 = np.zeros(args.hidden_dim)
    w2 = np.random.randn(args.hidden_dim, output_dim) * np.sqrt(2 / args.hidden_dim)
    b2 = np.zeros(output_dim)

    print('Start training:')
    print_freq = 100
    loss_curve = []

    for epoch in range(args.epochs):
        # train for one epoch
        print("[Epoch #{}]".format(epoch))

        # random shuffle dataset
        dataset = np.hstack((train_x, train_y))
        np.random.shuffle(dataset)
        train_x = dataset[:, :input_dim]
        train_y = dataset[:, input_dim:]

        n_iterations = train_x.shape[0] // args.batch_size

        for i in range(n_iterations):
            # load a mini-batch
            x_batch = train_x[i * args.batch_size: (i + 1) * args.batch_size, :]
            y_batch = train_y[i * args.batch_size: (i + 1) * args.batch_size, :]

            # TODO
            # compute loss and gradients
            # calc_loss_and_grad()
            loss, db2, dw2, db1, dw1 = calc_loss_and_grad(x_batch, y_batch, w1, b1, w2, b2)

            # TODO
            # update parameters
            w1 = w1 - args.lr * dw1
            b1 = b1 - args.lr * db1
            w2 = w2 - args.lr * dw2
            b2 = b2 - args.lr * db2

            loss_curve.append(loss)
            if i % print_freq == 0:
                print('[Iteration #{}/{}] [Loss #{:4f}]'.format(i, n_iterations, loss))

    # show learning curve
    plt.title('Training Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(range(len(loss_curve)), loss_curve)
    plt.show()

    # evaluate on the training set
    loss, y_hat = calc_loss_and_grad(train_x, train_y, w1, b1, w2, b2, eval_only=True)
    predictions = np.argmax(y_hat, axis=1)
    labels = np.argmax(train_y, axis=1)
    accuracy = np.sum(predictions == labels) / train_x.shape[0]
    print('Top-1 accuracy on the training set', accuracy)

    # evaluate on the test set
    loss, y_hat = calc_loss_and_grad(test_x, test_y, w1, b1, w2, b2, eval_only=True)
    predictions = np.argmax(y_hat, axis=1)
    labels = np.argmax(test_y, axis=1)
    accuracy = np.sum(predictions == labels) / test_x.shape[0]
    print('Top-1 accuracy on the test set', accuracy)


def main(args: argparse.Namespace):
    # print hyper-parameters
    print('Hyper-parameters:')
    print(args)

    # load training set and test set
    train_x, train_y = utils.load_data("train")
    test_x, test_y = utils.load_data("test")
    print('Dataset information:')
    print("training set size: {}".format(len(train_x)))
    print("test set size: {}".format(len(test_x)))

    # check your implementation of backward propagation before starting training
    utils.check_grad(calc_loss_and_grad)

    # train the network and report the accuracy on the training and the test set
    train(train_x, train_y, test_x, test_y, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multilayer Perceptron')
    parser.add_argument('--hidden-dim', default=50, type=int,
                        help='hidden dimension of the Multilayer Perceptron')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--batch-size', default=10, type=int,
                        help='mini-batch size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    args = parser.parse_args()
    main(args)
