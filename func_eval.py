import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim
import math

import torchvision
import os
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter

from util import AvgrageMeter, accuracy


class Net(nn.Module):
    # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    def __init__(self, n_input_channel: int = 1, n_targets: int = 10, kernel_size: int = 5):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(n_input_channel, 6, kernel_size)
        self.conv2 = nn.Conv2d(6, 16, kernel_size)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_targets)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet_Evaluator():
    def __init__(self,
                 dataset_path: str,
                 device: torch.device = torch.device('cpu')):
        """
        A wrapper for optimizing the hyperpameters of a neural network on the kmnist dataset
        Parameters:
            dataset_path: str
                the path where the dataset is stored
            device: torch.device
                where the model needs to be trained.
        """
        os.makedirs(dataset_path, exist_ok=True)
        transform = [torchvision.transforms.ToTensor(),
                     torchvision.transforms.Normalize((0.1918,), (0.3483,))]

        train_val_set = torchvision.datasets.KMNIST(dataset_path, train=True, download=True,
                                                    transform=torchvision.transforms.Compose(transform), )
        self.test_set = torchvision.datasets.KMNIST(dataset_path, train=False, download=True,
                                                    transform=torchvision.transforms.Compose(transform))
        self.train_set, self.val_set = torch.utils.data.random_split(train_val_set, [50000, 10000])
        self.device = device

    def train_and_eval(self,
                       batch_size: int,
                       learning_rate: float,
                       train_epoch: int = 10,
                       do_test_inference: bool = False):
        """
        train a lenet with the given hyperparameters and return the validation accuracy
        Parameters:
            batch_size: int
                batch size to train a neural network
            learning_rate: float
                learning rate of the optimizer. In this task, the optimizer is a SGD model
            train_epoch: int
                number of epochs that the network is trained
            do_test_inference: bool
                whether we want to perform a further test inference, this will give you an additional information about
                the generalization ability of your hyperparameters
        Return:
            validation_accuracy: float
                accuracy of the trained network on the validation dataset (score of the hyperparameter configuration).
        """
        # Define the data loaders
        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_set, batch_size=1000, shuffle=False)

        # Create the model, set the criterion and define the optimizer
        model = Net().to(device=self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        # train tne network for a certain number of epochs
        for epoch in range(train_epoch):
            num_iter = 0
            loss_epoch = 0.
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                num_iter += 1
                loss_epoch += loss.detach()
            print(f'training loss at epoch {epoch}: {loss_epoch / num_iter : .4f}')

        # again no gradients needed
        score_val = AvgrageMeter()
        val_loss = 0
        num_iter = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss
                num_iter += 1

                acc = accuracy(outputs, targets, topk=(1,))[0]
                score_val.update(acc.item(), inputs.size(0))

        if do_test_inference:
            score_test = AvgrageMeter()
            test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = model(inputs)
                    acc = accuracy(outputs, targets, topk=(1,))[0]
                    score_test.update(acc.item(), inputs.size(0))
            print(f'test accuracy: {score_val.avg:.4f}')
        print(f'final validation accuracy: {score_val.avg:.4f}')
        # we want ot do minimization
        return -score_val.avg

    def eval_config(self, cfg: Configuration, budget: float = 1.0, maximal_epoch: int = 10):
        """
        evaluate a hyperpameter configuration and return the validation accuracy for that hyperparameter configuration
        Parameters:
            cfg: Configuration
                the configuration to be evaluated. In this task, it contains two hyperparmeters: batch_size and
                learning_rate
            budget: float
                the amount of budgets assigned to the configuration. In this task, it controls the number of epochs that
                a neural network is trained. The higher it is, the longer the network will be trained.
            maximal_epoch: int
                maximal number of epochs that a neural network is trained. it represents the highest budgets.
        Return:
            validation_accuracy: float
                accuracy of the trained network on the validation dataset.
        """
        if budget == 0.:
            train_epoch = maximal_epoch
        else:
            train_epoch = math.ceil(budget * maximal_epoch)
        batch_size = cfg['batch_size']
        learning_rate = cfg['learning_rate']
        return self.train_and_eval(batch_size=batch_size, learning_rate=learning_rate, train_epoch=train_epoch)


if __name__ == '__main__':
    dataset_path = os.path.join('tmp', 'dataset')
    os.makedirs(dataset_path, exist_ok=True)

    transform = [torchvision.transforms.ToTensor(),
                 torchvision.transforms.Normalize((0.1918,), (0.3483,))]

    train_val_set = torchvision.datasets.KMNIST(dataset_path, train=True, download=True,
                                                transform=torchvision.transforms.Compose(transform), )
    test_set = torchvision.datasets.KMNIST(dataset_path, train=False, download=True,
                                           transform=torchvision.transforms.Compose(transform))

    train_set, val_set = torch.utils.data.random_split(train_val_set, [50000, 10000])

    evaluator = LeNet_Evaluator(dataset_path, )

    cs = ConfigurationSpace(seed=1)

    batch_size = UniformIntegerHyperparameter('batch_size', lower=32, upper=1024, default_value=512, log=True)
    learning_rate = UniformFloatHyperparameter('learning_rate', lower=1e-4, upper=1., default_value=1e-2, log=True)
    cs.add_hyperparameters([batch_size, learning_rate])

    evaluator.eval_config(cs.get_default_configuration())
