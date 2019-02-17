import pickle
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GraphNeuralNetwork(nn.Module):
    def __init__(self):
        super(GraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(hidden_layer)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(output_layer)])
        self.W_property = nn.Linear(dim, 1)

    def pad(self, matrices, pad_value):
        """Pad adjacency matrices for batch processing."""
        sizes = [m.shape[0] for m in matrices]
        M = sum(sizes)
        pad_matrices = pad_value + np.zeros((M, M))
        i = 0
        for j, m in enumerate(matrices):
            j = sizes[j]
            pad_matrices[i:i+j, i:i+j] = m
            i += j
        return torch.FloatTensor(pad_matrices).to(device)

    def sum_axis(self, xs, axis):
        y = list(map(lambda x: torch.sum(x, 0), torch.split(xs, axis)))
        return torch.stack(y)

    def mean_axis(self, xs, axis):
        y = list(map(lambda x: torch.mean(x, 0), torch.split(xs, axis)))
        return torch.stack(y)

    def update(self, xs, A, M, i):
        """Update the node vectors in a graph
        considering their neighboring node vectors (i.e., sum or mean),
        which are non-linear transformed by neural network."""
        hs = torch.relu(self.W_fingerprint[i](xs))
        if update == 'sum':
            return xs + torch.matmul(A, hs)
        if update == 'mean':
            return xs + torch.matmul(A, hs) / (M-1)

    def forward(self, inputs):

        Smiles, fingerprints, adjacencies = inputs
        axis = list(map(lambda x: len(x), fingerprints))

        M = np.concatenate([np.repeat(len(f), len(f)) for f in fingerprints])
        M = torch.unsqueeze(torch.FloatTensor(M), 1)

        fingerprints = torch.cat(fingerprints)
        fingerprint_vectors = self.embed_fingerprint(fingerprints)

        adjacencies = self.pad(adjacencies, 0)

        """GNN updates the fingerprint vectors."""
        for i in range(hidden_layer):
            fingerprint_vectors = self.update(fingerprint_vectors,
                                              adjacencies, M, i)

        if output == 'sum':
            molecular_vectors = self.sum_axis(fingerprint_vectors, axis)
        if output == 'mean':
            molecular_vectors = self.mean_axis(fingerprint_vectors, axis)

        for j in range(output_layer):
            molecular_vectors = torch.relu(self.W_output[j](molecular_vectors))

        molecular_properties = self.W_property(molecular_vectors)

        return Smiles, molecular_properties

    def __call__(self, data_batch, train=True):

        inputs = data_batch[:-1]
        correct_properties = torch.cat(data_batch[-1])
        Smiles, predicted_properties = self.forward(inputs)

        if train:
            loss = F.mse_loss(correct_properties, predicted_properties)
            return loss
        else:
            ts = std * correct_properties.to('cpu').data.numpy() + mean
            ys = std * predicted_properties.to('cpu').data.numpy() + mean
            ts, ys = np.concatenate(ts), np.concatenate(ys)
            return Smiles, ts, ys


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch):
            data_batch = list(zip(*dataset[i:i+batch]))
            loss = self.model(data_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        SMILES, Ys, Ts, SE_sum = '', [], [], 0
        for i in range(0, N, batch):
            data_batch = list(zip(*dataset[i:i+batch]))
            Smiles, ts, ys = self.model(data_batch, train=False)
            SMILES += ' '.join(Smiles) + ' '
            Ts.append(ts)
            Ys.append(ys)
            SE_sum += sum((ts-ys)**2)
        SMILES = SMILES.strip().split()
        T, Y = map(str, np.concatenate(Ts)), map(str, np.concatenate(Ys))
        MSE = SE_sum / N
        return MSE, zip(SMILES, T, Y)

    def save_MSEs(self, MSEs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, MSEs)) + '\n')

    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Smiles\tCorrect\tPredict\n')
            f.write('\n'.join(['\t'.join(p) for p in predictions]) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def load_tensor(filename, dtype):
    return [dtype(d).to(device) for d in np.load(filename + '.npy')]


def load_numpy(filename):
    return np.load(filename + '.npy')


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET, radius, update, output, dim, hidden_layer, output_layer, batch,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = sys.argv[1:]
    (dim, hidden_layer, output_layer, batch, decay_interval,
     iteration) = map(int, [dim, hidden_layer, output_layer, batch,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load each preprocessed data."""
    dir_input = ('../../dataset/regression/' + DATASET +
                 '/input/radius' + radius + '/')
    with open(dir_input + 'Smiles.txt') as f:
        Smiles = f.read().strip().split()
    molecules = load_tensor(dir_input + 'molecules', torch.LongTensor)
    adjacencies = load_numpy(dir_input + 'adjacencies')
    correct_properties = load_tensor(dir_input + 'properties',
                                     torch.FloatTensor)
    mean = load_numpy(dir_input + 'mean')
    std = load_numpy(dir_input + 'std')
    with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
        fingerprint_dict = pickle.load(f)
    n_fingerprint = len(fingerprint_dict)

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(Smiles, molecules, adjacencies, correct_properties))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    """Set a model."""
    torch.manual_seed(1234)
    model = GraphNeuralNetwork().to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    """Output files."""
    file_MSEs = '../../output/result/MSEs--' + setting + '.txt'
    file_predictions = '../../output/result/predictions--' + setting + '.txt'
    file_model = '../../output/model/' + setting
    MSEs = 'Epoch\tTime(sec)\tLoss_train\tMSE_dev\tMSE_test'
    with open(file_MSEs, 'w') as f:
        f.write(MSEs + '\n')
    print(MSEs)

    """Start training."""
    start = timeit.default_timer()
    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)
        MSE_dev = tester.test(dataset_dev)[0]
        MSE_test, predictions_test = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        MSEs = [epoch, time, loss_train, MSE_dev, MSE_test]
        tester.save_MSEs(MSEs, file_MSEs)
        tester.save_predictions(predictions_test, file_predictions)
        tester.save_model(model, file_model)

        print('\t'.join(map(str, MSEs)))
