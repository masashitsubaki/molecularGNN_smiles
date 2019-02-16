import pickle
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score, precision_score, recall_score


class GraphNeuralNetwork(nn.Module):
    def __init__(self):
        super(GraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(hidden_layer)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(output_layer)])
        self.W_property = nn.Linear(dim, 2)

    def pad(self, matrices, pad_value):
        """Pad adjacency matrices for batch processing."""
        sizes = [d.shape[0] for d in matrices]
        D = sum(sizes)
        pad_matrices = pad_value + np.zeros((D, D))
        m = 0
        for i, d in enumerate(matrices):
            s_i = sizes[i]
            pad_matrices[m:m+s_i, m:m+s_i] = d
            m += s_i
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
        which are non-linear transformed by a neural network."""
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

        """GNN updates fingerprint vectors."""
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
            loss = F.cross_entropy(predicted_properties, correct_properties)
            return loss
        else:
            ts = correct_properties.to('cpu').data.numpy()
            ys = F.softmax(predicted_properties, 1).to('cpu').data.numpy()
            correct_labels = ts
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores


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
        Correct_labels, Predicted_labels, Predicted_scores = [], [], []

        for i in range(0, N, batch):
            data_batch = list(zip(*dataset[i:i+batch]))

            (correct_labels, predicted_labels,
             predicted_scores) = self.model(data_batch, train=False)

            Correct_labels.append(correct_labels)
            Predicted_labels.append(predicted_labels)
            Predicted_scores.append(predicted_scores)

        correct_labels = np.concatenate(Correct_labels)
        predicted_labels = np.concatenate(Predicted_labels)
        predicted_scores = np.concatenate(Predicted_scores)

        AUC = roc_auc_score(correct_labels, predicted_scores)
        precision = precision_score(correct_labels, predicted_labels)
        recall = recall_score(correct_labels, predicted_labels)

        return AUC, precision, recall

    def result_AUC(self, epoch, time, loss_train, AUC_dev,
                   AUC_test, precision_test, recall_test, file_result):
        with open(file_result, 'a') as f:
            result = map(str, [epoch, time, loss_train, AUC_dev,
                               AUC_test, precision_test, recall_test])
            f.write('\t'.join(result) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def load_tensor(filename, dtype):
    return [dtype(d).to(device) for d in np.load(filename + '.npy')]


def load_numpy(filename):
    return np.load(filename + '.npy')


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


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

    """Load preprocessed data."""
    dir_input = ('../../dataset/classification/' + DATASET +
                 '/input/radius' + radius + '/')
    with open(dir_input + 'Smiles.txt') as f:
        Smiles = f.read().strip().split()
    Molecules = load_tensor(dir_input + 'Molecules', torch.LongTensor)
    adjacencies = load_numpy(dir_input + 'adjacencies')
    correct_properties = load_tensor(dir_input + 'properties',
                                     torch.LongTensor)
    with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
        fingerprint_dict = pickle.load(f)
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    n_fingerprint = len(fingerprint_dict)

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(Smiles, Molecules, adjacencies, correct_properties))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    """Set a model."""
    torch.manual_seed(1234)
    model = GraphNeuralNetwork().to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    """Output files."""
    file_AUC = '../../output/result/AUC--' + setting + '.txt'
    file_model = '../../output/model/' + setting
    result = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
              'AUC_test\tPrecision_test\tRecall_test\n')
    with open(file_AUC, 'w') as f:
        f.write(result + '\n')
    print(result)

    """Start training."""
    start = timeit.default_timer()
    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)
        AUC_dev = tester.test(dataset_dev)[0]
        AUC_test, precision_test, recall_test = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        tester.result_AUC(epoch, time, loss_train, AUC_dev,
                          AUC_test, precision_test, recall_test, file_AUC)
        tester.save_model(model, file_model)

        result = [epoch, time, loss_train, AUC_dev,
                  AUC_test, precision_test, recall_test]
        print('\t'.join(map(str, result)))
