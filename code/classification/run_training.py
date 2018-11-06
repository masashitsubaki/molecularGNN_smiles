import pickle
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score, precision_score, recall_score


class MolecularPropertyPrediction(nn.Module):
    def __init__(self):
        super(MolecularPropertyPrediction, self).__init__()
        self.embed_atom = nn.Embedding(n_fingerprint, dim)
        self.W_atom = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer)])
        self.W_property = nn.Linear(dim, 2)

    def pad(self, matrices, value):
        sizes = [d.shape[0] for d in matrices]
        D = sum(sizes)
        pad_matrices = value + np.zeros((D, D))
        m = 0
        for i, d in enumerate(matrices):
            s_i = sizes[i]
            pad_matrices[m:m+s_i, m:m+s_i] = d
            m += s_i
        return torch.FloatTensor(pad_matrices).to(device)

    def sum_axis(self, xs, axis):
        y = list(map(lambda x: torch.sum(x, 0), torch.split(xs, axis)))
        return torch.stack(y)

    def update(self, xs, adjacency, i):
        hs = torch.relu(self.W_atom[i](xs))
        return xs + torch.matmul(adjacency, hs)

    def forward(self, inputs):

        atoms, adjacency = inputs
        axis = list(map(lambda x: len(x), atoms))

        atoms = torch.cat(atoms)
        x_atoms = self.embed_atom(atoms)
        adjacency = self.pad(adjacency, 0)

        for i in range(layer):
            x_atoms = self.update(x_atoms, adjacency, i)

        y_molecules = self.sum_axis(x_atoms, axis)
        z_properties = self.W_property(y_molecules)

        return z_properties

    def __call__(self, data_batch, train=True):

        inputs, t_properties = data_batch[:-1], torch.cat(data_batch[-1])
        z_properties = self.forward(inputs)

        if train:
            loss = F.cross_entropy(z_properties, t_properties)
            return loss
        else:
            zs = F.softmax(z_properties, 1).to('cpu').data.numpy()
            ts = t_properties.to('cpu').data.numpy()
            scores = list(map(lambda x: x[1], zs))
            labels = list(map(lambda x: np.argmax(x), zs))
            return scores, labels, ts


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset_train):
        np.random.shuffle(dataset_train)
        N = len(dataset_train)
        loss_total = 0
        for i in range(0, N, batch):
            data_batch = list(zip(*dataset_train[i:i+batch]))
            loss = self.model(data_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset_test):

        N = len(dataset_test)
        score_list, label_list, ts_list = [], [], []

        for i in range(0, N, batch):
            data_batch = list(zip(*dataset_test[i:i+batch]))
            scores, labels, ts = self.model(data_batch, train=False)
            score_list = np.append(score_list, scores)
            label_list = np.append(label_list, labels)
            ts_list = np.append(ts_list, ts)

        auc = roc_auc_score(ts_list, score_list)
        precision = precision_score(ts_list, label_list)
        recall = recall_score(ts_list, label_list)

        return auc, precision, recall

    def result(self, epoch, time, loss_total, auc_dev,
               auc_test, precision, recall, file_result):
        with open(file_result, 'a') as f:
            result = map(str, [epoch, time, loss_total, auc_dev,
                               auc_test, precision, recall])
            f.write('\t'.join(result) + '\n')

    def save_model(self, model, file_name):
        torch.save(model.state_dict(), file_name)


def load_tensor(data, dtype):
    return [dtype(d).to(device) for d in np.load(dir_input + data + '.npy')]


def load_numpy(data):
    return np.load(dir_input + data + '.npy')


def load_pickle(data):
    with open(dir_input + data, 'rb') as f:
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

    (DATASET, radius, dim, layer, batch, lr, lr_decay, decay_interval,
     iteration, setting) = sys.argv[1:]
    (dim, layer, batch, decay_interval,
     iteration) = map(int, [dim, layer, batch, decay_interval, iteration])
    lr, lr_decay = map(float, [lr, lr_decay])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    dir_input = ('../../dataset/classification/' +
                 DATASET + '/input/radius' + radius + '/')
    molecules = load_tensor('molecules', torch.LongTensor)
    adjacencies = load_numpy('adjacencies')
    t_properties = load_tensor('properties', torch.LongTensor)
    with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
        fingerprint_dict = pickle.load(f)

    dataset = list(zip(molecules, adjacencies, t_properties))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    fingerprint_dict = load_pickle('fingerprint_dict.pickle')
    unknown = 100
    n_fingerprint = len(fingerprint_dict) + unknown

    torch.manual_seed(1234)
    model = MolecularPropertyPrediction().to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    file_result = '../../output/result/' + setting + '.txt'
    with open(file_result, 'w') as f:
        f.write('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
                'AUC_test\tPrecision_test\tRecall_test\n')

    file_model = '../../output/model/AUC--' + setting

    print('Training...')
    print('Epoch Time(sec) Loss_train AUC_dev '
          'AUC_test Precision_test Recall_test')

    start = timeit.default_timer()

    for epoch in range(iteration):

        if (epoch+1) % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_total = trainer.train(dataset_train)
        auc_dev = tester.test(dataset_dev)[0]
        auc_test, precision, recall = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        tester.result(epoch, time, loss_total, auc_dev,
                      auc_test, precision, recall, file_result)
        tester.save_model(model, file_model)

        print(epoch, time, loss_total, auc_dev, auc_test, precision, recall)
