import itertools
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F


def get_onehot(vec_len, pos):
    v = np.zeros(vec_len)
    v[pos] = 1
    return v


def save_to_file(vals, folder='', file=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_str = os.path.join(folder, file+'.pkl')
    with open(save_str, 'wb') as f1:
        pickle.dump(vals, f1)


def torch_save_to_file(to_save, folder='', file=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(to_save, os.path.join(folder, file))


def get_batch_indices(v, num_properties, types_per_property, device):
    v = torch.tensor(v, device=device, dtype=torch.long)
    batch_size = v.shape[0]
    v = v.view(batch_size, num_properties, types_per_property)
    out = []
    for prop in range(num_properties):
        v1 = v[:, prop]
        out.append(torch.argmax(v1, -1))
    out = torch.stack(out)
    return out


def make_dataset(num_properties=3, types_per_property=5, val_pct=0.1, test_pct=0.1):
    onehot_list = [get_onehot(types_per_property, i) for i in range(types_per_property)]
    train = list(itertools.product(onehot_list, repeat=num_properties))
    for i in range(len(train)):
        train[i] = np.concatenate(train[i])
    test_ind = int((1 - test_pct) * len(train))
    val_ind = int((1 - val_pct - test_pct) * len(train))
    test = train[test_ind:]
    val = train[val_ind: test_ind]
    train = train[:val_ind]
    return train, val, test


def sample_gumbel(shape, device, eps=1e-8):
    values = torch.empty(shape, device=device, dtype=torch.float).uniform_(0, 1)
    return -torch.log(-torch.log(values + eps) + eps)


def gumbel_softmax(logits, temperature, device):
    y = logits + sample_gumbel(logits.shape, device)
    return F.softmax(y / temperature, -1)
