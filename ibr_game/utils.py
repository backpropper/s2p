import os
import pickle
from collections import OrderedDict
import itertools
import random

import numpy as np
import torch
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_s2p_batch(seed_train_images, seed_train_captions, args, batch_size=None):
    data_size = seed_train_images.shape[0]
    if batch_size is None:
        batch_size = args.s2p_batch_size
    if batch_size > data_size:
        batch_size = data_size

    target_sample = random.sample(list(range(data_size)), batch_size)
    left_samples = list(range(data_size))
    distractor_samples = []
    for i in range(args.num_distrs):
        dsample = random.sample(left_samples, batch_size)
        distractor_samples.append(dsample)

    image_batch = seed_train_images[target_sample]
    distractor_image_batch = []
    for ds in distractor_samples:
        d_image_batch = seed_train_images[ds]
        distractor_image_batch.append(d_image_batch)
    caption_batch = [seed_train_captions[t] for t in target_sample]
    caption_len_batch = torch.tensor([len(c) for c in caption_batch], dtype=torch.long, device=device)

    caption_batch = torch.tensor([np.pad(c, (0, args.seq_len - len(c))) for c in caption_batch], dtype=torch.long, device=device)

    caption_batch_onehot = torch.zeros(caption_batch.shape[0], caption_batch.shape[1], args.vocab_size,
                             device=device).scatter_(-1, caption_batch.unsqueeze(-1), 1)

    return image_batch, distractor_image_batch, caption_batch_onehot, caption_len_batch


def get_batch_with_speaker(train_images, speaker, args, batch_size=None):
    data_size = train_images.shape[0]
    if batch_size is None:
        batch_size = args.s2p_batch_size
    if batch_size > data_size:
        batch_size = data_size

    target_sample = random.sample(list(range(data_size)), batch_size)
    left_samples = list(range(data_size))
    distractor_samples = []
    for i in range(args.num_distrs):
        dsample = random.sample(left_samples, batch_size)
        distractor_samples.append(dsample)

    image_batch = train_images[target_sample]
    distractor_image_batch = []
    for ds in distractor_samples:
        d_image_batch = train_images[ds]
        distractor_image_batch.append(d_image_batch)

    train_captions, train_captions_len = speaker.forward(image_batch)
    train_captions = train_captions.detach()
    train_captions_len = train_captions_len.detach()

    return image_batch, distractor_image_batch, train_captions, train_captions_len


def get_pop_batch(train_images, args, batch_size=None):
    data_size = train_images.shape[0]
    if batch_size is None:
        batch_size = args.pop_batch_size

    target_sample = random.sample(list(range(data_size)), batch_size)
    left_samples = list(range(data_size))
    distractor_samples = []
    for i in range(args.num_distrs):
        dsample = random.sample(left_samples, batch_size)
        distractor_samples.append(dsample)

    image_batch = train_images[target_sample]
    distractor_image_batch = []
    for ds in distractor_samples:
        d_image_batch = train_images[ds]
        distractor_image_batch.append(d_image_batch)

    return image_batch, distractor_image_batch


def trim_caps(caps, minlen, maxlen):
    new_cap = [[cap for cap in cap_i if maxlen >= len(cap) >= minlen] for cap_i in caps]
    return new_cap


def truncate_dicts(w2i, i2w, trunc_size):
    symbols_to_keep = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    inds_to_keep = [w2i[s] for s in symbols_to_keep]

    w2i_trunc = OrderedDict(itertools.islice(w2i.items(), trunc_size))
    i2w_trunc = OrderedDict(itertools.islice(i2w.items(), trunc_size))

    for s, i in zip(symbols_to_keep, inds_to_keep):
        w2i_trunc[s] = i
        i2w_trunc[i] = s

    return w2i_trunc, i2w_trunc


def truncate_captions(train_captions, valid_captions, test_captions, w2i, i2w):
    unk_ind = w2i["<UNK>"]

    def truncate_data(data):
        for i in range(len(data)):
            for ii in range(len(data[i])):
                for iii in range(len(data[i][ii])):
                    if data[i][ii][iii] not in i2w:
                        data[i][ii][iii] = unk_ind
        return data

    train_captions = truncate_data(train_captions)
    valid_captions = truncate_data(valid_captions)
    test_captions = truncate_data(test_captions)

    return train_captions, valid_captions, test_captions


def load_model(model_dir, model, device):
    model_dicts = torch.load(os.path.join(model_dir, 'model.pt'), map_location=device)
    model.load_state_dict(model_dicts)
    iters = model_dicts['iters']
    best_test_acc = model_dicts['test_acc']
    print("Best Test acc:", best_test_acc, " at", iters, "iters")


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


def to_sentence(inds_list, i2w, trim=False):
    sentences = []
    for inds in inds_list:
        if type(inds) is not list:
            inds = list(inds)
        sentence = []
        for i in inds:
            sentence.append(i2w[i])
            if i2w[i] == "<PAD>" and trim:
                break
        sentences.append(' '.join(sentence))
    return sentences


def filter_caps(captions, images, w2i, perc):
    new_train_captions = []
    new_train_images = []
    for ci, cap in enumerate(captions):
        if len(cap) > 0 and cap[0].count(w2i["<UNK>"]) / len(cap[0]) < perc:
            new_train_captions.append(cap[0])
            new_train_images.append(images[ci])
    return new_train_captions, torch.stack(new_train_images)


def sample_gumbel(shape, eps=1e-20):
    U = torch.empty(shape, device=device).uniform_(0, 1)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temp):
    y = (logits + sample_gumbel(logits.shape)) / temp
    return F.softmax(y, dim=-1)


def gumbel_softmax(logits, temp, hard):
    y = gumbel_softmax_sample(logits, temp)
    y_max, y_max_idx = torch.max(y, 1, keepdim=True)
    if hard:
        y_hard = torch.zeros(y.shape, device=device).scatter_(1, y_max_idx, 1)
        y = (y_hard - y).detach() + y
    return y, y_max_idx