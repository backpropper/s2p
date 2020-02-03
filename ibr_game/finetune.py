import argparse
import random
import os

import numpy as np
import copy

import torch

from models import Listener, Speaker, SpeakerListener, Beholder
import utils as U


def load_listener(curr_opts, trainpop_files):
    model = SpeakerListener(args).to(device=args.device)
    model.beholder = Beholder(curr_opts).to(device=args.device)
    model.listener = Listener(curr_opts, model.beholder).to(device=args.device)
    model_dict = torch.load(os.path.join(trainpop_files, f'pop{args.seed}.pt'), map_location=args.device)
    model.listener.load_state_dict(model_dict)
    return model


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    feat_path = args.coco_path
    data_path = args.coco_path

    (train_images, val_images, test_images) = [torch.load('{}/feats/{}'.format(feat_path, x)) for x in
                                                     "train_feats valid_feats test_feats".split()]
    (w2i, i2w) = [torch.load(data_path + 'dics/{}'.format(x)) for x in "w2i i2w".split()]
    (train_captions, val_captions, test_captions) = [torch.load('{}/labs/{}'.format(feat_path, x)) \
                                      for x in "train_org valid_org test_org".split()]

    train_images = train_images.to(device=args.device)
    val_images = val_images.to(device=args.device)
    test_images = test_images.to(device=args.device)

    train_captions = U.trim_caps(train_captions, 4, args.seq_len)
    val_captions = U.trim_caps(val_captions, 4, args.seq_len)
    test_captions = U.trim_caps(test_captions, 4, args.seq_len)

    w2i, i2w = U.truncate_dicts(w2i, i2w, args.num_words)
    train_captions, val_captions, test_captions = U.truncate_captions(train_captions, val_captions, test_captions, w2i, i2w)

    train_captions, train_images = U.filter_caps(train_captions, train_images, w2i, args.unk_perc)
    val_captions, val_images = U.filter_caps(val_captions, val_images, w2i, args.unk_perc)
    test_captions, test_images = U.filter_caps(test_captions, test_images, w2i, args.unk_perc)

    args.vocab_size = len(w2i)
    args.w2i = w2i
    args.i2w = i2w

    model = load_listener(args, args.trainpop_files)

    seed_train_images = train_images[args.num_seed_examples: args.num_total_seed_samples]
    seed_train_captions = train_captions[args.num_seed_examples: args.num_total_seed_samples]
    seed_val_images = train_images[int(args.num_seed_examples * (1 - args.seed_val_pct)): args.num_seed_examples]
    seed_val_captions = train_captions[int(args.num_seed_examples * (1 - args.seed_val_pct)): args.num_seed_examples]

    val_image_batch, val_distractor_image_batch, val_caption_batch, val_caption_len_batch = \
        U.get_s2p_batch(val_images, val_captions, args, batch_size=val_images.shape[0])

    test_image_batch, test_distractor_image_batch, test_caption_batch, test_caption_len_batch = \
        U.get_s2p_batch(test_images, test_captions, args, batch_size=test_images.shape[0])

    seed_val_image_batch, seed_val_distractor_image_batch, seed_val_caption_batch, seed_val_caption_len_batch = \
        U.get_s2p_batch(seed_val_images, seed_val_captions, args, batch_size=min(1000, seed_val_images.shape[0]))

    def listener_supervised_update(train_images, train_captions, args):
        image_batch, distractor_image_batch, caption_batch, caption_len_batch = \
            U.get_s2p_batch(train_images, train_captions, args)

        _, _ = model.listener.update(image_batch, distractor_image_batch, caption_batch,
                                     caption_len_batch)
        model.listener.optimizer.step()

    list_acc_list = [0]
    best_listener_model = None
    best_list_acc = 0
    for i in range(args.list_steps):
        listener_supervised_update(seed_train_images, seed_train_captions, args)

        if i % args.test_every == 0:
            _, acc = model.listener.test(seed_val_image_batch, seed_val_distractor_image_batch, seed_val_caption_batch,
                                         seed_val_caption_len_batch)
            if acc > best_list_acc:
                best_list_acc = acc
                best_listener_model = copy.deepcopy(model.listener)

            list_acc_list.append(acc)

    model = SpeakerListener(args).to(device=args.device)
    model.listener.load_state_dict(best_listener_model.state_dict())

    if args.save_dir != '':
        U.torch_save_to_file(model.listener.state_dict(), folder=os.path.join(args.save_dir, 'list_params'),
                       file=f"pop{args.seed}.pt")

    _, val_acc = model.listener.test(val_image_batch, val_distractor_image_batch, val_caption_batch,
                                          val_caption_len_batch)
    _, test_acc = model.listener.test(test_image_batch, test_distractor_image_batch, test_caption_batch,
                                           test_caption_len_batch)

    if args.save_dir != '':
        save_dict = {'seed_val_accs': list_acc_list,
                     'val_acc': val_acc,
                     'test_acc': test_acc
                     }
        U.save_to_file(save_dict, folder=args.save_dir, file=f'results{args.seed}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--trainpop_files', type=str, default='./list_params')
    parser.add_argument("--num_seed_examples", type=int, default=1000,
                        help="Number of seed examples")
    parser.add_argument("--num_total_seed_samples", type=int, default=10000,
                        help="Number of seed samples that will be trained on in total")
    parser.add_argument("--num_distrs", type=int, default=9,
                        help="Number of distractors")
    parser.add_argument("--pop_batch_size", type=int, default=1000,
                        help="Pop Batch size")
    parser.add_argument("--s2p_batch_size", type=int, default=1000,
                        help="s2p batch size")
    parser.add_argument("--num_words", type=int, default=100,
                        help="Number of words in the vocabulary")
    parser.add_argument("--seq_len", type=int, default=15,
                        help="Max Sequence length of speaker utterance")
    parser.add_argument("--unk_perc", type=float, default=0.3,
                        help="Max percentage of <UNK>")
    parser.add_argument("--max_iters", type=int, default=300,
                        help="max training iters")
    parser.add_argument("--D_img", type=int, default=2048,
                        help="ResNet feature dimensionality. Can't change this")
    parser.add_argument("--D_hid", type=int, default=512,
                        help="RNN hidden state dimensionality")
    parser.add_argument("--D_emb", type=int, default=256,
                        help="Token embedding (word) dimensionality")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="Dropout probability")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="Gumbel temperature")
    parser.add_argument("--hard", type=bool, default=True,
                        help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--list_steps", type=int, default=3000,
                        help="Min num of listener supervised steps")
    parser.add_argument("--test_every", type=int, default=100,
                        help="test interval")
    parser.add_argument("--seed_val_pct", type=float, default=0.1,
                        help="% of seed samples used as validation for early stopping")
    parser.add_argument('--coco_path', type=str, default="./coco/",
                        help="MSCOCO dir path")
    parser.add_argument("--save_dir", type=str, default="",
                        help="Save directory.")

    args = parser.parse_args()
    main(args)

