import argparse
import random
import os

import numpy as np
import copy
import torch

from models import Listener, Speaker, SpeakerListener
import utils as U


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    print("OPTS:\n", vars(args))

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

    model = SpeakerListener(args).to(device=args.device)

    seed_train_images = train_images[:int(args.num_seed_examples * (1 - args.seed_val_pct))]
    seed_train_captions = train_captions[:int(args.num_seed_examples * (1 - args.seed_val_pct))]
    seed_val_images = train_images[int(args.num_seed_examples * (1 - args.seed_val_pct)): args.num_seed_examples]
    seed_val_captions = train_captions[int(args.num_seed_examples * (1 - args.seed_val_pct)): args.num_seed_examples]

    s2p_train_images = train_images[args.num_seed_examples: -1000]
    s2p_train_captions = train_captions[args.num_seed_examples: -1000]
    s2p_val_images = train_images[-1000:]
    s2p_val_captions = train_captions[-1000:]

    val_image_batch, val_distractor_image_batch, val_caption_batch, val_caption_len_batch = \
        U.get_s2p_batch(val_images, val_captions, args, batch_size=val_images.shape[0])

    test_image_batch, test_distractor_image_batch, test_caption_batch, test_caption_len_batch = \
        U.get_s2p_batch(test_images, test_captions, args, batch_size=test_images.shape[0])

    s2p_val_image_batch, s2p_val_distractor_image_batch, s2p_val_captions_batch, s2p_val_captions_len_batch = \
        U.get_s2p_batch(s2p_val_images, s2p_val_captions, args, batch_size=1000)

    seed_val_image_batch, seed_val_distractor_image_batch, seed_val_caption_batch, seed_val_caption_len_batch = \
        U.get_s2p_batch(seed_val_images, seed_val_captions, args, batch_size=min(1000, seed_val_images.shape[0]))

    def listener_supervised_update(train_images, train_captions, args):
        image_batch, distractor_image_batch, caption_batch, caption_len_batch = \
            U.get_s2p_batch(train_images, train_captions, args)

        _, _ = model.listener.update(image_batch, distractor_image_batch, caption_batch,
                                                     caption_len_batch)
        model.listener.optimizer.step()

    def speaker_supervised_update(train_images, train_captions, args):
        image_batch, distractor_image_batch, caption_batch, caption_len_batch = \
            U.get_s2p_batch(train_images, train_captions, args)
        caption_batch = torch.argmax(caption_batch, dim=-1)

        _ = model.speaker.update(image_batch, caption_batch, caption_len_batch)
        model.speaker.optimizer.step()

    list_acc_list = [0]
    best_list_acc = 0

    if args.s2p_schedule == 'rand':
        for i in range(args.max_iters):
            p = random.uniform(0, 1)

            if p < args.rand_perc:
                listener_supervised_update(seed_train_images, seed_train_captions, args)
                speaker_supervised_update(seed_train_images, seed_train_captions, args)
            else:
                image_batch, distractor_image_batch = U.get_pop_batch(s2p_train_images, args)
                _, _ = model.update(image_batch, distractor_image_batch)
                model.optimizer.step()

            if i % args.test_every == 0:
                _, acc = model.listener.test(seed_val_image_batch, seed_val_distractor_image_batch, seed_val_caption_batch,
                                                seed_val_caption_len_batch)
                list_acc_list.append(acc)

                if acc > best_list_acc:
                    best_list_acc = acc
                    best_model = copy.deepcopy(model)

    elif args.s2p_schedule == 'sched' or args.s2p_schedule == 'sup2sp' or args.s2p_schedule == 'sched_frz' \
            or args.s2p_schedule == 'sched_rand_frz':

        best_list_acc = 0
        best_spk_loss = 99999999
        for i in range(args.min_list_steps):
            listener_supervised_update(seed_train_images, seed_train_captions, args)

            if i % args.test_every == 0:
                _, acc = model.listener.test(seed_val_image_batch, seed_val_distractor_image_batch, seed_val_caption_batch,
                                             seed_val_caption_len_batch)
                if acc > best_list_acc:
                    best_list_acc = acc
                    best_listener_model = copy.deepcopy(model.listener)

        for i in range(args.min_spk_steps):
            speaker_supervised_update(seed_train_images, seed_train_captions, args)

            if i % args.test_every == 0:
                seed_val_caption_batch_max = torch.argmax(seed_val_caption_batch, dim=-1)
                spk_loss = model.speaker.test(seed_val_image_batch, seed_val_caption_batch_max, seed_val_caption_len_batch)

                if spk_loss < best_spk_loss:
                    best_spk_loss = spk_loss
                    best_speaker_model = copy.deepcopy(model.speaker)

        model = SpeakerListener(args).to(device=args.device)
        model.listener.load_state_dict(best_listener_model.state_dict())
        model.speaker.load_state_dict(best_speaker_model.state_dict())
        model.beholder.load_state_dict(best_listener_model.beholder.state_dict())

        loss, acc = model.listener.test(seed_val_image_batch, seed_val_distractor_image_batch,
                                        seed_val_caption_batch,
                                        seed_val_caption_len_batch)
        list_acc_list.append(acc)

        best_seed_val_acc = 0
        best_s2p_acc = 0
        spk_list = -1
        if args.s2p_schedule == 'sched_frz':
            spk_list = 0

        for i in range(args.max_iters):
            for epoch in range(args.s2p_selfplay_updates):
                if args.s2p_schedule == 'sched_rand_frz':
                    p = random.uniform(0, 1)
                    if p < args.rand_frz_perc:
                        spk_list = 0
                    else:
                        spk_list = -1
                image_batch, distractor_image_batch = U.get_pop_batch(s2p_train_images, args)
                _, _ = model.update(image_batch, distractor_image_batch, spk_list)
                model.optimizer.step()

                if i % args.test_every == 0:
                    msg, msg_lens = model.speaker.forward(s2p_val_image_batch)
                    loss, acc = model.listener.test(s2p_val_image_batch, s2p_val_distractor_image_batch,
                                                                    msg, msg_lens)

                    if best_s2p_acc < acc:
                        best_s2p_model = copy.deepcopy(model)

                    loss, acc = model.listener.test(seed_val_image_batch, seed_val_distractor_image_batch,
                                                    seed_val_caption_batch,
                                                    seed_val_caption_len_batch)
                    if args.s2p_schedule == 'sup2sp':
                        list_acc_list.append(acc)

            if args.s2p_schedule == 'sup2sp':
                best_model = best_s2p_model
                break

            for lepoch in range(args.s2p_list_updates):
                listener_supervised_update(seed_train_images, seed_train_captions, args)

            if spk_list == -1:
                for sepoch in range(args.s2p_spk_updates):
                    speaker_supervised_update(seed_train_images, seed_train_captions, args)

            loss, acc = model.listener.test(seed_val_image_batch, seed_val_distractor_image_batch,
                                            seed_val_caption_batch,
                                            seed_val_caption_len_batch)

            list_acc_list.append(acc)
            if acc > best_seed_val_acc:
                best_seed_val_acc = acc
                best_model = copy.deepcopy(model)

    if args.save_dir != '':
        U.torch_save_to_file(best_model.speaker.state_dict(), folder=os.path.join(args.save_dir, 'spk_params'),
                           file=f"pop{args.seed}.pt")
        U.torch_save_to_file(best_model.listener.state_dict(), folder=os.path.join(args.save_dir, 'list_params'),
                           file=f"pop{args.seed}.pt")
        U.torch_save_to_file(best_model.beholder.state_dict(), folder=os.path.join(args.save_dir, 'bhd_params'),
                           file=f"pop{args.seed}.pt")

    _, val_acc = best_model.listener.test(val_image_batch, val_distractor_image_batch, val_caption_batch,
                                          val_caption_len_batch)

    _, test_acc = best_model.listener.test(test_image_batch, test_distractor_image_batch, test_caption_batch,
                                           test_caption_len_batch)

    if args.save_dir != '':
        save_dict = {'val_acc': val_acc,
                     'test_acc': test_acc,
                     'seed_val_accs': list_acc_list
                     }
        U.save_to_file(save_dict, folder=os.path.join(args.save_dir, 'lists'), file=f'results{args.seed}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0,
                        help='seed')
    parser.add_argument("--num_seed_examples", type=int, default=1000,
                        help="Number of seed examples")
    parser.add_argument("--num_distrs", type=int, default=9,
                        help="Number of distractors")
    parser.add_argument("--s2p_schedule", type=str, default="sched",
                        help="s2p schedule")
    parser.add_argument("--s2p_selfplay_updates", type=int, default=50,
                        help="s2p self-play updates")
    parser.add_argument("--s2p_list_updates", type=int, default=50,
                        help="s2p listener supervised updates")
    parser.add_argument("--s2p_spk_updates", type=int, default=50,
                        help="s2p speaker supervised updates")
    parser.add_argument("--s2p_batch_size", type=int, default=1000,
                        help="s2p batch size")
    parser.add_argument("--pop_batch_size", type=int, default=1000,
                        help="Pop Batch size")
    parser.add_argument("--rand_perc", type=int, default=0.75,
                        help="rand perc")
    parser.add_argument("--sched_rand_frz", type=int, default=0.5,
                        help="sched_rand_frz perc")
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
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout probability")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="Gumbel temperature")
    parser.add_argument("--hard", type=bool, default=True,
                        help="Hard Gumbel-Softmax Sampling.")
    parser.add_argument("--min_list_steps", type=int, default=2000,
                        help="Min num of listener supervised steps")
    parser.add_argument("--min_spk_steps", type=int, default=1000,
                        help="Min num of speaker supervised steps")
    parser.add_argument("--test_every", type=int, default=10,
                        help="test interval")
    parser.add_argument("--seed_val_pct", type=float, default=0.1,
                        help="% of seed samples used as validation for early stopping")
    parser.add_argument('--coco_path', type=str, default="./coco/",
                        help="MSCOCO dir path")
    parser.add_argument("--save_dir", type=str, default="",
                        help="Save directory.")

    args = parser.parse_args()
    main(args)

