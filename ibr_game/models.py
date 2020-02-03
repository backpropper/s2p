import numpy as np
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import utils as U


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Beholder(nn.Module):
    def __init__(self, args):
        super(Beholder, self).__init__()
        self.img_to_hid = nn.Linear(args.D_img, args.D_hid)
        self.drop = nn.Dropout(p=args.dropout)

    def forward(self, img):
        h_img = img
        h_img = self.img_to_hid(h_img)
        h_img = self.drop(h_img)
        return h_img


class Listener(nn.Module):

    def __init__(self, args, beholder):
        super(Listener, self).__init__()
        self.rnn = nn.GRU(args.D_emb, args.D_hid, 1, batch_first=True)
        self.emb = nn.Linear(args.vocab_size, args.D_emb)
        self.hid_to_hid = nn.Linear(args.D_hid, args.D_hid)
        self.drop = nn.Dropout(p=args.dropout)
        self.D_hid = args.D_hid
        self.D_emb = args.D_emb
        self.vocab_size = args.vocab_size
        self.i2w = args.i2w
        self.w2i = args.w2i
        self.beholder = beholder
        self.loss_fn = nn.CrossEntropyLoss().to(device=args.device)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, spk_msg, spk_msg_lens):
        batch_size = spk_msg.shape[0]

        h_0 = torch.zeros(1, batch_size, self.D_hid, device=device)

        spk_msg_emb = self.emb(spk_msg.float())
        spk_msg_emb = self.drop(spk_msg_emb)

        pack = nn.utils.rnn.pack_padded_sequence(spk_msg_emb, spk_msg_lens, batch_first=True)

        self.rnn.flatten_parameters()
        _, h_n = self.rnn(pack, h_0)
        h_n = h_n[-1:, :, :]
        out = h_n.transpose(0, 1).view(batch_size, self.D_hid)
        out = self.hid_to_hid(out)
        return out

    def get_loss_acc(self, image, distractor_images, spk_msg, spk_msg_lens):
        batch_size = spk_msg.shape[0]

        spk_msg_lens, sorted_indices = torch.sort(spk_msg_lens, descending=True)
        spk_msg = spk_msg.index_select(0, sorted_indices)
        image = image.index_select(0, sorted_indices)

        h_pred = self.forward(spk_msg, spk_msg_lens)
        h_pred = h_pred.unsqueeze(1).repeat(1, 1 + len(distractor_images), 1)

        all_images = len(distractor_images) + 1
        img_idx = [list(range(all_images)) for _ in range(batch_size)]
        for c in img_idx:
            random.shuffle(c)

        target_idx = torch.tensor(np.argmax(np.array(img_idx) == 0, -1), dtype=torch.long, device=device)

        h_img = [self.beholder(image)] + [self.beholder(img) for img in distractor_images]
        h_img = torch.stack(h_img, dim=0).permute(1, 0, 2)
        for i in range(batch_size):
            h_img[i] = h_img[i, img_idx[i], :]

        logits = 1 / torch.mean(torch.pow(h_pred - h_img, 2), 2).view(-1, 1 + len(distractor_images))

        pred_outs = torch.argmax(logits, dim=-1).cpu().numpy()
        batch_inds = target_idx.cpu().numpy()

        acc = np.mean(np.equal(batch_inds, pred_outs))
        loss = self.loss_fn(logits, target_idx)
        return loss, acc

    def test(self, image, distractor_images, spk_msg, spk_msg_lens):
        self.eval()
        loss, acc = self.get_loss_acc(image, distractor_images, spk_msg, spk_msg_lens)
        return loss.detach().cpu().numpy(), acc

    def update(self, image, distractor_images, spk_msg, spk_msg_lens):
        self.train()
        loss, acc = self.get_loss_acc(image, distractor_images, spk_msg, spk_msg_lens)
        self.optimizer.zero_grad()
        loss.backward()
        return loss, acc


class Speaker(nn.Module):

    def __init__(self, args, beholder):
        super(Speaker, self).__init__()
        self.rnn = nn.GRU(args.D_emb, args.D_hid, 1, batch_first=True)
        self.emb = nn.Embedding(args.vocab_size, args.D_emb, padding_idx=0)
        self.hid_to_voc = nn.Linear(args.D_hid, args.vocab_size)
        self.D_emb = args.D_emb
        self.D_hid = args.D_hid
        self.drop = nn.Dropout(p=args.dropout)
        self.vocab_size = args.vocab_size
        self.i2w = args.i2w
        self.w2i = args.w2i
        self.temp = args.temp
        self.hard = args.hard
        self.seq_len = args.seq_len
        self.beholder = beholder
        self.loss_fn = nn.CrossEntropyLoss(reduce=False)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, image):
        batch_size = image.shape[0]

        h_img = self.beholder(image).detach()

        start = [self.w2i["<BOS>"] for _ in range(batch_size)]
        gen_idx = []
        done = np.array([False for _ in range(batch_size)])

        h_img = h_img.unsqueeze(0).view(1, -1, self.D_hid).repeat(1, 1, 1)
        hid = h_img
        ft = torch.tensor(start, dtype=torch.long, device=device).view(-1).unsqueeze(1)
        input = self.emb(ft)
        msg_lens = [self.seq_len for _ in range(batch_size)]

        for idx in range(self.seq_len):
            input = F.relu(input)
            self.rnn.flatten_parameters()
            output, hid = self.rnn(input, hid)

            output = output.view(-1, self.D_hid)
            output = self.hid_to_voc(output)
            output = output.view(-1, self.vocab_size)

            top1, topi = U.gumbel_softmax(output, self.temp, self.hard)
            gen_idx.append(top1)

            for ii in range(batch_size):
                if topi[ii] == self.w2i["<EOS>"]:
                    done[ii] = True
                    msg_lens[ii] = idx + 1
                if np.array_equal(done, np.array([True for _ in range(batch_size)])):
                    break

            input = self.emb(topi)

        gen_idx = torch.stack(gen_idx).permute(1, 0, 2)
        msg_lens = torch.tensor(msg_lens, dtype=torch.long, device=device)
        return gen_idx, msg_lens

    def get_loss(self, image, caps, caps_lens):
        batch_size = caps.shape[0]
        mask = (torch.arange(self.seq_len, device=device).expand(batch_size, self.seq_len) < caps_lens.unsqueeze(
            1)).float()

        caps_in = caps[:, :-1]
        caps_out = caps[:, 1:]

        h_img = self.beholder(image).detach()
        h_img = h_img.view(1, batch_size, self.D_hid).repeat(1, 1, 1)

        caps_in_emb = self.emb(caps_in)
        caps_in_emb = self.drop(caps_in_emb)

        self.rnn.flatten_parameters()
        output, _ = self.rnn(caps_in_emb, h_img)
        logits = self.hid_to_voc(output)

        loss = 0
        for j in range(logits.size(1)):
            flat_score = logits[:, j, :]
            flat_mask = mask[:, j]
            flat_tgt = caps_out[:, j]
            nll = self.loss_fn(flat_score, flat_tgt)
            loss += (flat_mask * nll).sum()

        return loss

    def test(self, image, caps, caps_lens):
        self.eval()
        loss = self.get_loss(image, caps, caps_lens)
        return loss.detach().cpu().numpy()

    def update(self, image, caps, caps_lens):
        self.train()
        loss = self.get_loss(image, caps, caps_lens)
        self.optimizer.zero_grad()
        loss.backward()
        return loss


class SpeakerListener(nn.Module):

    def __init__(self, args):
        super(SpeakerListener, self).__init__()
        self.beholder = Beholder(args)
        self.speaker = Speaker(args, self.beholder)
        self.listener = Listener(args, self.beholder)
        self.i2w = args.i2w
        self.w2i = args.w2i
        self.D_hid = args.D_hid
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, image, spk_list=-1):
        msg, msg_lens = self.speaker.forward(image)

        if spk_list == 0:
            msg = msg.detach()

        msg_lens, sorted_indices = torch.sort(msg_lens, descending=True)
        msg = msg.index_select(0, sorted_indices)
        image = image.index_select(0, sorted_indices)

        h_pred = self.listener.forward(msg, msg_lens)
        return h_pred, image

    def get_loss_acc(self, image, distractor_images, spk_list=-1):
        batch_size = image.shape[0]

        h_pred, image = self.forward(image, spk_list)
        h_pred = h_pred.unsqueeze(1).repeat(1, 1 + len(distractor_images), 1)

        all_images = len(distractor_images) + 1
        img_idx = [list(range(all_images)) for _ in range(batch_size)]
        for c in img_idx:
            random.shuffle(c)

        target_idx = torch.tensor(np.argmax(np.array(img_idx) == 0, -1), dtype=torch.long, device=device)

        h_img = [self.beholder(image)] + [self.beholder(img) for img in distractor_images]
        h_img = torch.stack(h_img, dim=0).permute(1, 0, 2)
        for i in range(batch_size):
            h_img[i] = h_img[i, img_idx[i], :]

        logits = 1 / torch.mean(torch.pow(h_pred - h_img, 2), 2).view(-1, 1 + len(distractor_images))
        pred_outs = torch.argmax(logits, dim=-1).cpu().numpy()
        batch_inds = target_idx.cpu().numpy()

        acc = np.mean(np.equal(batch_inds, pred_outs))
        loss = self.loss_fn(logits, target_idx)
        return acc, loss

    def update(self, image, distractor_images):
        self.eval()
        acc, loss = self.get_loss_acc(image, distractor_images)
        return acc, loss

    def update(self, image, distractor_images, spk_list=-1):
        self.train()
        acc, loss = self.get_loss_acc(image, distractor_images, spk_list)
        self.optimizer.zero_grad()
        loss.backward()
        return acc, loss
