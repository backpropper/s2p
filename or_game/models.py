import random
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import utils as U


class CompEncoder(nn.Module):

    def __init__(self, opts):
        super(CompEncoder, self).__init__()
        self.config = opts

        self.encoder_embedding = nn.Linear(self.config['types_per_property'] * self.config['num_properties'],
                                           self.config['embedding_size_encoder'])
        self.encoder_out = nn.Linear(self.config['embedding_size_encoder'],
                                     self.config['num_latent_variables'] * self.config['output_size'])

        self.optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        input_embs = self.encoder_embedding(inputs)
        input_out = self.encoder_out(input_embs)

        input_out = input_out.view(batch_size, self.config['num_latent_variables'],
                                   self.config['output_size']).permute(1, 0, 2)

        return input_out

    def encode(self, inputs):
        input_out = self.forward(inputs)
        input_out = torch.softmax(input_out, dim=-1)
        return input_out

    def encoder(self, inputs):
        return self.forward(inputs).permute(1, 0, 2)

    def test(self, inputs, batch_inds):
        input_out = self.encode(inputs)

        pred_outs = torch.argmax(input_out, dim=-1).cpu().numpy()
        batch_inds = batch_inds.detach().cpu().numpy()

        batch_ind = batch_inds.transpose(1, 0)
        pred_outs = pred_outs.transpose(1, 0)

        train_acc = np.mean(np.all(np.equal(batch_ind, pred_outs), 1))
        return train_acc

    def update(self, out_list, batch_ind):
        batch_size = batch_ind.shape[0]
        criterion = nn.CrossEntropyLoss().to(device=self.config['device'])
        xent = torch.zeros(batch_size, device=self.config['device'])

        for i in range(self.config['num_latent_variables']):
            xent += criterion(out_list[i], batch_ind[i])

        self.optimizer.zero_grad()
        xent.mean().backward()
        self.optimizer.step()


class CompDecoder(nn.Module):

    def __init__(self, opts):
        super(CompDecoder, self).__init__()
        self.config = opts
        self.decoder_embedding = nn.Linear(self.config['num_latent_variables'] * self.config['output_size'],
                                           self.config['embedding_size_decoder'])
        self.decoder_out = nn.Linear(self.config['embedding_size_decoder'], self.config['types_per_property'] *
                                     self.config['num_properties'])
        self.optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])

    def decode(self, inputs):
        batch_size = inputs.shape[0]
        inputs = inputs.contiguous().view(batch_size, self.config['num_latent_variables'] * self.config['output_size'])
        input_embs = self.decoder_embedding(inputs)
        out = self.decoder_out(input_embs)

        out = out.view(batch_size, self.config['num_properties'],
                                   self.config['types_per_property']).permute(1, 0, 2)

        out = torch.softmax(out, dim=-1)
        return out

    def test(self, inputs, batch_inds):
        out = self.decode(inputs)

        pred_outs = torch.argmax(out, dim=-1).cpu().numpy()
        batch_inds = batch_inds.detach().cpu().numpy()

        batch_ind = batch_inds.transpose(1, 0)
        pred_outs = pred_outs.transpose(1, 0)

        train_acc = np.mean(np.all(np.equal(batch_ind, pred_outs), 1))
        return train_acc

    def update(self, out_list, batch_ind):
        batch_size = batch_ind.shape[0]
        criterion = nn.CrossEntropyLoss().to(device=self.config['device'])
        xent = torch.zeros(batch_size, device=self.config['device'])

        for i in range(self.config['num_properties']):
            xent += criterion(out_list[i], batch_ind[i])

        self.optimizer.zero_grad()
        xent.mean().backward()
        self.optimizer.step()


class CompEncoderDecoder(nn.Module):
    def __init__(self, opts):
        super(CompEncoderDecoder, self).__init__()
        self.config = opts
        self.enc = CompEncoder(opts)
        self.dec = CompDecoder(opts)
        self.temperature = torch.tensor([self.config['temp']] * self.config['num_latent_variables'],
                                       device=self.config['device'], dtype=torch.float)

    def forward(self, inputs):
        input_out = self.enc.forward(inputs)

        logits = []
        for num in range(self.config['num_latent_variables']):
            logit = U.gumbel_softmax(input_out[num], self.temperature[num], self.config['device'])
            logits.append(logit)

        logits = torch.stack(logits).permute(1, 0, 2)
        out = self.dec.decode(logits)
        return out

    def update(self, batch, batch_ind):
        out_list = self.forward(batch)
        batch_size = batch_ind.shape[0]

        criterion = nn.CrossEntropyLoss().to(device=self.config['device'])
        xent = torch.zeros(batch_size, device=self.config['device'])

        for i in range(self.config['num_properties']):
            xent += criterion(out_list[i], batch_ind[i])

        self.enc.optimizer.zero_grad()
        self.dec.optimizer.zero_grad()
        xent.mean().backward()
        self.enc.optimizer.step()
        self.dec.optimizer.step()

    def test(self, inputs, batch_inds):
        batch_size = inputs.shape[0]
        input_out = self.enc.encode(inputs)
        argmax_message = torch.argmax(input_out, dim=-1, keepdim=True).permute(1, 0, 2)
        message = torch.zeros(batch_size, self.config['num_latent_variables'], self.config['output_size'],
                              device=self.config['device']).scatter_(-1, argmax_message, 1)
        out = self.dec.decode(message)
        pred_outs = torch.argmax(out, dim=-1).cpu().numpy()
        batch_inds = batch_inds.detach().cpu().numpy()

        batch_ind = batch_inds.transpose(1, 0)
        pred_outs = pred_outs.transpose(1, 0)

        train_acc = np.mean(np.all(np.equal(batch_ind, pred_outs), 1))
        return train_acc


class CompositionalBot(object):
    def __init__(self, opts, seed=0):
        super(CompositionalBot, self).__init__()
        self.opts = opts
        torch_gen = torch.Generator()
        self.torch_gen = torch_gen.manual_seed(seed)
        self.random_gen = random.Random(seed)
        self.num_properties = opts['num_properties']
        self.types_per_property = opts['types_per_property']
        self.vocab_size = opts['output_size']
        self.num_latents = opts['num_latent_variables']
        self.word_idx, self.word_order = self.generate_language()

    def encoder(self, inputs):
        inputs = self.convert_to_latents(inputs)
        out = torch.matmul(inputs, self.word_idx)[:, self.word_order]
        return out

    def decoder(self, inputs):
        pass

    def generate_language(self):
        num_concepts = self.num_properties * self.types_per_property
        word_idx = torch.zeros(num_concepts, self.vocab_size, device=self.opts['device'])
        word_idx[:, :num_concepts] = torch.eye(num_concepts)
        word_idx = word_idx[torch.randperm(num_concepts, generator=self.torch_gen)][:,
                   torch.randperm(self.vocab_size, generator=self.torch_gen)]

        word_order = list(range(self.num_latents))
        self.random_gen.shuffle(word_order)
        word_order = torch.tensor(word_order, device=self.opts['device'])
        return word_idx, word_order

    def convert_to_latents(self, input):
        batch_size = input.shape[0]
        one_inds = torch.nonzero(input)[:, 1].reshape(batch_size, -1)
        perm_ones = one_inds.unsqueeze(-1)
        enc_out = torch.zeros(batch_size, self.num_latents, self.vocab_size,
                              device=self.opts['device']).scatter_(-1, perm_ones, 1)
        return enc_out
