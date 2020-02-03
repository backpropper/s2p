import argparse
import random
import os

import numpy as np
import copy
import torch

import utils as U
from models import CompositionalBot, CompEncoder, CompDecoder, CompEncoderDecoder


def main(opts):
    opts['cuda'] = torch.cuda.is_available()
    opts['device'] = torch.device('cuda' if opts['cuda'] else 'cpu')

    print("OPTS:\n", opts)

    if opts['num_compbot_samples_train'] == 0:
        opts['init_supervised_iters'] = 0
        opts['num_supervised_iters'] = 0

    random.seed(opts['seed'])
    np.random.seed(opts['seed'])
    torch.manual_seed(opts['seed'])

    comp_bot = CompositionalBot(opts, seed=opts['seed'])

    train_data, val_data, test_data = U.make_dataset(num_properties=opts['num_properties'],
                                                     types_per_property=opts['types_per_property'],
                                                     val_pct=opts['val_pct'],
                                                     test_pct=opts['test_pct'])

    n_samples_train = train_data[:opts['num_compbot_samples_train']]
    n_samples_val = val_data[:opts['num_compbot_samples_val']]
    n_samples_test = test_data[:opts['num_compbot_samples_test']]

    train_data = train_data[opts['num_compbot_samples_train']:]
    val_data = val_data[opts['num_compbot_samples_val']:]
    test_data = test_data[opts['num_compbot_samples_test']:]

    train_inds = U.get_batch_indices(train_data, opts['num_properties'], opts['types_per_property'],
                                   opts['device']).permute(1, 0)
    val_inds = U.get_batch_indices(val_data, opts['num_properties'], opts['types_per_property'],
                                           opts['device']).permute(1, 0)
    test_inds = U.get_batch_indices(test_data, opts['num_properties'], opts['types_per_property'],
                                           opts['device']).permute(1, 0)

    n_samples_train_inds = U.get_batch_indices(n_samples_train, opts['num_properties'], opts['types_per_property'],
                                           opts['device'])
    n_samples_val_inds = U.get_batch_indices(n_samples_val, opts['num_properties'], opts['types_per_property'],
                                         opts['device'])
    n_samples_test_inds = U.get_batch_indices(n_samples_test, opts['num_properties'], opts['types_per_property'],
                                             opts['device'])

    train_data = torch.tensor(train_data, device=opts['device'], dtype=torch.float)
    val_data = torch.tensor(val_data, device=opts['device'], dtype=torch.float)
    test_data = torch.tensor(test_data, device=opts['device'], dtype=torch.float)
    n_samples_train = torch.tensor(n_samples_train, device=opts['device'], dtype=torch.float)
    n_samples_val = torch.tensor(n_samples_val, device=opts['device'], dtype=torch.float)
    n_samples_test = torch.tensor(n_samples_test, device=opts['device'], dtype=torch.float)

    n_samples_train_words = comp_bot.encoder(n_samples_train)
    n_samples_val_words = comp_bot.encoder(n_samples_val)
    n_samples_test_words = comp_bot.encoder(n_samples_test)
    n_samples_train_target = torch.argmax(n_samples_train_words.permute(1, 0, 2), dim=-1)
    n_samples_val_target = torch.argmax(n_samples_val_words.permute(1, 0, 2), dim=-1)
    n_samples_test_target = torch.argmax(n_samples_test_words.permute(1, 0, 2), dim=-1)

    encdec_list = []
    for i in range(opts['num_encoders_train']):
        encdec_list.append(CompEncoderDecoder(opts).to(device=opts['device']))

    enc_acc = 0
    dec_acc = 0
    enc_acc_val = 0
    dec_acc_val = 0
    for it in range(opts['init_supervised_iters']):
        enc_acc_list = []
        dec_acc_list = []
        enc_acc_list_val = []
        dec_acc_list_val = []
        for j in range(opts['num_encoders_train']):
            enc_words = encdec_list[j].enc.encode(n_samples_train)
            encdec_list[j].enc.update(enc_words, n_samples_train_target)

            dec_out = encdec_list[j].dec.decode(n_samples_train_words)
            encdec_list[j].dec.update(dec_out, n_samples_train_inds)

            eacc_val = encdec_list[j].enc.test(n_samples_val, n_samples_val_target)
            eacc = encdec_list[j].enc.test(n_samples_train, n_samples_train_target)
            dacc_val = encdec_list[j].dec.test(n_samples_val_words, n_samples_val_inds)
            dacc = encdec_list[j].dec.test(n_samples_train_words, n_samples_train_inds)
            enc_acc_list_val.append(eacc_val)
            dec_acc_list_val.append(dacc_val)
            enc_acc_list.append(eacc)
            dec_acc_list.append(dacc)

        enc_acc = np.mean(enc_acc_list)
        dec_acc = np.mean(dec_acc_list)
        enc_acc_val = np.mean(enc_acc_list_val)
        dec_acc_val = np.mean(dec_acc_list_val)

        if (enc_acc > opts['max_val_acc']) and (dec_acc > opts['max_val_acc']):
            break

    encdec_acc_list = []
    for j in range(opts['num_encoders_train']):
        val_sample = random.sample(list(range(len(val_data))), opts['batch_size'])
        val_batch = val_data[val_sample]
        val_batch_inds = val_inds[val_sample].permute(1, 0)
        ed_acc = encdec_list[j].test(val_batch, val_batch_inds)
        encdec_acc_list.append(ed_acc)
    encdec_acc = np.mean(encdec_acc_list)

    print("\nInitial supervised stats:")
    print("Encoder Accuracy Train:", enc_acc, "Val:", enc_acc_val, "at", it, "iters")
    print("Decoder Accuracy Train:", dec_acc, "Val:", dec_acc_val, "at", it, "iters")
    print("EncDec Accuracy:", encdec_acc, "before self-play")
    print("\n Starting joint training")

    im_val_encdec_acc = encdec_acc
    enc_acc = 0
    dec_acc = 0
    encdec_acc = 0
    for it in range(opts['num_iters']):
        print("\nCombined iters", it)

        sp_encdec_acc = 0
        sp_val_encdec_acc = 0
        for fi in range(opts['num_selfplay_iters']):
            sp_encdec_acc_list = []
            val_encdec_acc_list = []
            for j in range(opts['num_encoders_train']):
                train_sample = random.sample(list(range(len(train_data))), opts['batch_size'])
                train_batch = train_data[train_sample]
                train_batch_inds = train_inds[train_sample].permute(1, 0)

                encdec_list[j].update(train_batch, train_batch_inds)

                sp_ed_acc = encdec_list[j].test(train_batch, train_batch_inds)
                sp_encdec_acc_list.append(sp_ed_acc)

                val_sample = random.sample(list(range(len(val_data))), opts['batch_size'])
                val_batch = val_data[val_sample]
                val_batch_inds = val_inds[val_sample].permute(1, 0)
                ed_acc = encdec_list[j].test(val_batch, val_batch_inds)
                val_encdec_acc_list.append(ed_acc)

            sp_encdec_acc = np.mean(sp_encdec_acc_list)
            sp_val_encdec_acc = np.mean(val_encdec_acc_list)

            if sp_val_encdec_acc > opts['max_val_acc']:
                break

        enc_acc_list = []
        dec_acc_list = []
        enc_acc_list_val = []
        dec_acc_list_val = []
        for j in range(opts['num_encoders_train']):
            eacc_val = encdec_list[j].enc.test(n_samples_val, n_samples_val_target)
            eacc = encdec_list[j].enc.test(n_samples_train, n_samples_train_target)
            dacc_val = encdec_list[j].dec.test(n_samples_val_words, n_samples_val_inds)
            dacc = encdec_list[j].dec.test(n_samples_train_words, n_samples_train_inds)
            enc_acc_list_val.append(eacc_val)
            dec_acc_list_val.append(dacc_val)
            enc_acc_list.append(eacc)
            dec_acc_list.append(dacc)

        sp_enc_acc = np.mean(enc_acc_list)
        sp_dec_acc = np.mean(dec_acc_list)
        sp_enc_acc_val = np.mean(enc_acc_list_val)
        sp_dec_acc_val = np.mean(dec_acc_list_val)

        print("\n After Self-Play stats:")
        print("Encoder Accuracy Train:", sp_enc_acc, "Val:", sp_enc_acc_val, "at", fi, "self-play iters")
        print("Decoder Accuracy Train:", sp_dec_acc, "Val:", sp_dec_acc_val, "at", fi, "self-play iters")
        print("EncDec Accuracy:", sp_val_encdec_acc, "at", fi, "self-play iters")

        im_enc_acc = 0
        im_dec_acc = 0
        im_enc_acc_val = 0
        im_dec_acc_val = 0
        for ii in range(opts['num_supervised_iters']):
            enc_acc_list = []
            dec_acc_list = []
            enc_acc_list_val = []
            dec_acc_list_val = []
            for j in range(opts['num_encoders_train']):
                enc_words = encdec_list[j].enc.encode(n_samples_train)
                encdec_list[j].enc.update(enc_words, n_samples_train_target)

                dec_out = encdec_list[j].dec.decode(n_samples_train_words)
                encdec_list[j].dec.update(dec_out, n_samples_train_inds)

                eacc_val = encdec_list[j].enc.test(n_samples_val, n_samples_val_target)
                eacc = encdec_list[j].enc.test(n_samples_train, n_samples_train_target)
                dacc_val = encdec_list[j].dec.test(n_samples_val_words, n_samples_val_inds)
                dacc = encdec_list[j].dec.test(n_samples_train_words, n_samples_train_inds)
                enc_acc_list_val.append(eacc_val)
                dec_acc_list_val.append(dacc_val)
                enc_acc_list.append(eacc)
                dec_acc_list.append(dacc)

            im_enc_acc = np.mean(enc_acc_list)
            im_dec_acc = np.mean(dec_acc_list)
            im_enc_acc_val = np.mean(enc_acc_list_val)
            im_dec_acc_val = np.mean(dec_acc_list_val)

            if (im_enc_acc > opts['max_val_acc']) and (im_dec_acc > opts['max_val_acc']):
                break

        val_encdec_acc_list = []
        for j in range(opts['num_encoders_train']):
            val_sample = random.sample(list(range(len(val_data))), opts['batch_size'])
            val_batch = val_data[val_sample]
            val_batch_inds = val_inds[val_sample].permute(1, 0)
            ed_acc = encdec_list[j].test(val_batch, val_batch_inds)
            val_encdec_acc_list.append(ed_acc)

        im_val_encdec_acc = np.mean(val_encdec_acc_list)

        print("\n After Supervised stats:")
        print("Encoder Accuracy Train:", im_enc_acc, "Val:", im_enc_acc_val, "at", ii, "supervised iters")
        print("Decoder Accuracy Train:", im_dec_acc, "Val:", im_dec_acc_val, "at", ii, "supervised iters")
        print("EncDec Accuracy:", im_val_encdec_acc, "at", ii, "supervised iters")

        if it % opts['test_interval'] == 0:
            enc_acc_list = []
            dec_acc_list = []
            encdec_acc_list = []
            for j in range(opts['num_encoders_train']):
                eacc = encdec_list[j].enc.test(n_samples_test, n_samples_test_target)
                dacc = encdec_list[j].dec.test(n_samples_test_words, n_samples_test_inds)
                test_sample = random.sample(list(range(len(test_data))), opts['batch_size'])
                test_batch = test_data[test_sample]
                test_batch_inds = test_inds[test_sample].permute(1, 0)
                ed_acc = encdec_list[j].test(test_batch, test_batch_inds)

                enc_acc_list.append(eacc)
                dec_acc_list.append(dacc)
                encdec_acc_list.append(ed_acc)

            enc_acc = np.mean(enc_acc_list)
            dec_acc = np.mean(dec_acc_list)
            encdec_acc = np.mean(encdec_acc_list)

            print(f'\n\n ###### At global iteration {it} Test stats: #######')
            print(f'Encoder accuracy on {opts["num_compbot_samples_test"]} samples is {enc_acc}')
            print(f'Decoder accuracy on {opts["num_compbot_samples_test"]} samples is {dec_acc}')
            print(f'Joint accuracy on test set is {encdec_acc}')

        if (sp_enc_acc > opts['max_val_acc']) and (sp_dec_acc > opts['max_val_acc']) and (im_val_encdec_acc > opts['max_val_acc']):
            break

    if opts['save_dir'] != '':
        for jk in range(opts['num_encoders_train']):
            U.torch_save_to_file(encdec_list[jk].enc.state_dict(),
                                 folder=os.path.join(opts['save_dir'], 'params', 'enc_params'),
                                 file=f"_pop{jk}_params.pt")
            U.torch_save_to_file(encdec_list[jk].dec.state_dict(),
                                 folder=os.path.join(opts['save_dir'], 'params', 'dec_params'),
                                 file=f"_pop{jk}_params.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-pct', type=float, default=0.1)
    parser.add_argument('--val-pct', type=float, default=0.1)
    parser.add_argument('--num-compbot-samples-train', type=int, default=0)
    parser.add_argument('--num-compbot-samples-val', type=int, default=1000)
    parser.add_argument('--num-compbot-samples-test', type=int, default=1000)
    parser.add_argument('--init-supervised-iters', type=int, default=1)
    parser.add_argument('--num-selfplay-iters', type=int, default=5)
    parser.add_argument('--num-supervised-iters', type=int, default=1)
    parser.add_argument('--num-iters', type=int, default=100)
    parser.add_argument('--max-val-acc', type=float, default=0.95)
    parser.add_argument('--num-encoders-train', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--temp', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--num-properties', type=int, default=6)
    parser.add_argument('--types-per-property', type=int, default=10)
    parser.add_argument('--num-latent-variables', type=int, default=6)
    parser.add_argument('--output-size', type=int, default=60)
    parser.add_argument('--embedding-size-encoder', type=int, default=500)
    parser.add_argument('--embedding-size-decoder', type=int, default=500)
    parser.add_argument('--test-interval', type=int, default=10)
    parser.add_argument('--save-dir', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)

    hyperparams = parser.parse_args()
    opts = vars(hyperparams)

    main(opts)
