On the interaction between supervision and self-play in emergent communication
==================================
Code repository of the models described in the paper accepted at ICLR 2020 
[On the interaction between supervision and self-play in emergent communication](https://openreview.net/pdf?id=rJxGLlBtwH "On the interaction between supervision and self-play in emergent communication").

Dependencies
------------------
### Python
* Python>=3.6
* PyTorch>=1.2

### GPU
* CUDA>=10.1
* cuDNN>=7.6


Downloading features for IBR game
-------
The preprocssed feature files for MS COCO images/captions can be downloaded from [here](https://drive.google.com/file/d/13m3SOqJkei8ozRwFa2y4cn3WzRqjpx5S/view?usp=sharing). The image features are obtained through a pretrained Resnet-50 model where each feature is of dimension 2048.
Extract the zip file and place the folder inside the `ibr_game` folder.



OR game
------------------
Change the directory to `or_game`.
```bash
$ cd or_game
```

#### Training a population of agents using `sched` S2P
```bash
$ python train.py --num-compbot-samples-train 1000 --init-supervised-iters 5 --num-selfplay-iters 20 --num-supervised-iters 5 --num-iters 100 --num-encoders-train 50
```
where `num-compbot-samples-train` is the size of seed dataset $\mathcal{D}$, `num-selfplay-iters` is $l$, `num-supervision-iters` is $m$, and `num-encoders-train` is the number of agents in the population.


IBR game
------------------
Change the directory to `ibr_game`.

```bash
$ cd ibr_game
```
#### Training a single set of agents using `sched` S2P
```bash
$ python train.py --num_distrs 9 --num_seed_examples 10000 --s2p_schedule sched --s2p_selfplay_updates 50 --s2p_spk_updates 50 --s2p_list_updates 50 --min_list_steps 2000 --min_spk_steps 1000 --max_iters 300
```
where `num_distrs` is the total number of distractors $D$, `num_seed_examples` is the size of seed dataset $\mathcal{D}$, `s2p_schedule` is the type of S2P, `s2p_selfplay_updates` is $l$, and `s2p_list_updates` is $m$.

#### Finetune listener over the whole seed data
```bash
$ python finetune.py --num_seed_examples 1000 --num_total_seed_samples 10000 --num_distrs 9  ----trainpop_files <TRAINPOP_FILES>
```
where `num_seed_examples` is the number of train samples in the seed dataset $\mathcal{D}_{train}$, `num_total_seed_samples` is the size of the whole seed dataset $\mathcal{D}$, and `<TRAINPOP_FILES>` is the path to the directory where the listener parameters are stored.

Dataset & Related Code Attribution
------------------
* MS COCO is licensed under Creative Commons.
* This project is licensed under the terms of the MIT license.



Citation
------------------
If you find the resources in this repository useful, please consider citing:
```
@inproceedings{lowe*2020on,
    title = {On the interaction between supervision and self-play in emergent communication},
    author = {Ryan Lowe* and Abhinav Gupta* and Jakob Foerster and Douwe Kiela and Joelle Pineau},
    booktitle = {International Conference on Learning Representations},
    year = {2020},
    url = {https://openreview.net/forum?id=rJxGLlBtwH}
}
```