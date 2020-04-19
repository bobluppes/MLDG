# Blogpost DL meta learning

bob [[Github](https://github.com/Bobr4y)]

wouter [[Github](https://github.com/h0uter)]

mats 

# Introduction
---
This blog post is part of the TU Delft Deep Learning Course. This blogpost will detail our process of reproducing the experiments and results from the paper "Learning to Generalize: Meta-Learning for Domain Generalization", [https://arxiv.org/abs/1710.03463](https://arxiv.org/abs/1710.03463) while observing and improving upon their available code.

Humans are adept at solving tasks under many different conditions. This is partly due to to fast adaptation, but also due to a lifetime of encountering new task conditions providing the opportunity to develop strategies which are robust to different task contexts. 

We would like artificial learning agents to do the same because this would make them much more versatile and perform better 'out-the-box'.

This paper proposes a novel meta learning approach for domain generalisation rather than proposing a specific model suited for DG. 

# Algorithm

---

The meta learning algorithm used in the paper is designed to make the model more robust for domain shifts. Therefore Domain Generalization methods are used. Hence, the algorithm is called Meta-Learning Domain Generalization. The high-level pseudocode of the algorithm can be found in Figure 1

![](https://i.imgur.com/VNZicw9.png)

Figure 1: Pseudocode Algorithm MLDG

As can be seen in line 2, the algorithm starts off by defining the domains S (i.e. Photo, Art painting, Cartoon, Sketch). Hereafter, the initial model parameters (Theta) and hyperparameters (Alpha, Beta, Gamma) are set. Line 4 denotes the start of the iterations. In each iteration, the training domain data are split in a meta-train set and a meta-test set as can be seen in Figure 2. It should be clear that the meta-test set is composed out of training data and not test data. 

![](https://i.imgur.com/shaKYDi.jpg)

Figure 2: Splitting the trainings domains

Subsequently, the gradients for meta-train are calculated using the loss function (F). With this gradient, the proposed updated parameters can be calculated for the meta train set. Thus far, nothing new occurs except for splitting the domains in sets, when compared to normal backpropagation. However, in line 8 the loss function (G) is calculated for the meta-test set as well. In line 9, the updated parameters (Theta) are calculated based on the meta-train set loss function (F) and the meta-test set loss function (G) times a constant gamma. One can intuitively interpret this as the usual parameter calculation based on the train set gradients and loss function (F). However, these parameters are corrected by the loss function of the meta-test set (G). Therefore, the model will not overfit on a set of domains. 

# Experiment to reproduce

---

From the paper, experiment 2 regarding object recognition had to be reproduced. The goal of this experiment is to recognize objects in one domain, while training the model in another. The goal is to obtain a domain-invariant feature representation.

For this experiment, the PACS multi-domain recognition benchmark was used. This dataset is designed specifically for cross-domain recognition problems (Li et al. 2017). The dataset contains 9991 images across 7 different categories spread over 4 domains. These categories and domains are listed below.

### Categories

- Dog
- Elephant
- Giraffe
- Guitar
- House
- Horse
- Person
- Sketch

### Domains

- Photo
- Art painting
- Cartoon


![](https://i.imgur.com/XOmolUK.png)

Figure 3: PACS (Domains S)

The proposed MLDG algorithm is compared against 4 baseline models. These are shown below.

### Baselines

- D-MTAE
- Deep-all
- DSN
- AlexNet+TF

The results which are to be reproduced are shown in table 1. In this table, the accuracy of the considered baselines on the four different domains are shown as well as the results for the proposed MLDG algorithm. The goal of this reproducability project is to reproduce the accuracy of the proposed MLDG algorithm on the different domains (right column of table 1).

![](https://i.imgur.com/80yZ6ih.png)


# Reproduction

---

## Understanding the code

there are 4 main files

1. `main_baseline.py`
2. `main_mldg.py`
3. `model.py`
4. `MLP.py`

run_baseline.sh (wat is een run/iteration)

run_mldg.sh

![](https://i.imgur.com/N82CWi8.jpg)

Dependencies

Lets start at the end and then walk our way backwards. The networks are ran from their respective `run_XXX.sh` run files. Here the following is specified:

- the learning rate lr
- test frequency: test_every
- batch_size
- the index of which domain should be kept 'unseen': unseen_index
- The number of training loops: inner_loops
- the step size: step_size


for MLDG only:

- the step size of the meta-learn step: meta_step_size
- the value for the hyper parameter beta: meta_val_beta

The models are defined in main_XXX.py by initialising their respective classes with a rather long list of parameters. These parameters include:

- 
-
-

Both classes are defined in `model.py`.
MLDG inherits it's initialization from the baseline.



## Efforts

First we dived into the existing code and rewrote it to python 3.

The first challenge we encountered was that google Colab stopped running too early to complete an entire run. Within each run crossvalidation is applied and thus several iterations are ran. To solve this issue we unwound the run loop so we could run each iteration separately. 

This allowed us to plot the meta-train and meta-test loss for each iteration within the run as can be seen in the results.



| Hyperparameter | Value | Specified in paper |
| -------- | -------- | -------- |
| Text     | Text     | Text     |


## Results

Using the code as described above (see [here](https://github.com/h0uter/MLDG) for the full repository), an attempt was made to reproduce the results from table 1.  A baseline (MLP) and the MLDG model are trained on the extracted features of the PACS dataset. The accuracy of both models for the four different domains can be seen below in table 2.


| Domain | MLP | MLDG |
| -------- | -------- | -------- |
| art_painting     | 69.27     | 71.75     |
| cartoon     | 52.01     | 48.30     |
| photo     | 89.61     | 95.74     |
| sketch     | 33.72     | 38.94     |
| Ave.     | 61.15     | 63.68     |
*Table 2*


These results were obtained by performing three independent runs and averaging the resulting accuracy per domain.

The comparison of the reproduced MLDG results against the original MLDG results from table 1 can be seen in table 3.

| Domain | Original | Reproduced | Difference (%) |
| -------- | -------- | -------- | -------- |
| art_painting     | 66.23     | 71.75     | +8.33
| cartoon     | 66.88     | 48.30     | -27.8
| photo     | 88.00     | 95.74     | +8.79
| sketch     | 57.51     | 38.94     | -32.3
| Ave.     | 70.01     | 63.68     |
*Table 3*

It can be argued that these reproduced accuracies improve when more independent runs are performed. In order to investigate this, a 95% confidence interval is calculated for the average accuracy of the reproduced MLDG model. This confidence interval is given by

    0.637, 95% CI [0.635, 0.639]

Apart from reproducing the results in table 1, the meta train loss and metal validation loss were investigated as a function of iteration in order to get a better understanding of the MLDG algorithm.

![](https://i.imgur.com/Xa9S5Wx.png)

photo domain

![](https://i.imgur.com/Wlvdgdr.png)

sketch domain


# Discussion
The first thing to notice is that the reproduced baseline model (MLP) is different from the baseline models in table 1. This is not vital for our reproducability project since the baseline results are not part of our reproducability goal (right most column in table 1). However, because the model is different, no clear comparison between the baselines can be made. The MLP baseline can however be used to compare against the reproduced MLDG results.

By comparing tables 1 and 2, it becomes clear that domains which were considered easy or difficult to meta learn remain the same in the reproduction. The sketch domain receives the lowest accuracy scores in both table 1 and 2, while the photo domain receives the highest scores.

While the ranking of easy and difficult domains to meta learn remain the same in the reproduction project, the achieved accuracy does not. In table 3 it can be seen that the accuracy of the reproduced MLDG model on the different domain is significantly different from the original results.



### **Future Work**

The code uses numpy whereas pyTorch might be a better candidate because it is capable of harnessing the power of the GPU, especially NVIDIA GPU's combined with CUDA parallel processing. 

## References

- [Li et al. 2017] Li, D.; Yang, Y.; Song, Y.-Z.; and
Hospedales, T. 2017. Deeper, broader and artier domain
generalization. In ICCV.
-