# Blogpost DL "Learning to Generalize: Meta-Learning for Domain Generalization"

bob [[Github](https://github.com/Bobr4y)]

mats

wouter [[Github](https://github.com/h0uter)]

# Introduction

---

This blog post is part of the TU Delft Deep Learning Course. This blogpost will detail our process of reproducing the experiments and results from the paper "Learning to Generalize: Meta-Learning for Domain Generalization", [https://arxiv.org/abs/1710.03463](https://arxiv.org/abs/1710.03463) while observing and improving upon their available code.

Humans are adept at solving tasks under many different conditions. This is partly due to to fast adaptation, but also due to a lifetime of encountering new task conditions providing the opportunity to develop strategies which are robust to different task contexts. 

We would like artificial learning agents to do the same because this would make them much more versatile and perform better 'out-the-box'.

This paper proposes a novel meta learning approach for domain generalisation rather than proposing a specific model suited for DG. 

# Algorithm

---

The meta learning algorithm used in the paper is designed to make the model more robust for domain shifts. Therefore Domain Generalization methods are used. Hence, the algorithm is called Meta-Learning Domain Generalization. The high-level pseudocode of the algorithm can be found in figure ......

![Blogpost%20DL%20Learning%20to%20Generalize%20Meta%20Learning%20f/pseudocode.png](Blogpost%20DL%20Learning%20to%20Generalize%20Meta%20Learning%20f/pseudocode.png)

As can be seen in line 2, the algorithm starts off by defining the domains S (i.e. Photo, Art painting, Cartoon, Sketch). Hereafter, the initial model parameters (Theta) and hyperparameters (Alpha, Beta, Gamma) are set. Line 4 denotes the start of the iterations. In each iteration, the training domain data are split in a meta-train set and a meta-test set as can be seen in figure ..... It should be clear that the meta-test set is composed out of training data and not test data. 

![Blogpost%20DL%20Learning%20to%20Generalize%20Meta%20Learning%20f/fig1.png](Blogpost%20DL%20Learning%20to%20Generalize%20Meta%20Learning%20f/fig1.png)

(Dit plaatje maar dan iets duidelijker)

Subsequently, the gradients for meta-train are calculated using the loss function (F). With this gradient, the proposed updated parameters can be calculated for the meta train set. Thus far, nothing new occurs except for splitting the domains in sets, when compared to normal backpropagation. However, in line 8 the loss function (G) is calculated for the meta-test set as well. In line 9, the updated parameters (Theta) are calculated based on the meta-train set loss function (F) and the meta-test set loss function (G) times a constant gamma. One can intuitively interpret this as the usual parameter calculation based on the train set gradients and loss function (F). However, these parameters are corrected by the loss function of the meta-test set (G). Therefore, the model will not overfit on a set of domains. 

# Experiment to reproduce

---

From the paper, experiment 2 regarding object recognition had to be reproduced. The goal of this experiment is to recognize objection in one domain, while only train in another domain.

For this experiment, the PACS multi-domain recognition benchmark was used. This is a dataset designed specifically for cross-domain recognition problems (Li et al. 2017). The dataset contains 9991 images across 7 different categories spread over 4 domains. These categories and domains are listed below.

### Categories

- Dog
- Elephant
- Giraffe
- Guitar
- House
- Horse
- Person
- Sketch

    ![Blogpost%20DL%20Learning%20to%20Generalize%20Meta%20Learning%20f/project_img1.png](Blogpost%20DL%20Learning%20to%20Generalize%20Meta%20Learning%20f/project_img1.png)

### Domains

- Photo
- Art painting
- Cartoon

The proposed MLDG algorithm is compared against 4 baseline models. These are shown below.

### Baselines

- D-MTAE
- Deep-all
- DSN
- AlexNet+TF

The results which are to be reproduced are shown in table 1. In this table, the accuracy of the considered baselines on the four different domains are shown as well as the results for the proposed MLDG algorithm. The goal of this reproducability project is to reproduce the accuracy of the proposed MLDG algorithm on the different domains (right column of table 1).

![Blogpost%20DL%20Learning%20to%20Generalize%20Meta%20Learning%20f/table_1.png](Blogpost%20DL%20Learning%20to%20Generalize%20Meta%20Learning%20f/table_1.png)

# Reproduction

---

## Efforts

First we dived into the existing code and rewrote it to python 3.

The first challenge we encountered was that google Colab stopped running too early to complete an entire run. Within each run crossvalidation is applied and thus several iterations are ran. To solve this issue we unwound the run loop so we could run each iteration separately. 

This allowed us to plot the meta-train and meta-test loss for each iteration within the run as can be seen in the results.

## Understanding the code

## Results

[Untitled](https://www.notion.so/929ed28f2b6b46db95c05c4c6987e647)

- [ ]  what accuracy did we achieve
- [ ]  how long did it train for
- [ ]  errors?
- [ ]  gpu temp

    These results were obtained by performing a single run with three iterations. The authors in the paper performed 5 runs with 3 iterations each. It is expected that increasing the runs will lead the model accuracy to converge towards the results in table 1.

    MLP: average test accuracy 69.5%

    ![Blogpost%20DL%20Learning%20to%20Generalize%20Meta%20Learning%20f/mlp.png](Blogpost%20DL%20Learning%20to%20Generalize%20Meta%20Learning%20f/mlp.png)

    MLDG: average test accuracy 71.7%

    ![Blogpost%20DL%20Learning%20to%20Generalize%20Meta%20Learning%20f/mldg.png](Blogpost%20DL%20Learning%20to%20Generalize%20Meta%20Learning%20f/mldg.png)

# Discussion

- [ ]  compare to orig results
- [ ]  absolute results
- [ ]  Baseline is different from the baselines in table 1

- [ ]  paper checken voor hoeveel runs nodig zijn
    - [ ]  accuracy

### **Future Work**

The code uses numpy whereas pyTorch might be a better candidate because it is capable of harnessing the power of the GPU

## References

-