# algo Goal / intro..............

- what is meta learning / learning to learn
- why do we want to L2L /generalise across multiple domains
- why do want do do it quickly

---

Humans are adept at solving tasks under many different conditions. This is partly due to to fast adaptation, but also due to a lifetime of encountering new task conditions providing the opportunity to develop strategies which are robust to different task contexts. 

We would like artificial learning agents to do the same because this would make them much more versatile and perform better 'out-the-box'.

# Algorithm

- meta steps for train and learn
- virtual train and test domains
- plaatje uit paper
- leuk

The meta learning algorithm used in the paper is designed to make the model more robust for domain shifts. Therefore Domain Generalization methods are used. Hence, the algorithm is called Meta-Learning Domain Generalization. The high-level pseudocode of the algorithm can be found in figure ......

![Blogpost%20skeleton/pseudocode.png](Blogpost%20skeleton/pseudocode.png)

As can be seen in line 2, the algorithm starts off by defining the domains S (i.e. Photo, Art painting, Cartoon, Sketch). Hereafter, the initial model parameters (Theta) and hyperparameters (Alpha, Beta, Gamma) are set. Line 4 denotes the start of the iterations. In each iteration, the training domain data are split in a meta-train set and a meta-test set as can be seen in figure ..... It should be clear that the meta-test set is composed out of training data and not test data. 

![Blogpost%20skeleton/fig1.png](Blogpost%20skeleton/fig1.png)

(Dit plaatje maar dan iets duidelijker)

Subsequently, the gradients for meta-train are calculated using the loss function (F). With this gradient, the proposed updated parameters can be calculated for the meta train set. Thus far, nothing new occurs except for splitting the domains in sets, when compared to normal backpropagation. However, in line 8 the loss function (G) is calculated for the meta-test set as well. In line 9, the updated parameters (Theta) are calculated based on the meta-train set loss function (F) and the meta-test set loss function (G) times a constant gamma. One can intuitively interpret this as the usual parameter calculation based on the train set gradients and loss function (F). However, these parameters are corrected by the loss function of the meta-test set (G). Therefore, the model will not overfit on a set of domains. 

# Experiment 2 repro

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

### Domains

- Photo
- Art painting
- Cartoon
- Sketch

    ![Blogpost%20skeleton/project_img1.png](Blogpost%20skeleton/project_img1.png)

The proposed MLDG algorithm is compared against 4 baseline models. These are shown below.

### Baselines

- D-MTAE
- Deep-all
- DSN
- AlexNet+TF

The results which are to be reproduced are shown in table 1. In this table, the accuracy of the considered baselines on the four different domains are shown as well as the results for the proposed MLDG algorithm. The goal of this reproducability project is to reproduce the accuracy of the proposed MLDG algorithm on the different domains (right column of table 1).

[Table 1]

![Blogpost%20skeleton/table_1.png](Blogpost%20skeleton/table_1.png)

# repro efforts

- [ ]  How did we tackle this challenge
    - [ ]  python 3
    - [ ]  pytorch rewrite
    - [ ]  remove numpy
    - [ ]  cuda handlers
- [ ]  strategy
- [ ]  methods used
- [ ]  expectations

# repro results

- [ ]  what accuracy did we achieve
- [ ]  how long did it train for
- [ ]  errors?
- [ ]  gpu temp

    These results were obtained by performing a single run with three iterations. The authors in the paper performed 5 runs with 3 iterations each. It is expected that increasing the runs will lead the model accuracy to converge towards the results in table 1.

    MLP: average test accuracy 69.5%

    ![Blogpost%20skeleton/mlp.png](Blogpost%20skeleton/mlp.png)

    MLDG: average test accuracy 71.7%

    ![Blogpost%20skeleton/mldg.png](Blogpost%20skeleton/mldg.png)

# repro discussion

- [ ]  compare to orig results
- [ ]  absolute results
- [ ]  Baseline is different from the baselines in table 1

# Efforts

---

1. omschrijven naar python 3
2. google colab stopte te vroeg (limited runtime)
3. Loop unrolling = loops opbreken voor zodat elke run los kan
4. plotten van meta-train loss en meta-validation loss → plots voor blog
5. 
6. 

- [ ]  pyTorch voor future work
- [ ]  paper checken voor hoeveel runs nodig zijn
    - [ ]  accuracy

1 run = 4 iteraties, elk domein is unseen geweest

k-fold crossvalidation

discussie
