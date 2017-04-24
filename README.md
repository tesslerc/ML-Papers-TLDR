# Machine Learning Papers TLDR

# Contents
[General](#General)

### General
- [DSD: Dense-Sparse-Dense Training for Deep Neural Networks](https://arxiv.org/abs/1607.04381)
  - The authors propose a new method for training Deep Neural Networks.
    > In the first D (Dense) step, we train a dense network to learn connection weights and importance. In the S (Sparse) step, we regularize the network by pruning the unimportant connections with small weights and retraining the network given the sparsity constraint. In the final D (re-Dense) step, we increase the model capacity by removing the sparsity constraint, re-initialize the pruned parameters from zero and retrain the whole dense network.

    Simply put, the initial stage: learns the most important connections in the network. The second stage: uses a slim network to learn a complex problem, this means the representation of the network will concentrate on the most important features + this avoids over fitting. The final stage: the network is already trained, we re-dense the network by returning the removed connections, now the network will perform fine turning around the learned model.
