# Machine Learning Papers TLDR

# Contents
[General](#General)
[Reinforcement Learning](#Reinforcement Learning)
[GANs](#GANs)

### General
- [DSD: Dense-Sparse-Dense Training for Deep Neural Networks](https://arxiv.org/abs/1607.04381)
  - The authors propose a new method for training Deep Neural Networks.
    > In the first D (Dense) step, we train a dense network to learn connection weights and importance. In the S (Sparse) step, we regularize the network by pruning the unimportant connections with small weights and retraining the network given the sparsity constraint. In the final D (re-Dense) step, we increase the model capacity by removing the sparsity constraint, re-initialize the pruned parameters from zero and retrain the whole dense network.

    Simply put, Dense (1): learns the most important connections in the network. Sparse (2): uses a slim network to learn a complex problem, this means the representation of the network will concentrate on the most important features + this avoids over fitting. Dense (3): the network is already trained, we re-dense the network by returning the removed connections, now the network will perform fine turning around the learned model.

### Reinforcement Learning
- [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) [website](https://blog.openai.com/evolution-strategies/)
  - Instead of running a single policy and optimizing it, we generate N random policies with a slight variance between them. We then evaluate them. Finally policies are combined as a weighted average where the weight is given with respect to the score in the evaluation. Combination is done at the weight level in the Neural Network.
  Advantage is a highly parallel solution, where evaluation of all policies can be done in parallel. Also this doesn't require backpropagation, which is a compute intensive solution.


### GANs
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) (CycleGAN)
  - Train two GANs (a) and (b). Discriminator (a) predicts if input is from domain (a). Generator (a) converts an input from domain (a) to domain (b). GAN (b) does the opposite direction.
  The main important contribution is the loss function. One loss is that generator (a) converts import x to B(x) that needs to fool discriminator (b). Second loss induced is that generator (b) needs to convert B(x) back to x.
