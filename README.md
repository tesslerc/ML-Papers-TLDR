# Machine Learning Papers TLDR

# Contents
[General](#General)

[Reinforcement Learning](#Reinforcement)

[GANs](#GANs)

### General
- [DSD: Dense-Sparse-Dense Training for Deep Neural Networks](https://arxiv.org/abs/1607.04381)
  - The authors propose a new method for training Deep Neural Networks.
    > In the first D (Dense) step, we train a dense network to learn connection weights and importance. In the S (Sparse) step, we regularize the network by pruning the unimportant connections with small weights and retraining the network given the sparsity constraint. In the final D (re-Dense) step, we increase the model capacity by removing the sparsity constraint, re-initialize the pruned parameters from zero and retrain the whole dense network.

    Simply put, Dense (1): learns the most important connections in the network. Sparse (2): uses a slim network to learn a complex problem, this means the representation of the network will concentrate on the most important features + this avoids over fitting. Dense (3): the network is already trained, we re-dense the network by returning the removed connections, now the network will perform fine turning around the learned model.

### Reinforcement Learning
- [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) - [website](https://blog.openai.com/evolution-strategies/)
  - Instead of running a single policy and optimizing it, we generate N random policies with a slight variance between them. We then evaluate them. Finally policies are combined as a weighted average where the weight is given with respect to the score in the evaluation. Combination is done at the weight level in the Neural Network.
  Advantage is a highly parallel solution, where evaluation of all policies can be done in parallel. Also this doesn't require backpropagation, which is a compute intensive solution.
- [Training Agent for First-Person Shooter Game with Actor-Critic Curriculum Learning](https://openreview.net/forum?id=Hk3mPK5gg)
  - They perform curiculim learning within ViZDoom, using the actor-critic approach.
  Their method uses the previous 4 images (no LSTM) as a state. Due to sparse reward they train the agent in domains with increasing difficulty (easy to hard), this makes sense since it is intially provided an easy to solve domain - and uses this knowledge to help it solve a harder domain once it is introduced. *Reward Shaping* - the authors note that they use reward shaping as another method to cope with sparse rewards, something to note is that reward shaping is a means of 'prior knowledge' or 'semi supervised' - it would be interesting see a solution with a more basic reward function like -X per step and +Y per kill.


### GANs
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) (CycleGAN)
  - They propose a model that can convert an input from domain (a) to domain (b). Example 1. given an image of a *horse*, return the same image with the horse converted to a *zebra* (and the other way around). Example 2. given an *drawing*, return the image as if it was a *photograph* (and the other way around).
  Train two GANs (a) and (b). Discriminator (a) predicts if input is from domain (a). Generator (a) converts an input from domain (a) to domain (b). GAN (b) does the opposite direction.
  The main important contribution is the loss function. One loss is that generator (a) converts import x to B(x) that needs to fool discriminator (b). Second loss induced is that generator (b) needs to convert B(x) back to x.
