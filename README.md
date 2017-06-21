# Machine Learning Papers TL;DR

# Contents
[General](#general)

[Reinforcement Learning](#reinforcement-learning)

[GANs](#gans)

# General
- [DSD: Dense-Sparse-Dense Training for Deep Neural Networks](https://arxiv.org/abs/1607.04381)
  - The authors propose a new method for training Deep Neural Networks.
    > In the first D (Dense) step, we train a dense network to learn connection weights and importance. In the S (Sparse) step, we regularize the network by pruning the unimportant connections with small weights and retraining the network given the sparsity constraint. In the final D (re-Dense) step, we increase the model capacity by removing the sparsity constraint, re-initialize the pruned parameters from zero and retrain the whole dense network.
    Simply put, Dense (1): learns the most important connections in the network. Sparse (2): uses a slim network to learn a complex problem, this means the representation of the network will concentrate on the most important features + this avoids over fitting. Dense (3): the network is already trained, we re-dense the network by returning the removed connections, now the network will perform fine turning around the learned model.
- [Machine Learning on Sequential Data Using a Recurrent Weighted Average](https://arxiv.org/abs/1703.01253)
  - Previous works have shown the power of the attention mechanism. This work proposes a similar approach which is easier to implement and which is trainable by simple gradient descent. Their approach takes a weighted average of the cell since the first time step - they show impressive results, especially in the training time.
- [Scalable and Sustainable Deep Learning via Randomized Hashing](https://arxiv.org/abs/1602.08194)
  - The authors build on the idea of [Adaptive Dropout](https://papers.nips.cc/paper/5032-adaptive-dropout-for-training-deep-neural-networks.pdf). In adaptive dropout, we sample which neurons to keep active based on a random bernouli variable and given the neuron activation value. Logically, similar inputs will provide similar activations. They exploit this by using [Locality-Sensitive Hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) which gives similar inputs similar hashed values. This allows keeping what neurons should be left active / the probability of each neuron to stay active - per input (or for N distinct inputs). Then the procedure is just hash -> find nearest neighbor -> dropout according to probabilities.

# Reinforcement Learning
- [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) - [website](https://blog.openai.com/evolution-strategies/)
  - Instead of running a single policy and optimizing it, we generate N random policies with a slight variance between them. We then evaluate them. Finally policies are combined as a weighted average where the weight is given with respect to the score in the evaluation. Combination is done at the weight level in the Neural Network.
  Advantage is a highly parallel solution, where evaluation of all policies can be done in parallel. Also this doesn't require backpropagation, which is a compute intensive solution.
- [Training Agent for First-Person Shooter Game with Actor-Critic Curriculum Learning](https://openreview.net/forum?id=Hk3mPK5gg)
  - They perform curriculum learning within ViZDoom, using the actor-critic approach.
  Their method uses the previous 4 images (no LSTM) as a state. Due to sparse reward they train the agent in domains with increasing difficulty (easy to hard), this makes sense since it is intially provided an easy to solve domain - and uses this knowledge to help it solve a harder domain once it is introduced. *Reward Shaping* - the authors note that they use reward shaping as another method to cope with sparse rewards, something to note is that reward shaping is a means of 'prior knowledge' or 'semi supervised' - it would be interesting see a solution with a more basic reward function like -X per step and +Y per kill.
- [Learning to Act by Predicting the Future](https://arxiv.org/abs/1611.01779)

# GANs
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
  - The authors propose to train the discriminator in a way such that the function it learns is the estimation of the Earth Mover (EM) distance (also known as Wasserstein distance). The benefit in this is that the normal GAN is trained to estimate the Jensen-Shannon (JS) distance, which isn't always defined and further more when trained to optimality - the discrimiators output looks like a 0-1 function (i.e gradients almost don't exist).
  The main benefit of the EM distance is that it is always defined (under certain conditions that are satisfied in current neural network architectures) and that the gradient always exists (doesn't go to zero as discriminator reaches optimality). Due to this, we actually benefit in training the discriminator to optimality.
  This is a remarkable paper with sound mathematical proof for convergence in neural networks.
  [This link](http://www.alexirpan.com/2017/02/22/wasserstein-gan.html) contains a nice explanation.
  [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) suggest further improvements. Weight clipping causes convergence issues, they suggest penalizing the norm of the gradient in order to keep it 1-Lipshitz + some other improvements.
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) (CycleGAN), [DiscoGAN](https://arxiv.org/abs/1703.05192), [DualGAN](https://arxiv.org/abs/1704.02510)
  - They propose a model that can convert an input from domain (a) to domain (b). Example 1. given an image of a *horse*, return the same image with the horse converted to a *zebra* (and the other way around). Example 2. given an *drawing*, return the image as if it was a *photograph* (and the other way around).
  Train two GANs (a) and (b). Discriminator (a) predicts if input is from domain (a). Generator (a) converts an input from domain (a) to domain (b). GAN (b) does the opposite direction.
  The main important contribution is the loss function. One loss is that generator (a) converts import x to B(x) that needs to fool discriminator (b). Second loss induced is that generator (b) needs to convert B(x) back to x.
- [Adversarial Neural Machine Translation](https://arxiv.org/abs/1704.06933) / [Improving Neural Machine Translation with Conditional Sequence Generative Adversarial Nets](https://arxiv.org/abs/1703.04887)
  - An interesting implementation of GAN. The discriminator receives both the input and the translated output. Supervised training.


# Non-summarized
- [Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car](https://arxiv.org/abs/1704.07911)
  - Last I checked, they perform deep imitation learning.
- [Going Wider: Recurrent Neural Network With Parallel Cells](https://arxiv.org/abs/1705.01346)
- [A New Type of Neurons for Machine Learning](https://arxiv.org/abs/1704.08362)
- [Parseval Networks: Improving Robustness to Adversarial Examples](https://arxiv.org/abs/1704.08847)
- [Visual Attribute Transfer through Deep Image Analogy](https://arxiv.org/abs/1705.01088)
- [Generative Adversarial Trainer: Defense to Adversarial Perturbations with GAN](https://arxiv.org/abs/1705.03387) + [Adversarial Transformation Networks: Learning to Generate Adversarial Examples](https://arxiv.org/pdf/1703.09387.pdf)
- [Model-based Adversarial Imitation Learning](https://arxiv.org/abs/1612.02179)
- [Real-Time Adaptive Image Compression](http://www.wave.one/icml2017)
- [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644)
- [Look, Listen and Learn](https://arxiv.org/abs/1705.08168)
- [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818) [code](https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589)
- [meProp: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting](https://arxiv.org/abs/1706.06197)
