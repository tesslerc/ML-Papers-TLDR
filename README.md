# Machine Learning Papers TL;DR

Papers are removed on a regular basis, based on how novel I feel they are.
Feel free to send push requests with your own paper summaries. The more people that contribute, the better this TL;DR page can become.

# Contents
[Concepts](#concepts)

[General](#general)

[Reinforcement Learning](#reinforcement-learning)

[GANs](#gans)

[Natural Language Processing](#natural-language-processing)

# Concepts
This section contains links to informative blogs regarding basic concepts in machine learning literature.

- [Reinforcement Learning](https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419)
- [VAE - Variational Auto-Encoder](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)
- [GAN - Generative Adversarial Network](https://medium.freecodecamp.org/an-intuitive-introduction-to-generative-adversarial-networks-gans-7a2264a81394)
- [LSTM - Long-Short term memory networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - a form of Recurrent neural networks.
- [Attention mechanism](https://distill.pub/2016/augmented-rnns/)

# General
## Theory
- [Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks](https://arxiv.org/abs/1806.05393)
  - They develop a new initialization scheme which allows the gradients to properly propagate through extremely deep networks.
- [ResNet with one-neuron hidden layers is a Universal Approximator](https://arxiv.org/abs/1806.10909)

## Application
- [DSD: Dense-Sparse-Dense Training for Deep Neural Networks](https://arxiv.org/abs/1607.04381)
  - The authors propose a new method for training Deep Neural Networks.
    > In the first D (Dense) step, we train a dense network to learn connection weights and importance. In the S (Sparse) step, we regularize the network by pruning the unimportant connections with small weights and retraining the network given the sparsity constraint. In the final D (re-Dense) step, we increase the model capacity by removing the sparsity constraint, re-initialize the pruned parameters from zero and retrain the whole dense network.
    Simply put, Dense (1): learns the most important connections in the network. Sparse (2): uses a slim network to learn a complex problem, this means the representation of the network will concentrate on the most important features + this avoids over fitting. Dense (3): the network is already trained, we re-dense the network by returning the removed connections, now the network will perform fine turning around the learned model.
- [Machine Learning on Sequential Data Using a Recurrent Weighted Average](https://arxiv.org/abs/1703.01253)
  - Previous works have shown the power of the attention mechanism. This work proposes a similar approach which is easier to implement and which is trainable by simple gradient descent. Their approach takes a weighted average of the cell since the first time step - they show impressive results, especially in the training time.
- [Safer Classification by Synthesis](https://arxiv.org/abs/1711.08534)
  - Discriminative classification models can easily be fooled to classify incorrectly with high confidence by out-of-distribution examples - this in itself can be hazardous in real world applications [(Adversarial examples in the physical world)](https://arxiv.org/abs/1607.02533).
  - The authors propose to overcome this by using a generative approach – for a given test image, find the latent variable that produces it in generators trained for each class. Then, choose the one with the highest confidence (i.e lowest error compared to the original image).
  - Intuitive error measures are for instance the MSE, we can define a threshold underwhich the generator is considered unconfident. If all classes are 'unconfident' we would prefer not to classify over being wrong with a high proability.
  - This is a general approach – and can be used with VAEs or GANs, or even with a k-Means generative model.
  - The paper shows competitive results for in-distribution test images, and very good results on out-of-distribution images (meaning, the model knows to reject the examples rather than misclassify them).
- [Tracking Emerges by Colorizing Videos](https://arxiv.org/abs/1806.09594) - [blog](https://ai.googleblog.com/2018/06/self-supervised-tracking-via-video.html)
  -  They train a network to colorize a grayscale video based on a colored initial frame. They find that this method allows for object tracking by coloring the initial object.
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501) - [blog](https://ai.googleblog.com/2018/06/improving-deep-learning-performance.html)
  - Using reinforcement learning techniques, the system learns how to optimally augment the data such that the network will generalize best. They achieve state of the art (83.54% on top1 accuracy on ImageNet) which is close to the results from [Facebook](https://code.fb.com/applied-machine-learning/advancing-state-of-the-art-image-recognition-with-deep-learning-on-hashtags/) in which they used the whole of Instagram as pre-training data.
- [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829) - [blog](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b)
  - Capsules introduce a new building block that can be used in deep learning to better model hierarchical relationships inside of internal knowledge representation of a neural network.

# Reinforcement Learning
## General
- [*(Draft)* Reinforcement Learning: An Introduction, Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/book/the-book-2nd.html)
  - The latest *draft* of the RL intro book, which provides an overview over many of the topics in the field.
- [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560)
  - The authors in this paper bring to light a large problem with Deep RL. I believe that **reading this paper is a requirement to anyone intending to work with Deep RL**, to properly understand the past / current issues with the research. The problems the authors bring up, are backed by extensive tests to prove their claims. The issues are regarding methods of reporting the results and model evaluation, which are inconsistent between papers and domains and in some cases are bad practice.
- [An Outsider's Tour of Reinforcement Learning](http://www.argmin.net/2018/06/25/outsider-rl/)
  - An excellent series of blog posts by Ben Recht in which he tackles many misconceptions in recent RL progress.
- [Measuring the Intrinsic Dimension of Objective Landscapes](https://arxiv.org/abs/1804.08838) - [blog](https://eng.uber.com/intrinsic-dimension/)
  - In the paper, they develop intrinsic dimension as a quantification of the complexity of a model in a manner decoupled from its raw parameter count, and provide a simple way of measuring this dimension using random projections.
- [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)
  - On each episode the agent receives a task (e.g. a location to reach). As the reward is sparse and there is no prior knowledge, learning such tasks is near impossible. This is overcome by using a hindsight experience replay which modifies the goal given for each trajectory such that the trajectory reaches the goal. (this can be seen as a human-like approach in which we learn from trial and error by observing the consequences of our actions).

## Value based methods
- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887), Follow up paper [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044), [Blog](https://flyyufelix.github.io/2017/10/24/distributional-bellman.html)
  - The output of the network is a the probability distribution (in the first paper) or a cummulative distribution (the second paper). They train the network to learn the distribution of the Q values. The problems they test are fully deterministic, hence the randomness which is learned can be from either the noise induced in the neural network (since it is an approximation, as opposed to a tabular representation, the value keeps changing) or from the boostrap (the policy keeps changing which adds additional noise).
- [The Uncertainty Bellman Equation and Exploration](https://arxiv.org/abs/1709.05380v1)
- [Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397) [Blog](https://deepmind.com/blog/reinforcement-learning-unsupervised-auxiliary-tasks/)
  - The authors propose a new archiecture named UNREAL (UNsupervised REinforcement and Auxiliary Learning), which combines the learning of auxiliary tasks to help the agent in learning the main task. They present two auxiliary tasks: (1) Learning to predict how your actions will change the environment i.e. predict s<sub>t+1</sub> given (s<sub>t</sub>, a<sub>t</sub>) (2) Predict the immediate reward, similar to learning the value function with a discount factor equal to 0. These prediction heads are connected to the feature extraction layer hence they do not interfere (bias) with the policy or the value prediction, but they do help shape the feature extraction layers to facilitate faster learning.
- [Many-Goals Reinforcement Learning](https://arxiv.org/abs/1806.09605)
  - Many-goals is a task of learning multiple goals (reward signals) at once by exploiting the off-policy nature of the Q-learning algorithm. In this work they present several algorithms for many-goal learning and in addition, they show that many-goal learning can be used as an auxillary task.
- [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868)
  - By using a density network they predict how many times a state has been seen (the pseudo-count). The Q values are then updated by providing an intrinstic motivating reward which decays monotonically in the pseudo-count.

## Policy based methods
- [Asynchronous Methods for Deep Reinforcement Learning (A3C)](https://arxiv.org/abs/1602.01783) - [blog](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
  - A3C is an asyncronous version of Advantage-Actor-Critic (A2C). A2C is a policy gradient algorithm in which the value function is subtracted from the reward (the agent attempts to maximize the advantage), in these algorithms the gradient is estimated by running multiple agents at once. In A3C the asyncronous part is achieved by having each agent send asyncronous gradient updates to a master node and periodically update the model from it.
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) - [blog](https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/)
  - Presents a way to adjust the trade-off between bias and variance in policy gradient methods by combining multi-step returns (monte-carlo) with bootstrapped value estimation (td methods).
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
  - TRPO and PPO are Trust Region search algorithms for reinforcement learning policy models. The idea is that as the model is non-linear, the further the weights move the lower the confidence we have that the new policy will be better than the previous one. This problem is tackled by constraining the distance the weights can move to a sub-space around the current weights. This can be seen as an adaptive learning rate per-step per weight which ensures a "safer" convergence to the fixed point.
- [Continuous control with deep reinforcement learning (Deep Deterministic Policy Gradients)](https://arxiv.org/abs/1509.02971)
- [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144)
  - In this paper the authors use [Kronecker-factored Approximate Curvature (K-FAC)](https://arxiv.org/abs/1503.05671) to approximate the inverse fisher matrix, this allows them to easily apply the *natural gradient* in a sample efficient way.
- [A Tour of Reinforcement Learning: The View from Continuous Control](https://arxiv.org/abs/1806.09460)
  - "This manuscript surveys reinforcement learning from the perspective of optimization and
control with a focus on continuous control applications. It surveys the general formulation,
terminology, and typical experimental implementations of reinforcement learning and reviews
competing solution paradigms.
...
This survey concludes with a discussion of some of the challenges in designing learning
systems that safely and reliably interact with complex and uncertain environments and how
tools from reinforcement learning and controls might be combined to approach these challenges."

## Evolution Strategies
- [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) - [website](https://blog.openai.com/evolution-strategies/)
  - Instead of running a single policy and optimizing it, we generate N random policies with a slight variance between them. We then evaluate them. Finally policies are combined as a weighted average where the weight is given with respect to the score in the evaluation. Combination is done at the weight level in the Neural Network.
  Advantage is a highly parallel solution, where evaluation of all policies can be done in parallel. Also this doesn't require backpropagation, which is a compute intensive solution.
- [Safe Mutations for Deep and Recurrent Neural Networks through Output Gradients](https://arxiv.org/abs/1712.06563)
  - They show how gradients can be combined with neuroevolution to improve the ability to evolve recurrent and very deep neural networks, enabling the evolution of DNNs with over one hundred layers, a level far beyond what was previously shown possible through neuroevolution. They do so by computing the gradient of network outputs with respect to the weights (i.e. not the gradient of error as in conventional deep learning), enabling the calibration of random mutations to treat the most sensitive parameters more delicately than the least, thereby solving a major problem with random mutation in large networks.
- [Improving Exploration in Evolution Strategies for Deep Reinforcement Learning via a Population of Novelty-Seeking Agents](https://arxiv.org/abs/1712.06560)
- [Guided evolutionary strategies: escaping the curse of dimensionality in random search](https://arxiv.org/abs/1806.10230)
  - While in many cases the real gradient is hard to estimate (e.g. simulation based estimation), a surrogate gradient is easy to define, such that it points in the general direction of the real gradient. They use this as a guiding signal for the evolution algorithm, such the covariance matrix of ES is projected onto the gradient directions.
  
## Model Based
- [Learning to Act by Predicting the Future](https://arxiv.org/abs/1611.01779)
  - They introduce two interesting elements: (A) A goal vector, essentially the priority of each task (pickup ammunition, kill enemies, stay alive), (B) they use a predicting element to predict the future values and select the action that best fits the goal.
- [World Models](https://arxiv.org/abs/1803.10122) - [blog](https://worldmodels.github.io/)
  - They train a model to predict the next latent state. Using this they can then learn a policy based on the models hallucinations.
- [Neural scene representation and rendering](https://deepmind.com/documents/211/Neural_Scene_Representation_and_Rendering_preprint.pdf) - [blog](https://deepmind.com/blog/neural-scene-representation-and-rendering/)
  - Given a few frame of the environment (from different viewpoints), they are able to recreate a full model allowing an agent to traverse the imaginary world.

## Bandits
- [Finite-time Analysis of the Multiarmed Bandit
Problem](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf)
  - Upper confidence bound algorithm for the multi armed bandit problem.
- [A Survey on Contextual Multi-armed Bandits](https://arxiv.org/abs/1508.03326)
  - In contextual MAB, each arm has a context provided to it. We assume there is some linear projection, i.e. a mapping from the context to the expected reward.
  - [Improved Algorithms for Linear Stochastic Bandits](http://david.palenica.com/papers/linear-bandit/linear-bandits-NIPS2011-camera-ready.pdf) - Confidence bounds for linear contextual bandits
- [X-Armed Bandits](http://www.jmlr.org//papers/v12/bubeck11a.html)
  - The X-armed bandit represents a problem in which we have infinitely many arms represented by some continuous subspace (e.g. the range \[0,1\]). The reward is assumed to be "locally Lipschitz". The goal is to find the optimal arm, or to be epsilon close to it.
- [Combinatorial Multi-Armed Bandit: General Framework and Applications](http://proceedings.mlr.press/v28/chen13a.html)
  - In CMAB they define a "super-arm" which is a combination of N "simple" arms. Upon playing a super-arm, the reward for each simple-arm is observed. The goal is to learn to find the optimal super-arm. *This is a form of batched bandits*
- [Batched bandit problems](https://arxiv.org/abs/1505.00369)
  - The limitation of this framework is that the arm selection is performed in batches. This means that a decision is made once every T steps for a total of T arms. They show that due to the uncertainty bound of the UCB algorithm, we can effectively select batches the size of log(t) without any impact on the performance (this leads to the fact that we can increase the batch size over time as we become more confident in the empirical estimates).

## Other interesting papers
- [Universal Option Models](https://papers.nips.cc/paper/5590-universal-option-models.pdf)
- [Universal Value Function Approximators](http://proceedings.mlr.press/v37/schaul15.pdf)
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
  - An upgrade to DPG (Deterministic Policy Gradient Algorithms, Silver et al. 2014), the algorithm proposed here uses an actor network and a critic network, with slowly updated matching target networks, to perform a policy gradient. An “experience replay” is used to enable training using mini-batches of state-action-reward-next state tuples. This breaks correlation between samples and enables convergence of the networks. The use of neural network function approximators allows the algorithm to be used in settings where the state and action spaces are continuous, without limiting discretization of the action space.
- [Grounded Language Learning in a Simulated 3D World](https://arxiv.org/abs/1706.06551)
  - The agent is located in a 3D domain (visual input only), and is required to perform as task given in textual format. Their architecture consists of: (1) a CNN to parse the visual input (2) an LSTM to process the task, textual input (3) a Mixing module which combines the outputs of both modules (4) an output LSTM which computes a probability distribution over possible actions &pi;(a<sub>t</sub>, s<sub>t</sub>), and a state-value function approximator V(s<sub>t</sub>). They claim that the simple architecture was not enough in order to acheive learning and that at least several additions were required:
    - Reward Prediction (RP) - In this auxillary task (*"Reinforcement Learning with Unsupervised Auxiliary Tasks"*), the agent is required to predict the immediate reward (similar to learning the value function with *&gamma; = 0*).
    - Value Replay (VR) - Since the A3C learns in an on-line manner, the VR uses the replay memory (similarly to the DQN) to resample recent historical sequences.
    - Language Prediction (LP) - Another output is received from the visual image processor, which is what the agent thinks will be the task. Intuitively this causes the agent to **understand** what are the important objects in the domain and to concentrate on them during the image parsing.
    - Temporal Autoencoding (tAE) - The tAE's objective is given the visual data v<sub>t</sub> and the action a<sub>t</sub>, to predict the next visual environment v<sub>t+1</sub>.
- [DeepMimic](https://arxiv.org/abs/1804.02717) - [website](https://xbpeng.github.io/projects/DeepMimic/index.html), [blog](http://bair.berkeley.edu/blog/2018/04/10/virtual-stuntman/), [video](https://www.youtube.com/watch?v=vppFvq2quQ0)
  - A recording of a human performing complex tasks is used as a regularizer for the agents behavior. They provide the agent with a task reward (how good it is at the given task) combined with a similarity reward (how similar is the behavior to that of the human). With the addition of several other nice tricks (e.g. early termination) they are able to achieve impressive empirical results.

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
- [Generative Adversarial Trainer: Defense to Adversarial Perturbations with GAN](https://arxiv.org/abs/1705.03387)
- [Adversarial Transformation Networks: Learning to Generate Adversarial Examples](https://arxiv.org/pdf/1703.09387.pdf)
# Natural Language Processing

# Non-summarized
