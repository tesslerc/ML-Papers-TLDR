# Machine Learning Papers TL;DR

# Contents
[General](#general)

[Reinforcement Learning](#reinforcement-learning)

[GANs](#gans)

[Natural Language Processing](#natural-language-processing)

# General
- [DSD: Dense-Sparse-Dense Training for Deep Neural Networks](https://arxiv.org/abs/1607.04381)
  - The authors propose a new method for training Deep Neural Networks.
    > In the first D (Dense) step, we train a dense network to learn connection weights and importance. In the S (Sparse) step, we regularize the network by pruning the unimportant connections with small weights and retraining the network given the sparsity constraint. In the final D (re-Dense) step, we increase the model capacity by removing the sparsity constraint, re-initialize the pruned parameters from zero and retrain the whole dense network.
    Simply put, Dense (1): learns the most important connections in the network. Sparse (2): uses a slim network to learn a complex problem, this means the representation of the network will concentrate on the most important features + this avoids over fitting. Dense (3): the network is already trained, we re-dense the network by returning the removed connections, now the network will perform fine turning around the learned model.
- [Machine Learning on Sequential Data Using a Recurrent Weighted Average](https://arxiv.org/abs/1703.01253)
  - Previous works have shown the power of the attention mechanism. This work proposes a similar approach which is easier to implement and which is trainable by simple gradient descent. Their approach takes a weighted average of the cell since the first time step - they show impressive results, especially in the training time.
- [Scalable and Sustainable Deep Learning via Randomized Hashing](https://arxiv.org/abs/1602.08194)
  - The authors build on the idea of [Adaptive Dropout](https://papers.nips.cc/paper/5032-adaptive-dropout-for-training-deep-neural-networks.pdf). In adaptive dropout, we sample which neurons to keep active based on a random bernouli variable and given the neuron activation value. Logically, similar inputs will provide similar activations. They exploit this by using [Locality-Sensitive Hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) which gives similar inputs similar hashed values. This allows keeping what neurons should be left active / the probability of each neuron to stay active - per input (or for N distinct inputs). Then the procedure is just hash -> find nearest neighbor -> dropout according to probabilities.
- [meProp: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting](https://arxiv.org/abs/1706.06197)
  - This paper suggests an interesting idea. They note that most of the time consumed during training is in the backpropagation stage - i.e propagating the gradient backwards through the network & updating the weights. They suggest a way to speedup learning by performing learning only on the samples that give the largest gradient (largest loss value). This intuitively can be explained that the smaller gradients will cause smaller steps which will be overwritten by the larger gradient steps.
- [YellowFin and the Art of Momentum Tuning](https://arxiv.org/abs/1706.03471)
  - An automatic tuner for momentum SGD - a short explanation is avilable at this [link](http://cs.stanford.edu/~zjian/project/YellowFin/).
    > "TLDR; Hand-tuned momentum SGD is competitive with state-of-the-art adaptive methods, like Adam. We introduce YellowFin, an automatic tuner for the hyperparameters of momentum SGD. YellowFin trains models such as large ResNets and LSTMs in fewer iterations than the state-of-the-art. It performs even better in asynchronous settings via an on-the-fly momentum adaptation scheme that uses a novel momentum measurement component along with a negative-feedback loop mechanism."
- [Safer Classification by Synthesis](https://arxiv.org/abs/1711.08534)
  - Discriminative classification models can easily be fooled to classify incorrectly with high confidence by out-of-distribution examples - this in itself can be hazardous in real world applications [(Adversarial examples in the physical world)](https://arxiv.org/abs/1607.02533).
  - The authors propose to overcome this by using a generative approach – for a given test image, find the latent variable that produces it in generators trained for each class. Then, choose the one with the highest confidence (i.e lowest error compared to the original image).
  - Intuitive error measures are for instance the MSE, we can define a threshold underwhich the generator is considered unconfident. If all classes are 'unconfident' we would prefer not to classify over being wrong with a high proability.
  - This is a general approach – and can be used with VAEs or GANs, or even with a k-Means generative model.
  - The paper shows competitive results for in-distribution test images, and very good results on out-of-distribution images (meaning, the model knows to reject the examples rather than misclassify them).  

# Reinforcement Learning
- [*(Draft)* Reinforcement Learning: An Introduction, Richard S. Sutton and Andrew G. Barto](http://incompleteideas.net/sutton/book/bookdraft2017nov5.pdf)
  - The latest *draft* of the RL intro book, which provides an overview over many of the topics in the field.
- [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560)
  - The authors in this paper bring to light a large problem with Deep RL. I believe that **reading this paper is a requirement to anyone intending to work with Deep RL**, to properly understand the past / current issues with the research. The problems the authors bring up, are backed by extensive tests to prove their claims. The issues are regarding methods of reporting the results and model evaluation, which are inconsistent between papers and domains and in some cases are bad practice.
- [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) - [website](https://blog.openai.com/evolution-strategies/)
  - Instead of running a single policy and optimizing it, we generate N random policies with a slight variance between them. We then evaluate them. Finally policies are combined as a weighted average where the weight is given with respect to the score in the evaluation. Combination is done at the weight level in the Neural Network.
  Advantage is a highly parallel solution, where evaluation of all policies can be done in parallel. Also this doesn't require backpropagation, which is a compute intensive solution.
- [Training Agent for First-Person Shooter Game with Actor-Critic Curriculum Learning](https://openreview.net/forum?id=Hk3mPK5gg)
  - They perform curriculum learning within ViZDoom, using the actor-critic approach.
  Their method uses the previous 4 images (no LSTM) as a state. Due to sparse reward they train the agent in domains with increasing difficulty (easy to hard), this makes sense since it is intially provided an easy to solve domain - and uses this knowledge to help it solve a harder domain once it is introduced. *Reward Shaping* - the authors note that they use reward shaping as another method to cope with sparse rewards, something to note is that reward shaping is a means of 'prior knowledge' or 'semi supervised' - it would be interesting see a solution with a more basic reward function like -X per step and +Y per kill.
- [Learning to Act by Predicting the Future](https://arxiv.org/abs/1611.01779)
  - They introduce two interesting elements: (A) A goal vector, essentially the priority of each task (pickup ammunition, kill enemies, stay alive), (B) they use a predicting element to predict the future values and select the action that best fits the goal.
- [Zero-Shot Task Generalization with Multi-Task Deep Reinforcement Learning](https://arxiv.org/abs/1706.05064)
  - The authors tackle a set of 'Multi-Task' problems, in which the tasks are given as a textual input (unordered set of tasks). To tackle this they build upon the hierarchical approach using the option framework. They present several interesting additions: (1) Generalization via analogy making - the agent has tasks {T} and objects {O} it encounters part of the state space G\` contained in G=TxO (to succeed it must learn to 'understand' the meaning of each object and task), (2) Dynamic 'stopping time' - unlike in other works where the task runs for T (constant) time steps or the skill learns to predict when it should come to an end, they present a novel way for the controller to switch between skills.
- [Improving Generalisation for Temporal Difference Learning: The Successor Representation](http://www.gatsby.ucl.ac.uk/~dayan/papers/d93b.pdf)
- [Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312)
  - Successor Features build upon the idea of Successor Representations (where SR are a special case of SF). The proposed formulation allows (theoretically) to decouple the state representation from the policy. The &Psi;<sub>i</sub>(s,a) contains state-action domain representation, where as w<sub>i</sub> contains the task specific knowledge. They suggest that when learning a new task, one can use the previous &Psi;<sub>k</sub>(s,a) k in {0, ..., i} in a manner of max<sub>k</sub> &Psi;<sub>k</sub>(s,a)\*w. This way, previous domain knowledge can be used and learning rate is improved.
- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887), Follow up paper [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044), [Blog](https://flyyufelix.github.io/2017/10/24/distributional-bellman.html)
- [Universal Option Models](https://papers.nips.cc/paper/5590-universal-option-models.pdf)
- [Universal Value Function Approximators](http://proceedings.mlr.press/v37/schaul15.pdf)
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- [Continuous control with deep reinforcement learning (Deep Deterministic Policy Gradients)](https://arxiv.org/abs/1509.02971)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Asynchronous Methods for Deep Reinforcement Learning (A3C)](https://arxiv.org/abs/1602.01783)
- [The Uncertainty Bellman Equation and Exploration](https://arxiv.org/abs/1709.05380v1)
- [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144)
  - In this paper the authors use [Kronecker-factored Approximate Curvature (K-FAC)](https://arxiv.org/abs/1503.05671) to approximate the inverse fisher matrix, this allows them to easily apply the *natural gradient* in a sample efficient way.
- [Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397) [Blog](https://deepmind.com/blog/reinforcement-learning-unsupervised-auxiliary-tasks/)
  - The authors propose a new archiecture named UNREAL (UNsupervised REinforcement and Auxiliary Learning), which combines the learning of auxiliary tasks to help the agent in learning the main task. They present two auxiliary tasks: (1) Learning to predict how your actions will change the environment i.e. predict s<sub>t+1</sub> given (s<sub>t</sub>, a<sub>t</sub>) (2) Predict the immediate reward, similar to learning the value function with a discount factor equal to 0. These prediction heads are connected to the feature extraction layer hence they do not interfere (bias) with the policy or the value prediction, but they do help shape the feature extraction layers to facilitate faster learning.
- [Grounded Language Learning in a Simulated 3D World](https://arxiv.org/abs/1706.06551)
  - The agent is located in a 3D domain (visual input only), and is required to perform as task given in textual format. Their architecture consists of: (1) a CNN to parse the visual input (2) an LSTM to process the task, textual input (3) a Mixing module which combines the outputs of both modules (4) an output LSTM which computes a probability distribution over possible actions &pi;(a<sub>t</sub>, s<sub>t</sub>), and a state-value function approximator V(s<sub>t</sub>). They claim that the simple architecture was not enough in order to acheive learning and that at least several additions were required:
    - Reward Prediction (RP) - In this auxillary task (*"Reinforcement Learning with Unsupervised Auxiliary Tasks"*), the agent is required to predict the immediate reward (similar to learning the value function with *&gamma; = 0*).
    - Value Replay (VR) - Since the A3C learns in an on-line manner, the VR uses the replay memory (similarly to the DQN) to resample recent historical sequences.
    - Language Prediction (LP) - Another output is received from the visual image processor, which is what the agent thinks will be the task. Intuitively this causes the agent to **understand** what are the important objects in the domain and to concentrate on them during the image parsing.
    - Temporal Autoencoding (tAE) - The tAE's objective is given the visual data v<sub>t</sub> and the action a<sub>t</sub>, to predict the next visual environment v<sub>t+1</sub>.
- [Understanding Grounded Language Learning Agents](https://arxiv.org/abs/1710.09867)
- [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)
  - An upgrade to DPG (Deterministic Policy Gradient Algorithms, Silver et al. 2014), the algorithm proposed here uses an actor network and a critic network, with slowly updated matching target networks, to perform a policy gradient. An “experience replay” is used to enable training using mini-batches of state-action-reward-next state tuples. This breaks correlation between samples and enables convergence of the networks. The use of neural network function approximators allows the algorithm to be used in settings where the state and action spaces are continuous, without limiting discretization of the action space.

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
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Nash Equilibrium](https://arxiv.org/abs/1706.08500)
  - They propose a two-time scale update rule for the generator & the discriminator. They prove that it converges for SGD and ADAM, they also show improved results on existing GAN architectures.
- [Fader Networks: Manipulating Images by Sliding Attributes](https://arxiv.org/abs/1706.00409)
  - The setup is of an Auto-Encoder. Input is of pairs {x<sub>i</sub>,y<sub>i</sub>} where x<sub>i</sub> is the input and y<sub>i</sub> is the label. *Encoder* receives input x<sub>i</sub> and translates it to latent representation (embedding) e<sub>i</sub>. *Decoder* receives {e<sub>i</sub>, y<sub>i</sub>} i.e the embedding and the label, it is required to generate back x<sub>i</sub> - the auto-encoder part of the scheme. The *Discriminator* receives the embedding e<sub>i</sub> and is required to determine the label y<sub>i</sub>. This setup causes the *Encoder* to learn a mapping invariant of the label y<sub>i</sub>, which in turn allows the *Decoder* to create new images based on a change of label (i.e x<sub>i</sub> is a man, we give *Decoder* the y<sub>i</sub> of a woman -> output will be of a woman with the given mans features).

# Natural Language Processing

# Non-summarized
- [Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car](https://arxiv.org/abs/1704.07911)
  - Last I checked, Nvidia perform deep imitation learning.
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
- [Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120)
