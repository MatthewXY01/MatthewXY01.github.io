---
title: MAML
date: 2019-10-29 17:24:54
categories: paper note
mathjax: true
tags:
  - meta-learning
  - few-shot learning
---

<center>
<p>
<a href="https://arxiv.org/abs/1703.03400">Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks</a></br>
Chelsea Finn, Pieter Abbeel, Sergey Levine</p></center>

这是一篇关于meta-learning的文章。作者提出了一种叫作'MAML'的算法，即'Model-Agnostic Meta-Learning'. 很显然，算法的优点在于'Model-Agnostic', 不受限于某一类具体任务或模型。同时作为一种meta-learning方法，在实际应用中能有效地处理few-shot learning问题——小样本训练，面对新任务数据能快速得到很好的泛化表现。  
>In effect, our method trains the model to be easy to fine-tune. —— *Chelsea Finn et al.*

这篇笔记首先记录学习过程中对meta-learning的理解，之后针对这篇文章记录些收获。  
<!-- more -->

## Meta-learning
---
Meta-learning的思想在于让机器学会去学习(learn to learn), 即通过过去任务的学习中得到“快速学习新任务”的**技巧**，并非得到处理特定问题本身的方法，而是获得快速掌握这些方法的能力。类似地，这一想法的产生也是“从人的角度出发”——只用见过几个例子就能完全认识一类物体，只用短短的一段经历就能收获某些技能，在一个问题上的处理方法很容易迁移应用到另一些差异很大看似毫不相关的问题上。

### Difference with traditional machining learning
既然是'learn to learn', 肯定与普通的'learn'有区别。相较于传统的机器学习过程，区别大概有以下：  

1. **学到的东西——f vs F**  
传统的机器学习是通过输入训练数据，不断调整模型参数，最终得到一个可以解决问题的尽可能优的方法，抽象成函数f. 而元学习同样通过输入训练数据，但最终希望得到一个能够找寻到f的函数F. 都是要找一个函数，但前者寻找的是解决一个问题的f, 后者寻找的是生成f的F. 
2. **对函数优劣的度量**  
在找寻可用于解决具体任务的f, loss function可以作为一种评价效果的尺度。在回归问题中常采用MSE而在分类问题中常采用cross entropy. 通常情况下，可以在完成一个batch的训练后计算一次，假设当前训练的函数记录为f1, loss记为l1。而对于一个F, 类似地，仍可以在完成一个batch的训练后计算一次，但这是对一个batch中的一批f的训练，loss的计算需要考虑各个f的loss. MAML中就是对各个f的loss简单求和。将F的loss记为L，则有：  
$$L(F)=\sum_{n=1}^{N} l^{n}$$  
想要找的一个最优的F：  
$$F^{*}=\arg \min _{F} L(F)$$
  
3. **训练资料**  
传统机器学习与元学习的最直观的区别应该是训练资料的构成。相比于“训练数据”，“训练资料”的说法更为贴切，也很好展示了为什么meta-learning训练的F是为了找寻解决问题的f.  
一般的机器学习任务是单任务的，我们将所有数据划分为训练数据和测试数据，也经常会多划分一个用于监视训练成果，调整超参数的验证集。无论是训练、验证还是测试，基本单元是**数据**。而对于meta-learning, 上升了一个层次，进行多任务训练。同样划分为训练、验证和测试，基本的单元是**任务**，每个任务中又划分有**用于训练**的数据和**用于测试**的数据，可将他们分别叫作**support set**和**query set**以避免混淆。而每个任务中的**query set**正是用于，在对F的单批训练完成后作为其损失函数L的一部分，计算该单个任务对应函数f的loss。每个task通过自己内部的**query set**计算出的loss综合起来就是F的loss. 无论是一般的机器学习任务还是meta-learning, 测试部分（在meta-learning中是指test tasks, 并非某个task中的query set）在验收最终成果前都是完全不可见。  

以上的细微差异可以看出'learn to learn'是比'learn'高了一个层次，有点像“套娃”。  

### N-way K-shot
N-way K-shot是few-shot learning中常见的experimental set-up. 以单个classification task为例，N代表该task下的类别总数，K代表每一类别提供的训练样本数。在N-way K-shot设定下的某task的support set大小是N x K.  

### Meta-learning vs Few-shot learning
- meta-learning本质是一种思想'learn to learn', few-shot learning是一种问题设定
- meta-learning在问题设定上与few-shot learning有相似处——新任务上少量样本快速适应，进一步地，**引入了更多假设**来克服单纯few-shot learning设定下可能会出现的overfitting的问题
- meta-learning的假设：存在一个任务分布，目标任务也满足该分布，可以通过在该分布采样数据来训练（support set and query set），相当于为目标任务提供先验知识  

总结来说，meta-learning可作为解决few-shot learning问题的一个典型方法，通过更多的假设，在即便没有大量服从目标任务分布的训练样本的情况下，用假设“理想情况”（任务分布相同）下的大量相似任务来弥补。  

### Reference
[Meta-learning和Few-shot learning的关系](https://www.zhihu.com/question/291656490)  

## MAML
---
**Outline:** propose an algorithm for meta-learning that is model-agnostic  
**Core idea:** train the model’s **initial parameters** such that the model has maximal performance on a new task after the parameters have been updated through **one or more gradient steps** computed with a small amount of data from that new task  
**Strength:** model-agnostic  
- compatible with any model trained with gradient descent
- applicable to a variety of different learning problems, including classification, regression, and reinforcement learning
- does not expand the number of learned parameters nor place constraints on the model architecture  

### Problem set-up and training protocol
Use $f$ to denote a model that maps observation $x$ to $a$. Formally, each task $\mathcal{T}$ consists of a loss function $\mathcal{L}$, a distribution over initial observations $q\left(\mathbf{x}\_{1}\right)$, a transition distribution $q\left(\mathbf{x}\_{t+1} | \mathbf{x}\_{t}, \mathbf{a}_{t}\right)$ , and an episode length $H$.  

$$\mathcal{T}=\left\\{\mathcal{L}\left(\mathbf{x}\_{1}, \mathbf{a}\_{1}, \ldots, \mathbf{x}\_{H}, \mathbf{a}\_{H}\right), q\left(\mathbf{x}\_{1}\right), q\left(\mathbf{x}\_{t+1} | \mathbf{x}\_{t}, \mathbf{a}_{t}\right), H\right\\}$$  

In supervised regression and classification problem, we can define the $H$=1 and drop the time subcript on $\mathbf{x}\_{t}$ if the model accepts a single input and produces a single output, rather than a sequence of inputs and outputs.  
Now we're going to pay attention to the details of data partitioning. In meta-training procedure:  
- sample a task $\mathcal{T}\_{i}$ from the distribution over tasks i.e. $p(\mathcal{T})$ with K samples drawn from ${q}\_{i}$ (support set under K-shot setting)
- train the model with the K samples and get feedback from the corresponding loss $\mathcal{L}\_{\mathcal{T}\_{i}}$ from $\mathcal{T}\_{i}$
- test on **new** samples from $\mathcal{T}\_{i}$(query set) and we will get a *test error*  
**Note:** The so-called *test error* which belongs to a certain $f$ in a certain training state is not equivalent to the exact test error of the meta-testing process.  
>In effect, the test error on sampled tasks $\mathcal{T}\_{i}$ serves as the training error of the meta-learning process. —— *Chelsea Finn et al.*

### Algorithm
Anyway, MAML's key idea is to train a powerful initial parameters that are **sensitive** to changes in the task such that small changes in the parameters will produce large improvements on the loss function of any task drawn from $p\left(\mathcal{T}\right)$. So let's get into the concrete algorithm.  
{% asset_img algorithm.png %}  
Nothing special except the update of the parameters. In the step 4 to 7, for each task, we take derivation of the loss in order to update the $\theta$ to $\theta_{i}^{\prime}$ with gradient descent.  
**Note:** 
- $\alpha$ may be fixed as a hyperparameter or meta-learned. 
- it's an identical copy of original $\theta$ with which each $\mathcal{T}\_{i}$ begins its parameter update. e.g. In the first loop, $\mathcal{T}\_2$ updates its parameters by $\theta\_{2}^{\prime}=\theta-\alpha \nabla\_{\theta} \mathcal{L}\_{\mathcal{T}\_{2}}\left(f\_{\theta}\right)$ but not $\theta\_{2}^{\prime}=\theta\_{1}^{\prime}-\alpha \nabla\_{\theta\_{1}^{\prime}} \mathcal{L}\_{\mathcal{T}\_{2}}\left(f\_{\theta\_{1}^{\prime}}\right)$
- we consider one gradient update for each $\mathcal{T}\_{i}$ in one iteration but using multiple gradient updates is a straightforward extension. Actually this 'one update' setting tends to make this only one update the most valuable update in a sense so that the updated parameters can be more sensitive.  
> That's one small step for a man, one giant leap for mankind. —— *Neil Armstrong*  

- the loss in $\theta_\{i}^{\prime}=\theta-\alpha \nabla\_{\theta} \mathcal{L}\_{\mathcal{T}\_{i}}\left(f\_{\theta}\right)$ computed by the support set of $\mathcal{T}\_{i}$ and corresponding outputs but not **new** data in query set. However, in the **meta-update** $\theta \leftarrow \theta-\beta \nabla\_{\theta} \sum\_{\mathcal{T}\_{i} \sim p(\mathcal{T})} \mathcal{L}\_{\mathcal{T}\_{i}}\left(f\_{\theta\_{i}^{\prime}}\right)$ we use query data which is invisible to the previous training and **updated parameters** $\theta^{\prime}$ to compute the loss. Such setting is conducive to improving the generalization ability of the model.  
**Approximation:** the MAML meta-gradient update involves a gradient through a gradient so that it requires second derivatives. In the process of the experiment from the authors, they do a **first-order approximation** i.e. The meta-optimization computes the meta-gradient at the post-update parameter values $\theta\_{i}^{\prime}$ instead of $\theta$:  
$$
\theta-\beta \nabla\_{\theta} \sum\_{\mathcal{T}\_{i} \sim p(\mathcal{T})} \mathcal{L}\_{\mathcal{T}\_{i}}\left(f\_{\theta\_{i}^{\prime}}\right)\ = \theta-\beta \sum\_{\mathcal{T}\_{i} \sim p(\mathcal{T})} \nabla\_{\theta} \mathcal{L}\_{\mathcal{T}\_{i}}\left(f\_{\theta\_{i}^{\prime}}\right) \approx \theta-\beta \sum\_{\mathcal{T}\_{i} \sim p(\mathcal{T})} \nabla\_{\theta\_{i}^{\prime}} \mathcal{L}\_{\mathcal{T}\_{i}}\left(f\_{\theta\_{i}^{\prime}}\right)
$$  

### Experiments
#### Regression experiments
**Task:** each task involves regressing from the input to the output of a sine wave, where the amplitude and phase of the sinusoid are varied between tasks. i.e. given K datapoints, evaluate the model to fit a sinusoid curve.  
e.g.  
task1: $y=3\sin(x)$  
task2: $y=2.5\sin(x+0.5\pi)$  
...  
**Distribution of tasks:** $p(\mathcal{T})$ is continuous, amplitude $\in$ [0.1, 5.0], phase $\in$ [0, π]  
**Baselines:**  
1. pretrained network: put all the tasks in the network together with their respective K samples into the network. i.e. train a network to regress to **random** sinusoid functions.  
2. an oracle which receives the true amplitude and phase as input.  

The regressor is a neural network model with 2 hidden layers of size 40 with ReLU nonlinearities. When training with MAML, we use one gradient update with K = 10 examples with a fixed step size α = 0.01, and use Adam as the meta-optimizer. The baselines are likewise trained with Adam.  
**Evaluation:** finetune the model on varying numbers of K examples.  
**Result:**  
{% asset_img regression.png %}  

MAML is better and able to estimate parts of the curve where there are **no datapoints**(figure 1), indicating that the model has learned about the **periodic structure** of sine waves. It's really amazing! When the K datapoints are all in the **right** half of the input range, pretrained net's  output in the **left** half is **almost unchanged** with finetuning(figure 3)—— It is blind!  
A resonalble explaination claims that too many contradictory outputs(e.g. y = 1 in $y=\sin(x)$ while y = -1 in $y=\sin(x+\pi)$ when x = π/2) just disturbed the model so that pretrained model without MAML is unable to recover suitable representation.  

#### Classification experiments
**Task:** N-way k-shot learning task. i.e. we're given K (e.g. 1 or 5) labelled examples for N classes that we have not previously trained on and asked to classify new instances into the N classes.  
**Datasets:** Omniglot([Lake et al. 2011](http://web.mit.edu/jgross/Public/lake_etal_cogsci2011.pdf)) and MiniImagenet([Ravi & Larochelle, 2017](https://openreview.net/forum?id=rJY0-Kcll)). In this note I skip the experiments on MiniImagenet(lazy...)  
{% asset_img omniglot.png %}  
Omniglot is a MNIST-like scribbles dataset with 1623 characters with 20 examples each.  
**Architecture:** model's architecture is the same as embedding function in *Matching network*: a CNN with 4 modules of [3x3 CONV 64 filters, batchnorm, ReLU, 2x2 max pool] but using strided convolutions instead of max-pooling for experiments on omniglot. The last layer is fed into softmax.  
**Baselines and results:**  
{% asset_img classification.png %}  
MAML achieves results that are comparable to or outperform SOTA convolutional and recurrent models for few-shot classification. But the authors emphasize that MAML is more 'outstanding' because Siamese networks, matching networks, memory module approaches are all specific to classification, and are not directly applicable to regression or RL scenarios.

## Conclusion
---
It's a great idea to work on initial parameters so that this computing framework is universal(you need to initialize parameters anyhow). However, MAML still requires that the model is trained based on gradient decent.  
I skipped some details such as experiments on MiniImagenet, MAML applied in RL, more comparison in experiments...
It's actually my first 'kind of formal' paper note and by the way, I test the *mathjax* in hexo blog which makes the note with math formulas more elegant.  
I cited some relevant content about meta-learning in the course video by [Hung-yi Lee](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML19.html) and imitated the note style of Andrej Karpathy in [this](https://github.com/karpathy/paper-notes/blob/master/matching_networks.md#the-training-protocol). I have also learnt something from his note.  
Hope to make more paper notes! :)