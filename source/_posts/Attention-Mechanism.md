---
title: Attention Mechanism
date: 2019-10-23 15:42:35
categories: Deep learning
tags:
  - attention mechanism
  - RNN
---

## 前言  
第一次听说**Attention**机制是在音视频处理课上，在视觉问答领域中有用到 **Attention based model** 后来在学习Matching Network的时候又一次接触到了。查询一些资料，做点总结和记录。  

## 人类的注意力机制  
人类在感知东西的时候，一般不会是一个场景从到头看到尾每次全部都看，而往往是根据需求观察注意特定的一部分。观察一幅画面，重点看颜色鲜艳的部分；看一篇文章，先看大字标题和开头几句；听英语听力，在提问前的引导语句结束后开始全神贯注。我们对信息本身分配了不同程度的专注度。  
观察许多流行的深度学习模型，很多创新点都是“从人的角度出发”被提出。这一点感觉很像仿生学（鲨鱼皮泳衣、蝙蝠和雷达等等）。'Attention'本身就是个很human-like的词，深度学习中的**Attention mechanism** 也是从人类视觉注意力机制中获得灵感。  

## 深度学习中的注意力机制
深度学习中的注意力机制从本质上讲和人类的选择性视觉注意力机制类似，核心目标也是从众多信息中选择出对当前任务目标更关键的信息。具体来说，也只是利用一系列权重参数构建注意类分配系数。  
### Encoder-Decoder 框架
注意力机制作为一种通用思想，不依赖于具体框架存在，但又通常附着在Encoder-Decoder框架下。如图是文本处理领域里常用的Encoder-Decoder框架最抽象的一种表示：  
{% asset_img encoder-decoder.jpg %}  

可以把它看作适合处理由一个句子（或篇章）生成另外一个句子（或篇章）的通用处理模型。对于句子对<Source,Target>，我们的目标是给定输入句子Source，期待通过Encoder-Decoder框架来生成目标句子Target。Source和Target可以是同一种语言，也可以是两种不同的语言。而Source和Target分别由各自的单词序列构成：  
{% asset_img source-target.jpg %}  

Encoder对单词序列Source进行编码，进行非线性变换转换为中间语义c，再将编码结果c送入Decoder从而生成目标序列Target.   
采用这种结构的模型在许多比较难的序列预测问题（如文本翻译）上都取得了最好的结果：  
- [Sequence to Sequence Learning with Neural Networks, 2014](https://arxiv.org/abs/1409.3215)
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation, 2014](https://arxiv.org/abs/1406.1078)  

其中，文本处理和语音识别的Encoder通常采用RNN模型，而图像处理的Encoder采用CNN模型，Decoder通常都由RNN/LSTM实现。中间语义编码c为**定长**向量，这引发了一系列问题：  
- RNN模型本身存在长程梯度消失的问题
- 输入序列不论长短都会被编码成一个固定长度的向量表示，而解码则受限于该固定长度的向量表示，输入序列过长则编码结果很难保留全部必要信息  

### 引入Attention mechanism
还是以文本翻译为例，对基本模型(Soft attention)进行说明。
#### 基本思想
打破了传统Encoder-Decoder结构在编解码时都依赖于内部一个固定长度向量的限制。原先的模型中未引入注意力机制，对于最终输出序列，有以下表示：  
{% asset_img no-attention.jpg %}  
显然，在文本翻译的例子中，对于输出序列中的任意单词的生成过程，中间语义作为生成的参与者没有发生变化。换言之，每一个输出单词都“平均”地考虑了输入句子中的每一个单词，盲目而缺少针对性。以'Tom Chase Jerry'这句为例，理想输出应该是“汤姆”、“追逐”、“杰瑞”。事实上，“杰瑞”为音译，原输入对这一输出的影响应该只存在于'Jerry'，在传统的Encoder-Dcoder框架中，“杰瑞”收到了'Tom chases Jerry'的平均影响。  
{% asset_img tom-chases-jerry.jpg %}  
显然，在输入为短句的情况下，未引入注意力机制并不太影响，但不难理解，长句的输入将会影响模型的性能。对于上面的例子，如果引入attention，为体现输入对输出的不同影响程度，给以给出类似的概率分布值： （Tom,0.3）(Chase,0.2) (Jerry,0.5). 考虑到概率分布的影响，编码结果c的产生发生了变化。具体来说，将注意力机制引入原先模型后，运行如图：  
{% asset_img with_attention.jpg %}  
对应的输出序列可表示为：  
{% asset_img attention.jpg %}  

#### 考虑了概率分布的中间语义C
在引入了attention机制后，C的编码变得更有针对性，还是以上面“猫捉老鼠”为例，新模型下C的可能对应以下：  
{% asset_img newc.jpg %}  
其中，f2是Encoder对raw input word的一种变化，以RNN为例，由于每个cell间对应的参数是共享的，f2可认为是恒定的一种变换，结果可能（以）是hidden value. g可认为是将Encoder接收的单词们的中间表示整合为接收的句子的中间语义C的操作。**一般的做法中，g函数就是对构成元素加权求和：**  
{% asset_img g.jpg %}  
其中，Lx是source的长度，此例为3，而a_ij则是对于每个Ci中的h_ij的注意力分配系数（概率）。翻译“汤姆”时，C汤姆的计算过程如图：  
{% asset_img Ctom.jpg %}  

#### Attention系数a
编码C的生成考虑了注意力机制，那么Attention系数a从哪来？  
观察到系数和为一，自然想到这是在经过某种带有“归一化”操作的计算后产生的，具体的过程可以通过展开模型结构来理解。将传统Encoder-Decoder框架展开：  
{% asset_img detailE-C.jpg %}  
其中，EOS代表输入序列的中止。也可以从中观察到当前时刻的输出y参与了下一时刻输出的生成。注意力系数的计算如图：  
{% asset_img coefficient.jpg %}  
可以看出，Soft attention模型中，Attention系数的归一化是通过Softmax函数实现的，Softmax既能实现归一化，又能突出重要元素的权重。而在送入Softmax函数之前，“待归一化”的结果是通过F函数同时作用于Encoder与Decoder产生。F是个度量similarity的函数，可以有多种选择：
- 普通点积
- Cosine distance(可以视为归一化后的点积——向量夹角Cos值可由向量点乘除以向量模长计算)
- L1 or L2 distance
- ...
到此，我大概理解为什么Matching Network利用了 *'external memories'*(用到LSTM)同时被归为Metric-based方法了  
>In this work, we employ ideas from metric learning based on deep neural features and from recent advances that augment neural networks with external memories  
——Vinyals, O et al. [Matching networks for one shot learning, 2016](https://arxiv.org/abs/1606.04080)

Attention因为涉及到了度量距离，其在文本翻译问题中的物理意义可以看作是“短语对齐”。 

### Attention mechanism的本质
前面说过Attention机制不依赖于具体框架存在，如果抛开Encoder-Decoder框架，可以这样看待Attention机制：Source中的构成元素想象成是由一系列的<Key,Value>数据对构成，此时给定Target中的某个元素Query，通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。更概括地说，**Attention本质是一种加权求和：**  
{% asset_img nature_attention.jpg %}  

总结下来，Attention机制的具体计算过程可如下：  
1. 计算Query和Key的similarity
2. 对上一步的结果归一化处理，得到系数
3. 根据系数对Value加权求和

{% asset_img procedure.jpg %}  
目前绝大多数具体的注意力机制计算方法都符合上述的三阶段抽象计算过程，以上仅是Soft attention的基本模型，还有一种Self attention未说明（懒）。  

## 总结
1. 传统Encoder-Decoder框架应用于RNN使用定长中间向量，这限制了对长序列输入的学习
2. Attentinon mechanism通过让网络学会“对于每个输出项，应当将注意力放在输入的哪部分”来克服传统Encoder-Decoder框架的缺陷
3. Attention mechanism应用于多种类型的序列预测问题，包括文本翻译、语音识别等  

## Reference
[Attention in Long Short-Term Memory Recurrent Neural Networks](https://machinelearningmastery.com/attention-long-short-term-memory-recurrent-neural-networks/)
[目前主流的attention方法都有哪些](https://www.zhihu.com/question/68482809/answer/264632289)
