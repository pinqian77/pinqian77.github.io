# Stanford CS234

## Useful Info from Others

- “在 reinforcement learning 里面，environment 跟 reward function 不是你可以控制的，environment 跟 reward function 是在开始学习之前，就已经事先给定的。你唯一能做的事情是调整 actor 里面的 policy，使得 actor 可以得到最大的 reward。”



- “如果用 deep learning 的技术来做 reinforcement learning ，policy 就是一个 network。要让你的 machine，你的 policy 看到什么样的画面， 这个是自己决定的。应该考虑给机器看到什么样的游戏画面，可能是比较有效的。”



- “在做 policy gradient 的时候，output 其实是 stochastic 的。我们 output 一个 action 的 distribution，根据这个 action 的distribution 去做sample， 所以在 policy gradient 里面，你每次采取的 action 是不一样的，是有随机性的。”



- “Q-learning 是 value-based 的。在 value based 的方法里面，我们 learn 的不是 policy，一个 critic。 ”



- “对随机性的策略来说，输入某一个状态 s，采取某一个 action 的可能性并不是百分之百，而是有一个概率 P 的，就好像抽奖一样，根据概率随机抽取一个动作；而对于确定性的策略来说，它没有概率的影响。当神经网络的参数固定下来了之后，输入同样的 state，必然输出同样的 action，这就是确定性的策略。”

<img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\ll.png" alt="ll" style="zoom:50%;" />



# ———————————————







## My Questions

### 1. 关于输入作为observation（state）数据的处理

​		玩Atari虽然前后也是有时间联系的，但是用CNN来处理输入的图片，单独地作为输入似乎也可以实现。那像股票交易这种按每次独立的observation可能不太行。所以我觉得可能需要RNN或者LSTM先来处理一下要作为state的数据，让他们成为前后有关联的数据。

### 2. Return回报G， V函s数， Q函数的关系



### 3. 对给定一个Policy，某个轨迹出现的概率的计算公式理解   

​        $p_\theta(τ)=p(s_1)p_\theta(a_1∣s_1)p(s_2∣s_1,a_1)p_\theta(a_2∣s_2)p(s_3∣s_2,a_2)⋯$

​                  $=p(s_1)∏_{t=1}^Tp_\theta(a_t∣s_t)p(s_{t+1}∣s_t,a_t) $

​		这里有两个概率，一个是基于环境的状态转移概率$p(s_{t+1}∣s_t,a_t)$ ，还有一个是基于Agent 的Policy $p_\theta(a_t∣s_t)$. 所以这就可以算出某条轨迹的概率了。

### 4. Advantage Function 

​		就是给给每一个state给出一个好坏的评价。



## ——————————————





## Lec_1

这节课主要讲了强化学习的两个主体，Agent和Environment的组成，以及对Agent的分类。

### Overview 

####  1. Environment
​		Environment是Agent所处的环境，也是我们对现实场景的建模（Model the world）。

​		而这个环境的 **Model** 是否包含 Transition dynamics 和 Reward function，决定了之后我们的强化学习是 Model-based 还是 Model-free。

​		对于 **Transition dynamics**，是指在这个环境中，从某一State转换到其他State的概率分布。

​		这个概率分布用满足Markov Property，指的是状态转移的方程与历史状态无关而仅仅和当前状态相关，数学表达为：

​                                                                  $P(s_{t+1}|s_t,a_t, ..., s_1, a_1) = P(s_{t+1}|s_t,a_t)$

​		对于 **Reward function**，有多种建立方式，具体怎么建就要结合相应的专业知识。

​														      	$R(s,a,s^,)=E[r_t|s_t=s,a_t=a,s_{t+1}=s^,]$

​												      			$R(s,a)=E[r_t|s_t=s,a_t=a]$

​													      		$R(s)=E[r_t|s_t=s]$

 

####  2. Agent

​		一般 Agent 由 Policy，Value function，Model（optional，就是上面说的model）组成。

​		**Policy** $\pi$ is a mapping from the agent state to an action. 就是指由当前状态去选取一个动作的概率。

​		**Value function** is an expected sum of discounted rewards. 给定一个 $\pi$ ，用未来状态的累计折扣奖励对当前状态s的评估值，其中 $\gamma$ 为折扣因子。

​                                                                 $V^\pi(s)=E_\pi[r_r+\gamma r_{t+1}+\gamma^2 r_{t+2}+ …|s_t=s]$

​		**Model** 指的就是Transition dynamics 和 Reward function。



#### 3. Taxonomy of Agent

​		

|  Agent type  |  Policy  | Value Function | Model |
| :----------: | :------: | :------------: | :---: |
| Value Based  | Implicit |       √        |  ？   |
| Policy Based |    √     |       ×        |  ？   |
| Actor Critic |    √     |       √        |  ？   |
| Model Based  |    ？    |       ？       |   √   |
|  Model Free  |    ？    |       ？       |   ×   |







## Lec_2

### Markov

这节课主要讲了MP-MRP-MDP的构成，并在最后给出通过MDP control 求解最优策略的方法。

#### 1. Markov process

​		MP is a stochasitc process that satisfies the Markov porperty.

​		在强化学习中，我们一般会给这个过程两个假设 （Markov chain）：

​	    (1) **Finite state space**， $|S|<\infin$。

​	    (2) **Stationary transition probabilities** ，意思是说，状态转移概率是time independent的。比如在 $T_i$ 时由“在天上”转到“在地上”的概率，和 $T_j$ 时由“在天上”转到”在地上“的概率时相同的（与遇到状态的时间无关）。



​		我们用tuple $(S,P)$ 来定义MP，其中：

- $S$ is finite state space
- $P$ is transition probability model

​    



#### 2. Markov reward process

​		MRP 即在 MP 的基础上加上 Reward function 和 Discount factor，同时，新增一条假设：

​		(3) **Stationary rewards**, 和假设2可类比，这里即 reward 是 time independent的。

​	

​		我们用tuple$(S,P,R,\gamma)$ 定义MRP，其中：

- $R$ is a funtion that maps states to rewards(real number). 这个$R$ 和 $r$ 是等价的，会互换

  ​																	$R(s)=E[r_t|s_t=s]$

- $\gamma$ is dicount factor



​		除此之外，还有几个重要的概念需要定义：

​		**Horizon** $H$​：the number of time steps in each episode of the process

​		**Return **$G_t$: the discounted sum of rewards starting at time t up to the horizon H

​																     		$G_t=\sum_{i=t}^{H-1}\gamma^{i-t}r_i$

​		**State value function** $V_t(s)$: the expected return starting from state s at time t

​															<u>V值函数的作用是判断这个状态的好坏程度</u>

​														        			$V^\pi_t(s)=E[G_t|s_t=s]$

​		**State-action value function** $Q^\pi_t(s,a)$:

​															<u>Q值函数的作用是通过它让Agent知道在这个状态下做什么动作能够得到最大奖励</u>

​																			$Q^\pi_t(s,a)=E[G_t|s_t=s,a_t=a]$ 



#### 3. Markov decision process

​		MDP is defined as tuple $(S,A,P,R,\gamma)$.

- $S,\gamma,H,G_t$ remain the same

- $A$ is finite action space

- $P,R$ need to take $a$ into account

  ​                               										$R(s,a)=E[r_t|s_t=s,a_t=a]$ 

  ​	

  ​	有了MDP，我们给它Policy（stationary policy），就可以通过计算每个状态的值函数去寻得最佳 $\pi$（课程中讲到某个给定的policy，通过Bellman Backup 和Bellman Operator 可以证明值函数收敛）。状态值函数的计算式如下：

  ​																	$V^\pi(s)=r(s,\pi(s))+\gamma\sum_{s'\in S}P(s^,|s,\pi(s))V^\pi(s^,)$



#### 4. MDP control

​		这个control，在这里，我觉得可以翻译成“**做决策**”的意思。

​		MDP Control的目的是 <u>找到一个最优的策略 $\pi$ 使得在该策略下的状态价值函数最大</u>:

​										        							$\pi^\star(s)=arg\ max_\pi V^\pi(s)$

​		课程里说这个最优状态值函数一定存在且唯一，并且是一个deterministic stationary policy（是time independent的）。

​		寻找最优Policy有三种方法，这里重点讲两种。

##### 4.1 Policy Iteration

​		PI算法首先随机初始化一个策略，通过evaluate这个策略得到关于这个策略的值函数，然后再不断优化这个策略（取得更大的值函数值）达到拟合。

<img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\PI.png" alt="PI" style="zoom:60%;" />

​		其中policy improvement 的算法如下：

​                          	<img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\PI improvement.png" alt="PI improvement" style="zoom:60%;" />



##### 4.2 Value Iteration

​		VI则是将价值函数初始化为0向量，每一次迭代计算当前值函数的Bellman Optimality Operator，由于该算子是一个压缩算子，因此具有唯一的不动点，并且该不动点是最优点，最后通过值函数来取得最优的策略。

<img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\VI.png" alt="VI" style="zoom:60%;" />



## Lec_3 + Lec_4

#### Policy Evaluation and Control in Model-free Env

这节课主要讲怎么样在缺少MDP模型内部的状态转移函数的情况下进行Policy Evaluation。

之前我们是通过动态规划来policy evaluation的，可是动态规划（PI, VI）需要知道状态转移函数。

<img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\DP.png" alt="DP" style="zoom:60%;" />



Policy Evaluation 指的是：对一个状态 $s$, 要去计算 $V^\pi(s)=E_\pi[G_t|s_t=s]$.  在已知状态转移函数的情况下，我们可以通过这个函数算出下一个可能的状态，那没有没有这个函数怎么办呢？大量采样，获取经验。



#### 1. Monte Carlo

​		要去估计一个指标的quantity，可以通过大量的采样，把每次得到的结果求和取平均，得到的即是quantity。在RL中，若采用MC方法，则这个quantity就是 $V^\pi(s)$, 而 $V^\pi(s)$ 本就是在 $\pi$ 下 $s$ 时 $G_t$ 的期望值，所以在这里MC每一次返回的就是 $G_t$, 最后把所有的 $G_t$ 求和取平均，就可以得到 $V^\pi(s)$ 了。

​		为了让这个算法更具灵活性，我们更改取平均为乘一个 $\alpha$，

- 当 $\alpha=1/N(s)$ 时，等同于MC
- 当 $\alpha>1/N(s)$ 时，会倾向于忽略之前的data sample  

​		升级后这个方法叫做Incremental MC。

##### 1.1First-Visit MC

​		                    <img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\First MC.png" alt="First MC" style="zoom:60%;" />

##### 1.2 Every-Time MC

<img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\Every MC.png" alt="Every MC" style="zoom:60%;" />



#### 2. Temporal Difference

TD 算法结合了DP和MC，在每一步之后进行更新。

<img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\TD.png" alt="TD" style="zoom:60%;" />



##### 2.1 With SARSA

##### 2.2 With Q-learning



#### 3. Batch MC & Batch TD

​		就是事先采样一批 $h$, 之后投喂给这些算法。

​		课程中说Batch MC收敛于min MSE。给定一个确定的episode set，我们可以算出什么时候Batch MC的更新为0，也就是说最终每个状态的价值会收敛到sample出的batch的该状态的平均Return值。这和优化min MSE的结果相同。

​		而TD利用了马尔科夫性质，当解决具有马尔科夫性质的问题时，用TD会比较有帮助。



#### 4.Model free control

L3 解决了在没有 model 的情况下怎么去做policy evaluation，L4讲怎么在没有model 的情况下去做决策（model-free control）。



## ——————————————



## 插播Ⅰ Q-learning到底是什么

#### 1. Q-learning 是什么

​		Q-learning的本质就是选个最大的Q值函数然后做出action。通过最大化Q值函数得到的下一个$\pi$ 一定比上一个好。

#### 2. 从 Q-table 到 Q-network

​		Q-learning是在一张Q-table里记录了所有的state和action对应的Q-function，那么在之后遇到某个state就可以通过“查表”的方式去选择Q值函数最大的action。

​		然而，如果state和action空间组合太大，这时候Q-table就很难容纳所有的组合，所以就想能不能用一个函数来拟合这张表（这就是Value Function Approximation）。于是就引入了一个神经网络来进行泛化（一个regression的问题）。对于这个NN来说，**input是 $state的数据$，output是每一个action的 $Q-function$** ，然后就又可以通过“查网”的方式去选择Q值函数最大的action了。

#### 3. 怎么得到 table 和 net

​		然后最重要的其实是到底怎么去得到那张Q-table，那个Q-network。

- 精髓：用 $Q-function$ 去逼近我们想要的Target目标值 $G_t$

- Q-table：是用TD进行更新的（**<u>但是这个公式还是没看懂，以后再说</u>**），主要是要得到reward和下一个Q值函数。

  <img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\Qlearning.png" alt="Qlearning" style="zoom:50%;" />

- Q-network：用一个function去拟合Q-table，我们用一个神经网络去找到这个function。

  ​		既然是要训练一个神经网络，肯定是要有Loss function的；要有Loss function，那必然先得有标签。而这个标签就是我们要去逼近的Target 目标值 $G_t$:

  ​																		$R_t+\gamma\ Q(S_{t+1},A_{t+1})$ 

  (**<u>为什么R上面是t+1，下面是t？</u>**)

  那么形如

  ​																		$J(\theta)=(()-())^2$

  的Loss函数就出来了。

  ​		接下来去最小化它，就需要训练样本，样本哪来呢？通过**Epsilon Greedy**或者**Boltzmann Exploration**这样均衡Exploration-Exploitation的采样方法采样。

  ​		采的样用一次就没用了吗？那也太浪费了。这就用到了**Replay Buffer**，它是一个储存experience的地方。不同的 $\pi$ 与环境交互的轨迹都会放在里面，之后通过采batch的方法，去训练得到我们的Q-network。

  ​		最终，这样理论上就可以找到这个能够拟合Q-table 的 function了。

  ```markdown
  这个replay buffer对存放谁的玩耍经历是不在意的。
  我们训练的时候在意那条轨迹的什么？
  每一次的状态、动作，最主要的是对应的奖励。
  就像我在b站上存了很多大神打乒乓球的比赛，这些比赛就是不同的policy去交互存在replay buffer里的experience，而我就是那个要被训练的神经网络，我能从这些经验里学习吗？当然可以。因为这些就算不是我自己的经验，也有对应state和action的reward，这些是通用的东西。
  ```

  

#### 4. DQN 变种

		##### 4.1 Double DQN（DDQN）

- 为什么需要Double DQN？
  
- DQN有缺陷，它 estimate 的Q函数往往会偏大，估计过于乐观。
  
- 怎么实现Double DQN？

  - 与DQN不同的是，Double DQN 有两个NN，一个用来选根据Q函数选 action，一个用来算 Q函数的value。
  - Remind that 其实在DQN中，我们其实就有两个NN，一个target network是固定不变的，且兼顾算value 和做action的职责；另一个会更新参数的network要去逼近这个target。
  - Double DQN 只是把算value的工作给了那个target network，做action的工作给更新参数的network。
  - 形式上，

  <img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\Double dqn.png" alt="Double dqn" style="zoom:50%;" />



##### 4.2 Duel DQN

- Duel DQN 的好处
  - 不用穷举所有的action，采样比较效率。

- 怎么实现Duel DQN
  - 原本的DQN输出直接是不同action的Q值函数，但是Duel DQN 输出分为两部分，再把这两部分加起来。

<img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\Duel DQN.png" alt="Duel DQN" style="zoom:50%;" />



##### 4.3 Prioritized Experience Replay

- 为什么需要这个优化方法
  - 之前说到过的经验池，我们是在其中进行无差别采样。自然而然就能想到，要是我们可以把那些训练的不太好的多采到样去训练几次，效果一定会更好，这就用到了这个对Replay Buffer 优化的方法。
- 实现方法
  - TD error 比较大（TD error 就是 network 的 output 跟 target 之间的差距）的给一个较大的概率被采样到。



##### 4.4 Muti-step：balance between MC and TD

- MC和TD
  - MC是采了一个episode的样再去更新，TD是一步一个更新。
- 结合起来实现
  - 从这一个state开始，跨几步（TD），然后后面的全算进去（MC）。这个跨几步就是个超参数了。



##### 4.5 Noisy net

- 是什么Noisy net
  - 这个操作就是为了能让exploration更多元化一点。
  - 实现的方法主要有两个方向
    - 不管哪一episode，当遇到相似state的时候会做出不同action（Noisy on action）——乱试
    - 一episode中遇到相似state的时候action相同，但不同的episode会不同（Noisy on parameters）——符合某分布地试
- 怎么实现
  - Noisy on action：$\epsilon-greedy$ 就是个例子。
  - Noisy on parameters：给做决定的那个target NN的每个神经元权重参数加个高斯模糊，每次episode就固定住这个被模糊了的target NN，那么算的value，做出的action就都是服从高斯模糊分布的。



##### 4.6 Distributional Q-function

- 普通DQN输出的Q函数值是一个期望值，我们并不知道其中的分布。而这个方法就是model 了每个输出的分布（**<u>不是很懂</u>**）。



##### 4.7 Ranbow DQN

- 综合了以上6种加原视DQN的方法

<img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\ranbow DQN.png" alt="ranbow DQN" style="zoom:50%;" />



#### 5. Q-learning 不足

​		只在离散的动作环境下使用起来比较方便，到连续的情况下，将其拓展，得到DDPG。





## 插播Ⅱ Q-learning 进化到 DDPG

​		DDPG = Deep Deterministic Policy Gradient，

​		借鉴了 DQN 的技巧：目标网络和经验回放。经验回放跟 DQN 一样，但 target network 的更新跟 DQN 不一样。

#### 1. 对比Q-learning

<img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\dqnDDPG.png" alt="dqnDDPG" style="zoom:50%;" />

- DQN：是value-based的。给一个state，Q-network会计算出当前状态对应所有动作的Q值函数，选一个Q值函数最大的action
- DDPG：是actor-critic的。给一个state，直接通过policy-network输出一个action，Q-network会根据当前state和action算出唯一一个Q值函数。

#### 2. DDPG中两个net的优化

<img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\op2.png" alt="op2" style="zoom:50%;" />

- Q-network的优化区别就在于下一次的动作选取是根据当前的Policy-network来的。采样什么的还是通过经验池来采。
- 因为这是个actor-critic结构的，在这里Q-net就是critic，Policy-net就是actor；critic的评判标准是从环境（采样的样本）中学到的，而actor表演（产生动作）的好坏是从critic那儿知道的。所以思路很清晰，critic要去逼近当前s和a下的期望值，actor要去逼近critic给的要求actor要去达到的Q值函数。

```makedown
就像老师和学生的关系。
```





## 插播Ⅲ Sparse reward 如何让Agent合理探索

#### 1.  Intrinsic Curiosity Module

- 为什么要有这个？

  - 就像人一样，人大多数时候就是活着，做的一些“探索”得不到什么奖励反馈，而又“探索”不到新的领域，就得不到进步。所以要赋予Agent（人）好奇心，带脑子地去大胆探索未知领域。

- 如何实现

  <img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\ICM.png" alt="ICM" style="zoom:50%;" />

  - ICM本质是个对“好奇心”建了模的奖励函数。

    - 我们给赋予Agent的Curiosity 建模便得到一个函数，这个函数也是的奖励函数，输入是 $s_t,a_t,s_{t+1}$ , 输出一个reward值，最终也会被加入到Return中去。

  - ICM设计

    - ICM中有个network，这个network 输入 $s_t,a_t$, 输出预测的 $s_{t+1}$ . 我们本来就是有这段episode的，所以就能拿这个预测出来的 $s_{t+1}$ 和真实的 $s_{t+1}$ 去做一个比较，差得越多那么给的ICM reward就越大，reward大了自然之后再采到这条episode 的概率就大了。但是这样有问题啊，人说难被探索到的状态一定是好的状态呢？所以接下来我们要给Agent装两个脑子。
    - 两脑子还是两个network，一个负责输入 $s_t$ 输出一个逼近真实 $a_t$ 的action，一个负责输入 $s_{t+1}$ 输出一个逼近真实 $a_{t+1}$ 的action。其实起到的是对state的关键特征提取的作用，因为在去逼近真实action的时候，network就是会把这几个关键特征的权重搞大，其他和这个预测真实action的特征就弱化掉。

    <img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\1603963622(1).png" alt="1603963622(1)" style="zoom:50%;" />

#### 2. Curriculum learning

- 就是像人一样由简单到难去学



#### 3. Hierarchical learning

- 在上级的旨意下，遵循上级的旨意，分步去实现一件事情

<img src="C:\Users\Symmetric_QIAN\Desktop\CS234 笔记\src\1603964246(1).png" alt="1603964246(1)" style="zoom:50%;" />


## ——————————————



