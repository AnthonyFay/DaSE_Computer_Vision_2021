# 1 计算机视觉简介



## 定义和基本任务

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211221193027270.png" alt="image-20211221193027270" style="zoom: 50%;" />

### 定义

用计算机模拟人类的视觉系统，去完成人类的视觉任务。

### 五大任务

1. **分类**
2. **语义分割**
3. **分类+定位**
4. **目标检测**
5. **实例分割**



## 发展历史

- 1960s: The Summer Vision Project
- 1970s: CV 算法基础的形成
- 1980s: 更严格的数学框架
- 1990s: 研究聚焦和技术融合
- 2000s: Machine Learning
- 2010s: Deep Learning

---



# 2 图像分类

## $\text{KNN}$

从 $K$ 个距离最近的邻居中找占据多数的类别作为预测结果。

### 时间复杂度

- 训练：$O(1)$
- 测试：$O(N)$ （无论是“几NN”，预测都需要计算与每个样本的距离，区别只是选择最近点的个数不同）

### 为什么有白色？

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211221194752041.png" alt="image-20211221194752041" style="zoom:50%;" />

当 K 值比较大时，图中出现了白色区域。白色区域是 KNN 无法决策的区域，可能这个区域离 2 个或 2 个以上的类的中心点是最近的。可以调整 K 值、或修改最近邻的决策条件、或修正距离公式，来减少白色区域。

### 超参数

- 选取的邻居个数：K

  - 选取的 K 值太小

    会有模型过拟合的风险。直观地看，过拟合就是学习到了很多“局部信息”，或者是“噪音”，使得模型中包含很多“不是规律的规律”。在 KNN 中，K 越小，就越有可能让模型的学习结果被“局部信息”所左右。在极端情况下，K = 1，KNN 算法的结果只由离待预测样本最近的那个点决定，这使得 KNN 的结果大概率被“有偏差的信息”或者“噪音”所左右，是一种过拟合。

  - 选取的 K 值太大

    会有模型欠拟合的风险。与 K 值太小的情况相反，K 值过大意味着离待预测样本点较远的点也会被包含进来对其判别产生影响，此时就会欠拟合。

- 选取的距离计算方式：$d(x, y)$

  明氏距离：
  $$
  d(x,y)=(\sum_{i=1}^N\left | x_i-y_i \right |^p)^{\frac1p}
  $$

#### 调参方法

可以根据样本的分布，先选择一个较小的值，然后通过**交叉验证**（也有人选择贝叶斯、bootstrap 等）选择一个合适的 K 值（一般 K 超过 20，上限是 $\sqrt n$，随着数据集的增大，K 的值也要增大）。

对于大型数据集和深度学习较少使用交叉验证。

### 算法评价

#### 优点

- 简单

#### 缺点

- 训练阶段是简单的标签记忆(non-parametric)
  - 预测效率低下，时间复杂度 $O(N)$
  - lazy learner，训练阶段不做任何泛化，而图片分类需要具备泛化能力的分类器(eager learner)
- 像素距离和图像信息的语义鸿沟
  - 图片像素距离相近 $\neq$ 图像信息相近
- 训练集需要在整个像素空间中均匀分布，导致 curse of dimensionality



## Linear Classifier

$$
f(\boldsymbol x,W)=W\boldsymbol x+\boldsymbol b
$$

取分数最高的类别为预测类别。

### 缺陷

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211221200301976.png" alt="image-20211221200301976" style="zoom:50%;" />

---



# 3 损失函数和优化

## 损失函数

定义一个衡量输出分数好坏的函数：损失函数（目标函数），最小化损失函数来获得更好的参数。

令 $\boldsymbol s=f(\boldsymbol x_i,W)$，即图片 $i$ 的所有类别分数（一个向量）。

- hinge loss
  $$
  L_i=\sum_{j\neq y_i}\text{max}(0,\ s_j-s_{y_i}+\Delta)
  $$

- cross-entropy loss
  $$
  L_i=-\text{log}\frac{e^{\boldsymbol s_{y_i}}}{\sum_je^{\boldsymbol s_j}}
  $$

练习题：

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211221203135939.png" alt="image-20211221203135939" style="zoom:50%;" />

- hinge loss
  $$
  \begin{align}
  &L_1=\text{max}(0,5.8-7.9+1)+\text{max}(0,-1.9-7.9+1)=0\\
  &L_2=\text{max}(0,3.3+0.8+1)+\text{max}(0,6.5+0.8+1)=13.4\\
  &L_3=\text{max}(0,-1.1-2.5+1)+\text{max}(0,3.4-2.5+1)=1.9\\
  &L_4=\text{max}(0,1.5-2.3+1)+\text{max}(0,4.4-2.3+1)=3.3\\
  &L=\frac1N\sum_{i=1}^NL_i=4.65
  \end{align}
  $$

- cross-entropy loss
  $$
  \begin{align}
  
  &L_1=-\text{log}(\frac{e^{7.9}}{e^{7.9}+e^{5.8}+e^{-1.9}})\\
  &L_2=-\text{log}(\frac{e^{-0.8}}{e^{3.3}+e^{-0.8}+e^{6.5}})\\
  &L_3=-\text{log}(\frac{e^{2.5}}{e^{-1.1}+e^{3.4}+e^{2.5}})\\
  &L_4=-\text{log}(\frac{e^{2.3}}{e^{2.3}+e^{1.5}+e^{4.4}})\\
  &L=\frac1N\sum_{i=1}^NL_i
  \end{align}
  $$

### 问答题

#### hinge loss

- **Q1：假设一个 SVM 分类器已经能够在数据集上正确分类，那么微调该分类器，使得输出分数发生小幅变化（比如 0.001），是否会改变损失函数的值？为什么？**

  当 $s_j-s_{y_j}<-\Delta$ 时，对损失函数的值没有影响；反之，则会略微改变损失函数的值。

  这是由 hinge loss 的计算公式 $L_i=\sum_{j\neq y_i}\textrm{max}(0, \boldsymbol s_j-\boldsymbol s_{y_j}+\Delta)$ 决定的。

- **Q2：hinge loss 的最大值和最小值分别是多少？**

  最小值为 0，最大值为正无穷（理论上）

- **Q3：假如初始化 $W$ 接近 0，导致所有输出分数都 $\approx$ 0，那么 $L_i$ 约等于多少？**

  $C-1$，$C$ 为类别数

- **Q4：假如去掉 $j\neq y_i$ 的限制，损失函数如何变化？**

  增加 1

- **Q5：假如在 $L_i$ 中使用 $\textrm{max}(0, \boldsymbol s_j-\boldsymbol s_{y_j}+2)$ 代替 $\textrm{max}(0, \boldsymbol s_j-\boldsymbol {s}_{y_j}+1)$，有什么影响？**

  没有影响，hinge loss 只关注输出分数之间的差异，这里的常数只起到 scale 参数的作用。

- **Q6：假如 $W$ 使得 $L=0$（完美），请问 $W$ 是否唯一？**

  不唯一，$c\times W$ 也使得 $L=0$，$c$ 为任意正整数

#### cross-entropy loss

- **Q1：如果有输出分数发生微小改变（比如±0.1），损失函数是否发生改变？**

  是的，正确类别和错误类别输出分数差距越大，损失函数越小

- **Q2：损失函数 $L_i$ 的最大值和最小值分别是多少？**

  最小值为 0（理论上），最大值为正无穷（理论上）

- **Q3：假如初始化 $W$ 接近 0，导致所有输出分数都 $\approx$ 0，那么 $L_i$ 约等于多少？**

  $\text{log}\ C$，$C$ 为类别数

## 正则化项

$$
L(W)=\frac1N\sum_{i=1}^Nl(f(\boldsymbol x_i,W), y_i)+\lambda R(W)
$$

- $\frac1N\sum_{i=1}^Nl(f(\boldsymbol x_i,W), y_i)$ 使模型尽可能拟合数据集
- $\lambda R(W)$ 防止模型过度拟合训练集

L1 偏向于使参数集中在**少数**输入像素上；L2 偏向于使参数分布在**所有**像素上

### 意义

- 缩小参数空间
- 调整参数偏好的分布
- 提高模型泛化能力

## 图像特征抽取

线性分类器应用于图像，往往也需要对原始像素做特征抽取，利用抽取的特征训练模型，提高预测性能。

### Colour Histogram

1. 建立色相哈希表
2. 哈希每个像素值，并计算每个 key 中像素的个数
3. 将哈希结果作为模型输入

### Histogram of Oriented Gradients

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211221210745405.png" alt="image-20211221210745405" style="zoom:50%;" />

## 优化

### 梯度下降

$$
W_{\text{new}}=W-\lambda\nabla_W L
$$

#### 数值梯度

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211221212803063.png" alt="image-20211221212803063" style="zoom:50%;" />

#### 解析梯度

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211221212849968.png" alt="image-20211221212849968" style="zoom:50%;" />

##### hinge loss

$$
L_i=\sum_{j\neq y_i}\text{max}(0,\ w_j^Tx_i-w_{y_i}^Tx_i+\Delta)
$$

其中，$w_j$ 是 $W$ 的第 $j$ 行的转置（列向量）。
$$
\frac{\partial L_i}{\partial w_j}=\boldsymbol 1(w_j^Tx_i-w_{y_i}^Tx_i+\Delta>0)x_i
$$

$$
\frac{\partial L_i}{\partial w_{y_i}}=-\left (\sum_{j\neq y_i}\boldsymbol 1\left(w_j^Tx_i-w_{y_i}^Tx_i+\Delta>0\right)\right )x_i
$$

##### cross-entropy loss

$$
L_i=-\text{log}\frac{e^{w_{y_i}^Tx_i}}{\sum_{j}e^{w_j^Tx_i}}
$$

其中，$w_j$ 是 $W$ 的第 $j$ 行的转置（列向量）。
$$
\frac{\partial L_i}{\partial w_{j}}=
\left\{\begin{matrix}
\begin{aligned}
&\frac{e^{w_{j}^Tx_i}}{\sum_je^{w_j^Tx_i}}x_i,\ y_i\neq j\\ 
&(\frac{e^{w_{y_i}^Tx_i}}{\sum_je^{w_j^Tx_i}}-1)x_i,\ y_i=j
\end{aligned}
\end{matrix}\right.
$$

### 随机梯度下降

使用梯度下降：

- 每次更新 $W$ 需要遍历所有数据。
  - 优势：每次迭代 loss 下降快
  - 劣势：一次迭代需要遍历所有数据，并且容易陷入 local minima

随机梯度下降：

- 每次选取一个 sample 集（minibatch，大小一般为 32/64/128/256）
- 利用在sample集上的损失计算近似梯度
  - 优势：迭代更新速度快，并且往往因为 minibatch 含有噪声而避开 local minima
  - 劣势：每次迭代 loss 下降较慢

---



# 4 神经网络和反向传播

## 激活函数

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222155605350.png" alt="image-20211222155605350" style="zoom:50%;" />

## 全连接的神经网络

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222155651460.png" alt="image-20211222155651460" style="zoom:50%;" />

## 反向传播

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222155947445.png" alt="image-20211222155947445" style="zoom:50%;" />

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222164357075.png" alt="image-20211222164357075" style="zoom:50%;" />

---



# 5 卷积神经网络

## 网络结构

### 卷积层

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222165035394.png" alt="image-20211222165035394" style="zoom:50%;" />

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222165116385.png" alt="image-20211222165116385" style="zoom:50%;" />

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222165134397.png" alt="image-20211222165134397" style="zoom:50%;" />

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222165159245.png" alt="image-20211222165159245" style="zoom:50%;" />

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222165218356.png" alt="image-20211222165218356" style="zoom:50%;" />

如何计算 activation maps 的大小：

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222165415572.png" alt="image-20211222165415572" style="zoom:50%;" />

### 池化层

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222170333293.png" alt="image-20211222170333293" style="zoom:50%;" />

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222170427684.png" alt="image-20211222170427684" style="zoom:50%;" />

如何计算 feature maps 的大小：

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222170453440.png" alt="image-20211222170453440" style="zoom:50%;" />

### 完整的 CNN

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222171158872.png" alt="image-20211222171158872" style="zoom:50%;" />

## 反向传播

### 池化层

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222171253922.png" alt="image-20211222171253922" style="zoom:50%;" />

### 激活层

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222171459568.png" alt="image-20211222171459568" style="zoom:50%;" />

### 卷积层

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222203359854.png" alt="image-20211222203359854" style="zoom:50%;" />

---



# 6 神经网络的训练

## 激活函数

### Sigmoid

$$
\sigma(x)=\frac1{1+e^{-x}}
$$

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222205546382.png" alt="image-20211222205546382" style="zoom:50%;" />

#### 优点

- 将数值压缩到 $(0,1)$
- 曲线平滑，易于求导

#### 缺点

- 容易饱和输出

  当 $x$ 稍微大一点或小一点，

  - $\sigma(x)$ 的值都接近 $0$ 或 $1$，且基本维持不变
  - 局部梯度接近于 $0$，造成回传梯度消失，参数无法更新

- 不是零均值（zero-centred）

  <img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222205718299.png" alt="image-20211222205718299" style="zoom:50%;" />

- `exp()` 函数计算复杂度高

### tanh

$$
\text{tanh}(x)=\frac{\text{sinh}(x)}{\text{cosh}(x)}=\frac{e^{2x}-1}{e^{2x}+1}
$$

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222210046430.png" alt="image-20211222210046430" style="zoom:50%;" />

#### 优点

- 将数值压缩到 $(0,1)$
- 曲线平滑，易于求导

#### 缺点

- 容易饱和输出
- `exp()` 函数计算复杂度高

### ReLU

$$
f(x)=\text{max}(0,x)
$$

#### 优点

- 在正区间不会饱和
- 计算复杂度极低
- 收敛速度比 Sigmoid 和 tanh 快

#### 缺点

- 不是零均值
- 不压缩数据，数据幅度会随着网络加深不断增大
- 神经元坏死（Dead ReLU）
  - 由于参数初始化或者学习率设置不当，导致某些神经元的输入永远是负数
  - 导致相应的参数永远不会更新
  - 采用合适的参数初始化和调整学习率可以缓解这种现象

### Leaky ReLU

$$
f(x)=\text{max}(0.01x,x)
$$

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222210905681.png" alt="image-20211222210905681" style="zoom:50%;" />

#### 优点

- 不会造成饱和
- 计算复杂度低
- 收敛速度比Sigmoid和tanh快
- 近似零均值
- 解决ReLU的神经元坏死问题

#### 缺点

- 数值幅度不断增大
- 实际表现不一定比 ReLU 好

### ELU

$$
f(x)=
\left\{\begin{matrix}
\begin{aligned}
&x,&x>0\\
&\alpha(\text{ exp}(x)-1),&x\leq0
\end{aligned}
\end{matrix}\right.
$$

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222211305223.png" alt="image-20211222211305223" style="zoom:50%;" />

#### 优点

- 不易造成饱和
- 收敛速度比 Sigmoid 和 tanh 快
- 近似零均值
- 解决 ReLU 的神经元坏死问题

#### 缺点

- `exp()` 计算复杂度高
- 表现不一定比 ReLU 好

### 实际搭建模型的时候

- 首选 ReLU，但是要注意初始化和学习率设置
- 不要使用 Sigmoid
- 可以使用 tanh，不过效果通常一般
- 尝试其余激活函数

## 数据预处理

### 调整图像大小

- 一般将图像裁剪为大小一致的正方形
- 可以通过 downscale 或者 upscale 调整大小

### 图像序列化

将图片转化为像素值数组，并附上相应标签

### 零均值化

- 将原始像素值从 $[0, 255]$ 调整为 $[-128, 127]$
- 平均
  - 计算所有图像的平均，得到 mean image（mean image 和原始图像的大小一致），将每个图像减去 mean image（e.g. AlexNet）
  - 每个 channel 减去各自的平均（e.g. VGGNet）
  - 每个 channel 减去各自的平均，再除以 std（e.g. ResNet）

### 标准化

将数值压缩到一个较小的区间

- 减小损失函数对权重参数变化的敏感度
- 方便优化参数

实际搭建模型的时候，一般先不标准化

## 权重参数的初始化

### 意义

- 参数初始化过小（$\approx$ 0）
  - 回传梯度快速接近 0，梯度消失，导致靠近输入层的梯度无法更新
- 参数初始化过大（$$>$$ 1）
  - 回传梯度快速增大，梯度爆炸，导致靠近输入层的梯度更新太快

### 参数初始化方法

#### 全部初始化为 0

- 每一层的神经元输出完全一样
- 每一层的参数梯度完全一样
- 每一层的参数永远相同
- 无法学习数据特征

#### 完全随机初始化

- 零均值，方差较小的正态分布随机数
  - 网络越深，所有激活越靠近 0
  - 越靠近输出层的梯度 $\frac{\partial L}{\partial w}$ 越接近 0
  - 靠近输出层的 $w$ 无法更新
- 零均值，方差较大的正态分布随机数
  - 网络越深，所有激活越饱和
  - 激活门的局部梯度接近 0，回传梯度消失
- 尽可能保持 $y$ 和 $x$ 的分布保持一致

#### Xavier 初始化

$$
W\sim U\left( -\sqrt\frac{6}{n_i+n_{i+1}},\sqrt\frac{6}{n_i+n_{i+1}} \right)
$$
$n_i$ 是第 $i$ 层网络的大小

激活函数替换为 ReLU

缺点：

- 网络越深，所有激活越靠近 0
- 靠近输出层参数无法更新

#### He 初始化

$$
W\sim \left( U\left( -\sqrt\frac{6}{n_i+n_{i+1}},\sqrt\frac{6}{n_i+n_{i+1}} \right)\right)/2
$$

$n_i$ 是第 $i$ 层网络的大小

激活函数替换为 ReLU

### 实际搭建模型的时候

优先使用 ReLU+He 初始化

## Batch Normalization

> 深度神经网络涉及到很多层的叠加，而每一层的参数更新会导致上层的输入数据分布发生变化，通过层层叠加，高层的输入分布变化会非常剧烈，这就使得高层需要不断去重新适应底层的参数更新。为了训好模型，我们需要非常谨慎地去设定学习率、初始化权重、以及尽可能细致的参数更新策略。
> Google 将这一现象总结为 Internal Covariate Shift，简称 ICS
>
> - 内部输出的分布由于参数变化而不停变化
> - 导致激活容易饱和或者趋近 0
> - 神经网络训练不易收敛

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222220623933.png" alt="image-20211222220623933" style="zoom:50%;" />

但是可以看到，batch normalization 针对的是一整个 batch 进行一阶统计量及二阶统计量的计算，即是隐式的默认了每个 batch 之间的分布是大体一致的。

推理时可能只有一个或者几个样本，无法有效计算 $\mu$ 和 $\sigma^2$。希望使用固定的 $\mu$ 和 $\sigma^2$，使用训练过程中保存的 $\mu$ 和 $\sigma^2$ 来估计真实值。

<img src="E:\undergraduate\junior\SEM1\计算机视觉\复习.assets\image-20211222220841223.png" alt="image-20211222220841223" style="zoom:50%;" />

