# Task0 机器学习概述

## 1、机器学习分类

##### 1.1监督学习（supervised learning）

Applications in which the training data comprises examples of the input vectors along with their corresponding target vectors are known as *supervised learning* problems.

监督学习的训练集要求包括输入和输出

主要应用于分类和预测。常见的监督学习算法包括回归分析和统计分类。

##### 1.2非监督学习（unsupervised learning）

In other pattern recognition problems, the training data consists of a set of input vectors x without any corresponding target values.

在非监督学习中，无须对数据集进行标记，即没有输出。其需要从数据集中发现隐含的某种结构，从而获得样本数据的结构特征，判断哪些数据比较相似。

无监督学习的典型算法有自动编码器、受限玻尔兹曼机、深度置信网络等；典型应用有：聚类和异常检测等。

##### 1.3半监督学习

半监督学习是监督学习和非监督学习的结合，其在训练阶段使用的是未标记的数据和已标记的数据，不仅要学习属性之间的结构关系，也要输出分类模型进行预测。与使用所有标签数据的模型相比，使用训练集的训练模型在训练时可以更为准确，而且训练成本更低，在实际运用中也更为普遍。

##### 1.4强化学习（reinforcement learning）

Here the learning algorithm is not given examples of optimal outputs, in contrast to supervised learning, but must instead discover them by a process of trial and error

用于描述和解决智能体（agent）在与环境的交互过程中通过学习策略以达成回报最大化或实现特定目标的问题. 

## 2.泛化能力（generalization）与过拟合（Overfitting）

##### 2.1 定义

The ability to categorize correctly new examples that differ from those used for training is known as *generalization*.

Overfitting is the tendency of data mining procedures to tailor models to the training data, at the expense of generalization to previously unseen data points

##### 2.2 bias和variance

total error = bias + variance

variance：组间方差

bias：实际误差

<img src="https://pic3.zhimg.com/80/v2-e20cd1183ec930a3edc94b30274be29e_hd.jpg">

##### 2.3 过拟合与欠拟合

**欠拟合**一般表示模型对数据的表现能力不足，通常是模型的复杂度不够，并且Bias高，训练集的损失值高，测试集的损失值也高.

**过拟合**一般表示模型对数据的表现能力过好，通常是模型的复杂度过高，并且Variance高，训练集的损失值低，测试集的损失值高.

<img src="https://pic1.zhimg.com/80/v2-22287dec5b6205a5cd45cf6c24773aac_hd.jpg">

##### 2.4 一般解决方案

高variance：增加训练样本、减少特征维数、减小模型复杂度

高bias：增加特征维数、增加模型复杂度

## 3.机器学习常见算法

#### 3.1导图

<img src="https://blog.griddynamics.com/content/images/2018/04/machinelearningalgorithms.png">

#### 3.2 常见算法介绍

##### 3.2.1 Linear Algorithms

https://mp.weixin.qq.com/s/MOuC5nNiv0GKvFv3hIC77A

1. Linear Regression
2. Lasso Regression 
3. Ridge Regression
4. Logistic Regression

##### 3.2.2 Decision Tree

1. ID3
2. C4.5
3. CART

##### 3.2.3 SVM 支持向量机

##### 3.2.4 Naive Bayes Algorithms

1. Naive Bayes
2. Gaussian Naive Bayes
3. Multinomial Naive Bayes
4. Bayesian Belief Network (BBN)
5. Bayesian Network (BN)

##### 3.2.5 kNN

##### 3.2.6 Clustering Algorithms

1.  k-Means
2.  k-Medians
3.  Expectation Maximisation (EM)
4.  Hierarchical Clustering

##### 3.2.7 K-Means

##### 3.2.8 Random Forest

##### 3.2.9 Dimensionality Reduction Algorithms

##### 3.2.10 Gradient Boosting algorithms

1. GBM

2. XGBoost

3. LightGBM

4. CatBoost

   ##### 3.2.11Deep Learning Algorithms

1.  Convolutional Neural Network (CNN)
2.  Recurrent Neural Networks (RNNs)
3.  Long Short-Term Memory Networks (LSTMs)
4.  Stacked Auto-Encoders
5.  Deep Boltzmann Machine (DBM)
6.  Deep Belief Networks (DBN)

## 4.机器学习常见模型评价指标

1. MSE(Mean Squared Error)

   均方误差是指参数估计值与参数真值之差平方的期望值; MSE可以评价数据的变化程度，MSE的值越小，说明预测模型描述实验数据具有更好的精确度。（ 𝑖i 表示第 𝑖i 个样本，𝑁N 表示样本总数）通常用来做回归问题的代价函数。

$$
MSE(y,f(x))=\frac{1}{N}\sum_{i=1}^{N}(y-f(x))^2
$$

2. MAE(Mean Absolute Error)

$$
MSE(y,f(x))=\frac{1}{N}\sum_{i=1}^{N}|y-f(x)|
$$

3. RMSE(Root Mean Squard Error)

$$
RMSE(y,f(x))=\frac{1}{1+MSE(y,f(x))}
$$

4. Top-k准确率

$$
Top_k(y,pre_y)=\begin{cases}
1, {y \in pre_y}  \\
0, {y \notin pre_y}
\end{cases}
$$

5. 混淆矩阵

   * 真正例(True Positive, TP):真实类别为正例, 预测类别为正例
   * 假负例(False Negative, FN): 真实类别为正例, 预测类别为负例
   * 假正例(False Positive, FP): 真实类别为负例, 预测类别为正例 
   * 真负例(True Negative, TN): 真实类别为负例, 预测类别为负例

   * 真正率(True Positive Rate, TPR): 被预测为正的正样本数 / 正样本实际数

   $$
   TPR=\frac{TP}{TP+FN}
   $$

   * 假负率(False Negative Rate, FNR): 被预测为负的正样本数/正样本实际数

   $$
   FNR=\frac{FN}{TP+FN}
   $$

   * 假正率(False Positive Rate, FPR): 被预测为正的负样本数/负样本实际数，

   $$
   FPR=\frac{FP}{FP+TN}
   $$

   * 真负率(True Negative Rate, TNR): 被预测为负的负样本数/负样本实际数，

   $$
   TNR=\frac{TN}{FP+TN}
   $$

   * 准确率(Accuracy)

   $$
   ACC=\frac{TP+TN}{TP+FN+FP+TN}
   $$

   * 精准率

   $$
   P=\frac{TP}{TP+FP}
   $$

   * 召回率

   $$
   R=\frac{TP}{TP+FN}
   $$

   * F1-Score

   $$
   \frac{2}{F_1}=\frac{1}{P}+\frac{1}{R}
   $$

   * ROC

   ROC曲线的横轴为“假正例率”，纵轴为“真正例率”. 以FPR为横坐标，TPR为纵坐标，那么ROC曲线就是改变各种阈值后得到的所有坐标点 (FPR,TPR) 的连线，画出来如下。红线是随机乱猜情况下的ROC，曲线越靠左上角，分类器越佳. 


   * AUC(Area Under Curve)

   AUC就是ROC曲线下的面积. 真实情况下，由于数据是一个一个的，阈值被离散化，呈现的曲线便是锯齿状的，当然数据越多，阈值分的越细，”曲线”越光滑. 

   

## 5.机器学习模型选择

##### 5.1交叉验证

所有数据分为三部分：训练集、交叉验证集和测试集

##### 5.2 k-折叠交叉验证

![image-20200108224943019](/Users/sissi/Library/Application Support/typora-user-images/image-20200108224943019.png)

- 设训练集为S ，将训练集等分为k份:$\{S_1, S_2, ..., S_k\}$. 
- 然后每次从集合中拿出k-1份进行训练
- 利用集合中剩下的那一份来进行测试并计算损失值
- 最后得到k次测试得到的损失值，并选择平均损失值最小的模型

##### 5.3 nested cross validation

![image-20200108225144444](/Users/sissi/Library/Application Support/typora-user-images/image-20200108225144444.png)

![](/Users/sissi/Documents/机器学习/微信学习小组/team-learning-master/初级算法梳理/502650-20171215192759371-739543867.png)



参考书：

指标：https://www.cnblogs.com/lliuye/p/9549881.html

​          https://blog.csdn.net/qq_20011607/article/details/81712811