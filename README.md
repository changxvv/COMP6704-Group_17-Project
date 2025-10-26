# COMP 6704 Group Project

选择**一个**问题进行研究：

# 研究问题：Supply Chain Network Optimization

研究的子问题：multi-commodity network flow (MCNF) problem

问题定义：

>在由工厂/仓库/客户组成的有向网络上，为多种商品安排运输，把各自的需求从供给地送到需求地。
>
>成本按件线性计费（单位成本 $\times$ 流量），允许同一商品分流走多条路径；每条运输通道的容量为所有商品共享，即各商品在该通道上的流量之和不能超过其容量。必须满足每种商品的总需求，并且仓库只做中转（中间节点对每种商品遵守流量守恒）。
>
>最小化总运输成本。

具体来说，是线性费用、可分流（连续变量）、各商品共享弧容量的**最小成本多商品网络流**。

## 弧式（arc-based）最小成本 MCNF（标准 LP）

给定有向图 $G=(V,A)$（ $V$ 为节点（仓库）集， $A$ 为弧（运输通道）集），商品集合 $K$。弧 $(i,j)\in A$ 的共享容量（运输通道的运输数量上限）为 $u_{ij}\ge0$。商品 $k$ 在弧上的单位成本为 $c^{k}_{ij}$。

用 $b_i^k$ 表示节点 $i$ 对商品 $k$ 的净供给（源点为正、汇点为负、中转为 0 且 $\sum_i b_i^k=0$）。决策变量 $x^{k}_{ij}\ge0$ 为商品 $k$ 在弧 $(i,j)$ 上的运输量：

$$
\begin{aligned}
\min \ & \sum_{k\in K}\sum_{(i,j)\in A} c^{k}_{ij}\,x^{k}_{ij}\\
\text{s.t. }
& \sum_{(i,j)\in A} x^{k}_{ij}-\sum_{(j,i)\in A} x^{k}_{ji}=b_i^k, &&\forall i\in V,\ \forall k\in K \quad(\text{流量平衡})\\
& \sum_{k\in K} x^{k}_{ij}\le u_{ij}, &&\forall (i,j)\in A \quad(\text{容量共享})\\
& x^{k}_{ij}\ge0, &&\forall (i,j)\in A,\ \forall k\in K .
\end{aligned}
$$

等价地，也常用 $(s_k,t_k,d_k)$ 指定每个商品的源、汇与需求（把 $b_{s_k}^k=d_k,\ b_{t_k}^k=-d_k$，其余为 0）。

## 实现算法

参考，自己也可以找其他的，**至少**实现一个。

### Simplex Method（原始/对偶单纯形）

- **原始单纯形（Primal）**：在可行域的边上移动，选入“最能降目标”的非基变量，维持原始可行。
- **对偶单纯形（Dual）**：维持对偶可行，逐步修复原始可行（在存在“不可行基”时更稳）。


### Interior-Point Method（障碍法 / 预测-校正）

- 在可行域内部沿**障碍轨道**逼近最优边界；Mehrotra Predictor–Corrector 是工业标准。
- 每步解 KKT 线性方程组（稀疏对称不定/正定化后求解），数十步内收敛。

### Column Generation（Dantzig–Wolfe 分解）

- 将每个商品 $k$ 的流改用**路径变量**表示（从源到汇的可行路径），容量约束作耦合。
- 解一个**受限主问题（RMP）**：只含当前选取的少量路径列；
- 定价子问题（**Pricing**）：对每个商品解一次**最短路**（边权为**降低成本**：原始成本减去容量约束的对偶价），若出现负降低成本路径，则把它作为新列加入 RMP，迭代直到无改进列。

### First-Order Primal–Dual（PDHG / PDLP 一类）

把 LP 写成鞍点： $\min_x f(x)+g(Ax)$ 处理。

PDHG: https://link.springer.com/article/10.1007/s10851-010-0251-1

PDLP: https://arxiv.org/abs/2106.04756

## 需要做的工作

1. 搜文献：conducting a comprehensive **literature review**
2. 代码跑算法：various optimization algorithms using platforms such as Python or MATLAB
3. 实验画图：performing experimental comparisons of their performance. **Ablation studies and sensitivity analysis are highly valued.**
4. 总结：Finally, you should summarize the strengths and weaknesses of the different algorithms.

>[!important]
>each team member should independently implement at least one algorithm related to the chosen optimization problem.

## 最终成果
1. Project Report (page limit: up to 15 pages, _NeurlIPS 2025_ Style): Introduction, Methodology, Experimental Results and Analysis, and Conclusions.
2. Group Presentation (time requirement: 30 minutes presentation + 10 minutes Q&A): recommended to follow the same structure as the Project Report for consistency and clarity. **Each group member is required to present and answer questions.**

>[!note]
>**GitHub repository link** must be included in the report (either in the abstract or as a footnote)
> - A comprehensive README documentation
> - make sure you have included everything that allows to reproduce all experimental results presented in the report.

## 计划

  | 阶段       | 任务           | 具体内容                                                | 截止日期 |
  |------------|----------------|---------------------------------------------------------|----------|
  | **Week 1** | 文献调研       | 研究MCNF问题相关文献，确定自己的算法任务                | 10.2     |
  | **Week 2** | 算法实现       | 每个成员实现自己的算法，并进行简单测试，确定dataset     | 10.9     |
  | **Week 3** | 运行测试       | 完善算法实现，进行统一的测试，跑dataset                 | 10.16    |
  | **Week 4** | 准备报告与展示 | 制作slides，准备presentation，撰写项目报告（DDL 10.26） | 10.23    |
