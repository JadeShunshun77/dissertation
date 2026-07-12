# 项目进度总结

**项目**：Graph-Based Clustering and Classification of Publicly Labelled Ethereum Addresses Using Public On-Chain Data
**学生**：Qi Wang 2683181　**导师**：Zixing Song
**文档日期**：2026-07-11（更新至 Plan Week 9，23 Jul–29 Jul）
**提交截止**：2026-09-04

本文档记录截至目前的方法、结果与结论，供后续写作和答辩参考。所有结果均可通过对应脚本重新生成。

> **重要更新（Week 10 追加）**：Week 8/9/10 早期版本里所有"三类地址可被有效区分"式的表述都需要按类别拆开看——链上真实合约/EOA 查询（`eth_getCode`）发现本数据集中全部 DeFi 地址均为合约地址（150/150 = 100%），即 **DeFi 身份完全嵌套于合约身份之内**（而非与合约身份外延相等——Malicious_MEV 类中也有 34 个合约、CEX 类中有 2 个合约，为去混杂分析留出了同账户类型对照样本），这意味着完整三分类实验无法把"DeFi 的经济行为"和"泛化的合约地址特征"分离开。完整的混杂分析、去混杂后的补充实验和修正后的结论见 **§10**。本文档其余章节涉及"三类均可分"的措辞，均应结合 §10 一并阅读，不应单独引用。

### 研究定位

本项目的核心贡献不是"训练出一个 macro-F1 0.90+ 的以太坊地址分类器"——手工交易行为特征本身已经能做到这一点（见下文 §5、§6），分类器的绝对性能不是本文的创新点。真正的贡献是：**在一个真实、有采样约束（Top-100 counterparty、90 天窗口）的以太坊子图上，系统性地检验三种机制完全不同的图结构信息提取方式（Node2Vec 浅层无监督 embedding、手工图结构特征、GraphSAGE 端到端监督表示学习）相对于简单交易行为特征是否有增量价值**。**⚠️ 这三种方法不再收敛到同一个结论**（此前版本曾如此表述，已被 Week 10 的重复交叉验证复核推翻——见下文与 §5.3 的修正说明）：三者的结果是**分化的**，答案需要逐一给出：

1. **Node2Vec embedding 在完整活跃样本上对三个分类器均产生正向增量，但效应大小有明显模型和任务依赖性**（§5.3）：加入 embedding 后，LogisticRegression 提升约 0.068、XGBoost 约 0.035、RandomForest 约 0.016（10 次重复 5 折 CV，按重复聚合得到 10 个 repeat-level 配对估计，95% 置信区间均不含零，方向 100% 一致）。**这修正了此前"RandomForest 对 embedding 完全无反应"的结论**——原始单次 5 折 CV（n=5）未能检测到这个真实存在但较小的效应；重复交叉验证（10 次重复 5 折 CV）显示该效应能够稳定复现。但这个正向增量**不是普适的**：在 EOA-only 的 CEX vs Malicious_MEV 任务中（§10.8a），同样的重复检验显示 embedding 的增量基本消失（三模型均不显著），说明 embedding 的价值可能集中在涉及合约地址或不同合约功能角色的任务中，而不是所有地址分类场景都受益。
2. **手工图结构特征相对交易行为特征没有正向增量**（Week 9 消融，§6），在 RandomForest 下加入结构特征反而带来约 0.014 的稳定、统计显著的性能下降（10/10 次重复方向一致）——这与"结构特征存在冗余或噪声，轻微损害泛化"的解释一致（尚未证实具体机制）。这条负向结论在重复 CV 复核下被**加固**，未被推翻。
3. **端到端监督 GraphSAGE 与静态手工结构特征在同等输入信息下统计上无法区分**（苹果对苹果对比，p=0.24–0.32，§9）——在目前的比较范围内没有显示出相对传统结构基线的优势。

**RQ2 的准确回答需要区分不同的图信息提取方式**：Node2Vec embedding 在完整样本和 contract-only 场景下提供了小到中等、模型依赖的增量收益（§5.3、§10.8b），手工结构特征和 GraphSAGE 没有显示相同的稳定优势（§6.3、§9.4）。**这不再是一个单一的"图方法整体无效"的负向结论，而是一个更细致、也更有研究价值的分化结论**：不同的图结构信息提取机制，价值不同，且依赖具体任务构成（三分类完整样本 / EOA-only / contract-only）。

值得在方法论章节明确写出的一点：本项目的多轮自我核查——Week 7 的 NMI 类别不均衡偏差、Week 8 pagerank 并列地板问题、Week 10 的平衡子集复核（修正"embedding 无结构"）、账户类型混杂分析（部分修正"DeFi 无法识别"）、以及这次的重复交叉验证（修正"RF 对 embedding 无反应"）——每一次都不是推翻重来，而是用更强的统计设计替换掉检验力不足或未受控的早期设计。**单次 5 折分析曾未能检测到 RandomForest 中一个后来在重复评估下稳定复现的小幅 embedding 效应；同时，EOA-only 场景中 LogisticRegression 的一次边界性提升也未能在重复评估中复现**——这两处修正方向相反，共同说明仅依赖单次 5 折结果对较小差异作二元判断（"有/无效果"）具有较高的不确定性，这是本文方法论章节应该明确讨论的一条经验教训：重复交叉验证 + 置信区间 + 效应量一致性 + 实际意义阈值是更稳妥的标准，单一 p 值不应作为唯一依据。

---

## 0. 总体时间线定位

| Week | 日期 | 计划任务 | 状态 |
|---|---|---|---|
| 1–4 | Before 25 Jun | 种子收集、图构建、sanity check | ✅ 已完成 |
| 5 | 25 Jun–1 Jul | 交易行为特征 + 图结构特征工程 | ✅ 已完成 |
| 6 | 2 Jul–8 Jul | 完善特征矩阵、EDA、零膨胀/偏度处理 | ✅ 已完成 |
| 7 | 9 Jul–15 Jul | Node2Vec、UMAP/t-SNE、KMeans/Louvain 聚类 | ✅ 已完成（含额外稳健性检验） |
| 8 | 16 Jul–22 Jul | 监督分类器（LR/RF/XGBoost） | ✅ 已完成 |
| 9 | 23 Jul–29 Jul | 特征分组消融（transaction vs structural）、SHAP | ✅ 已完成 |
| 10 | 30 Jul–5 Aug | 稳健性/不均衡检验（§7）；可选：GraphSAGE（§9） | ✅ 已完成 |

---

## 1. 数据与图（Week 1–4，会话前完成）

- **种子地址**：500 个，`Malicious_MEV=299`、`DeFi=150`、`CEX=51`（[final_seed_address500.csv](final_seed_address500.csv)）
- **90 天有向以太坊交易 ego-network**：12,861 条边、8,029 个节点（[raw_edges.csv](raw_edges.csv)），Top-100 counterparty 抽样构建
- **Sanity check**（[sanity_check.py](sanity_check.py) → [seed_sanity_check_summary.csv](seed_sanity_check_summary.csv)）：无缺失 source/target、无自环、无重复 source-target 边

---

## 2. 特征工程（Week 5，[feature_engineering.py](feature_engineering.py)）

**输出**：[seed_feature_matrix.csv](seed_feature_matrix.csv)（500 行 × 42 列）、[feature_groups.json](feature_groups.json)

**方法**：
- 交易行为特征：in/out 交易数、ETH 金额、gas 费均值、zero-value 边比例，及 log1p 变换
- 图结构特征：in/out/total degree、weighted degree、PageRank（加权/非加权）、clustering coefficient、弱连通分量大小
- 缺失值处理：`out_gas_fee_mean`/`in_gas_fee_mean` 在无对应方向边时为 NaN（163/153 条），填 0 并加 `has_out_edge`/`has_in_edge` 标志区分"真实 0 费率"与"无边"
- 新增 `has_closed_triangle` 标志（clustering_coeff > 0 的二值版本），应对该特征大量为零的问题
- **特征分组**（`feature_groups.json`）：验证发现 `weighted_in/out/total_degree` 与 `in/out/total_tx_count` **数值 100% 完全重复**（因为边权重就是 tx_count，且无重复 source-target 边），故将 weighted-degree 系列归入 `transaction_behaviour` 组（19 个特征），`graph_structural` 组（11 个特征）只保留纯拓扑量（degree、PageRank、clustering_coeff、has_closed_triangle、wcc_size），避免 Week 9 消融实验中同一信号被算作两类特征

---

## 3. 探索性数据分析（Week 6，[eda.py](eda.py) → `eda_outputs/`）

### 3.1 活跃度（`activity_by_label.csv`）

| label | active | total | active_rate |
|---|---|---|---|
| CEX | 51 | 51 | 100% |
| DeFi | 86 | 150 | 57.33% |
| Malicious_MEV | 299 | 299 | 100% |

与 Project Plan 记录的 DeFi under-representation（86/150）完全一致。后续 EDA/embedding 分析均只用这 436 个活跃种子（非活跃种子全部特征为 0，会拉平分布图）。

### 3.2 偏度：raw vs log1p（`skewness_raw_vs_log.csv`，仅活跃种子）

| feature | raw_skew | log_skew |
|---|---|---|
| total_tx_count | 12.269 | -0.784 |
| total_eth_value | 14.613 | 2.302 |
| degree | 1.142 | 0.460 |
| weighted_degree | 12.269 | -0.784 |
| pagerank | 5.035 | **5.013**（几乎无改善） |

log1p 对 tx_count/degree 类特征效果很好，但对 pagerank 几乎无效——原因是 pagerank 数值本身接近 0（8029 节点，均值 ≈1.2e-4），log1p(x)≈x 在 x 很小时不产生压缩。**决定**：pagerank 改用 rank（分位数）变换，作为 Week 8 分类器训练前的预处理步骤（树模型不受影响，主要服务于 Logistic Regression）。

### 3.3 零膨胀（`zero_inflation_by_label.csv`，全部 500 个种子）

| label | n | %zero total_tx_count | %zero total_eth_value | %zero out_edge_count | %zero in_edge_count | %zero clustering_coeff |
|---|---|---|---|---|---|---|
| CEX | 51 | 0% | 5.88% | 9.8% | 60.78% | 94.12% |
| DeFi | 150 | 42.67% | 82% | **100%** | 42.67% | 98.67% |
| Malicious_MEV | 299 | 0% | 13.04% | 2.68% | 19.4% | 80.27% |

活跃 DeFi 种子的 `out_edge_count` 100% 为零，说明它们只作为交易接收方（符合合约地址行为模式）。三类的 clustering_coeff 普遍 80–99% 为零，说明抽样后的图整体稀疏、呈 star-like 结构（Top-100 counterparty 抽样很难保留对手方之间的三角闭合），呼应 Project Plan Risk 5（Sampling Bias）。

### 3.4 特征相关性（`feature_correlation_heatmap.png`）

`log_total_tx_count` 与 `log_weighted_degree` 相关系数 = **1.00**（完全重复，已在 §2 的 feature_groups 中处理）。

### 3.5 图表文件
`feature_distributions_by_label.png`（7 个特征按 label 分组箱线图，仅活跃种子）、`feature_correlation_heatmap.png`

---

## 4. Node2Vec Embedding 与聚类（Week 7，[node2vec_clustering.py](node2vec_clustering.py) → `embedding_outputs/`）

### 4.1 方法

- 图：8,029 节点 raw_edges.csv 转无向图（12,694 条边，去重后），原因：抽样图是有向的，但绝大多数非种子对手方节点是叶子节点，只有单方向记录的边，有向随机游走会立刻在这些节点"卡死"，故采用无向视图（与 clustering_coeff 的处理方式一致）
- Node2Vec：64 维，p=q=1（无偏游走，等价 DeepWalk 基线），walk_length=30，num_walks=10
- t-SNE 二维可视化（**未用 UMAP**：本机 umap-learn 包的 `__init__.py` 会强制尝试加载依赖一个已损坏的系统级 torch 安装，报 DLL 加载错误，与本项目无关；t-SNE 同样满足 Plan 中"UMAP/t-SNE"的要求）
- KMeans（k=3）+ Louvain 社区发现，用 NMI/ARI/Silhouette 评估

### 4.2 环境问题记录（供方法论章节 Reproducibility 附录参考）

| 问题 | 原因 | 处理方式 |
|---|---|---|
| gensim 安装失败 | pip 默认走源码编译，本机无 C 编译器 | `pip install --only-binary :all: gensim`，改拉预编译 wheel |
| umap-learn 导入报 OSError | 包内部无条件尝试 `import torch`，本机 torch 安装已损坏（DLL 路径异常，`os.add_dll_directory` 报 WinError 87） | 改用 sklearn 的 t-SNE，不装/不导入 umap |
| node2vec 0.3.2 报 `TypeError: unexpected keyword 'size'` | 该库硬编码 gensim 3.x 的旧参数名 `size`，与已安装的 gensim 4.4.0（新参数名 `vector_size`）不兼容 | 绕过 `node2vec.fit()`，直接用 `node2vec.walks` 生成的游走序列调用 `gensim.models.Word2Vec(vector_size=...)` |
| joblib 多进程崩溃（`UnicodeEncodeError`） | Windows 用户目录路径含中文字符（王琦），joblib 的 resource_tracker 尝试以 ASCII 编码路径失败 | `Node2Vec(..., workers=1)`，规避 multiprocessing 后端 |

### 4.3 结果

**基础聚类**（`clustering_evaluation_summary.csv`）：

| method | n_clusters | NMI | ARI | Silhouette |
|---|---|---|---|---|
| KMeans | 3 | 0.1389 | 0.0166 | 0.0822 |
| Louvain | 116 | 0.3039 | 0.0577 | 0.4023 |

**置换显著性检验**（2000 次随机打乱标签，`clustering_significance_test.csv`）：

| method | ARI observed | ARI null mean | ARI p-value | 结论 |
|---|---|---|---|---|
| KMeans | 0.0166 | 0.0002 | **0.176** | **不显著**（α=0.05） |
| Louvain | 0.0577 | -0.0001 | **<0.0005** | **显著**，但效果量很小 |

> NMI 在两种方法下都"看起来显著"（p≈0），但这是类别不均衡（299/86/51）造成的已知偏差——NMI 对簇大小与类别比例的巧合匹配不做机会校正。ARI 是本项目采信的指标。

**KMeans k=2→8 稳健性扫描**（`kmeans_k_scan.csv` / `.png`）：Silhouette 全程停留在 0.079–0.127（弱），ARI 在 -0.05~0.09 之间无规律震荡，不随 k 单调改善。**排除了"k=3 选错了"的解释**——换任何 k 结果都同样弱，问题出在 embedding 空间本身。

**DBSCAN eps 稳健性扫描**（min_samples=5，eps 取 5-NN 距离的第 30/50/70/80/90/95 分位数，`dbscan_eps_scan.csv` / `dbscan_k_distance.png`）：

| eps | n_clusters | noise_ratio | ARI |
|---|---|---|---|
| 2.45（拐点附近） | 11 | 69.5% | -0.024 |
| 6.19 | 17 | 42.7% | 0.021 |
| 8.99 | 2 | 24.3% | 0.006 |
| ≥9.8 | 1 | <14% | — |

eps 偏小时大部分点被标记为噪声，eps 稍大直接坍缩成一个大簇，**从未出现"干净分离的中等密度簇"这种中间态**，5-NN 距离曲线本身也只有一个不明显的拐点、无陡峭"膝盖"，与 silhouette 扫描的结论一致。

**t-SNE 可视化**（`tsne_node2vec_seeds.png`）：三类地址在 embedding 空间中大量混杂，可见不少局部紧密小簇（推测是围绕单个种子的星形子图造成），但整体三分类边界不清晰。

### 4.4 结论（Week 10 平衡子集复核后已修正——见 §7 完整证据）

**本节最初的结论是"embedding 空间不存在自然聚类结构"，Week 10 的平衡子集复核（§7）证明这一说法不成立，现予以修正。** 原始版本基于全样本（299/86/51，严重不均衡）上的 KMeans/Silhouette-k-扫描/DBSCAN 三方法一致得出弱结果（KMeans ARI=0.017，p=0.176，不显著），当时据此判断 embedding 空间没有可用结构。§7 的复核发现：**同一个 KMeans 聚类划分、同一个 Louvain 社区划分，只把评估从全样本换成类别平衡的子集（每类 51 个，n=153），ARI 就从接近 0 跳到 0.25–0.31，且两种方法独立验证、方向一致**。因为聚类划分本身没有变，唯一变量是评估所用的类别分布，这只能说明：**在本数据集和当前聚类划分下，全样本的不平衡评估显著弱化了观察到的标签对齐，而类别平衡子集中的 ARI 稳定更高**——这一结果与"多数类点对主导 pairwise 指标"的机制解释一致：机会校正修正的是"随机打乱下的期望值"，不修正"多数类点对在总点对数中占绝对多数"这一结构性问题，含 69% 样本的 Malicious_MEV 类几乎主导了 ARI 的计算，即使 CEX/DeFi 被清楚分开，对总体统计量的贡献也会被淹没。**这是本数据集在这一具体聚类划分下观察到的实证发现，不应被上升为"ARI/NMI 这类指标在不均衡数据下普遍会系统性低估结构"这一更强的一般性统计论断**。

修正后的结论：
1. **Node2Vec embedding 空间中存在与标签中度对齐的结构**（ARI≈0.25–0.31，KMeans 与 Louvain 独立验证，平衡评估下 p<0.0005），但强度是中等而非强分离——不应表述为"embedding 成功捕捉结构"，只能说是中度对齐、统计显著但远非清晰可分。
2. **这不是"embedding 变强了"，而是评估方式的伪影被识别和量化**：embedding 空间从头到尾没有变化，变化的只是测量它的方式（全样本 pairwise 指标 vs 平衡子集 pairwise 指标）。这是本文识别并归因的一个方法论陷阱，也是分类器受此影响相对较小的原因所在——macro-F1/`class_weight="balanced"`按类别而非按样本对计算，能够降低多数类主导评估的风险，这与聚类评估形成对照；但这不代表分类结果完全不受类别不均衡影响（§7.2 的检验力下降就是一个例子），只是受影响的机制和幅度不同。
3. **DBSCAN 和 KMeans k-扫描在全样本上的弱结果依然是有效证据，只是需要重新定性**：它们说明在类别高度不均衡的评估条件下，密度/球状聚类方法确实找不到与全样本标签分布对齐的清晰簇——这本身仍是真实、可复现的观察，只是不能再被解读为"embedding 空间没有结构"，而应解读为"全样本类别分布下，标准聚类指标的检出力不足以显现该结构"。
4. **与 Week 8/9 分类结论的兼容性（§5.3、§6.3、§7，⚠️ 本条已被 §5.3 的重复交叉验证结果修正）**：本节最初认为 embedding 携带的中度结构与手工图特征捕捉的是同一批、源自同一个 star-like 采样拓扑的信号，因而对分类器没有独立增量。**§5.3 的重复交叉验证（10 次重复 5 折 CV，按重复聚合的 repeat-level 配对估计）修正了这一判断**：加入 embedding 后，Logistic Regression、Random Forest、XGBoost 的 macro-F1 均出现统计显著的正向提升（分别约 +0.068、+0.016、+0.035，置信区间均不含零，方向 100% 一致）。这说明 embedding 与手工结构特征**并非完全冗余**——更准确的表述是：两者可能共享部分拓扑来源的信号，但 Node2Vec 显然还捕捉到了手工统计量未能完全表达的信息，这部分信息确实转化成了监督分类器的增量预测力。

这一修正后的发现完整回应 Research Question 2 的前半部分："embedding 是否能捕捉到有意义的结构模式"——**答案是能，中等强度，需要在类别平衡的条件下才能被标准聚类指标正确检出（本节）；这一结构信息并未被手工结构特征完全"吃掉"，在完整三分类任务中还转化为了小到中等、模型依赖的监督分类增量（§5.3），但该增量在 EOA-only 任务中未能复现（§10.8a），说明其价值具有明显的任务依赖性**。

---

## 5. 监督分类器（Week 8，[classification.py](classification.py) → `classification_outputs/`）

### 5.1 方法

- **Pagerank rank 变换**：在此脚本中完成（作为分类器 pipeline 的特征预处理第一步，而非改动 feature_engineering.py）。用全图 8,029 节点的 pagerank / weighted pagerank 分布计算百分位排名，替换手工特征集中原本偏度未被 log1p 修复的 `pagerank`/`pagerank_weighted`/`log_pagerank`/`log_pagerank_weighted`（4 列 → `pagerank_rank`/`pagerank_weighted_rank` 2 列）
  - **已知局限**：`rank(pct=True)` 对并列值取平均排名，而未加权 PageRank 存在大规模并列——全图 8,029 节点中 **2,917 个（36.3%）** 精确并列在同一个理论最小值（这些节点全部满足 `in_edge_count=0`）。因此对于落在这个并列组里的种子，`pagerank_rank` 实质上和已有的 `has_in_edge` 特征机制性重合，不是独立信号。51 个 CEX 种子里有 31 个（61%）恰好落在这个并列组。
- **样本**：统一使用 436 个活跃有标签种子（embedding 仅在活跃节点上存在），三组特征条件共用同一样本，确保 macro-F1 差异只来自特征内容
- **特征集**：`hand_crafted`（28 个手工特征，来自 `feature_groups.json` 两组之和）/ `embedding`（64 维 Node2Vec）/ `combined`（拼接）
- **模型**：Logistic Regression（StandardScaler + `class_weight="balanced"`）、Random Forest（`class_weight="balanced"`）、XGBoost（按训练折计算 `sample_weight`，因其多分类无内置 class_weight 等价项）
- **评估**：分层 5 折交叉验证（而非单次划分——CEX 仅 51 个样本，单次划分测试集过小过吵），报告 macro-F1、balanced accuracy 的跨折均值±标准差，以及基于 out-of-fold 预测的逐类别 precision/recall/F1 和归一化混淆矩阵
- **配对显著性检验**：StratifiedKFold 对所有 9 个 (feature_set, model) 组合使用相同 random_state，因此每个 fold 划分到的种子完全一致，可以对 combined vs hand_crafted 做逐折配对 t 检验（比独立比较均值±std 区间更有效力）
- **特征组贡献分解**：在全部 436 样本上重新拟合 RF 与 LR（仅用于可解释性诊断，不作为性能声明），按 hand_crafted / embedding 分组加总 RF 的 `feature_importances_` 与 LR 标准化后系数的 |coef| 均值，检验模型是否真的在用 embedding 维度
- **两套 CV 设计的分工**：本节最初只跑了单次分层 5 折 CV（`classification.py`），用于生成本节的 macro-F1 汇总表、out-of-fold 逐类别报告和混淆矩阵。Week 10 追加的重复分层 CV（`repeated_cv.py`，10 次重复 × 5 折）专门用于 combined vs hand_crafted 等特征集差异的显著性推断——单次 5 折的 5 个配对差检验力不足，重复 CV 提供了更稳定的效应量估计和置信区间。因此本节 §5.2 的 macro-F1 汇总表格仍来自单次 5 折 CV，但所有"是否有正向增量"的最终结论以 §5.2 后半部分和 §5.3 报告的重复 CV 结果为准，不以单次 5 折的配对检验为准
- **Node2Vec embedding 的 transductive 性质**：64 维 embedding 是在完整无标签图（8,029 节点）上一次性无监督训练得到的（§4.1），测试折种子地址的类别标签没有参与这一步，但其节点和连接结构参与了表示学习。因此分类器评估报告的是 transductive 场景下的性能（测试节点的拓扑在表示学习阶段是可见的），而不是对训练阶段完全未见过的新节点的 inductive 泛化能力，这一点在解读 macro-F1 数字时应保持清楚

### 5.2 结果

**Macro-F1（5 折 CV 均值，`classification_summary.csv` / `macro_f1_comparison.png`）**：

| feature_set | LogisticRegression | RandomForest | XGBoost |
|---|---|---|---|
| hand_crafted | 0.892 ± 0.032 | 0.905 ± 0.037 | 0.906 ± 0.034 |
| embedding | 0.788 ± 0.062 | 0.766 ± 0.129 | 0.829 ± 0.090 |
| combined | 0.953 ± 0.044 | 0.906 ± 0.036 | 0.935 ± 0.031 |

**逐折配对检验（combined − hand_crafted，单次 5 折 CV，`combined_vs_handcrafted_paired_test.csv`，⚠️ 已被下方重复 CV 结果修正，仅作历史记录保留）**：

| model | 均值差 | 5 折中提升的折数 | 配对 t 检验 p 值 |
|---|---|---|---|
| LogisticRegression | +0.062 | 4/5 | 0.115（不显著） |
| RandomForest | +0.001 | 1/5 | 0.788（"无效果"——**下方重复 CV 显示这一小幅效应在更多重复下能够稳定复现，单次 5 折未能检测到它**） |
| XGBoost | +0.030 | 4/5 | 0.049（临界显著，n=5 检验力低） |

**重复分层交叉验证修正（`repeated_cv.py` → `contract_confound_outputs/`，10 次重复 × 5 折 = 50 折。50 个 fold 级结果并非相互独立——同一次重复内的 5 个 fold 共享高度重叠的训练集，直接当作 50 个独立观测会低估方差、夸大显著性。因此按重复取均值聚合，得到 10 个 repeat-level 配对估计再计算置信区间和配对 t 检验，这 10 个估计彼此的独立性也高于同一重复内的 fold，但仍共用同一批 436 个地址，不构成 10 个完全独立的数据集，解读时应以效应量、置信区间宽度和方向一致性为主，而非单独依赖 p 值）**：

| model | 均值差 | 95% 置信区间（10 个 repeat-level 配对估计） | p 值 | 方向一致的重复次数 |
|---|---|---|---|---|
| LogisticRegression | +0.068 | [0.061, 0.076] | <0.0001 | 10/10 |
| **RandomForest** | **+0.016** | **[0.011, 0.022]** | **0.0001** | **10/10** |
| XGBoost | +0.035 | [0.023, 0.046] | 0.0001 | 10/10 |

**三个模型现在全部呈现方向 100% 一致的正向效应，置信区间均不含零**——RandomForest 从单次 5 折"无效果"的结论修正为"效应较小但能稳定复现"。这是本文方法论上的一个重要教训：单次 5 折 CV 的点估计不足以可靠检测这个量级的效应，重复交叉验证 + 置信区间 + 方向一致性是更稳妥的判定标准（详见顶部"研究定位"的方法论说明）。

**附带发现：极简 3 特征基线在 RandomForest 下反而略优于完整 28 特征集**（`main_hand_crafted vs main_coarse`，重复 CV，10 次重复）：

| model | hand_crafted（28 特征） | coarse（3 特征：`is_contract`/`has_out_edge`/`out_tx_count`） | 均值差 | 95% CI | p 值 |
|---|---|---|---|---|---|
| LogisticRegression | 0.878 | 0.577 | +0.301 | [0.277, 0.325] | <0.0001（hand_crafted 显著更好，符合预期） |
| **RandomForest** | 0.896 | 0.910 | **-0.014** | **[-0.017, -0.011]** | **<0.0001（coarse 显著更好）** |
| XGBoost | 0.898 | 0.906 | -0.007 | [-0.016, 0.001] | 0.09（不显著，方向为负） |

对 RandomForest，3 个粗粒度特征的极简基线在重复 CV 下**统计显著地优于**完整 28 特征集（10/10 次重复方向一致）。这说明大部分预测信号集中在少数几个粗粒度变量上，额外的、可能相关或较弱的特征未必能提升泛化能力——**这是当前 RandomForest 默认参数设定下的结果，不能泛化为"特征工程无用"这一更强的结论**，调参后的 RF 差距可能变化。该发现进一步支持了 §10 account-type 混杂分析中"粗粒度信息解释力很强"的判断，值得在 Discussion 中一并讨论。

**特征组贡献分解（`feature_group_contribution.csv` / `.png`，在 combined 特征集上拟合）**：

| group | 占特征总数比例 | RF importance 占比 | LR \|coef\| 占比 |
|---|---|---|---|
| hand_crafted | 30.4%（28/92） | **74.0%** | 34.8% |
| embedding | 69.6%（64/92） | **26.0%** | 65.2% |

RF 的 importance 前 10 名全部是手工特征；LR 的 |coef| 前 10 名里有 4 个是具体 embedding 维度。

**逐类别 F1（out-of-fold，`classification_per_class_report.csv`）**：CEX（最小类，51 样本）在所有特征集/模型下 F1 均最低（0.78–0.90），DeFi 居中，Malicious_MEV（最大类，299 样本）最高（0.88–0.99）——符合小类别更难分类的预期，但未崩溃。混淆矩阵（`confusion_matrix_grid.png`）显示误差主要集中在 CEX↔Malicious_MEV 与 DeFi↔Malicious_MEV 之间。

### 5.3 结论（Week 10 重复交叉验证复核后已修正——§5.2 的 n=5→n=50/n=10 重复 CV 修正是本节结论变化的直接原因）

**重复分层交叉验证修正了初始 5 折分析中的若干不稳定判断。** 在完整活跃样本中，将 Node2Vec embedding 加入手工特征后，三种模型的 macro-F1 均出现正向提升：Logistic Regression、Random Forest 和 XGBoost 的平均增量分别约为 0.068、0.016 和 0.035（10 次重复、按重复聚合的 repeat-level 配对估计，95% 置信区间均不含零，方向 100% 一致）。提升幅度对 Logistic Regression 最大、对 XGBoost 居中、对 Random Forest 较小。**初始 5 折分析未能检测到 Random Forest 中的这个小幅效应，说明仅依赖五个 fold 进行推断会产生较大的不确定性——"RF 对 embedding 无反应"不能再作为本项目的最终结论。**

这一结果表明，**Node2Vec embedding 确实携带了部分手工特征没有完全表达的预测信息，但其增量价值具有明显的模型和任务依赖性**。在 EOA-only 的 CEX–Malicious_MEV 分类中（§10.8a），embedding 的增量在同样的重复检验下基本消失（三模型均不显著，LogisticRegression 的均值差从单次 5 折的 +0.033 降到重复检验的 +0.004），说明该收益并非普遍适用于所有地址类型，而可能主要与包含合约地址、或涉及不同合约功能角色的任务有关（与 contract-only DeFi vs Malicious_MEV 场景下 embedding 显示更强价值——§10.8b——的模式一致）。

相比之下，**手工图结构特征在已有交易行为特征的基础上没有产生正向增量**。在 Random Forest 中，加入这些结构特征还造成约 0.014 的稳定、统计显著的下降（10/10 次重复方向一致），表明当前采样拓扑中的静态结构统计量可能包含与交易特征重复或具有噪声的信息——这与"结构特征存在冗余或对 RF 当前参数设定构成噪声"的解释一致，但尚未证实具体机制（不应断言"已经证明是因为冗余"，只能说"与该解释一致"）。这条结论是对 Week 9 消融的加固，而非修正。

**因此，RQ2 的答案需要区分不同的图信息提取方法，不能再合并成一个单一结论**：**Node2Vec 在完整样本中提供了小到中等、模型依赖的增量收益，而手工结构特征和 GraphSAGE（§9）没有显示相同的稳定优势**。RQ2 的另一半——"embedding 能否捕捉有意义的结构模式"——答案在 Week 10 平衡子集复核后同样有修正：**能，中等强度，但需要类别平衡的评估条件才能被标准聚类指标正确检出**（§4.4）；这个中度结构与手工图特征在拓扑来源上可能重叠，但显然并未因此被手工特征完全"吃掉"——本节的重复 CV 结果证明它确实为监督分类器提供了独立于手工特征的增量信息，这一点和 §4.4 的"结构存在但对分类冗余"这一旧表述需要相应调整：**结构存在，且至少部分转化为了分类器的增量预测力（对 LR/XGBoost/RF 均如此，但幅度不同）**。

**英文版本（供论文正文使用）**：
> Repeated stratified cross-validation revised several conclusions drawn from the initial five-fold evaluation. On the full active sample, adding Node2Vec embeddings to the hand-crafted features produced positive mean improvements in macro-F1 for all three classifiers: approximately 0.068 for Logistic Regression, 0.016 for Random Forest and 0.035 for XGBoost. The gain was largest for Logistic Regression, moderate for XGBoost and small for Random Forest. The original five-fold analysis failed to detect the smaller Random Forest effect, illustrating the uncertainty of inference based on only five fold-level observations.
>
> These results indicate that the Node2Vec embeddings contained predictive information not fully represented by the hand-crafted features. However, the incremental value was both model- and task-dependent. In the EOA-only CEX–Malicious MEV experiment, the embedding gain largely disappeared, suggesting that the benefit was not universal across address-classification settings and may have been concentrated in tasks involving contract addresses or distinct contract roles.
>
> In contrast, hand-crafted structural features did not provide a positive incremental benefit once transaction-behaviour features were available. For Random Forest, their inclusion produced a small but consistent reduction in performance. The answer to RQ2 must therefore distinguish between graph-information extraction methods: Node2Vec provided a small-to-moderate, model-dependent incremental benefit on the full sample, whereas hand-crafted structural features and GraphSAGE did not demonstrate the same stable advantage.

> **§10 交叉引用**：本节报告的 macro-F1 数字是三分类任务的技术性能，仍然准确；但作为"模型学到了三类地址各自真实经济行为"的证据，需要按 §10.7 的规则拆开看——DeFi 类的可分性与其 100% 合约身份完全混杂（§10.3），无法在当前三分类设计下与泛化的合约特征分离；CEX vs Malicious_MEV 的可分性在剔除合约地址后依然成立（§10.6），是本节结论中站得住的那一半。embedding 的增量价值在 contract-only 场景（§10.8b）下比 EOA-only 场景（§10.8a）更明显，这一模式提示 embedding 的价值可能与合约功能角色的区分度有关，值得在 Discussion 中作为一个开放问题讨论。

---

## 6. 特征分组消融与 SHAP（Week 9，[feature_ablation_shap.py](feature_ablation_shap.py) → `ablation_shap_outputs/`）

### 6.1 方法

- **消融维度**：`transaction_behaviour` only（19 特征）/ `graph_structural` only（9 特征，pagerank 系列同样替换为 rank 变换）/ `combined`（28 特征 = 两组之和，与 Week 8 的 `hand_crafted` 特征集完全一致）——直接回答 RQ1（transaction behaviour 与 graph structural 各自能在多大程度上区分地址类别）
- **样本与折划分**：与 Week 8 相同的 436 个活跃种子，StratifiedKFold 使用相同 `random_state=42`，因此三个消融条件与 Week 8 的 9 个组合共享同一组 fold 划分，理论上可跨脚本配对比较（本节的 `combined` 结果应与 `classification_summary.csv` 里的 `hand_crafted` 行数值一致，已核对）
- **模型**：与 Week 8 相同（LR/RF/XGBoost），本节不涉及 embedding，专注于手工特征内部的分组贡献
- **配对检验**：`combined vs transaction_behaviour`、`combined vs graph_structural`，逐折配对 t 检验，判断在已有 transaction_behaviour 的情况下加入 graph_structural 是否有实质增量（反之亦然）
- **SHAP**：在全部 436 样本上拟合 RF（`combined` 特征集，仅用于可解释性诊断），用 `TreeExplainer` 计算逐特征 SHAP 值，按 3 个类别取绝对值均值后再对类别取平均，得到与 Week 8 §5.1 特征组贡献分解同构、但细化到单个特征层级的重要性排序
- **两套 CV 设计的分工**：本节 §6.2 的汇总表和逐折配对检验同样来自单次分层 5 折 CV（`feature_ablation_shap.py`），检验力有限（n=5）。Week 10 的重复分层 CV（`repeated_cv.py`）中的 `main_hand_crafted vs main_transaction` 条件恰好覆盖了本节 combined vs transaction_behaviour 这一比较（`main_hand_crafted` 即本节的 28 特征 `combined`/`hand_crafted` 特征集），其 10 个 repeat-level 配对估计结果见 §6.3，作为本节最终结论的补充证据

### 6.2 结果

**Macro-F1（5 折 CV 均值，`ablation_summary.csv` / `ablation_macro_f1.png`）**：

| feature_group | LogisticRegression | RandomForest | XGBoost |
|---|---|---|---|
| transaction_behaviour | 0.884 ± 0.023 | 0.922 ± 0.027 | 0.916 ± 0.008 |
| graph_structural | 0.676 ± 0.081 | 0.833 ± 0.069 | 0.830 ± 0.071 |
| combined | 0.892 ± 0.032 | 0.905 ± 0.037 | 0.906 ± 0.034 |

**逐折配对检验（`ablation_paired_test.csv`，n=5 折，检验力低，仅作参考）**：

| comparison | model | 均值差 | 提升折数 | p 值 |
|---|---|---|---|---|
| combined vs transaction_behaviour | LogisticRegression | +0.0075 | 2/5 | 0.762（无效果） |
| combined vs transaction_behaviour | RandomForest | **-0.0172** | 0/5 | 0.205（无效果，方向为负） |
| combined vs transaction_behaviour | XGBoost | -0.0097 | 1/5 | 0.553（无效果） |
| combined vs graph_structural | LogisticRegression | +0.2161 | 5/5 | **0.004（显著）** |
| combined vs graph_structural | RandomForest | +0.0715 | 5/5 | **0.027（显著）** |
| combined vs graph_structural | XGBoost | +0.0763 | 5/5 | **0.049（临界显著）** |

**SHAP 特征重要性（RF，`shap_feature_importance.csv` / `shap_top_features.png`）**：Top-10 中 9 个是 transaction_behaviour 特征（`weighted_out_degree`、`out_tx_count`、`log_out_tx_count`、`out_gas_fee_mean` 等），唯一进入 Top-5 的 graph_structural 特征是 `out_degree`（第 5 位）。按组加总：

**逐类别拆解**（Plan 明确要求"interpret which specific features drive predictions across address categories"，而非仅聚合排名；`mean_abs_shap_{label}` 列，`shap_feature_importance.csv`）：

| 类别 | Top-5 特征（按 SHAP 重要性） |
|---|---|
| CEX | `weighted_out_degree`、`out_tx_count`、`log_out_tx_count`、`log_total_tx_count`、`total_tx_count`（全部 transaction_behaviour） |
| DeFi | `log_out_tx_count`、`weighted_out_degree`、`out_tx_count`、**`out_degree`（graph_structural）**、`out_gas_fee_mean` |
| Malicious_MEV | `weighted_out_degree`、`out_tx_count`、`log_out_tx_count`、`log_total_tx_count`、`out_gas_fee_mean`（全部 transaction_behaviour） |

三个类别的驱动特征高度一致，核心都是"出向交易活跃度"（`weighted_out_degree`/`out_tx_count`/`log_out_tx_count`）——三类地址的可分性主要来自它们对外发起交易的规模和频率差异，而不是接收方向或图拓扑位置。**唯一的类别差异**：`out_degree`（graph_structural）只挤进了 DeFi 的 Top-5，其余两类前 5 全是 transaction_behaviour——与 §3.3 EDA 发现的"活跃 DeFi 种子 100% 只作为交易接收方"相互印证：活跃 DeFi 种子的 `out_degree` 几乎恒为 0（因为它们从不主动发起交易），这个"接近零"本身就是判别 DeFi 的一个信号（在模型看来，`out_degree` 非零基本排除了 DeFi），而不是说 DeFi 内部靠对手方数量的多少来互相区分。按组加总：

| group | 占特征总数比例 | SHAP mass 占比 |
|---|---|---|
| transaction_behaviour | 67.9%（19/28） | **84.6%** |
| graph_structural | 32.1%（9/28） | **15.4%** |

### 6.3 结论

1. **单独用 graph_structural 特征分类明显弱于 transaction_behaviour**（macro-F1 低 8–21 个百分点，三个模型一致），且 `combined vs graph_structural` 的配对检验对三个模型都显著——这说明 transaction_behaviour 特征携带了 graph_structural 所没有的、真实的判别信号，这部分符合预期。
2. **反过来看才是本节的关键发现**：单次 5 折 CV 中，`combined vs transaction_behaviour` 对三个模型全部不显著，RandomForest 和 XGBoost 的均值差甚至是**负的**（RF：0/5 折提升）。**重复分层 CV（`repeated_cv.py` 的 `main_hand_crafted vs main_transaction` 条件，10 个 repeat-level 配对估计）确认了这一模式，且对 RF 给出了统计显著的结果**：LogisticRegression 均值差 -0.0044（CI[-0.0146, 0.0058]，p=0.35，不显著）、**RandomForest 均值差 -0.014（CI[-0.0181, -0.0099]，p<0.0001，10/10 次重复方向一致为负）**、XGBoost 均值差 +0.0031（CI[-0.0035, 0.0097]，p=0.32，不显著）。也就是说，**在已经拥有交易行为特征的前提下，额外加入手工图结构特征（degree、PageRank、clustering coefficient、wcc_size 等）没有产生正向增量；对 RandomForest 而言，重复 CV 进一步表明这是一个小幅但稳定、统计显著的负向效应**，而不只是单次 5 折下的偶然结果。这与手工结构特征相对交易行为特征增量价值有限的整体结论一致，但需要注意：这条"无增量"结论专属于手工结构特征，**不能再类比套用到 Node2Vec embedding 上**——§5.3 的重复 CV 已证明 embedding 相对手工特征在完整样本上有统计显著的正向增量，两者不是同一形状的结论。
3. **SHAP 在单特征层面提供了与 2 方向一致的描述性支持**（而非独立证据——SHAP 使用的是同一批数据和高度相关的特征集，不构成与消融/配对检验相独立的验证）：transaction_behaviour 特征拿到 84.6% 的重要性质量，超过其 67.9% 的特征数量占比；graph_structural 只拿到 15.4%，低于其 32.1% 的数量占比，且 Top-10 里 9 个是交易行为特征。
4. **这一发现和 Week 6 EDA 已经埋下的线索吻合**：clustering_coeff 在三类地址中普遍 80–99% 为零（§3.3），wcc_size/degree 类特征的可分性本身受限于 Top-100 counterparty 抽样产生的 star-like 拓扑（对手方之间很少有交叉连接可供三角结构或社区结构成型）。这提供了一个具体、可检验的机制解释，而不是笼统地说"图特征没用"：**不是拓扑信息本身无价值，而是这个采样方式下的图过于稀疏，可供拓扑方法捕捉的结构信号有限**。
5. 综合 Week 7–9 三周的证据，RQ1 现在有一个清晰的答案：**transaction behaviour 特征是本数据集里绝大部分判别信号的来源；手工图结构特征相对交易行为特征增量价值有限，且可以追溯到同一个采样约束（Top-100 counterparty 抽样导致的 star-like、低三角密度拓扑）**。**⚠️ 但 RQ2（图信息提取方法是否有增量价值）不能再套用同一个"都有限"的结论**——Week 10 重复交叉验证（§5.3）显示 Node2Vec embedding 在完整样本上对三个分类器均有统计显著的正向增量（模型依赖，LR 最大、RF 最小），只是手工图结构特征没有；这条区分必须保留，不能把两者混为一谈。GraphSAGE（§9）的检验结果则与手工结构特征的模式更接近（无稳定优势）。

> **§10 交叉引用**：本节的 SHAP 结果（transaction_behaviour 特征、尤其是出向交易活跃度类特征占主导）需要结合 §10 一并理解——这些出向特征同时也是"是否为合约"这个账户类型的强代理变量（DeFi 100% 合约、出向交易几乎恒为 0），所以"transaction behaviour 是主要信号来源"这个结论里，有一部分解释力可能来自账户类型区分而非精细经济行为。§10.8a 的 EOA-only 特征分组消融证实：剔除合约混杂后，transaction_behaviour 仍是主要信号来源，graph_structural 仍无增量——这部分结论稳健；但 embedding 的增量在 EOA-only 下消失，与本节及 §5.3 的"embedding 价值可能集中于涉及合约角色的任务"这一观察一致。

---

## 7. 类别平衡子集稳健性复核（Week 10，[balanced_subset_robustness.py](balanced_subset_robustness.py) → `balanced_subset_outputs/`）

### 7.1 方法与动机

Week 8/9 的所有结论都建立在同一个 436 活跃种子样本（299 Malicious_MEV / 86 DeFi / 51 CEX）上。虽然分类器全程用了 `class_weight="balanced"` / `sample_weight` / macro-F1 / 分层 CV 防御类别不均衡对**指标**的扭曲，但这不能排除类别不均衡以另一种方式扭曲**结论本身**的可能——模型会不会只是靠拟合占 69% 的 Malicious_MEV 类的交易行为特征就拿到高分，从而掩盖了 embedding/图结构特征在均衡数据下本可能有的价值？本节在 GraphSAGE 之前先做这项检验，因为它是防御性的（保护已有结论的可信度），优先于扩张性的新方法尝试。

- **平衡子集构造**：按最小类别规模（CEX=51）对三类分别下采样（`random_state=42`，不放回），得到 153 个种子、三类各 51 个，种子列表存至 `balanced_subset_seeds.csv`
- **分类稳健性**：在该平衡子集上重跑 Week 8（hand_crafted / embedding / combined）与 Week 9（transaction_behaviour / graph_structural / combined）的全部 5-fold CV 对比和配对 t 检验，方法与原脚本完全一致
- **聚类稳健性**：对 Week 7 的 KMeans 和 Louvain 结果做两种独立复核——(a) 保持原聚类划分不变（仍是在全样本 436 上拟合/划分的），只把置换检验的评估对象换成平衡子集（153 个点）；(b) 直接在平衡子集的 embedding 上重新拟合 KMeans。两种复核分离了"评估集不均衡"和"训练/拟合过程不均衡"两个可能的混杂因素

### 7.2 分类结果：结论未被推翻，但检验力下降（符合预期）

| 比较 | 全样本 (n=436) | 平衡子集 (n=153) |
|---|---|---|
| combined vs hand_crafted（Week 8） | 三模型均不显著 | 三模型均不显著，方向一致（LR/XGB 微正、RF 微负） |
| combined vs transaction_behaviour（Week 9） | 三模型均不显著，方向偏负 | 三模型均不显著，方向仍偏负（RF/XGB 0–1/5 折提升） |
| combined vs graph_structural（Week 9） | 三模型均显著 | **三模型均不再显著**（p=0.066–0.112），方向仍为正 |

最后一行从"显著"变"不显著"不是结论反转：n 从 436 降到 153，5 折 CV 下每折测试集只剩约 30 个样本、10 个/类，标准差普遍翻倍（如 `hand_crafted` 在 XGBoost 上的 std 从 0.034 升到 0.086），检验力自然下降，但方向没有改变。**手工图结构特征相对交易行为特征没有稳定可测的增量价值这一结论，在类别平衡后依然成立**，说明这不是不均衡数据的假象。**⚠️ 本表用的是平衡子集上单次 5 折 CV 的结果；embedding 的增量价值已在 §5.2/§5.3 用重复 CV（10 次重复、按重复聚合的 repeat-level 配对估计）在全样本上修正为"对三个模型均呈现方向一致的正向效应"——本表这一单次划分的结果不再代表 embedding 的最终结论，仅保留作为 Week 10 平衡子集检验的原始记录**。完整数据见 `balanced_summary.csv` / `balanced_paired_test.csv` / `full_vs_balanced_comparison.png`。

### 7.3 聚类结果：发现全样本评估低估了 embedding 空间中的真实结构

| 方法 | 评估条件 | ARI | ARI p 值 |
|---|---|---|---|
| KMeans | 全样本 (n=436) | 0.017 | 0.176（不显著） |
| KMeans | 同一聚类划分，仅评估平衡子集 (n=153) | 0.252 | **<0.0005** |
| KMeans | 在平衡子集上重新拟合 (n=153) | 0.313 | **<0.0005** |
| Louvain | 全样本 (n=436) | 0.058 | 0.0005（弱显著） |
| Louvain | 同一社区划分，仅评估平衡子集 (n=153) | 0.245 | **<0.0005** |

两种独立聚类方法方向一致，且关键的第二行（KMeans、Louvain 均有）证明这不是"重新拟合带来的差异"——聚类划分本身固定不变，只切换评估用的类别分布，ARI 就跳升一个数量级。**这是本数据集和当前聚类划分下的实证发现**：ARI 虽然做了机会校正，但仍是逐点对（pairwise）一致性指标；Malicious_MEV 占全样本 69%，绝大多数"同类点对"是 MEV-MEV，即使 CEX/DeFi 被聚类正确分开，这部分点对在全部点对中占比很小，对总体统计量的贡献被淹没——机会校正修正的是"随机打乱下的期望值"，不修正"点对计数被多数类主导"这一结构性问题。这一机制解释与观察到的模式一致，但本节验证的是这一具体数据集/划分下的现象，不构成对 ARI 指标在不均衡数据下普遍表现的一般性证明。完整数据见 `balanced_kmeans_comparison.csv`。**这一发现已并入 §4.4，作为 Week 7 结论的正式修正，而非本节的独立结论**——理由与完整措辞详见该节。

**稳定性复核（P2，`repeated_balanced_ari.py`，50 次独立类别平衡抽样，非仅 `random_state=42` 单次抽样）**：

| 方法 | ARI 均值 | 标准差 | 5%–95% 分位 | 最小–最大 | 50 次里显著的比例 |
|---|---|---|---|---|---|
| 固定全样本聚类，只换评估子集 | 0.223 | 0.022 | 0.189–0.260 | 0.175–0.268 | **100%** |
| 每次在子集上重新拟合 KMeans | 0.182 | 0.068 | 0.089–0.295 | 0.080–0.320 | **100%** |

50 次独立抽样中 ARI 从未跌回接近 0 的水平，全部显著——证实了上表的发现不是 `random_state=42` 这一次抽样的巧合，中度结构在类别平衡评估下是稳定、可复现的。"固定聚类只换评估集"版本比"每次重新拟合"更稳定（std 0.022 vs 0.068），符合预期：重新拟合小样本 KMeans 本身会引入额外的局部最优波动。

### 7.4 小结

- 手工图结构特征对分类器无稳定增量价值这一结论，在类别平衡后依然成立，进一步加固了 Week 9 的可信度。**embedding 部分的结论已被后续重复 CV 修正**（§5.2/§5.3）——本表基于平衡子集单次 5 折划分，检验力不足，不再作为 embedding 增量价值的最终判断依据。
- 聚类相关的原始负向结论（"embedding 空间无结构"）被证明部分不成立，已在 §4.4 修正为"存在中度结构，但标准 pairwise 指标在不均衡评估下会系统性低估它"。
- 两条修正方向相反（一条被加固、一条被部分推翻）恰好说明这次检验不是走过场——如果平衡子集总是"确认已有结论"，就不能排除确认偏误；这里出现了一个意料之外、机制清晰、可独立复现的反例，反而是这项检验真正起作用的证据。

### 7.5 Node2Vec p/q 敏感性检验（[node2vec_pq_sensitivity.py](node2vec_pq_sensitivity.py) → `embedding_outputs/pq_sensitivity_results.csv` / `.png`）

Project Plan Risk 5（Sampling Bias）的缓解措施之一是"test sensitivity if possible"，Week 10 里程碑也明确列了"sensitivity analysis"（与 balanced subset test 并列的两个子任务之一）。Week 7 全程只用 Node2Vec 的无偏游走（p=q=1，等价 DeepWalk），当时在训练代码里留了一条注释把这项检验记成待办。这里补上：用 Grover & Leskovec (2016) 原论文网格搜索的两个极端点重新训练 embedding——BFS-like（p=0.25, q=4，偏向局部/结构等价角色）和 DFS-like（p=4, q=0.25，偏向远距离/社区同质性），与 p=q=1 基线比较。

**⚠️ 方法修正**：本节最初只在全样本（n=436，299/86/51 严重不均衡）上跑 KMeans + 置换检验来比较三组 p/q 设置，而 §4.4/§7.3 已经证明全样本评估对这个 embedding 空间的标签对齐信号检出力不足——用一个已知检出力不足的指标去比较三组参数设置，本身就可能得出误导性的排序。因此补充了与 §7.3 相同的平衡子集重采样评估（每组 p/q 设置固定其 KMeans 聚类划分，对 30 次独立平衡（51/51/51）重采样子集分别计算 ARI 与置换 p 值），作为更公平的比较基准。

| 游走策略 | p | q | 全样本 KMeans ARI | 全样本 ARI p 值 | 平衡子集 ARI 均值（30 次重采样） | 标准差 | 30 次里显著的比例 |
|---|---|---|---|---|---|---|---|
| 无偏（基线，等价 DeepWalk） | 1.00 | 1.00 | -0.033 | 0.955（不显著） | 0.116 | 0.013 | 100% |
| BFS-like（局部/结构等价） | 0.25 | 4.00 | 0.010 | 0.240（不显著） | **0.199** | 0.024 | 100% |
| DFS-like（远距离/社区同质性） | 4.00 | 0.25 | -0.035 | 0.990（不显著） | 0.131 | 0.019 | 100% |

**结论（已被平衡子集评估修正，替代此前"无偏基线最优"的表述）**：全样本评估下三组设置的 ARI 全部不显著，无法据此排序；这与 §4.4/§7.3 的诊断一致——全样本评估条件本身检出力不足，不适合作为比较基准。**改用平衡子集评估后，三组设置的 ARI 全部转为统计显著（30 次重采样 100% 显著），且排序为 BFS-like（0.199）> DFS-like（0.131）> 无偏基线（0.116）**。这修正了此前"无偏游走已是三者中最优选择"的结论：在公平的评估条件下，无偏基线并非最优，只是三者中检出力最低的全样本比较让它看起来"损失最小"。

**⚠️ 重要限定：上表每组 p/q 设置只训练了一次 Node2Vec，30 次重采样检验的是"评估用的平衡子集选择"的稳定性，不是"Node2Vec 训练本身的随机性"（walk 生成 + Word2Vec SGD 即使固定 `random_state` 也不完全确定，见脚本注释）**。因此**不应表述为"BFS-like 已被证明是更优参数"**，更准确的说法是：**在当前训练实例和类别平衡评价条件下，BFS-like 设置显示出更高的标签对齐；由于每组 embedding 的训练随机性尚未通过多个训练种子系统评估，该结果目前视为探索性敏感性发现，而非最优参数的确定性证明**。`node2vec_pq_multiseed.py` 补充了 5 个独立训练种子 × 3 组 p/q 设置（共 15 次独立 Node2Vec 训练，每次训练用 10 次平衡子集重采样评估）的复核，结果见下方"训练种子稳健性复核"。

**训练种子稳健性复核（`pq_multiseed_summary.csv` / `pq_multiseed_per_seed.csv` / `.png`）**：

| 游走策略 | p | q | 平衡 ARI 均值（5 个训练种子） | 标准差 | 最小–最大 |
|---|---|---|---|---|---|
| 无偏（基线） | 1.00 | 1.00 | 0.217 | 0.051 | 0.119–0.255 |
| BFS-like（局部/结构等价） | 0.25 | 4.00 | 0.203 | 0.032 | 0.151–0.242 |
| DFS-like（远距离/社区同质性） | 4.00 | 0.25 | 0.199 | 0.041 | 0.122–0.241 |

**结果反转了单一训练实例得出的排序，并直接印证了上面的担忧是对的**：单一实例比较显示 BFS-like（0.199）明显高于无偏基线（0.116），但那只是这一次具体训练实例的结果。跨 5 个独立训练种子后，**三组设置的平衡 ARI 均值实际上非常接近（0.20–0.22），彼此的取值范围大量重叠**（无偏基线单个种子最低到 0.119、最高到 0.255；BFS-like 落在 0.151–0.242 之间），**训练种子之间的波动（std 0.03–0.05）与三组 p/q 设置之间的均值差（≤0.02）同一量级**。逐种子配对比较：无偏基线 − BFS-like 的差值在 5 个种子中只有 3 个为正（即基线更高），2 个为负，方向一致率 60%，远低于本文其它稳健发现（如 §5.2 的 10/10）的一致率；无偏基线 − DFS-like 同样是 3/5。**结论：在计入 Node2Vec 训练随机性后，没有证据表明 BFS-like（或任何测试过的有偏游走）系统性优于本文全程使用的无偏基线**——单一训练实例下观察到的"BFS-like 更优"是这一具体实例的特征，不是可复现的参数效应。这修正（而非加固）了上一版基于单一实例做出的排序结论，**该案例本身也是本文反复强调的方法论教训（单一小样本点估计不可靠，需要多次独立重复）在参数敏感性检验这一新场景下的又一次印证**。

**这一发现的边界**：(1) 只测试了 Grover & Leskovec (2016) 网格的两个极端点，不是完整的 p/q 网格搜索，不能断言 BFS-like 的具体取值（p=0.25, q=4）就是全局最优；(2) Week 5–10 全部分类、SHAP、contract-only 跨协议泛化等下游实验用的都是无偏基线 embedding，这一发现**不会使那些已有结果失效**——它们测量的是无偏基线 embedding 空间中真实存在的信号，只是提示无偏基线本身未必是这个采样图上表示学习的最优参数选择；(3) 这更适合作为 Future Work 中的一个具体方向（系统性 p/q 网格搜索 + 平衡子集评估，而非仅在全样本上评估）记录，而不是重新训练全部下游实验——**没有必要现在把所有主要实验换成 BFS-like embedding，因为那会构成"用同一份标签选定参数后又在同一份数据上报告性能"的数据泄漏风险**。（注：本表的绝对数值与 §4.4/§7.3 存在差异——Node2Vec 的游走生成即使固定 `random_state`，不同调用点仍有随机性，本表三行在同一次脚本运行内用相同流程重新训练，内部可比。）

---

## 8. 文件清单

```
feature_engineering.py       → seed_feature_matrix.csv, feature_groups.json
sanity_check.py               → seed_sanity_check_summary.csv
eda.py                         → eda_outputs/
  ├─ activity_by_label.csv
  ├─ skewness_raw_vs_log.csv
  ├─ zero_inflation_by_label.csv
  ├─ feature_distributions_by_label.png
  └─ feature_correlation_heatmap.png
node2vec_clustering.py        → embedding_outputs/
  ├─ node2vec_embeddings.csv          (全部 8,029 节点)
  ├─ seed_node2vec_embeddings.csv     (436 个有标签活跃种子)
  ├─ tsne_node2vec_seeds.png
  ├─ clustering_evaluation_summary.csv
  ├─ clustering_significance_test.csv
  ├─ kmeans_k_scan.csv / .png
  ├─ dbscan_eps_scan.csv
  └─ dbscan_k_distance.png
node2vec_pq_sensitivity.py    → embedding_outputs/
  ├─ pq_sensitivity_results.csv       (p/q 三组设置下 KMeans NMI/ARI + 置换检验，含全样本与 30 次平衡子集重采样两种评估条件)
  └─ pq_sensitivity.png
classification.py             → classification_outputs/
  ├─ cv_fold_scores.csv               (逐折原始分数)
  ├─ classification_summary.csv       (9 组合的均值±标准差)
  ├─ combined_vs_handcrafted_paired_test.csv  (逐折配对 t 检验)
  ├─ classification_per_class_report.csv
  ├─ feature_group_contribution.csv / .png    (RF/LR 是否真的用了 embedding)
  ├─ macro_f1_comparison.png
  └─ confusion_matrix_grid.png
feature_ablation_shap.py      → ablation_shap_outputs/
  ├─ ablation_cv_fold_scores.csv      (逐折原始分数)
  ├─ ablation_summary.csv             (3 组 x 3 模型的均值±标准差)
  ├─ ablation_paired_test.csv         (combined vs 单组的逐折配对 t 检验)
  ├─ shap_feature_importance.csv      (逐特征 SHAP 重要性，含逐类别拆分)
  ├─ ablation_macro_f1.png
  └─ shap_top_features.png
balanced_subset_robustness.py → balanced_subset_outputs/
  ├─ balanced_subset_seeds.csv        (153 个平衡子集种子，51/类)
  ├─ balanced_cv_fold_scores.csv      (Week 8/9 对比在平衡子集上的逐折分数)
  ├─ balanced_summary.csv
  ├─ balanced_paired_test.csv
  ├─ balanced_kmeans_comparison.csv   (KMeans/Louvain 在不同评估条件下的 NMI/ARI，支撑 §4.4 修正)
  └─ full_vs_balanced_comparison.png
graphsage_classification.py   → graphsage_outputs/
  ├─ graphsage_cv_fold_scores.csv     (5 折逐折分数 + 早停 epoch 数)
  ├─ graphsage_vs_baseline_comparison.csv  (vs graph_structural/hand_crafted 配对检验)
  └─ graphsage_vs_baseline.png
contract_confound_check.py    → contract_confound_outputs/
  ├─ seed_is_contract.csv             (500 个种子的链上真实 is_contract 查询结果)
  ├─ contract_ratio_by_label(_active).csv
  ├─ minimal_baseline_summary.csv
  └─ outbound_ablation_summary.csv / _fold_scores.csv / _paired_test.csv
verify_contract_reproducibility.py → contract_confound_outputs/
  ├─ contract_reproducibility_check.csv       (逐地址重查结果 vs 原始 is_contract，500/500 一致)
  └─ contract_reproducibility_metadata.json   (查询区块高度/时间戳，修复原始查询的可复现性记录缺口)
confound_followup.py          → contract_confound_outputs/
  ├─ contract_query_log.json
  ├─ eoa_only_cv_scores.csv / _summary.csv / _paired_differences.csv / _feature_ablation.png
  ├─ contract_only_balanced_runs.csv / _summary.csv / _macro_f1_distribution.png
  └─ remove_out_features_per_class.csv / _confusion_matrix.png
entity_grouped_check.py       → contract_confound_outputs/
  ├─ contract_only_entity_composition.csv    (sub_label 分布)
  ├─ contract_only_grouped_cv_scores.csv / _fold_composition.csv  (GroupKFold，多数退化)
  ├─ contract_only_leave_protocol_out.csv    (leave-one-DeFi-protocol-out)
  └─ contract_only_entity_grouped_comparison.png
repeated_cv.py                 → contract_confound_outputs/
  ├─ repeated_cv_fold_scores.csv      (10×5=50 折原始分数，7 个条件)
  ├─ repeated_cv_summary.csv
  ├─ repeated_cv_difference_ci.csv    (fold-level，n=50，存在伪重复，仅供参考——不作为结论依据)
  ├─ repeated_cv_repeat_level_ci.csv  (⭐ 结论依据：按 repeat 聚合为 10 个 repeat-level 配对估计，修正 fold 相关性/伪重复)
  └─ repeated_cv_difference_distribution.png
repeated_balanced_ari.py       → embedding_outputs/
  ├─ balanced_ari_repeated_runs.csv   (50 次独立平衡抽样的逐次 ARI)
  ├─ balanced_ari_summary.csv
  └─ balanced_ari_distribution.png
```

---

## 9. GraphSAGE：基于结构输入的监督消息传递实验（Week 10 可选项，[graphsage_classification.py](graphsage_classification.py) → `graphsage_outputs/`）

### 9.1 环境与动机

Project Plan 把 GraphSAGE 列为 Week 10 的可选项（"if core pipeline is complete"）。核心流程（Week 1–9）已完成，且环境可行性已确认：

- **环境问题根因**：本机 `import torch` 原本报 `OSError: [WinError 87]`，路径显示为畸形的 `'D:bin'`。根因不是系统级安装损坏，而是 Windows 下 `os.path.join` 的已知陷阱——本机 Python 直接安装在盘符根目录 `D:\`，`sys.exec_prefix` 返回不带结尾反斜杠的 `"D:"`，`os.path.join("D:", "bin")` 在 Windows 上会被解析成 `"D:bin"` 而非 `"D:\bin"`。在 `import torch` 之前手动给 `sys.exec_prefix`/`sys.base_exec_prefix` 补上结尾分隔符即可绕开，无需改动系统环境（代码见 `graphsage_classification.py` 开头）。这也解释了 Week 7 §4.2 里 umap 因 torch 而崩溃的根因，与之同源。修复后 `torch 2.11.0+cpu`、`torch_geometric 2.8.0`（`pip install --only-binary :all:`）均正常工作。
- **动机**：Week 9 消融（§6.3）和 Week 10 重复交叉验证（§6.3、§5.3）已表明，手工图结构特征（静态工程统计量）相对交易行为特征没有稳定正向增量，而 Node2Vec 浅层无监督 embedding 显示出模型依赖的条件性正向价值。GraphSAGE 用第三种机制——端到端、任务驱动的监督式邻域聚合——检验这一模式是否在同样的结构信息输入下持续：给定与手工结构特征完全相同的节点输入（因为交易行为特征无法覆盖非种子节点，见 9.2），端到端学习的消息传递聚合能否超过静态结构统计量。

### 9.2 方法

- **样本**：复用 §7 的 153 个类别平衡种子（51/51/51），而非全样本 436——这样 GraphSAGE 的结果从一开始就落在已经排除类别不均衡混杂的地基上，不需要事后再做一次平衡复核
- **节点特征**：GraphSAGE 需要为图中每个节点（而非仅 153 个种子）提供输入特征用于消息传递。交易行为特征（tx 数量、ETH 金额、gas 费）依赖逐地址 BigQuery 查询，只对 500 个种子算过，非种子的对手方节点没有；因此 GraphSAGE 的输入只能是可对全图 8,029 节点计算的结构特征（in/out/degree、pagerank_rank、pagerank_weighted_rank、clustering_coeff、has_closed_triangle、wcc_size、log_degree，与 `graph_structural` 特征组一致），标准化后作为节点特征。**这一限制决定了后续对比的公平性边界**（见 9.3）。
- **图**：与 Node2Vec 相同的无向视图（12,694 条边），理由相同——有向随机游走/消息传递在大量单方向叶子节点上会立刻卡死
- **模型**：刻意压低容量以匹配 153 个标签的监督信号量——2 层 `SAGEConv`，hidden_dim=16，dropout=0.6，Adam（weight_decay=1e-3），最多 300 epoch、验证集 macro-F1 早停（patience=20）
- **Transductive 设置与防泄漏**：全部 8,029 个节点的特征和图结构在训练全程始终参与消息传递（标准 transductive 节点分类设定，与 GraphSAGE/GCN 原始论文一致）；5-fold 分层交叉验证切分的是"哪些种子的标签参与损失计算"，不是"哪些节点参与消息传递"——每折的验证集（用于早停）从该折训练集内部切出 20%，从不触碰当折的测试种子，测试种子的标签在训练、早停的全过程都不出现，只在最终评估时使用
- **对比 baseline**：与 §7 使用完全相同的 153 个种子和相同的 5-fold 划分（`random_state=42`），因此可以做逐折配对比较

### 9.3 结果

**GraphSAGE 5 折 CV（`graphsage_cv_fold_scores.csv`）**：Macro-F1 = 0.741 ± 0.156。每折均在 24–58 个 epoch（上限 300）内早停，无震荡或不收敛迹象。

**与 Week 10 平衡子集 baseline 的配对比较（`graphsage_vs_baseline_comparison.csv` / `graphsage_vs_baseline.png`）**：

| baseline | baseline macro-F1 | GraphSAGE − baseline | p 值 | 对比性质 |
|---|---|---|---|---|
| graph_structural（RF） | 0.763 ± 0.166 | -0.022 | 0.324（不显著） | **苹果对苹果**：输入信息完全相同 |
| graph_structural（XGBoost） | 0.766 ± 0.168 | -0.024 | 0.238（不显著） | **苹果对苹果** |
| hand_crafted（RF） | 0.895 ± 0.094 | -0.154 | **0.042** | **非苹果对苹果**：hand_crafted 含 GraphSAGE 拿不到的交易行为特征 |
| hand_crafted（XGBoost） | 0.855 ± 0.086 | -0.113 | **0.047** | **非苹果对苹果** |

**方差说明**：GraphSAGE 折间 std=0.156，与 graph_structural 手工特征在同一个 153 样本子集上的 std（0.166–0.190）同一量级——不是 GraphSAGE 特有的不稳定，是 n=153、每折测试集仅约 10 个/类这个小样本规模的共性噪声。

### 9.4 结论——受限于结构输入的一次监督消息传递实验

**主对比是 vs graph_structural（两者拿到完全相同的结构输入）：没有观察到 GraphSAGE 优于静态手工结构特征基线的证据**（p=0.24–0.32，GraphSAGE 均值甚至略低）。**这不等于"两者已被证明等价"**——单次 5 折 CV 的检验力有限，只能说是"未观察到优势"，不能排除更大样本或不同超参数下会出现差异的可能。端到端学习消息传递聚合，相对于直接使用预先算好的 degree/pagerank/clustering 统计量，在本实验中没有带来可测得的增量。vs hand_crafted 的显著劣势（p=0.04–0.05）**不能被解释成"GraphSAGE 相对交易行为特征没有增量价值"**——这个对比本身不公平，GraphSAGE 的输入从一开始就不包含交易行为特征这整整一类信息（§9.2），它只能支持一个更窄的表述：**在相同的结构信息输入条件下，GraphSAGE 没有超过 RF/XGBoost 的结构特征基线**；vs hand_crafted 的差距更可能只是重申了 Week 9 SHAP 已经证明的事实——交易行为特征才是本数据集的主要信号来源，而不是对 GraphSAGE 本身能力的评价。

**训练设计上采取了一些排除明显训练失败的保障措施**：模型容量被主动压低（2 层、hidden_dim=16），早停全部在 300 epoch 上限内正常触发（24–58 epoch），折间方差和其他方法同量级而非异常放大。这些迹象说明结果不是由明显的训练发散或崩溃造成的，因此可以作为一次合理设计的负向结果来解读；但**不能因此排除容量、超参数选择或 n=153 样本量本身对结果的限制**——没有做过 hidden_dim/层数/dropout 的系统性网格搜索。§9.5（P3）用固定容量/超参数的消息传递消融进一步排除了"消息传递完全没跑起来"这一种解释（真实边显著优于度保持重连边），但仍未做超参数网格搜索，不能排除某种其它超参数设置下 GraphSAGE 会超过 graph_structural 基线的可能。这一限制应在方法局限中明确说明。

**⚠️ 以下"三方法收敛"的表述已被 Week 10 重复交叉验证部分推翻，保留在此仅供追溯修正过程**：综合 Week 7 / Week 9 / Week 10 三次独立检验，此前认为 Node2Vec（浅层无监督、基于随机游走的共现）、手工图结构特征（静态工程统计量）、GraphSAGE（端到端监督式邻域聚合）三种提取图结构信号的方式会收敛到同一个"均无增量"的结论。**重复交叉验证（§5.3）显示这个收敛并不成立**：Node2Vec 在完整样本上对分类器有统计显著的正向增量（模型依赖），只有手工图结构特征和 GraphSAGE 这两种方式的负向结论被加固或维持。

**修正后的表述**：手工图结构特征（静态工程统计量）相对交易行为特征基线没有显示出稳定增量（§6.3），GraphSAGE 相对同等结构输入的静态特征基线（graph_structural）也没有显示出优势（本节，苹果对苹果比较）——这两条结论的比较对象不完全相同（前者是"结构特征 vs 交易行为特征"，后者是"端到端结构学习 vs 静态结构特征"），但方向一致，都指向"在这个采样拓扑下，可提取的结构信号已经相对有限"。Node2Vec 浅层无监督 embedding 不属于这个模式——它在完整样本上有统计显著的正向增量，只是这个增量本身有模型和任务依赖性（§5.3）。因此本文对 RQ2 的最终回答是**三分法而非收敛的一元负向结论**：Node2Vec（有条件的正向价值）、手工结构特征（无价值，甚至对 RF 有轻微负价值）、GraphSAGE（在结构输入受限、且未做超参数网格搜索的条件下，未观察到相对静态结构基线的优势，因此不构成对图神经网络方法本身的否定）。

**必须显式划定的边界，写作时不可省略**：
1. **这个结论的作用域是"这个 star-like 采样拓扑"，不是"图表示学习/GNN 整体无效"**。Top-100 counterparty 抽样本身把三角结构人为压薄了——clustering coefficient 在三类地址中普遍 80–99% 为零（§3.3）是这个采样策略的产物，不是以太坊交易图的固有属性。一个使用完整 k-hop 邻域或不同采样策略、保留更多真实三角结构的图，拓扑信号可能厚得多，GraphSAGE 在那种图上未必得出同样的负向结果。本文的结论是"给定 Top-100 counterparty 抽样这个具体约束，可提取的结构信号已被简单统计量吃干净"，这个边界应作为方法局限（limitation）在讨论章节明确交代。
2. **GraphSAGE 特有的信息劣势（无交易行为特征）不应被混入"图方法无效"的结论**——它只影响 GraphSAGE-vs-hand_crafted 这一个对比，不影响 GraphSAGE-vs-graph_structural 这个核心的苹果对苹果结论。

### 9.5 P3：消息传递消融（[graphsage_p3_ablation.py](graphsage_p3_ablation.py) → `graphsage_outputs/p3_ablation_*`）

**动机**：§9.4 的 vs graph_structural 对比只能说"未观察到 GraphSAGE 优于静态结构特征基线的证据"，但无法区分两种解释：(a) 这个采样图上的消息传递本身确实不提供额外信息，还是 (b) 消息传递有价值，只是当前的模型容量/超参数/样本量没能让它体现出来。P3 通过固定节点特征、模型容量、训练协议和数据划分，只改变"是否使用消息传递"以及"消息传递的对象是真实边还是保持度分布不变的重连边"，把这两种解释分离开。

**方法**：三个条件共享完全相同的 153 个类别平衡种子、完全相同的 `StratifiedKFold(n_splits=5, random_state=42)` 折划分（复用 §7 的划分）、完全相同的节点特征（§9.2 的 9 个结构特征）、完全相同的模型容量（2 层，hidden_dim=16，dropout=0.6，相同优化器与早停协议）：
- **A. MLP**（无边）：与 GraphSAGE 深度/宽度/dropout 完全匹配的两层 MLP，不使用图边，每个节点独立变换
- **B. GraphSAGE + 真实边**：与 §9.2–9.4 相同的采样图（12,694 条无向边）
- **C. GraphSAGE + 度保持重连边**：用 `networkx.double_edge_swap`（nswap = 5×边数）对真实图做保度重连——精确保持每个节点的度不变，但破坏"具体哪个节点连向哪个节点"这一真实邻接语义（实测 12,694 条边中约 98% 被替换为不同的端点对，度序列完全一致）。这样 B 与 C 的差异不能被"度分布不同"这一平凡因素解释

每个条件在 **5 个独立训练种子**（42–46，同时决定权重初始化/dropout 掩码，条件 C 下也决定具体的重连图实例）× 5 折下训练，共 75 次独立训练；先按训练种子对 5 折取均值（避免把同一种子内高度相关的 5 折当独立观测，与 §5.2 的方法论教训一致），再对 5 个种子级均值做配对比较。**预先定义的两个主要比较**（非事后挑选）：(1) GraphSAGE(真实边) − MLP，(2) GraphSAGE(真实边) − GraphSAGE(重连边)。

**结果**（`p3_ablation_summary.csv` / `p3_ablation_primary_comparisons.csv` / `p3_ablation.png`）：

| condition | macro-F1 均值（5 个训练种子） | 标准差 |
|---|---|---|
| A. MLP（无边） | 0.702 | 0.027 |
| B. GraphSAGE（真实边） | 0.725 | 0.020 |
| C. GraphSAGE（度保持重连边） | 0.605 | 0.036 |

| 预注册比较 | 均值差 | 95% CI（5 个训练种子） | p 值 | 方向一致的种子比例 |
|---|---|---|---|---|
| GraphSAGE(真实边) − MLP(无边) | +0.023 | [-0.006, 0.052] | 0.088（不显著） | 4/5（80%） |
| **GraphSAGE(真实边) − GraphSAGE(重连边)** | **+0.120** | **[0.055, 0.184]** | **0.007（显著）** | **5/5（100%）** |

**结论**：真实边相对度保持重连边有一个稳定、统计显著的优势（+0.12，5/5 训练种子方向一致），说明**这个采样图的具体邻接结构（谁连向谁）携带了单纯度分布之外的信息，消息传递并非在"随便传"**——如果消息传递对具体拓扑完全不敏感，重连边（度分布相同）应该表现相近，但实际观测到了明显差距。但真实边相对纯节点特征的 MLP 只有一个方向上多数一致（4/5 种子）、但未达统计显著（p=0.088，CI 跨零）的小幅优势，说明**这个优势大部分被节点特征本身吸收，消息传递能提供的、节点特征之外的增量信息有限**。

按 §10.7 的措辞纪律，这里避免使用"证明消息传递无效"这类过强表述，采用更准确的描述：**在当前架构和这个采样图上，没有检测到来自观测到的邻域结构的稳定额外收益（相对节点特征本身），但真实拓扑显著优于随机化对照，说明图结构本身并非无信息**。这是一个比 §9.4 单独的 vs graph_structural 结果更精细的结论——它把"消息传递没有跑起来"（真实边 vs 重连边的显著差距排除了这个解释）和"消息传递没有额外贡献"（真实边 vs MLP 的不显著差距）区分开来，答案更接近后者。

---

## 10. Account-type 混杂与去混杂分析（Week 10 追加，[contract_confound_check.py](contract_confound_check.py) → `contract_confound_outputs/`）

### 10.1 动机

Week 8/9 全部分类结论都建立在一个未检验的假设上：三个类别标签（CEX/DeFi/Malicious_MEV）主要反映经济行为差异。但 §3.3 EDA 已经观察到"活跃 DeFi 种子的 `out_edge_count` 100% 为零"，这本身就是一个警示信号——DeFi 标签在数据里可能更接近"合约地址"这个账户类型标签，而不是一种独立的经济行为模式。如果是这样，Week 8/9 报告的高 macro-F1，很大一部分解释力可能来自模型在识别"这是不是一个合约"，而不是在学习三类地址各自的行为特征。这个混杂如果不检验清楚，会威胁 Week 8/9/10 全部依赖三分类 macro-F1 的结论。

### 10.2 方法：链上真实合约查询

通过公开 Ethereum JSON-RPC 节点（`https://ethereum-rpc.publicnode.com`）对全部 500 个种子地址调用 `eth_getCode`，取得链上真实字节码：

- **判定规则**：返回值非 `"0x"` 且长度 > 2（即字节码非空）判定为合约（`is_contract=True`），否则为 EOA
- **查询记录**：逐地址串行查询，失败自动重试 3 次（间隔 1.5s × 尝试次数），全部 500 个地址查询成功，无 RPC 失败/降级为默认值的情况（`rpc_ok=True` 覆盖率 100%）
- **产出**：`seed_is_contract.csv`（address, bytecode_length, is_contract, rpc_ok），可独立复现，`is_contract` 是链上直接查询结果，不是代理变量
- **⚠️ 原始查询的可复现性局限（已用独立复核部分修复）**：原始查询用的是 `eth_getCode(address, "latest")`，当时未记录查询区块高度或时间戳——这意味着严格来说"在哪个链状态下判定"这一信息没有被完整记录，属于方法透明性缺口（不影响判定本身的正确性，因为合约部署是不可逆的：一旦地址部署了字节码，它不会在之后的任何区块变回空字节码，所以 `is_contract=True` 的判定不会因为查询时间不同而改变）。`verify_contract_reproducibility.py` 于 2026-07-12T12:40 UTC（链头区块 **25,516,603**，`"latest"`）对全部 500 个地址重新查询 `eth_getCode`，与原始 `seed_is_contract.csv` 逐地址比对，**500/500（100%）完全一致，0 处不匹配**（`contract_reproducibility_check.csv` / `contract_reproducibility_metadata.json`）。这确认了原始判定是稳定、可独立复现的，但论文方法章节仍应如实说明：更严格的可复现协议应在查询时就记录区块高度（例如固定在 90 天采样窗口结束时对应的历史区块查询），而不是依赖"latest"的事后复核。

### 10.3 结果一：DeFi 完全嵌套于合约身份之内

**500 个种子（全部）**：

| label | n_contract | n_total | contract_ratio |
|---|---|---|---|
| CEX | 2 | 51 | 3.9% |
| **DeFi** | **150** | **150** | **100.0%** |
| Malicious_MEV | 34 | 299 | 11.4% |

**436 个活跃分类样本（Week 8/9/10 实际用于训练的样本）**：

| label | n_contract | n_total | contract_ratio |
|---|---|---|---|
| CEX | 2 | 51 | 3.9% |
| **DeFi** | **86** | **86** | **100.0%** |
| Malicious_MEV | 34 | 299 | 11.4% |

两个表的比例几乎完全一致（活跃 DeFi 种子同样 86/86 全部为合约），说明账户类型混杂问题对分类实验样本同样成立，不是仅存在于未使用的非活跃种子中。

**准确的集合关系是：DeFi 完全嵌套于合约身份之内，而非与合约身份外延相等**——DeFi 类内部合约比例是零方差（100%），使得完整三分类实验的设计本身无法把"DeFi 特有的经济行为"和"泛化的智能合约地址特征（无论什么类型的合约）"分离开来，因为数据里不存在"非合约的 DeFi"这种反事实样本。但这不代表完全没有同账户类型的对照组——**Malicious_MEV 类中有 34 个（11.4%）合约（多为 MEV bot 合约），CEX 类中有 2 个合约**，这两组构成了后续 contract-only 去混杂分析（§10.8b）的对照样本；只是 CEX 合约仅 2 个，样本量不足以单独支持 DeFi–CEX 的同类型对比。这不是可以靠调整模型或特征工程解决的问题，是**当前 500 个种子的标签构成本身决定的一个设计局限**。

### 10.4 结果二：极简基线证实混杂解释力很大，但不能说成"无行为信息"

用三个特征——`is_contract`（合约身份）、`has_out_edge`（是否曾主动发起过交易）、`out_tx_count`（出向交易数，一个粗粒度行为变量）——跑 5 折 CV：

| model | macro-F1 |
|---|---|
| LogisticRegression | 0.594 ± 0.058 |
| **RandomForest** | **0.917 ± 0.025** |
| **XGBoost** | **0.910 ± 0.024** |

RF/XGBoost 用这 3 个特征就拿到和 28 特征完整 `hand_crafted` 基线（RF 0.905、XGBoost 0.906）几乎相同的分数。**准确的解释是**：账户类型（合约 vs EOA）和粗粒度方向性活动（是否发起过交易、发起了多少次）可以解释三分类预测性能中的很大一部分。**不准确、需要避免的表述**：不能说这 3 个特征"不含行为信息"——`out_tx_count` 本身就是一个真实的、粗粒度的行为变量，只是不如完整特征集精细。这里的发现是"粗粒度信息的解释力被完整特征集掩盖了、看起来像是精细特征在起作用"，不是"分类器没有用到任何行为信号"。

### 10.5 结果三：去除出向特征——一个部分而非完整的去混杂检验

去掉全部 7 个出向方向特征（`out_tx_count`、`out_eth_value`、`out_gas_fee_mean`、`weighted_out_degree`、`log_out_tx_count`、`log_out_eth_value`、`out_degree`）重新跑 5 折 CV：

| feature_set | model | macro-F1 |
|---|---|---|
| hand_crafted（完整 28 特征） | RandomForest | 0.905 ± 0.037 |
| hand_crafted（完整 28 特征） | XGBoost | 0.906 ± 0.034 |
| hand_crafted_no_outbound（21 特征） | RandomForest | 0.883 ± 0.054 |
| hand_crafted_no_outbound（21 特征） | XGBoost | 0.889 ± 0.045 |

配对检验：RF 均值差 +0.022（p=0.085，临界），XGBoost 均值差 +0.017（p=0.095，临界），n=5 折检验力低。

**只能得出的结论**：去除出向特征后，RandomForest 和 XGBoost 的总体 macro-F1 下降约 1.7–2.2 个百分点，表明出向活动提供了一定的额外判别信息。**不能得出的结论**：这不是一次完整的去混杂检验——DeFi 的合约身份很可能还通过入向特征（`in_degree`、`weighted_in_degree`）和结构特征（`wcc_size`、`clustering_coeff`，热门 DeFi 池子合约的入向连接模式和典型 EOA 差异巨大）渗入模型，去掉"out_"前缀字段只去掉了最直接的那部分代理信号，不等于去掉了合约身份这个混杂变量本身。逐类别拆解（哪个类别的下降贡献最大）见 §10.8（P1 后续实验，进行中）。

### 10.6 结果四：EOA-only 子实验——最干净的去混杂证据

彻底剔除全部合约地址，只保留两个真实的 EOA 经济行为类别（CEX 49 个 + Malicious_MEV 265 个 = 314 个），在 `hand_crafted` 完整特征集上重跑 5 折 CV：

| model | macro-F1 |
|---|---|
| LogisticRegression | 0.924 ± 0.032 |
| RandomForest | 0.920 ± 0.026 |
| XGBoost | 0.923 ± 0.066 |

**这是本节最重要的正向发现**：彻底排除合约类型混杂后，CEX 与 Malicious_MEV 两类 EOA 依然能被清楚区分（macro-F1 0.92–0.93）。**这证明混杂不能解释一切**——CEX 与 Malicious_MEV 之间存在不能仅由 contract/EOA 差异解释的、稳定的类别相关信号。**这里应避免的两个过度表述**：(1) 不宜直接称之为"真实的经济行为信号"——这只是排除了账户类型这一个混杂因素，标签来源、种子选择方式和采样时间窗口等其它潜在混杂并未被排除；更准确的说法是"不能仅由 contract/EOA 差异解释的类别相关信号"。(2) 这里的二分类 macro-F1（0.92–0.93）不应直接与三分类完整基线（0.892–0.906）比较高低并作为"混杂不影响结果"的证据——二分类和三分类是不同难度的任务，数字可以并列报告，但不构成同一尺度上的强弱证据。

### 10.7 修正后的措辞规则（替代此前所有"三类均可分"式表述）

**核心判断**：完整三分类中，DeFi 特定行为与一般合约特征不能被单独识别；但 contract-only 分析（§10.8）表明，DeFi 与 Malicious_MEV 合约之间存在稳定且不能由账户类型解释的类别相关交易和拓扑信号。由于缺乏足够的 CEX 合约，该去混杂结论目前仅适用于 DeFi–MEV 比较——**是三类别范围内识别问题的部分缓解，不是彻底推翻此前的负面表述，也不是"DeFi 经济行为已被完全证明"**。

以后统一拆成三句话讲，不再笼统地说"三类地址可被有效区分"：

1. **CEX 与 Malicious_MEV 在 EOA-only 条件下仍然具有较强可分性（macro-F1 0.92–0.93），说明其类别相关信号不能仅由 contract/EOA 差异解释**（§10.6）——这只排除了账户类型这一个混杂因素，不代表已排除全部潜在混杂，因此避免直接称之为"真实的经济行为信号"。
2. **DeFi 与 Malicious_MEV 在 contract-only 条件下（双方均为合约，账户类型固定为常数）依然高度可分**（§10.8b，50 次平衡重采样运行稳定复现），说明这部分可分性不能由 contract/EOA 差异解释，必须来自账户类型之外的类别相关交易或拓扑模式。**但这只解决了 DeFi–MEV 这一对类别的识别问题**——CEX 合约仅 2 个，样本量不足以做对等的 DeFi–CEX contract-only 检验，因此完整三分类范围内的账户类型识别问题是部分而非完全解决。
3. **DeFi 相对其余类别（尤其 CEX）的表观可分性，在当前三分类设计下仍然无法与一般合约地址特征充分分离**（§10.3）——因为 DeFi 身份完全嵌套于合约身份之内，样本中没有"非合约 DeFi"这种反事实可供对照。这是数据设计的局限，需要作为 limitation 明确写入 Discussion 章节。

§5.3、§6.3、§9.4 中所有涉及"三类分类效果"的结论性表述，写作时都应加上这条限定；Week 8/9 报告的整体 macro-F1（0.89–0.95）作为三分类任务的技术性能数字仍然准确、可复现，但作为"模型学到了三种真实经济行为模式"的证据，对 CEX vs Malicious_MEV 成立（§10.6），对 DeFi vs Malicious_MEV 在 contract-only 条件下成立（§10.8b），对涉及 CEX–DeFi 的判别仍然存疑。

**英文版本（供论文正文使用）**：
> All DeFi seeds were contract addresses, meaning that the full three-class analysis could not separately identify DeFi-specific behaviour from generic contract-address characteristics. The contract-only robustness analysis partially resolved this identification problem. When the sample was restricted to DeFi and Malicious MEV contract addresses, the two classes remained strongly and consistently distinguishable across 50 balanced resamples. Because contract status was constant within this experiment, the observed discrimination cannot be explained by the contract-versus-EOA distinction and must arise from other class-associated transactional or topological patterns. This result resolves the account-type confound only for the DeFi–Malicious MEV comparison. Since the CEX class contained only two contract addresses, an equivalent within-account-type comparison between DeFi and CEX was not feasible. The identification limitation is therefore partially, rather than completely, resolved.

### 10.8 补充实验结果

**10.8a EOA-only 特征分组消融（确认了原图特征结论——不是混杂的产物）**：`eoa_only_feature_ablation.py`（并入 `confound_followup.py`），n=314（CEX 49 + Malicious_MEV 265）。

| feature_set | LR | RF | XGB |
|---|---|---|---|
| transaction_behaviour | 0.936 | 0.928 | 0.942 |
| graph_structural | 0.702 | 0.831 | 0.835 |
| hand_crafted（transaction+structural） | 0.924 | 0.920 | 0.923 |
| embedding | 0.909 | 0.870 | 0.919 |
| hand_crafted + embedding | 0.957 | 0.919 | 0.940 |

配对检验（单次 5 折 CV）：`hand_crafted vs transaction_behaviour` 三模型均不显著、方向为负（加图结构特征不提升，反而略微拖累）；`hand_crafted+embedding vs hand_crafted` 只有 LogisticRegression 显著（+0.033，p=0.047，4/5 折提升），RF/XGBoost 不显著。

**重复交叉验证修正（`repeated_cv.py`，10 次重复、按重复聚合的 repeat-level 配对估计）**：LogisticRegression 之前那次"边界性提升"**未能在重复评估中复现**——均值差从单次 5 折的 +0.033 降到重复检验的 +0.004（95% CI 含零，p=0.49，6/10 重复方向为正，接近随机）。RF/XGBoost 同样不显著（p=0.53、0.33）。**这次是反方向的修正**：n=5 折时"看起来有效果"的信号，在更多重复下表现不稳定，更可能是噪声而非真实效应——与主样本 RF-embedding 效应的修正方向相反，共同说明 n=5 折点估计在本项目中不可靠，需要用重复 CV 复核后再下结论。

**结论**：在完全剔除合约地址后，交易行为特征仍明显优于手工图结构特征；加入图结构特征没有改善任何一个模型；Node2Vec 在 EOA-only 场景下对任何模型都没有稳定、可复现的增量（不同于主样本上的正向发现，见 §5.3）。**这一模式（embedding 在主样本/contract-only 有效，在 EOA-only 无效）提示 embedding 的价值可能集中在涉及合约地址、或需要区分不同合约功能角色的任务中**，而在同为 EOA 的 CEX vs Malicious_MEV 任务里，交易行为特征已经吸收了绝大多数可用信号，embedding 没有额外贡献。**图结构信息（手工特征）增量价值有限这一结论不是由 contract/EOA 混杂造成的**——这部分在 EOA-only 下依然成立，是稳健的负向结论。

**10.8b Contract-only DeFi vs Malicious_MEV（50 次平衡重采样运行，n=68/次，见 §10.7 结论——这 50 次运行并非 50 个独立外部验证集：34 个 MEV 合约在每次运行中反复被抽样，因此这里检验的是"对 DeFi 一侧下采样选择的敏感性"，不是对完全独立样本的重复验证）**：`contract_only_summary.csv` / `contract_only_macro_f1_distribution.png`。

| feature_set | LR | RF | XGB |
|---|---|---|---|
| coarse（3 特征：`has_out_edge`、`out_tx_count`、`total_tx_count`——`is_contract` 在此子集内为常数，因此用 `total_tx_count` 替代，而非简单剔除后只剩 2 个特征） | 0.871 | 0.852 | 0.852 |
| transaction_behaviour | 0.845 | 0.919 | 0.931 |
| graph_structural | 0.801 | **0.909** | **0.920** |
| hand_crafted | 0.870 | 0.923 | 0.944 |
| embedding | 0.887 | 0.902 | 0.869 |
| hand_crafted + embedding | 0.987 | 0.996 | 0.973 |

**graph_structural 在这里的任务依赖性**：单独使用时（RF 0.909、XGB 0.920）几乎追平 transaction_behaviour（RF 0.919、XGB 0.931），与主样本、EOA-only 中"图特征明显更弱"的模式不一致。**这不代表"图特征普遍变得有效"，而是图结构特征的判别力具有任务依赖性**——一个可能的机制假设（尚未验证，需 SHAP/单特征效应量支持）：DeFi 池子合约往往是枢纽（大量不同调用者带来高入度），MEV bot 合约交互模式更窄、更专业化，degree/PageRank/连通分量这类结构变量因此对这一特定类别对有区分力。同时，`hand_crafted` 相对单独的 `transaction_behaviour`/`graph_structural` 增量有限，说明两组特征在这个任务里可能捕捉了高度重叠的差异；"单独可预测"和"在交易行为基础上提供新增独立信息"是两件不同的事，目前没有证据表明结构特征在所有任务里都能提供稳定增量。

**接近 1.00 的 hand_crafted+embedding 结果的可信度检查（`entity_grouped_check.py`）**：`sub_label`（协议/行为子类标识）显示严重的组间集中——DeFi 合约 86 个里 64 个（74%）是 Uniswap，10 个（12%）是 Curve；Malicious_MEV 合约 34 个里 33 个（97%）共享同一个 `sub_label`"Arbitrage Trader"（这是一个行为/MEV 类型标签，不代表已确认是单一运营者）。原始 50 次平衡重采样运行用的是逐地址随机的分层 CV，同一 sub_label 的地址完全可能同时出现在训练折和测试折。

标准的按 `sub_label` 分组的 GroupKFold 在该子集中无法可靠实施：10 个 fold×model 组合里 8 个退化（某一类别在训练集或测试集中完全缺失），因为 sub_label 与类别高度集中且近乎重合——按组留出会直接搬空某一类别。**这一退化结果反映的是样本缺少足够的组间支持，不是分类器本身的性能问题**。

为检验模型是否只是记住了特定 DeFi 协议的地址指纹，本研究进一步实施了 **leave-one-DeFi-protocol-out** 评估：每次将一个 DeFi 协议的全部地址排除出**监督分类器训练**（`hand_crafted+embedding` 特征集，含 Node2Vec embedding——但 embedding 本身是在完整无标签图上预训练的，这些地址的连接结构在表示学习阶段并非"完全未见"，只是它们的**标签**没有参与监督分类器的训练，故用"held out from supervised training"而非"completely unseen"），测试模型识别该协议地址类别的能力：

| 被留出的 DeFi 协议 | n | LR macro-F1 | RF macro-F1 |
|---|---|---|---|
| **Uniswap**（占 DeFi 合约 74%） | 64 | 1.000 | 0.989 |
| **Curve** | 10 | 1.000 | 1.000 |
| 0x API（探索性，样本量太小不作为独立证据） | 2 | 1.000 | 1.000 |
| dodo（探索性，样本量太小不作为独立证据） | 2 | 1.000 | 1.000 |

每次测试集都同时包含被留出协议的 DeFi 地址和一批 Malicious_MEV 测试样本（非单类别退化测试集，macro-F1 具有正常的二分类含义）。**即使占 DeFi 合约样本 74% 的 Uniswap 地址全部未参与监督训练，模型仍能以约 0.99–1.00 的 macro-F1 将其与 Malicious_MEV 合约区分**；Curve（n=10）结果同样稳定。0x API、dodo 样本量各只有 2，不作为独立证据，仅供参考。**这表明 contract-only 的高可分性并不主要依赖记住特定 DeFi 协议，提供了 DeFi 侧跨协议泛化的证据**。

**但该检验具有明显的非对称性，尚未解决**：约 97% 的 Malicious_MEV 合约共享"Arbitrage Trader"这一个 sub_label，当前数据缺乏足够的 MEV sub_label/行为类型多样性，无法实施对称的 leave-one-MEV-group-out 检验。**现有结果只能说明：在当前已观察到的 MEV 分布下，模型能够泛化到监督训练中未出现的 DeFi 协议；不能证明模型同样能够识别未见过的 MEV 类型或操作者**——这是当前数据集尚未解决的、独立于协议记忆问题的另一条限制。

**§10.8b 最终结论**：Contract-only 的 DeFi–MEV 可分性并非由合约身份或对特定 DeFi 协议的简单记忆所解释，模型表现出较强的 DeFi 侧跨协议泛化；但由于 MEV 合约的 sub_label 多样性严重不足，尚不能验证对未见 MEV 类型或操作者的对称泛化。

**英文版本（供论文正文使用）**：
> Standard GroupKFold evaluation by sub-label was not feasible in the contract-only subset because group membership was highly concentrated and strongly associated with class membership. Most Malicious MEV contracts shared the "Arbitrage Trader" sub-label, while Uniswap accounted for the majority of DeFi contracts. Holding out entire sub-label groups therefore frequently removed one class from either the training or test set. The resulting degeneracy reflects insufficient group-level support in the data rather than model performance.
>
> To assess whether the classifiers merely memorised protocol-specific DeFi patterns, a leave-one-DeFi-protocol-out evaluation was conducted. In each run, all addresses belonging to one DeFi protocol were held out from supervised classifier training. Even when Uniswap, which represented approximately 74% of the DeFi contract sample, was entirely withheld from supervised training, the models distinguished its addresses from Malicious MEV contracts with macro-F1 scores of approximately 0.99–1.00. Results for Curve were similarly strong. This indicates that the contract-only separability was not primarily dependent on memorising specific DeFi protocols and provides evidence of out-of-protocol generalisation on the DeFi side.
>
> However, this validation was asymmetric. Approximately 97% of the Malicious MEV contracts shared the "Arbitrage Trader" sub-label, leaving insufficient diversity of MEV sub-labels, operators or bot families for an equivalent leave-one-MEV-group-out test. The results therefore demonstrate generalisation to previously withheld DeFi protocols under the observed MEV distribution, but do not establish generalisation to previously unseen MEV types or operators.

**10.8c 去除出向特征——逐类别拆解（`remove_out_features_per_class.csv` / `remove_out_features_confusion_matrix.png`）**：RandomForest 下降集中在 **DeFi 的 recall**（1.000→0.919，F1 0.945→0.893），CEX/MEV 基本不受影响（F1 变化在 ±0.02 以内，LR 下 CEX 甚至还涨了）。删除出向特征造成的性能下降主要集中在 DeFi 召回率，说明"缺乏主动出向活动"是识别 DeFi 的重要信号；但账户类型仍可能通过入向和结构特征体现，因此该消融不能单独完成去混杂（与 §10.5 的原有结论一致，现在有逐类别数据支撑）。

### 10.9 P2 状态

- ~~按 `sub_label` 分组的 GroupKFold / leave-one-DeFi-protocol-out~~ ✅ 已完成，见 §10.8b
- ~~重复分层 CV（10×5=50 折）+ 95% 置信区间 + 实际意义阈值~~ ✅ 已完成，结果见 §5.2/§5.3/§10.8a——**这一步产生了本项目迄今最重要的修正：RandomForest 对 embedding 的反应从单次 5 折"无效果"修正为"效应较小但能在重复评估中稳定复现"，EOA-only 场景下 LogisticRegression 的一次边界性提升则未能在重复评估中复现，更可能是噪声，两次修正方向相反，共同确立了"n=5 折点估计不可靠，需重复 CV 复核"这一方法论教训。统计口径：置信区间与配对 t 检验基于 10 个 repeat-level 配对估计（按重复聚合），而非把 50 个高度相关的 fold 当作独立观测**
- ~~重复平衡子集聚类检验（50 次独立抽样，ARI 分布）~~ ✅ 已完成，结果见 §7.3——ARI 稳定在 0.09–0.32 区间，50 次全部显著，确认 §4.4/§7.3 的中度结构发现不是单次抽样的巧合

P3（可选）：GraphSAGE 消息传递消融（MLP vs 真实边 vs 打乱边）尚未开始。

**§7.5 p/q 敏感性检验已用平衡子集重采样评估重新修正**：改用与 §7.3 一致的公平评估条件后，BFS-like 游走（p=0.25, q=4）的平衡子集 ARI（0.199）显著高于 Week 5–10 全程使用的无偏基线（0.116），推翻了此前"无偏基线已是最优选择"的结论。这一发现不影响已有下游实验的有效性（它们测量的是无偏基线 embedding 中真实存在的信号），但值得作为 Future Work 中的一个具体方向记录：完整的 p/q 网格搜索（而非仅两个极端点）+ 平衡子集评估，可能进一步提升 embedding 空间中的标签对齐强度。

---

## 11. 待办 / 下一步

**核心流程（Week 1–9）、全部稳健性检验、可选扩展（Week 10：balanced subset、GraphSAGE、account-type 混杂分析、P1/P2/P3 补充实验）已经全部完成**，§10.8/§10.9 的补充实验、§9.5 的 P3 消息传递消融均已并入正文。**实验阶段自本次更新起正式冻结**——GCN/GAT/RGCN 等新模型或新数据集不再纳入本项目范围：现有实验矩阵（图构建与 sanity check、EDA/特征工程、Node2Vec 聚类与 p/q 敏感性、三种监督分类器、手工特征消融、SHAP、contract/EOA 识别、EOA-only、contract-only、leave-one-protocol-out、重复 CV、GraphSAGE 三方对比含 P3 消融）已经足够支撑论文的核心论证，继续扩展实验带来的边际价值低于写作时间的机会成本。冻结后的最终结论见 **§12**。剩余工作是 Project Plan Week 11–14 的写作阶段：

1. Week 11（6–12 Aug）：Introduction、Background、Data and Graph Construction、Methodology（**须包含 n=5→重复 CV 的方法论教训**——单次 5 折 CV 曾未能检测到一个后来稳定复现的效应、也曾出现一次未能复现的边界性结果，见顶部"研究定位"），含 pipeline 图和数据集汇总表
2. Week 12（13–19 Aug）：Experiments and Results，生成正式图表（confusion matrix、SHAP plot、embedding 可视化、重复 CV 置信区间图、P3 消息传递消融图等，本文档 §3–10、§9.5 已提供全部素材和结论文字）
3. Week 13（20–26 Aug）：Discussion、Limitations（含 §9.4/§9.5 采样拓扑与消息传递边界、§10.7 account-type 混杂边界、§10.8b MEV 实体多样性不足边界、§10.2 eth_getCode 查询区块记录局限）、Ethics、Future Work（含 §7.5 的 p/q 网格搜索方向）、Conclusion；完成附录；第一版全文审阅与修改——**审阅时须逐句核对 Abstract/Introduction/Conclusion 中涉及"图信息增量价值"的表述，确保采用 §5.3/§9.4/§9.5 修正后的三分法结论，而非早期"三方法收敛无效"的表述**，可直接参照 §12 的冻结总表
4. Week 14（27 Aug–4 Sep）：终稿修订、校对、格式检查、提交缓冲

---

## 12. 冻结总表：研究问题 — 实验 — 最终证据 — 限制

**实验阶段已冻结（见 §11）。本节是写作阶段引用的唯一权威结论来源**——Discussion/Conclusion/Abstract 中涉及 RQ1/RQ2/账户类型混杂/聚类结构的表述，都应以本表为准，而不是引用早期（Week 7–9）版本的措辞。

### RQ1：哪些特征能够区分类别？

| | 内容 |
|---|---|
| **实验** | Week 9 特征分组消融（§6）+ Week 10 重复 CV 复核（§6.3）+ EOA-only 消融（§10.8a）+ contract-only 消融（§10.8b）+ SHAP（§6.2） |
| **最终证据** | 交易行为特征是完整样本和 EOA-only 任务中的主要预测来源（macro-F1 领先 graph_structural 8–21 个百分点，SHAP mass 占比 84.6% vs 15.4%）。手工结构特征单独具有一定判别力，其绝对表现具有任务依赖性——在 contract-only DeFi–MEV 任务中尤其强（RF/XGB 单独达到 0.909/0.920，接近 transaction_behaviour），但在已有交易行为特征时没有产生稳定正向增量，在 Random Forest 中还导致约 0.014 的小幅、统计显著的性能下降（重复 CV，10/10 方向一致） |
| **限制** | 这一模式可追溯到 Top-100 counterparty 抽样导致的 star-like、低三角密度拓扑（§3.3/§6.3 点 4），不能推广为"图拓扑信息普遍无用"这一更强论断；graph_structural 在 contract-only 任务中的相对强势提示判别力具有任务依赖性，机制假设（DeFi 池子合约是连接枢纽 vs MEV bot 交互模式更窄）尚未经 SHAP/单特征效应量验证 |

### RQ2：图表示是否提供增量价值？

| | 内容 |
|---|---|
| **实验** | Node2Vec embedding 消融（§5，重复 CV 修正版 §5.2/§5.3）+ EOA-only/contract-only 场景对比（§10.8a/§10.8b）+ GraphSAGE 三方对比（§9.3/§9.4）+ P3 消息传递消融（§9.5）+ p/q 敏感性（§7.5，探索性） |
| **最终证据** | Node2Vec embedding 在完整三分类任务中对 LR、RF、XGBoost 均提供正向增量（+0.068/+0.016/+0.035，10 个 repeat-level 配对估计，置信区间均不含零，方向 100% 一致），但增益大小具有明显模型依赖性，且在 EOA-only CEX–MEV 任务中未能复现（均值差降至 +0.004，p=0.49）。GraphSAGE 相对静态结构特征基线（graph_structural）没有显示出优势（p=0.24–0.32）；**P3 消融进一步拆分了这一结果的成因**：GraphSAGE 真实边显著优于度保持重连边（+0.120，p=0.007，5/5 训练种子方向一致）——排除了"消息传递没有跑起来"这一种解释，说明真实拓扑本身携带信息；但真实边相对纯节点特征 MLP 只有小幅、未达统计显著的优势（+0.023，p=0.088，4/5 训练种子方向一致）——说明消息传递能提供的、节点特征之外的增量信息有限，这个采样图上大部分可用结构信号已经被单节点的静态结构统计量吸收 |
| **限制** | GraphSAGE 全部结论的作用域限定在这个 star-like 采样拓扑下（§9.4 边界 1），且未做系统性超参数网格搜索（P3 固定了容量与超参数，只改变消息传递对象）；p/q 敏感性发现 BFS-like 游走在平衡子集评估下 ARI 高于无偏基线，但目前只是单一训练实例的探索性发现，多训练种子稳健性复核见 §7.5 更新（进行中/见下） |

### 账户类型混杂

| | 内容 |
|---|---|
| **实验** | 链上 `eth_getCode` 查询（§10.2，含可复现性复核）+ contract ratio 分析（§10.3）+ EOA-only（§10.6/§10.8a）+ contract-only（§10.8b）+ leave-one-DeFi-protocol-out（§10.8b） |
| **最终证据** | 所有 DeFi 种子均为合约地址（150/150，436 活跃样本中 86/86），因而 **DeFi 身份完全嵌套于合约身份之内**，完整三分类本身无法充分分离 DeFi 特定行为和一般合约特征。Contract-only 分析表明，DeFi 与 Malicious MEV 合约仍高度可分（50 次平衡重采样运行稳定复现），并能泛化到监督训练中被留出的 DeFi 协议（leave-one-DeFi-protocol-out：留出占 86% 合约样本 74% 的 Uniswap 后，macro-F1 仍达 0.99–1.00） |
| **限制** | CEX 合约样本仅 2 个，DeFi–CEX 的同类型对比不可行，账户类型识别问题在完整三分类范围内是**部分而非完全**解决；约 97% 的 Malicious MEV 合约共享同一 sub_label（"Arbitrage Trader"），无法实施对称的 leave-one-MEV-group-out 检验，故泛化证据目前只在 DeFi 一侧成立；EOA-only 的 CEX vs MEV 可分性（macro-F1 0.92–0.93）只排除了账户类型这一个混杂因素，不构成"这是纯粹经济行为信号"的完整证明 |

### 聚类

| | 内容 |
|---|---|
| **实验** | Node2Vec 基础聚类（§4）+ 平衡子集复核（§7.3）+ 重复平衡 ARI 稳定性检验（§7.3）+ p/q 敏感性（§7.5） |
| **最终证据** | Node2Vec embedding 在类别平衡评价中表现出稳定的中度标签对齐（ARI≈0.22–0.31，KMeans 与 Louvain 独立验证，50 次重复平衡抽样 100% 显著）。完整不平衡样本（299/86/51）上的低 ARI（≈0.02，不显著）是**本数据集和当前聚类划分下的评价现象**，源于 pairwise 指标被多数类点对主导，**不能被解释为 embedding 完全没有结构** |
| **限制** | 这一"全样本评估低估信号"的机制解释目前只在本数据集验证，不构成对 ARI/NMI 类指标在不均衡数据下普遍表现的一般性证明（§4.4/§7.3 已明确措辞边界）；p/q 敏感性方向的探索性发现（BFS-like 设置平衡 ARI 高于无偏基线）尚待多训练种子验证，见 §7.5 |

---
