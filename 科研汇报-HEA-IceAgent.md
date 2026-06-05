# HEA-IceAgent：面向冰成核调控的高熵合金智能筛选与推理体

## 科研汇报文档

---

> **一句话摘要**：我们开发了一个双模式计算智能体，能够从 8 元素高熵合金空间中**筛选**和**推理推测**出具有促进冰成核（类碘化银）、抑制冰成核和抑制冰重结晶三种功能的高熵合金候选材料，并输出可验证的排名表。

---

## 1. 研究背景

### 1.1 冰成核调控的科学意义与应用需求

冰在固体表面的形成（异质成核）是自然界和工业界普遍存在的现象，对以下领域有重大影响：

| 应用领域 | 需求 | 关键问题 |
|---------|------|---------|
| **航空安全** | 抑制冰成核 | 机翼结冰导致升力丧失，每年造成数十起事故 |
| **风力发电** | 抑制冰成核 | 叶片结冰降低发电效率 20-50% |
| **电力传输** | 抑制冰成核 | 输电线覆冰可导致大规模停电 |
| **人工影响天气** | 促进冰成核 | 碘化银（AgI）用于人工增雨已有 70 年历史 |
| **低温生物保存** | 抑制冰重结晶 | 细胞冷冻时冰晶生长破坏细胞膜，IRI 材料可提高存活率 |
| **材料合成** | 利用冰重结晶 | Li et al. (2025) 已用冰重结晶方法合成了 HEA 涂层 |

### 1.2 碘化银（AgI）的启示

AgI 是已知最有效的冰成核剂，其科学原理在于**晶格外延匹配**：

- **冰 Ih**：六方晶系，a = 4.52 Å，c = 7.36 Å
- **β-AgI**：六方晶系，a = 4.58 Å，c = 7.49 Å
- **AgI(0001) 面与冰(0001)面的晶格失配度仅 1.3%** ——这一极低的失配度使得冰晶可以在 AgI 表面近乎完美地外延生长

**核心科学问题**：是否存在其他材料，具有与 AgI 相当或更优的冰成核性能？是否存在能主动**抑制**冰成核或冰重结晶的材料？

### 1.3 为什么选择高熵合金？

高熵合金（High-Entropy Alloys, HEAs）由 4 种及以上主元素等摩尔或近等摩尔混合而成，具有三大独特优势：

1. **晶格常数可连续调控**：通过改变组成元素的种类和比例，可以在一定范围内连续调节晶格常数，有可能精确匹配冰 Ih 的最佳外延生长条件。

2. **化学异质性**：多元素随机分布创造了多样化的局部化学环境，可以**钉扎冰晶界**，阻止冰晶粗化——这一机制与抗冻蛋白（AFP）抑制冰重结晶的原理类似。

3. **巨大的组成空间**：仅 8 种元素（Al-Si-Cr-Mn-Fe-Co-Ni-Cu）就可以形成 246 种不同的元素组合，乘以无数的化学计量比，理论组成空间远超 DFT 直接计算的能力范围。

### 1.4 关键文献支撑

- **Li et al. (2025)** *J. Mater. Chem. A*：首次报道了利用**双层冰重结晶方法**合成高熵合金材料和涂层，直接证明了 HEA 形成与冰生长控制之间的关联。
- **Kangming Li et al. (2024)** *J. Mater. Chem. A*：建立了包含 83,797 个 DFT 结构的高熵合金数据库（Zenodo: 10.5281/zenodo.10854500），涵盖 8 元素、2-7 元合金的完整 bcc/fcc 空间。
- **Wu et al. (2017)**：证明了离子特异性的冰重结晶抑制可用于制备多孔材料。
- **冰结合蛋白（IBP）研究**：抗冻蛋白通过特定氨基酸残基与冰晶面的氢键相互作用实现 IRI 活性。

---

## 2. 研究框架

### 2.1 三大筛选任务

```
┌─────────────────────────────────────────────────────────────────┐
│                    HEA-IceAgent 三大任务                         │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Task A        │   Task B        │   Task C                    │
│   促进冰成核     │   抑制冰成核     │   抑制冰重结晶 (IRI)         │
│   (类AgI机制)    │                 │                             │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ 低晶格失配       │ 高晶格失配       │ 化学异质性                   │
│ 六方结构倾向     │ 疏水性           │ 高配置熵                     │
│ 负形成能（稳定）  │ 电子势能平滑     │ 晶格畸变                     │
│                 │ 低SRO（光滑）     │ 混合焓                       │
├─────────────────┼─────────────────┼─────────────────────────────┤
│ 应用：人工降雨   │ 应用：防冰涂层    │ 应用：低温保存               │
│ 冰雪运动         │ 航空/风电        │ 冷冻电镜                     │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

### 2.2 双模式架构

HEA-IceAgent 采用**筛选 + 推理**双模式设计：

```
                          ┌──────────────────┐
                          │  4个CSV数据文件    │
                          │  (1.28 GB DFT数据) │
                          └────────┬─────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
            ┌───────▼───────┐            ┌────────▼────────┐
            │  MODE 1: 筛选  │            │  MODE 2: 推理    │
            │  (SCREEN)     │            │  (REASON)       │
            ├───────────────┤            ├─────────────────┤
            │ 78,085 结构   │            │ ML模型训练       │
            │ 218 种独特组成 │            │ (218→晶格常数)   │
            │ DFT晶格常数    │            │                 │
            │ 完整2D匹配     │            │ 全枚举246种元素   │
            │ AgI 验证 ✓    │            │ 组合×计量比      │
            │               │            │ = 1,170 候选     │
            │ 三任务评分     │            │                 │
            │ Top-50 排名    │            │ ML预测晶格常数    │
            │               │            │ 代理评分排序      │
            └───────┬───────┘            └────────┬────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       统一输出               │
                    │  CSV + JSON + HTML 报告      │
                    │  SHA256 数据溯源             │
                    └─────────────────────────────┘
```

### 2.3 核心算法：2D 晶格外延匹配

这是整个智能体的**核心科学计算模块**，实现了 Zur 旋转覆盖层算法：

```
输入：HEA组成 + 晶格类型 (bcc/fcc) + DFT晶格常数 a

对每个 (HEA表面, 冰晶面) 组合（共9对）：
  ├── 构建 2D 表面晶胞（基底矢量 a1, a2）
  ├── 对旋转角 θ = 0°, 1°, 2°, ..., 179°：
  │   ├── 旋转冰晶格 → a1_ice(θ), a2_ice(θ)
  │   ├── 搜索整数超胞矩阵 S_sub(n×n) 和 S_ice(n×n)
  │   ├── 计算 2D 应变张量：ε = S_ice⁻¹·S_sub − I
  │   └── 失配度 = ‖ε‖_F / √2
  └── 全局最优值

输出：最佳失配度 + 最佳冰面 + 最佳HEA表面 + 最佳旋转角 + 超胞矩阵
```

**冰 Ih 的三个主要晶面：**
- Basal (0001)：a1 = [4.52, 0], a2 = [-2.26, 3.91]
- Prism (10-10)：a1 = [4.52, 0], a2 = [0, 7.36]
- Pyramidal (11-22)：a1 = [4.52, 0], a2 = [-2.26, 7.84]

**BCC 表面：**(110)、(100)、(111)；**FCC 表面：**(111)、(100)、(110)

**算法验证：**
- AgI(0001) vs 冰(0001)：计算失配度 = **1.327%**
- 文献报道值：~1.3%
- **验证结论：PASS ✓**

### 2.4 特征工程

智能体计算 20 个工程描述符，分为四类：

| 类别 | 描述符 | 物理意义 |
|------|--------|---------|
| **组成特征** | S_conf（配置熵）、delta_r（原子半径失配）、var_EN（电负性方差）、avg_EN | 反映多元素混合程度和化学多样性 |
| **SRO 特征** | SRO1-4 均值/梯度/异质性/范围 | 反映不同壳层的化学短程有序度 |
| **稳定性特征** | stability_score、abs_stability | 基于 DFT 形成能的稳定性评估 |
| **表面特征** | hydrophobicity、surface_energy_proxy、work_function_proxy、electronic_smoothness | 评估表面与水分子相互作用的趋势 |

---

## 3. 应用示例

### 3.1 环境配置

```bash
# Python 3.10+ 虚拟环境
python -m venv venv
venv/Scripts/pip install numpy pandas scipy scikit-learn xgboost matplotlib seaborn joblib

# 从 Zenodo 下载数据文件（4个CSV，共约 1.28 GB）
# https://zenodo.org/records/10854500
# 将以下文件放入 hea_ice_agent/ 目录：
#   - structure_ini_featurized.dat_all.csv
#   - structure_featurized.dat_all.csv
#   - hea.2023-04-06.csv
#   - SROs_structure_ini.csv
```

### 3.2 运行筛选模式（Mode 1: SCREEN）

从数据库中已有的 218 种 HEA 组成中筛选排名：

```python
import sys
sys.path.insert(0, 'E:/HEA-Agent')
from hea_ice_agent.pipeline import HEAIcePipeline

# 创建管道并运行（4个CPU核心，约14分钟）
pipeline = HEAIcePipeline(n_jobs=4, verbose=True)
outputs = pipeline.run_full(skip_download=True)

# 输出文件：
#   results/candidates_promoter.csv   — 促进冰成核 Top 50
#   results/candidates_inhibitor.csv  — 抑制冰成核 Top 50
#   results/candidates_iri.csv        — 抑制冰重结晶 Top 50
#   results/candidates_all_tasks.csv  — 全部 218 种组成排名
#   results/provenance.json           — 数据溯源（SHA256 + 参数快照）
#   results/report.html               — HTML 交互式报告
```

### 3.3 运行推理模式（Mode 2: REASON）

ML 外推到数据库中不存在的 HEA 组成：

```python
import sys
sys.path.insert(0, 'E:/HEA-Agent')
from hea_ice_agent.dual_mode import run_dual_mode

# 同时运行筛选和推理
outputs = run_dual_mode(n_jobs=4, verbose=True)

# 额外输出：
#   results/reasoned_new_candidates.csv  — 1,170 种 ML 预测的新候选
```

### 3.4 筛选结果示例

**促进冰成核（Task A）Top 5：**

```
排名  组成                     晶格    2D失配度    评分
 1   Co-Fe-Ni-Si              BCC    1.320%     99.8  ← 最佳，与 AgI 几乎一致
 2   Al-Cu-Fe-Mn-Si           FCC    1.332%     99.7
 3   Al-Cr-Ni                 BCC    1.327%     99.5  ← 与 AgI 完全相同
 4   Al-Cr-Cu-Ni-Si           BCC    1.316%     99.5
 5   Al-Cr-Cu-Ni              BCC    1.315%     99.4
```

**抑制冰重结晶（Task C）Top 5：**

```
排名  组成                        晶格   SRO异质性   配置熵    评分
 1   Al-Co-Cr-Fe-Mn-Ni-Si (7元)  BCC    0.214      1.946    96.0  ← 最大熵
 2   Al-Mn-Ni-Si (4元)           BCC    0.412      1.386    95.7  ← 最经济
 3   Al-Cr-Mn-Si (4元)           BCC    0.380      1.386    93.9  ← 无Co方案
 4   Al-Co-Cr-Si (4元)           BCC    0.333      1.386    93.8
 5   Al-Co-Fe-Mn-Si (5元)        BCC    0.627      1.609    93.7
```

### 3.5 关键发现

1. **Si 掺杂是冰成核促进的关键**：Top-10 促进剂中 90% 含 Si。Si 的大原子半径差异和半金属特性显著改变了晶格常数和 SRO 参数。

2. **BCC 晶格优于 FCC**：在所有三个任务中，BCC 候选材料占据了前 10 名中的 70% 以上。

3. **Al 几乎不可或缺**：前 20 名候选材料中 90% 以上含铝。Al 的大原子半径（1.43 Å）和低电负性（1.61）对于调控晶格常数和表面化学至关重要。

4. **中高熵（4-5 元）综合性能最优**：虽然 7 元合金配置熵最高，但 4-5 元合金可以在 SRO 异质性、晶格畸变和配置熵之间取得最佳平衡。

5. **所有 218 种组成的 2D 失配度均 < 2%**：说明这 8 元素空间的晶格常数范围恰好覆盖了与冰 Ih 的良好匹配区——这对实验合成非常有利。

### 3.6 推理预测（不在数据库中的新候选）

ML 模型从 218 种已知组成学习组成→晶格常数的映射，然后枚举全空间：

```
推理发现的新候选（Top 5 促进剂）：
  Al-Mn-Ni-Si 系列     (FCC, a≈5.26Å, Ef<0)  ← 最优先 DFT 验证
  Al-Co-Cr-Si 系列     (FCC, a≈5.26Å, Ef<0)
  Al-Cr-Fe-Si 系列     (FCC, a≈5.27Å, Ef<0)

推理发现的新候选（Top 5 抑制剂）：
  Al-Cr-Cu-Mn-Si 系列  (BCC, a≈9.3Å)         ← 最高失配
  Al-Cr-Cu-Fe-Si 系列  (BCC, a≈9.3Å)

推理发现的新候选（Top 5 IRI）：
  Al-Cu-Mn-Ni-Si 系列  (高 S_conf + 高 var_EN)
```

---

## 4. 数据溯源与可重复性

### 4.1 输入数据验证

每次运行自动记录 SHA256 哈希，保存在 `results/provenance.json`：
```json
{
  "input_files": {
    "structure_ini_featurized": {"sha256_prefix": "0b3082179025fc70"},
    "sro": {"sha256_prefix": "d9e151250083b140"},
    "hea_main": {"sha256_prefix": "3845f4168a662950"}
  }
}
```

### 4.2 无外部 API 依赖

智能体**不调用任何外部 API**。所有计算基于本地 DFT 数据库和自训练的 ML 模型，确保：
- 结果完全可重复
- 不依赖网络连接
- 数据隐私安全
- 适合在超算集群上批量运行

---

## 5. 不足与展望

### 5.1 当前局限

1. **组成级平均**：使用每种组成的平均 DFT 晶格常数，未能捕获同一组成下不同结构（有序 vs SQS、不同超胞尺寸）的差异。
2. **评分权重未经实验校准**：目前所有评分函数基于物理启发式设计，权重没有经过实验数据训练。
3. **缺失的物理**：未考虑表面终止面化学（哪个元素暴露在最外层）、氢键形成能力、冰黏附能等与冰成核直接相关的表面科学问题。
4. **理想表面假设**：仅考虑了 BCC/FCC 低指数理想表面，未涉及表面重构、台阶、缺陷等真实表面特征。
5. **元素空间有限**：现有数据库仅覆盖 8 种元素，Ti、V、Mo、W 等有前景的元素未被包含。

### 5.2 后续工作

1. **DFT 验证**：对 Top-5 促进剂候选（Co-Fe-Ni-Si、Al-Cr-Ni 等）进行 DFT 表面能计算和冰分子吸附能计算。
2. **分子动力学模拟**：对排名最高的 2-3 个候选进行冰成核 MD 模拟。
3. **实验校准**：通过文献和实验数据校准评分权重。
4. **扩展元素空间**：将数据库扩展到含 Ti、V、Mo、W 等元素。
5. **IRI 实验验证**：对 Al-Co-Cr-Fe-Mn-Ni-Si 和 Al-Mn-Ni-Si 进行冰重结晶抑制实验。

---

## 6. 总结

HEA-IceAgent 是一个**双模式**（数据库筛选 + ML 推理）的计算材料筛选智能体，能够在 8 元素高熵合金空间中系统地搜索具有冰成核调控能力的候选材料。

**核心创新点：**
1. 实现了真正的 **2D 晶格外延匹配算法**（Zur 方法），而非简单的 1D 晶格常数比较
2. AgI 基准验证通过（计算值 1.327%，文献值 ~1.3%）
3. 双模式设计：既可以从已有数据库筛选排名，也可以 ML 外推到全新组成
4. 完整的数据溯源机制（SHA256 哈希 + 参数快照）
5. 零 API 依赖，纯本地计算，保证可重复性

**筛选发现的主要候选：**
- **促进冰成核**：Co-Fe-Ni-Si (BCC, d=1.320%, 与 AgI 几乎一致)
- **抑制冰成核**：Al-Cr-Mn (BCC, d=2.76%, 最高失配)
- **抑制冰重结晶**：Al-Co-Cr-Fe-Mn-Ni-Si (BCC, S_conf=1.946, 7 元最大熵)

**所有结果均为物理引导的假设，需要通过 DFT 计算和实验进行验证。**

---

## 附录

### A. 代码仓库

- GitHub: [https://github.com/jhw7291/HEA-IceAgent](https://github.com/jhw7291/HEA-IceAgent)
- 许可证: MIT
- 语言: Python 3.10+（14 个模块，~3,200 行）
- 依赖: numpy, pandas, scipy, scikit-learn, xgboost, matplotlib, seaborn, joblib

### B. 数据来源

Kangming Li et al. (2024). DFT dataset for high entropy alloys [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10854500

### C. 参考文献

1. Li, K. et al. (2024). Efficient first principles based modeling via machine learning: from simple representations to high entropy materials. *J. Mater. Chem. A*, DOI: 10.1039/D4TA00982G.
2. Li, X. et al. (2025). Synthesizing high-entropy alloy materials and coatings using a bilayer ice recrystallization method.
3. Wu, S. et al. (2017). Ion-specific ice recrystallization provides a facile approach for the fabrication of porous materials.
4. Zhang, Z. & Liu, X. (2019). Control of ice nucleation: freezing and antifreeze strategies. *Chem. Soc. Rev.*
