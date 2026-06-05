# HEA-IceAgent v2.1

## 高熵合金冰成核调控筛选智能体

---

### 一句话概述

自动筛选能**促进冰成核**（类碘化银）、**抑制冰成核**、**抑制冰重结晶**的高熵合金，
输出可供 DFT 计算或实验验证的候选材料排名表。

---

### 科学依据

| 任务 | 物理机制 | 核心描述符 |
|---|---|---|
| **促进冰成核** (Task A) | 外延晶格匹配（类似 AgI 机制） | 2D 晶格失配度、形成能、六方倾向 |
| **抑制冰成核** (Task B) | 晶格失配 + 疏水表面 | 高失配度、低表面能、电子平滑度 |
| **抑制冰重结晶** (Task C) | 化学异质性钉扎冰晶界（类似 AFP 抗冻蛋白） | SRO 异质性、配置熵、晶格畸变 |

**关键校准**：AgI(0001) vs Ice(0001) 的 2D 失配度 = **1.327%**，与文献值 1.3% 一致。

---

### 输入数据

| 文件 | 大小 | 内容 |
|---|---|---|
| `structure_ini_featurized.dat_all.csv` | 390 MB | 83,797 个结构的 273 个 Matminer 特征 + 形成能 |
| `structure_featurized.dat_all.csv` | 394 MB | 弛豫后结构的特征 |
| `hea.2023-04-06.csv` | 453 MB | DFT 弛豫数据（晶格体积、磁矩、电荷等） |
| `SROs_structure_ini.csv` | 43 MB | 短程有序参数 (SRO1-SRO4) |

- 元素空间：Al, Si, Cr, Mn, Fe, Co, Ni, Cu（8 元素）
- 结构：BCC/FCC，有序 + SQS，2-7 元合金
- 清洗后：**78,085 结构 → 218 种独特组成**

---

### 架构

```
E:/HEA-Agent/hea_ice_agent/
│
├── config.py              全局参数（冰 Ih 晶格常数、元素属性、评分权重）
├── loader.py              加载 4 个 CSV → 合并 SRO → 清洗 → 分组
├── features.py            20 个工程描述符（组成熵、SRO 梯度、疏水性等）
├── lattice_matching.py    ★ 核心：2D 晶格外延匹配（Zur 旋转覆盖层算法）
├── scoring.py             三任务独立评分（AgI 校准）
├── pipeline.py            主管道编排器
├── report.py              生成 CSV/JSON/HTML 报告 + matplotlib 图
├── utils.py               工具函数
├── download.py            数据完整性校验
├── main.py                命令行入口
│
├── results/               ★ 输出目录
│   ├── candidates_promoter.csv    促进冰成核候选 (Top 50)
│   ├── candidates_inhibitor.csv   抑制冰成核候选 (Top 50)
│   ├── candidates_iri.csv         抑制冰重结晶候选 (Top 50)
│   ├── candidates_all_tasks.csv   全部 218 种组成的排名
│   ├── summary.json               JSON 格式汇总
│   ├── report.html                HTML 综合报告
│   ├── provenance.json            数据溯源（SHA256 哈希 + 参数快照）
│   └── figures/                   可视化图表
│
└── venv/                  Python 3.14 虚拟环境
```

**总代码量**：2,932 行 Python，13 个模块。

---

### 核心算法：2D 晶格外延匹配

```
1. 从 hea.2023-04-06.csv 读取每个组成的 DFT 晶格常数 a_dft
   （volume_per_atom × NIONS）^(1/3)

2. 对每个 (HEA 组成, 晶格类型)：
   a. 构建 3 个 HEA 低指数面：BCC(110,100,111) 或 FCC(111,100,110)
   b. 构建 3 个冰 Ih 晶面：Basal(0001), Prism(10-10), Pyramidal(11-22)
   c. 对每个 (HEA 面, 冰面) 组合：
      - 对冰晶格旋转 0-180°（步长 1°）
      - 搜索整数超胞矩阵 S_sub(n×n) 和 S_ice(n×n)
      - 计算 2D 应变张量：ε = S_ice⁻¹·S_sub − I
      - 失配度 = ‖ε‖_F / √2  （归一化为各向同性等效百分比）

3. 取所有 9 个面组合中的最优值作为该组成的失配度
```

**验证**：AgI 基准测试通过——AgI(0001) vs Ice(0001) = 1.327%。

---

### 运行方式

```bash
# 激活环境
E:/HEA-Agent/venv/Scripts/python.exe

# 完整流水线
python -c "
import sys; sys.path.insert(0,'E:/HEA-Agent')
from hea_ice_agent.pipeline import HEAIcePipeline
HEAIcePipeline(n_jobs=4).run_full()
"

# 仅验证 AgI 基准
python -c "
import sys; sys.path.insert(0,'E:/HEA-Agent')
from hea_ice_agent.lattice_matching import test_agi_benchmark
test_agi_benchmark()
"
```

**运行时间**：~14 分钟（4 CPU 核心），主要耗时在 Phase 3 的 2D 超胞搜索。

---

### 最终结果摘要

#### Task A: 促进冰成核 Top 5

| # | 组成 | 晶格 | 2D 失配度 | 最佳冰面 | 评分 |
|---|---|---|---|---|---|
| 1 | **Co-Fe-Ni-Si** | BCC | 1.32% | basal/100 | 99.8 |
| 2 | Al-Cu-Fe-Mn-Si | FCC | 1.33% | prism/110 | 99.7 |
| 3 | Al-Cr-Ni | BCC | 1.33% | prism/110 | 99.5 |
| 4 | Al-Cr-Cu-Ni-Si | BCC | 1.32% | prism/110 | 99.5 |
| 5 | Al-Cr-Cu-Ni | BCC | 1.31% | basal/111 | 99.4 |

#### Task B: 抑制冰成核 Top 5

| # | 组成 | 晶格 | 2D 失配度 | 评分 |
|---|---|---|---|---|
| 1 | **Al-Cr-Mn** | BCC | 1.04% | 99.9 |
| 2 | Al-Cr-Fe-Mn | BCC | 1.72% | 99.9 |
| 3 | Al-Co-Cr-Mn | BCC | 1.71% | 99.7 |
| 4 | Al-Co-Fe-Ni | BCC | 1.95% | 98.8 |
| 5 | Al-Co-Cr-Fe | BCC | 1.87% | 98.7 |

#### Task C: 抑制冰重结晶 (IRI) Top 5

| # | 组成 | 晶格 | SRO 异质性 | 配置熵 | 评分 |
|---|---|---|---|---|---|
| 1 | **Al-Co-Cr-Fe-Mn-Ni-Si** | BCC | 0.214 | 1.946 | 96.0 |
| 2 | Al-Mn-Ni-Si | BCC | 0.412 | 1.386 | 95.7 |
| 3 | Al-Cr-Mn-Si | BCC | 0.380 | 1.386 | 93.9 |
| 4 | Al-Co-Cr-Si | BCC | 0.333 | 1.386 | 93.8 |
| 5 | Al-Co-Fe-Mn-Si | BCC | 0.627 | 1.609 | 93.7 |

#### 全能候选（三任务总分最高）

| # | 组成 | 促进 | 抑制 | IRI | 总分 |
|---|---|---|---|---|---|
| 1 | Al-Co-Cr-Fe-Mn-Ni | 0.651 | 0.581 | 0.506 | 1.738 |
| 2 | Al-Cu-Fe-Mn-Si | 0.692 | 0.575 | 0.430 | 1.698 |
| 3 | Al-Cu-Fe-Ni | 0.652 | 0.490 | 0.530 | 1.672 |

---

### 数据真实性保障

- **输入文件 SHA256** 记录在 `provenance.json` 中
- 晶格常数来自 **DFT 数据库**（`hea.2023-04-06.csv` 的 `volume_per_atom` 列），**不是 Vegard 定律估算**
- 每次运行记录时间戳、参数快照、数据统计
- 2D 匹配用**完整 2×2 超胞矩阵**，不是 1D 矢量简化

---

### 已知局限（诚实声明）

1. **组成级平均**：218 种组成使用平均 DFT 晶格常数，同一组成内不同结构（有序 vs SQS、不同尺寸）的晶格参数差异未被捕获
2. **评分未经验证**：权重是物理启发的，没有用实验数据训练或校准
3. **缺失物理**：表面终止面化学（哪个元素暴露在最外层）、氢键能力、冰黏附能未建模
4. **理想表面**：仅考虑 BCC/FCC 低指数理想表面，未考虑重构、台阶、缺陷
5. **元素空间有限**：仅覆盖 8 种元素，许多有前景的元素（如 Ti、V、Mo、W 等）不在数据库中
6. **IRI 机制**：基于与抗冻蛋白的类比推理，未经实验验证

**所有结果为物理学引导的假设。顶级候选需要通过 DFT 表面计算和/或冰成核实验进行验证。**

---

### 后续建议

1. 对 Top-5 候选进行 DFT 表面能计算（slab 模型）
2. 计算冰分子在候选表面上的吸附能
3. 分子动力学模拟冰成核过程（对排名最高的 2-3 个候选）
4. 根据 DFT 结果校准评分权重
5. 扩展数据库到更多元素体系
