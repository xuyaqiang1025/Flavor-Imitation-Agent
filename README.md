# 仿香 Agent (Flavor Imitation Agent) v2.0

> **电子烟香精配方逆向工程与自动化辅助设计系统**  
> *基于 GC-MS 图谱解析、化学指纹识别与 LLM 智能推理*

---

## 📖 项目简介

本系统是专为香精香料工程师打造的高阶 AI 助手，旨在解决配方逆向工程中的核心挑战：**从复杂的 GC-MS 图谱中精准剥离天然提取物与合成单体**。

通过集成 **ChromaDB 向量数据库** 与 **多模态 LLM 推理**，本系统实现了从“数据清洗”到“解卷积分析”再到“感官反馈微调”的全链路自动化。

---

## 🧠 核心原理 (Core Hybrid Engine)

系统采用 **Hybrid Deconvolution (混合解卷积)** 架构，将严谨的数学定量与灵活的 AI 定性推理相结合：

### 1. 虚拟反应器溯源 (Reactor Traceback)
自动识别 e-liquid 存放过程中产生的副产物：
*   **识别**：通过 LLM 化学 Agent 识别缩醛（Acetal）、酯化等反应产物。
*   **溯源**：基于分子量比例关系（如 152:226），将副产物的质量自动还原至父本物质（如香兰素），确保配方还原度。

### 2. CAS 稀疏向量矩阵匹配 (Vector Similarity)
*   **编码**：将 GC-MS 图谱数字化为基于 CAS 号的高维稀疏向量。
*   **匹配**：利用余弦相似度（Cosine Similarity）在本地知识库中检索潜在天然提取物（如柠檬油、烟草浸膏）。支持变长采样频率，无论精油含 5 个还是 150 个组分均可精准匹配。

### 3. LLM 定性证据链确认 (Qualitative Qualification)
*   **验证**：LLM 扫描图谱中的 **Marker Molecules (标志物)**（如：检出瓦伦烯和辛醛是确诊甜橙油的关键证据）。
*   **判定**：只有当定性证据链闭环时，系统才会将其标记为 `Confirmed`，有效避免误判。

### 4. 迭代递减定量计算 (Quantitative Subtraction)
*   **计算**：采用中位数反算法确定各天然提取物的添加量。
*   **剥离**：从总图谱中递归扣除已录入的天然组分占比。
*   **残差判定**：扣除后剩余的“溢出”浓度，被自动归类为**人工添加的合成单体补强**。

---

## 🛠 功能模块详解

### 📁 1. 数据清洗与还原 (`src/ingestion/`)
*   `parser.py`: 兼容多平台（Agilent/Shimadzu）的图谱解析。
*   `cleaner.py`: 自动滤除溶剂、尼古丁及柱流失杂质。
*   `reactor_agent.py`: [NEW] 基于 LLM 的反应副产物分析与溯源引擎。

### ⚙️ 2. 混合计算引擎 (`src/engine/`)
*   `hybrid_deconvoluter.py`: 核心解卷积管线，执行 Traceback -> CAS Match -> LLM Qualify -> Math Subtraction 的四阶段分析。

### 📚 3. 多源知识库 (`src/knowledge/`)
*   `vector_db_builder.py`: [UPGRADE] 支持从 **CSV / Excel / Markdown** 批量导入自定义精油指纹，管理本地 **ChromaDB** 向量存储。
*   `inventory_manager.py`: 将化学分子高效映射至您的真实采购仓库 (Real Inventory)。

### ⚖️ 4. 感官与合规校验 (`src/logic/`)
*   `sensory_validator.py`: 检查组分的沸点、LogP 及稳定性，预警雾化积碳与合规风险。
*   `llm_agent.py`: 闭环感官反馈（Sensory Loop），理解“厚实感”、“击喉感”等描述并给出化学建议。
### ⚖️ 4. 逻辑验证 (Logic)
*   `sensory_validator.py`: **雾化一致性检查**。
    *   检查物理属性（沸点、LogP）。
    *   **示例警告**: "警告：*棕榈酸乙酯* 沸点 >300°C，在陶瓷芯上可能由于温度不足无法完全雾化，导致积碳或风味缺失。"

---

## 🚀 操作手册与使用指南 (Operational Manual)

本系统结合了**计算化学（迭代减法）**与**大语言模型 RAG 推理**，专门用于深度解析 GC-MS 电子烟油及香精图谱，实现天然提取物与合成单体的高精度剥离。

### 📍 一、 核心工作流 (Standard Workflow)

1. **环境准备与启动**
   * 确保 Python 环境已安装最新依赖：`pip install pandas streamlit plotly chromadb openai openpyxl`
   * 在项目根目录终端执行命令启动仪表盘：`streamlit run app.py`

2. **配置 AI 智能引擎 (必加项)**
   * 在界面左侧的【AI 智能配置】栏，一键切换各大大模型服务商。
   * **已集成服务商**：`DeepSeek (推荐)`、`OpenAI`、`Anthropic (Claude)`、`Google (Gemini)`、`阿里云 (Qwen)`、`硅基流动 (SiliconFlow)`。
   * 切换服务商后，系统会自动预填对应的 **Base URL**，您只需填入 🔑 **API Key**。
   * *注意：不填写 API Key 系统将降级为纯数学减法，失去动态发现未知副产物和高阶 RAG 定性推理的能力。*

3. **上传色谱数据**
   * 上传通过安捷伦/岛津仪器导出的 GC-MS CSV 汇总表。建议包含：`cas`、`concentration_mg_kg`（或面积/含量列）、`compound_name_cn/en`。
   * 引擎会自动开启 **虚拟反应器 (Virtual Reactor)**，在后台按分子量原理进行亲本还原。

4. **查看与导出结果**
   * **仪表盘看板**：采用现代 Glassmorphism 设计，快速洞察原始峰数量、有效原料数及解析置信度。
   * **导出配方**：点击主界面的 **【📥 下载 CSV 配方】** 或 **【📄 下载 Excel 配方】**，一键保存分析成果。
   * **推理日志**：展开 **【🛠️ 查看 AI 解析推理日志】** 面板，查看分阶式（Step-by-Step）的逻辑演化过程。

---

### 🧬 二、 核心进阶：如何扩充“天然特征指纹库”

系统内置了部分基础精油数据，但**本系统越用越强的关键，在于持续喂给它您实验室的独家 GC-MS 数据或文献数据。**

#### 1. 准备您的数据模板
使用 Excel 打开项目目录下的模板文件：`data/inventory/my_extracts_template.csv`。

| 列名 (必须小写) | 填写说明 (非常重要) |
| :--- | :--- |
| `id` | 天然精油/浸膏的唯一英文标识，如：`orange_oil_brazil` |
| `name_cn` | 中文名称，将直接显示在最终的配方结果中，如：`甜橙精油 (巴西)` |
| `description` | 简要描述该精油的感官特征。 |
| **`markers`** | **最核心字段！这是告诉大模型用来“定性”的绝对凭证。** 请填写该精油**最独有的微量标志物**（如：橙油填 `Octanal, Decanal`，勿填大家都有的 `Limonene`）。 |
| **`composition_json`** | **数学定量矩阵。** 必须为合法 JSON 数组。**成分维度不限**（3个主成分或150个全成分均可），系统会自动降维算余弦相似度。 |

**`composition_json` 填写示例**：
`[{"cas":"5989-27-5", "name":"Limonene", "pct":90.0}, {"cas":"112-31-2", "name":"Decanal", "pct":1.5}]`

#### 2. 导入与更新向量库 (ChromaDB)
1. 在网页界面左侧边栏，找到并展开 **【📥 导入自定义指纹数据】**。
2. 支持 **CSV / Excel (.xlsx) / Markdown** 三种格式。
3. 拖拽文件后点击 **【🔄 导入并更新向量库】**。
4. 系统会将高维浓度分布转化为数学向量，永久写入本地 ChromaDB。断电不丢失。

---

### ⚠️ 三、常见排错 (Troubleshooting)

1. **配方加起来不到或超过 100%？**
   * 正常现象。系统采用绝对浓度 (mg/kg) 独立换算。如果原始 GC-MS 数据的总检出物质未完全覆盖，或天然提取物组分天然存在比例波动，都会导致总和不等于 100%。您可以在生成的配方表中手动干预归一化。
2. **某种天然精油没被识别出来（即使库里有）？**
   * 极大概率是检测图谱里遗漏了您在 CSV 中设定的 `markers` (标志物)。系统硬性规定：**大模型必须验证 Marker 存在，才能“确诊”该天然物。** 建议修改 CSV 中的 markers 设定（选择浓度更高或更普适的专属成分），然后重新导入。
3. **遇到 ImportError 模块冲突**
   * 若二次开发，请确保遵循“自下而上”的引用关系。避免核心计算引擎（如 `deconvoluter.py` 和 `rag_inference.py`）发生循环导入。

---

## 目录结构说明
```text
Flavor Imitation Agent/
├── data/
│   ├── raw_gcms/          # [输入] GC-MS 原始文件
│   └── inventory/         # [配置] 您的原料清单及精油导入模板
├── src/
│   ├── ingestion/         # 包括 Virtual Reactor 与清洗器
│   ├── engine/            # 包括 Hybrid Deconvoluter (四步解卷积混合引擎)
│   ├── knowledge/         # Inventory Manager 与 Vector DB Builder
│   └── logic/             # 验证与 LLM Sensory Agent
└── app.py                 # UI 界面入口 (Streamlit)
```
