# MatCreator / PFD_Agent：语义向量检索实验说明文档

## 1. 目标与术语对照

| 口头/文档用语 | 本仓库中的对应物 |
|----------------|------------------|
| `node_info.db`（如 `guidance.txt`） | 项目中的 **SQLite 信息库**：由环境变量 `INFO_DB_PATH` 指定，或通过 `db_tools.py` 的 `--info-db` 传入。仓库内**没有**硬编码固定文件名；常见命名为 `info.db` 等。 |
| domain dataset | `datasets.path` 指向的 **ASE `.db` 文件**（解压 `database/domain_datasets.tar.gz` 后的各域数据）。 |
| 当前「LLM 查表再选库」流程 | 见 `agents/MatCreator/skills/database/SKILL.md` 中 **Typical workflow for dataset search**（`validate-sql` → `query-info` → `query-compounds` → `export-entries`）。 |

核心信息库 **schema** 在 `db_tools.py` 的 `_ensure_schema` 中定义，与 `SKILL.md` 文档一致：`nodes`、`datasets`、`dataset_elements` 三张表。

---

## 2. 与本实验直接相关的文件说明

### 2.1 `agents/MatCreator/skills/database/db_tools.py`

- **作用**：数据库技能的唯一 CLI 入口；所有子命令打印 **JSON 到 stdout**。
- **环境变量**：`INFO_DB_PATH`（信息库）、`ASE_DB_PATH`（单个 ASE 库，用于 `query-compounds` / `export-entries`）。
- **与语义检索相关的现有能力**：
  - `query-info`：对信息库执行 **只读 `SELECT`**（`cmd_query_info`）；若结果中含列名 `path`（大小写不敏感），会相对于 **信息库文件所在目录** 解析为绝对路径（便于后续打开 `.db`）。
  - `query-compounds` / `export-entries`：在已确定的 `path` 上操作 ASE 数据。
- **Schema 要点**（嵌入文本拼接时应覆盖的字段）：
  - **`nodes`**：`name`, `description`, `functional`, `code`, `cutoff_eV`, `pseudopot`, `kspacing`, `spin_pol`, `vdw`, `extra_params` 等。
  - **`datasets`**：`elements`, `n_elements`, `system_type`, `field`, `entries`, `source`, `path`, `has_forces`, `has_stress`, `has_energy`, `energy_min`, `energy_max` 等。
  - **`dataset_elements`**：`(dataset_id, element)` 逐元素索引。

### 2.2 `agents/MatCreator/skills/database/SKILL.md`

- **作用**：给 Agent/用户写的 **操作手册**：SQL 规则、`query-info` 的示例、按元素/泛函/体系类型筛选的典型语句。
- **语义检索实验**：本文件描述的是 **符号/SQL 流程**；你的新工作是在此之前或并行增加 **向量检索 → 缩小候选 `dataset_id` / `path`**，再仍用此处命令做精查与导出。

### 2.3 `agents/MatCreator/skills/database/database.md`

- 与 `SKILL.md` 内容同源类文档（若两处并存，以你实际引用的为准；命令与 `INFO_DB_PATH` 约定相同）。

### 2.4 `database/domain_datasets.tar.gz`

- **作用**：README 所述默认 **域数据集** 归档；需解压后，信息库中的 `path` 才能解析到真实 `.db` 文件。
- **实验注意**：构建向量索引前，应保证 **信息库路径与解压目录的相对关系** 与 `query-info` 解析逻辑一致（`path` 相对 `info.db` 父目录）。

### 2.5 `agents/MatCreator/skill.py`

- **作用**：从 workspace 加载各 skill 目录下的 `SKILL.md`；**不**包含数据库实现细节。
- **论文/系统层面**：将来若把「语义检索」固化为技能，会涉及 **新 skill 或扩展现有 database skill 的说明**，与这里加载机制相关，但当前仓库**尚无**向量检索代码。

### 2.6 根目录 `README.md`

- 说明 MatCreator 定位、安装、`INFO_DB_PATH` 与 domain 数据在 `database/domain_datasets.tar.gz` 的说明；语义检索是对 **数据发现层** 的增强，不改变 MLFF 训练对 `export-entries --fmt extxyz` 的要求。

---

## 3. 向量检索在架构中的位置（结合现有流程）

现有顺序（摘自 `SKILL.md`）大致为：

1. `query-info` 列出/筛选 nodes 与 datasets；
2. 用 `dataset_elements` 等做元素级筛选；
3. `query-compounds --db-path <datasets.path>`；
4. `export-entries ... --fmt extxyz` 供力场训练。

**建议的语义检索插入点**：

- **离线**：对每个 `dataset_id`（或每个 `datasets` 行）构造一段 **可检索文本**（JOIN `nodes` 后拼接字段），用预训练模型得到向量，写入向量索引（FAISS / 内存 numpy / 其他库），并保存 `dataset_id` ↔ 向量 ↔ 可选原文快照。
- **在线/实验脚本**：用户自然语言 query → 同模型嵌入 → Top-k 最近邻 → 得到 `dataset_id` 或已解析的 `path` → 再用 **`query-compounds` / `export-entries`** 做后续实验。

这样 **不修改** `db_tools.py` 也能先做独立实验脚本；接入 Agent 则是后续在 skill 层增加调用说明或封装。

---

## 4. 分步实验构建（可复现实验清单）

下列步骤按 **准备数据 → 建索引 → 检索 → 评估 → 可选端到端案例** 排列。

### 步骤 0：环境准备

- 按根目录 `README.md` 创建虚拟环境并 `pip install`/`uv pip install` 项目依赖。
- 解压 `database/domain_datasets.tar.gz` 到文档期望位置。
- 准备 **信息库** `.db` 文件，设置 `INFO_DB_PATH`（或在命令中显式 `--info-db`）。
- 用只读 SQL 确认数据可见（与 `SKILL.md` 一致）：
  - `validate-sql` + `query-info` 查询 `nodes`、`datasets` 行数及示例 `path`。

### 步骤 1：定义「一条向量」的粒度

- **推荐默认**：**一行 `datasets` = 一条向量**（对应一个 domain ASE `.db`），与 `path` 一一对应。
- 若某些 `domain_*` 在库中拆成多行，需在文档中说明是 **按行嵌入** 还是 **合并为逻辑数据集**（会影响检索与论文表述）。

### 步骤 2：构造嵌入用文本（核心设计）

- 对每个 `dataset_id`，执行 JOIN 查询（示例逻辑，非固定 SQL）：
  `SELECT d.*, n.name, n.description, n.functional, n.code, ... FROM datasets d JOIN nodes n ON d.node_id = n.node_id`
- 将选中列 **线性拼接成一段英文/中英混合文档**（可模板化，例如：`"Node: ... Functional: ... Elements: ... Field: ... System: ... Source: ..."`）。
- **记录版本**：嵌入模型名、模板版本、信息库文件 checksum，便于复现实验。

### 步骤 3：选择预训练嵌入模型并固定超参

- 选用公开句向量模型（如 Sentence-Transformers 系等），固定：模型名、向量维度、是否截断长文本、batch 大小。
- **目的**：满足「先试预训练」的对比前提；论文中单独一小节可写模型选择与局限。

### 步骤 4：离线建索引

- 对步骤 2 中每条文本计算向量，建立 **向量矩阵 + 元数据表**（至少含 `dataset_id`、`path`、可选拼接文本摘要）。
- 持久化方式任选：`.npy` + JSON 元数据、FAISS 索引文件、HDF5 等（由你后续实现决定；本文档不写死文件名）。

### 步骤 5：在线检索原型

- 输入：模拟用户问题字符串（如 *“I need data for zinc-blende gallium arsenide”*）。
- 用同一模型嵌入 query，对索引做 **Top-k**（k 如 5、10、20）。
- 输出：候选 `dataset_id`、`path`、相似度分数；可选：用 `query-info` 再拉一次完整行供人工核对。

### 步骤 6：与基线对比（论文实验）

- **基线 A**：纯 LLM + `query-info` + SQL/`LIKE`（复现当前流程）。
- **基线 B**（可选）：BM25 或关键词检索（若在拼接文本上建倒排索引）。
- **方法**：预训练向量 Top-k。
- **指标**（需自建小评测集）：例如 Recall@k、MRR；或 **人工判定**「Top-5 是否包含合理 domain 数据集」。
- **案例集**：包含 **词汇不对齐** 样例（GaAs vs zinc-blende gallium arsenide）、中英文混合（若目标用户会混用）等。

### 步骤 7：与下游工具链对齐（应用层）

- 对 Top-k 中的某个 `path`，用现有 CLI 验证：
  `query-compounds --db-path <resolved_path> --selection "..."`
  `export-entries --db-path <resolved_path> --mode selection ... --fmt extxyz`
- 若做「力场微调 → 数据回流」故事：可衔接 `save-extxyz`（`db_tools.py` 中 `cmd_save_extxyz`）将新数据注册回 **同一信息库 schema** 下的 `User Upload` 节点，形成闭环叙述（具体训练命令取决于 `deepmd` 等其它 skill，不在 `db_tools.py` 内）。

### 步骤 8：记录与论文素材

- 固定随机种子（若有 sampling）、记录硬件、嵌入耗时、索引大小。
- 截图或日志：**用户 query → Top-k 结果表 → 一次成功的 `export-entries` 输出路径**，作为应用案例图。

---

## 5. 风险与注意事项（结合本仓库行为）

- **`path` 解析**：`query-info` 将 `path` 转为相对于 **信息库文件父目录** 的绝对路径；向量索引中应存储与之一致的解析结果，或在实验脚本中复用相同规则，避免「索引里是相对路径、运行时找不到文件」。
- **元信息过短**：若拼接后文本几乎只有 `elements`，语义模型可能无法关联「zinc-blende」等表述；可通过扩充 `description`/`source` 字段利用，或在论文中讨论 **metadata 覆盖度** 对检索的影响。
- **安全与只读**：`validate-sql` 限制为单条 `SELECT`；自定义检索脚本若直接 `sqlite3` 连接信息库，建议保持只读，与现有工具哲学一致。

---

## 6. 文档小结

- **仓库内与「信息库 + domain `.db`」相关的权威说明**：`SKILL.md` + `db_tools.py`（schema 与命令）。
- **语义向量实验**：在仓库外或独立脚本中完成 **文本拼接 → 预训练嵌入 → Top-k**；通过现有 **`query-info` / `query-compounds` / `export-entries`** 验证下游；用 **对比实验 + 典型案例** 支撑论文叙事。
