# OPK RAG Architecture

> **Document type:** Architecture specification
> **Architecture status:** Canonical Retrieval Runtime v2 active; legacy UUID/SearchResponse runtime historical/deprecated
> **Current implementation status:** Migration baseline, Supabase access boundary, retrieval RPC, local Vault discovery/change detection, document-state sync, deterministic Markdown parser/chunker, pre-embedding chunk persistence, local embedding, Canonical Retrieval Runtime v2, vector/BM25/hybrid historical retrieval, optional reranking, Evidence Bundle selection, answerability, citation parsing, and grounding validation implemented
> **Last updated:** 2026-08-16

## 1. 架构目标

`opk-rag` 是一个 Supabase-first、PostgreSQL/pgvector-based、local-development-friendly 的 RAG 系统。TASK-0098 后，长期架构定位为 Heterogeneous Document Intelligence + Adaptive Agentic RAG；Markdown 是当前实现的 representation，不再是唯一 canonical input。

当前正式参考运行环境是：

```text
Local Vault
→ Local CLI orchestration
→ Remote Supabase PostgreSQL / pgvector
→ Local retrieval / answerability / grounding
→ Remote DeepSeek generation
→ CLI answer with citations
```

下一阶段权威路线图见 `docs/HETEROGENEOUS_DOCUMENT_AND_ADAPTIVE_GRAPHRAG_ROADMAP.md`。该路线图冻结 `CanonicalDocument`、Knowledge Identity / Representation Identity、pdfQA / WildGraphBench / GraphRAG-Bench benchmark authority，以及 GraphRAG 不替代 Agent RAG 的决策。

### 1.1 长期 Canonical Document 架构

```text
Knowledge Source
      ↓
Representation
      ├── Markdown
      ├── HTML
      ├── TeX
      ├── PDF
      ├── XLSX
      └── Future Formats
              ↓
Format-specific Adapter / Parser
              ↓
CanonicalDocument IR
              ↓
CanonicalSection / CanonicalBlock
              ↓
Unified Chunking
              ↓
CanonicalChunk
              ↓
Retrieval
              ↓
Reranking / Rank Fusion
              ↓
Agentic Retrieval
              ↓
Optional Graph Retrieval
              ↓
Evidence Fusion
              ↓
Grounded Generation
```

TASK-0098 does not implement `CanonicalDocument`; it freezes it as the next upstream IR direction. The existing TASK-0090 `CanonicalChunkV2` remains the active retrieval identity foundation.

### 1.2 Agent / Graph Boundary

GraphRAG is not a replacement for Agent RAG. Graph retrieval is an optional retrieval capability selected by an agent when query and evidence structure make graph traversal useful or necessary. The final target is Adaptive Agentic RAG, not a standalone GraphRAG runtime replacing the current agent path.

## 2. 正式数据库架构

正式数据库方案为：

```text
Supabase
└── PostgreSQL
    └── pgvector
```

Supabase 负责：

* 文档元数据；
* Markdown Chunk；
* 标题路径；
* 内容哈希；
* 文档稳定身份；
* Chunk 稳定身份；
* 索引配置；
* 索引运行记录；
* 索引失败记录；
* Embedding 向量；
* 检索所需字段；
* 引用和来源定位所需字段。

SQLite 不是当前正式架构。

## 3. 本地优先与部署模式

### 3.1 本地优先

本地优先指：

* Vault 在本地文件系统；
* 扫描、解析、分块、哈希和索引计划在本地执行；
* Answerability、Citation Parsing 和 Grounding Validation 在本地执行；
* CLI 是主要入口。

本地优先不代表当前参考运行环境中的数据库和生成模型都必须本地部署。

### 3.2 云端参考模式

Phase 2 的正式参考运行环境使用：

* 远程 Supabase；
* PostgreSQL + pgvector；
* 远程 DeepSeek；
* 本地 Vault；
* 本地 CLI。

这不是完全离线系统，也不是完全本地系统。

## 4. 组件边界

### 4.0 Canonical Retrieval Runtime v2

TASK-0090 makes `canonical_retrieval_runtime_v2` the active retrieval architecture for new development. The logical Chunk and Retrieval Candidate authority is `canonical_chunk_id`, not a database UUID or legacy `SearchResponse` identity.

The active v2 path is:

```text
Corpus Snapshot
→ CanonicalChunkV2
→ Vector Retrieval
→ RetrievalCandidateV2
→ Optional Reranking
→ EvidenceContextV2
→ Downstream RAG
```

`canonical_chunk_id` is deterministic for an unchanged snapshot and is derived from canonical JSON containing the corpus snapshot id, normalized source path, source span, and content digest. Storage rows may still have physical ids, but those ids are implementation details and are not the business-level retrieval identity.

Legacy UUID/SearchResponse retrieval remains in the repository for historical verification and audit of earlier tasks. It is not an equal active runtime for new features, and v2 does not silently fall back to the legacy candidate path, TASK-0088 candidate injection, or TASK-0089 identity bridge. Existing knowledge bases require a clean v2 reindex rather than UUID-preserving migration.

### 4.1 Scanner

负责：

* 读取本地 Vault；
* 枚举 Markdown 文件；
* 收集文件元数据；
* 计算源文件内容哈希；
* 与外部传入的既有文档状态比较，生成 `added` / `modified` / `deleted` / `unchanged`。

不负责：

* Markdown 语义解析；
* Chunk 切分；
* Embedding；
* 数据库事务；
* Supabase 调用。

当前已实现的 `opk_rag.vault` 扫描器采用以下语义：

* Vault 根路径解析为绝对真实路径；
* 支持 `.md`，扩展名大小写不敏感；
* 相对路径使用 POSIX `/` 并保留文件名大小写；
* 默认忽略隐藏目录、`.git/`、`.obsidian/`、缓存和构建目录；
* 默认不跟随目录符号链接，也不索引文件符号链接；
* 内容哈希为流式 SHA-256；
* 单文件失败记录为结构化 `ScanFailure`。

### 4.2 Parser / Chunker

负责：

* Markdown 解析；
* 标题路径提取；
* 结构化分块；
* 行号定位。

不负责：

* 直接数据库存储；
* 网络提交。

当前实现位于 `opk_rag.chunking`：

* Frontmatter 支持简单 YAML 标量和列表；
* 标题识别支持 `#` 到 `######`，并忽略 fenced code block 内的 `#`；
* Chunk 长度单位为字符数，默认 `target_size=1200`、`max_size=1800`、`overlap=0`；
* 代码块在可行时保持完整，单个超长代码块会作为明确的超长 Chunk 保留；
* Chunk checksum 使用标准化正文的 SHA-256。

### 4.3 Index Planner

负责：

* 读取 Supabase 当前索引状态；
* 比较本地 Vault 与数据库状态；
* 生成 `added` / `modified` / `deleted` / `moved` / `unchanged` 计划；
* 绑定 configuration fingerprint。

当前 TASK-0003 的本地变化检测只实现 `added` / `modified` / `deleted` / `unchanged`。重命名不会自动推断为同一文档；未来完整 Index Planner 如要支持 `moved`，必须在独立任务中定义唯一匹配和歧义处理规则。

### 4.4 Index Executor

负责：

* 执行索引计划；
* 调用本地 Embedding；
* 组织文档级事务提交；
* 更新 `index_runs` 和 `index_failures`。

### 4.5 Repository / Storage

负责：

* 与 Supabase 或 PostgreSQL 交互；
* 文档级 upsert；
* 事务控制；
* 检索查询；
* RPC / function 调用。

### 4.6 Retrieval

负责：

* pgvector 语义检索；
* 去重与排序；
* 返回真实 `document_id` / `chunk_id` / 路径 / 片段。

当前实现范围包括 pgvector 向量检索、PostgreSQL BM25、RRF hybrid fusion、默认本地 CrossEncoder Rank-Fusion reranking、去重、context token 预算选择和 Evidence Bundle。Reranker 默认开启，可通过 `OPK_RAG_RERANK_ENABLED=false` 或 CLI `--no-rerank` 显式关闭。Answer 层已经提供基础 pre-generation gate、严格结构化 LLM 输出、本地引用校验和确定性 unsupported-claim 检查；当前不具备完整语义蕴含验证。

### 4.7 Answer

负责：

* 基于已检索证据组织输出；
* 执行引用校验；
* 证据不足时拒答。
* 区分正常知识拒答和 provider/database/runtime 系统错误。

不负责：

* 自行生成文件路径；
* 自行生成标题路径；
* 自行生成 Chunk ID；
* 伪造原文片段。

## 5. 正式数据流

```text
本地 Obsidian / Markdown Vault
        ↓
本地文件扫描
        ↓
本地 Markdown 解析与结构化分块
        ↓
本地内容哈希与稳定身份计算
        ↓
本地增量索引计划
        ↓
本地 Embedding
        ↓
Supabase PostgreSQL
  ├── documents
  ├── chunks
  ├── index_configurations
  ├── index_runs
  ├── index_failures
  └── pgvector embeddings
        ↓
PostgreSQL lexical/BM25 / pgvector retrieval
        ↓
候选合并与排序
        ↓
本地 Answerability / Grounding
        ↓
答案、文件路径、标题路径和原文片段
```

### 5.1 数据边界

* `OPK_RAG_LLM_API_KEY` 只从环境变量或本地 `.env` 读取，不写入仓库内容。
* DeepSeek 请求只发送当前问题、选中的 Evidence 片段和必要的标题路径元数据。
* Knowledge Base 隔离由独立 `knowledge_bases` 记录和固定 Vault root 维持。
* 任何实验都必须明确当前 KB ID，不能混用不同 Vault 的 Evidence。

## 6. 索引事务路径

索引事务路径应为：

```text
读取数据库状态
→ 扫描本地 Vault
→ 生成 IndexPlan
→ 本地解析与 Embedding
→ 单文档 PostgreSQL 事务提交
→ 更新 index_runs
→ 写入 index_failures
```

正式要求：

* 不将整个 Vault 放进单个大事务；
* 以文档级原子更新为最小提交单元；
* 单文档失败不得污染其他文档；
* 网络中断不得留下半更新文档。

## 7. migration 管理

正式 schema 变更方式为 Supabase migration。

当前已实现目录：

```text
supabase/
└── migrations/
    └── 202607170001_create_core_rag_schema.sql
```

migration 至少需要覆盖：

* 启用 pgvector；
* 创建 `documents`；
* 创建 `chunks`；
* 创建 `index_configurations`；
* 创建 `index_runs`；
* 创建 `index_failures`；
* 外键与唯一约束；
* 向量索引；
* 必要 PostgreSQL function / RPC。

当前已经实现：

* pgvector 扩展启用声明；
* `knowledge_bases` / `documents` / `chunks` / `index_configurations` / `index_runs` / `index_failures`；
* 外键、唯一约束、级联删除；
* `chunks.embedding` 的 `vector(1024)` 列；
* 基于 cosine 的 HNSW 向量索引；
* public 表 RLS 安全边界；
* 受控向量检索 RPC `public.match_chunks(...)`。
* 本地 Vault 文件发现、文件快照和纯内存变化检测。

当前尚未实现：

* 事务式 Repository 写入路径；
* Markdown Parser / Chunker / Embedding 运行时；
* 完整检索服务；
* 自动化真实数据库迁移回归 CI 流水线。

## 8. 配置与密钥边界

建议环境变量：

* `DATABASE_URL`
* `SUPABASE_URL`
* `SUPABASE_PUBLISHABLE_KEY`
* `SUPABASE_ANON_KEY`，仅用于兼容旧命名
* `SUPABASE_SERVICE_ROLE_KEY`

约束：

* 数据库迁移、健康检查和索引写入默认使用 `DATABASE_URL` 直连 PostgreSQL；
* public 核心表默认不对 `anon` / `authenticated` 开放 Data API 表访问；
* 查询路径不应默认使用 `service_role`；
* `service_role` 仅用于可信本地索引进程、migration、受控管理流程或明确授权的 RPC；
* 不提交真实密钥到 Git；
* 本地开发和托管环境必须隔离配置。
