# ruff: noqa: RUF001
"""Apply audited human curation decisions to the RAG-papers benchmark v1.

The script is intentionally deterministic and model-free. It accepts only the pinned
``rag-papers-benchmark-v1.json`` payload, verifies both its SHA-256 digest and every
curated case's original position/question, and writes a v2 artifact with an embedded
audit trail.

All 60 source cases are in scope and retained: the current policy records 21 accepts,
39 rewrites, and zero drops. Rewritten cases may reuse evidence from other source cases
through explicit, zero-based field selectors; the selectors and the resulting
source-to-output mapping are preserved in the embedded audit trail.

Usage::

    uv run scripts/curate_rag_benchmark.py
    uv run scripts/curate_rag_benchmark.py --dry-run
    uv run scripts/curate_rag_benchmark.py --output artifacts/ragas/custom-v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts" / "ragas" / "rag-papers-benchmark-v1.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "ragas" / "rag-papers-benchmark-v2.json"

EXPECTED_SOURCE_SHA256 = "51dafa1e56af0d6f785053d469ca4d1cb62274215e0b6980093f8527412a6a92"
EXPECTED_CORPUS_CONTENT_HASH = "a809ebc5c4ad960466663cee899ece9c4e016703f94fa5c5d4838ea9998077f4"
EXPECTED_SOURCE_CASE_COUNT = 60
CURATED_SOURCE_CASE_INDEXES = frozenset(range(1, EXPECTED_SOURCE_CASE_COUNT + 1))
CURATION_POLICY_ID = "rag-papers-benchmark-v1-to-v2-manual-2026-07-21"

Action = Literal["accept", "rewrite", "drop"]


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """Zero-based selectors used to rebuild one curated case's evidence fields."""

    source_case_index: int
    reference_context_indexes: tuple[int, ...]
    context_evidence_id_indexes: tuple[int, ...]
    evidence_id_indexes: tuple[int, ...]
    document_id_indexes: tuple[int, ...]
    evidence_quote_indexes: tuple[int, ...] = ()
    evidence_quote_context_indexes: tuple[int, ...] = ()
    evidence_id_field: Literal["evidence_ids", "context_evidence_ids"] = "evidence_ids"


@dataclass(frozen=True, slots=True)
class CaseDecision:
    """One human decision pinned to the source case's one-based position and question."""

    action: Action
    expected_question: str
    review_notes: str
    new_question: str | None = None
    new_reference: str | None = None
    expected_evidence_ids: tuple[str, ...] | None = None
    new_evidence_ids: tuple[str, ...] | None = None
    keep_evidence_quote_indexes: tuple[int, ...] | None = None
    evidence_sources: tuple[EvidenceSource, ...] = ()


DECISIONS: dict[int, CaseDecision] = {
    1: CaseDecision(
        action="accept",
        expected_question="GraphRAG在构建图索引时使用了哪两个阶段？",
        review_notes=(
            "接受：单条引文直接覆盖图索引构建的两个阶段；与 case 21 的全局问答流程边界清晰。"
        ),
    ),
    2: CaseDecision(
        action="rewrite",
        expected_question="RAG技术最初是如何与Transformer架构结合以增强语言模型的？",
        new_question=(
            "根据《Retrieval-Augmented Generation for Large Language Models: A Survey》，"
            "RAG在Transformer兴起之初的研究重点是什么？"
        ),
        new_reference=(
            "该早期阶段主要通过预训练模型（PTM）融入额外知识来增强语言模型，"
            "并开展改进预训练技术的基础性研究。"
        ),
        review_notes=(
            "重写：原问题将历史阶段描述表达成了Transformer集成机制；改为引文"
            "直接支持的早期研究重点。"
        ),
    ),
    3: CaseDecision(
        action="rewrite",
        expected_question="大型语言模型（LLMs）在推理时面临的主要限制是什么？",
        new_question="RQ-RAG论文指出，仅依赖预编码参数知识会给LLM推理带来哪些问题？",
        new_reference=(
            "模型知识在训练或更新后保持静态，无法纳入最新的实时信息；因此容易"
            "产生幻觉，并难以对需要最新信息的查询给出准确、及时的回答。"
        ),
        review_notes=(
            "重写：原问题过于宽泛，语料中多篇论文都能提供其他有效答案；增加"
            "RQ-RAG来源锚点以使 exact evidence 唯一。"
        ),
    ),
    4: CaseDecision(
        action="accept",
        expected_question="RAPTOR系统如何通过树结构来捕捉文本的高层和低层细节？",
        review_notes=(
            "接受：建树步骤及跨层检索目的均由单条引文直接支持；与 case 14的主贡献题侧重不同。"
        ),
    ),
    5: CaseDecision(
        action="accept",
        expected_question="什么是SELF-RAG中的反思令牌（reflection tokens）？它们分为哪两类？",
        review_notes=(
            "接受：retrieval token与critique token的分类及用途均被直接引用；与"
            "case 25的推理时解码定制不重复。"
        ),
    ),
    6: CaseDecision(
        action="rewrite",
        expected_question="RankRAG框架中，指令微调的LLM在上下文排序和答案生成方面表现如何？",
        new_question=(
            "RankRAG论文报告，加入少量排序数据后，指令微调LLM在上下文排序和"
            "答案生成上分别优于哪些基线？"
        ),
        new_reference=(
            "在上下文排序上，它优于现有专业排序模型，也优于仅用大量排序数据微调"
            "的同一LLM；在答案生成上，Llama3-RankRAG在九个知识密集型基准上显著优于"
            "Llama3-ChatQA-1.5和所比较的GPT-4模型。"
        ),
        review_notes=(
            "重写：将含混的‘表现如何’改为两类明确的比较基线，并与 case 16 的框架流程题分离。"
        ),
    ),
    7: CaseDecision(
        action="rewrite",
        expected_question="本文旨在构建什么样的密集检索系统？",
        new_question="HyDE论文旨在构建怎样的零样本密集检索系统？",
        review_notes="重写：只替换失去上下文后无法定位的‘本文’指代；reference保持不变。",
    ),
    8: CaseDecision(
        action="rewrite",
        expected_question="混合模型如何解决纯参数化模型的缺点？",
        new_question="原始RAG论文指出，结合参数化记忆与基于检索的非参数化记忆有哪些直接优势？",
        new_reference=(
            "这种混合方式使知识可以被直接修订和扩展，并使模型访问到的知识可以被检查和解释。"
        ),
        review_notes=(
            "重写：当前evidence quote以‘these issues’回指未收录的前句，不能独立支持"
            "原reference枚举的所有纯参数模型缺点；改问引文显式陈述的优势。"
        ),
    ),
    9: CaseDecision(
        action="rewrite",
        expected_question="检索增强生成（RAG）的有效性取决于什么？",
        new_question="CRAG论文指出，RAG的有效性取决于检索文档的哪两个属性？",
        review_notes=(
            "重写：只增加CRAG来源锚点，避免语料中多个页面都能回答同一泛化命题；reference保持不变。"
        ),
    ),
    10: CaseDecision(
        action="rewrite",
        expected_question="分块在检索增强生成过程中扮演什么角色？",
        new_question="LightRAG论文指出，分块如何提升RAG的信息检索准确性？",
        review_notes=(
            "重写：只增加LightRAG来源锚点，使 exact evidence 不再被多篇论文中的"
            "通用分块描述所歧义；reference保持不变。"
        ),
    ),
    11: CaseDecision(
        action="rewrite",
        expected_question="在典型的RAG设置中，系统如何利用大型外部语料库来回答查询？",
        new_question=(
            "GraphRAG论文对典型RAG设置的描述中，从大型外部语料库选出的记录子集需要满足哪两个条件？"
        ),
        new_reference=("各条记录应分别与查询相关，并且该子集整体足够小，能够放入LLM的上下文窗口。"),
        review_notes=(
            "重写：原quote在‘The LLM then’处被页眉截断，无法直接支持原reference"
            "的最后生成步骤；改问quote完整覆盖的两个筛选条件。"
        ),
    ),
    12: CaseDecision(
        action="rewrite",
        expected_question="这篇论文的主要贡献是什么？",
        new_question=(
            "《Retrieval-Augmented Generation for Large Language Models: A Survey》列出的首项"
            "贡献是什么？"
        ),
        new_reference=(
            "对最新RAG方法进行全面、系统的综述，并梳理RAG在不同范式中的演进，其中包括朴素RAG。"
        ),
        review_notes=(
            "重写：消除‘这篇论文’的指代歧义，并将复数‘主要贡献’收窄到截断quote实际覆盖的首项贡献。"
        ),
    ),
    13: CaseDecision(
        action="rewrite",
        expected_question="根据论文，模型在什么情况下应该直接回应而不进行检索？",
        new_question="在RQ-RAG提出的按需搜索策略中，模型在什么情况下应直接回答而不检索？",
        review_notes=("重写：只用RQ-RAG及按需搜索策略替换无来源的‘根据论文’；reference保持不变。"),
    ),
    14: CaseDecision(
        action="accept",
        expected_question="RAPTOR的主要贡献是什么？",
        review_notes=(
            "接受：不同尺度的摘要检索增强及实验验证均被单条引文直接覆盖；"
            "与 case 4 的建树机制题不重复。"
        ),
    ),
    15: CaseDecision(
        action="rewrite",
        expected_question="根据论文，加利福尼亚州的名称源自什么？",
        new_question="SELF-RAG在决定检索有帮助后，接下来如何处理多个检索段落？",
        new_reference=(
            "它输出retrieval token按需调用检索器，然后并行处理多个检索段落，"
            "评估其相关性，并为每个段落生成相应任务输出。"
        ),
        evidence_sources=(
            EvidenceSource(
                source_case_index=5,
                reference_context_indexes=(0,),
                context_evidence_id_indexes=(0,),
                evidence_id_indexes=(0,),
                document_id_indexes=(0,),
                evidence_quote_indexes=(0,),
            ),
        ),
        review_notes=(
            "重写并复用 case 5 证据：移除PDF多栏图示中的偶然州名事实，改为同一论文"
            "摘要明确支持的按需检索与多段落并行处理流程。"
        ),
    ),
    16: CaseDecision(
        action="accept",
        expected_question="RankRAG框架的主要贡献是什么？",
        expected_evidence_ids=(
            "evd_eec9b514e32068da1e91",
            "evd_27c812d65ea8225cc034",
        ),
        new_evidence_ids=("evd_27c812d65ea8225cc034",),
        review_notes=(
            "接受内容并最小化gold：贡献、训练任务和推理流程均由"
            "evd_27c812d65ea8225cc034的引文单独支持，删除不必要的摘要页gold ID。"
        ),
    ),
    17: CaseDecision(
        action="rewrite",
        expected_question="根据论文，GPT-3模型如何通过少量数据实现对齐？",
        new_question="HyDE为什么让无监督对比编码器对假设文档进行编码？",
        new_reference=(
            "编码器的稠密瓶颈充当有损压缩器，过滤假设文档中额外的幻觉细节；"
            "所得向量再用于搜索语料库嵌入。"
        ),
        evidence_sources=(
            EvidenceSource(
                source_case_index=27,
                reference_context_indexes=(0,),
                context_evidence_id_indexes=(0,),
                evidence_id_indexes=(0,),
                document_id_indexes=(0,),
                evidence_quote_indexes=(1,),
            ),
        ),
        review_notes=(
            "重写并复用 case 27 证据：删除无法回答‘如何对齐’的旁引，改为HyDE中"
            "dense bottleneck过滤幻觉细节并形成检索向量的明确机制。"
        ),
    ),
    18: CaseDecision(
        action="rewrite",
        expected_question="什么是检索增强生成（RAG）模型？",
        new_question="Lewis等提出的原始RAG模型如何组合参数化记忆与非参数化记忆？",
        new_reference=(
            "参数化记忆是预训练的seq2seq Transformer；非参数化记忆是由预训练神经检索器"
            "访问的Wikipedia稠密向量索引；两者被组合进一个端到端训练的概率模型。"
        ),
        review_notes=(
            "重写：增加Lewis等原始RAG的来源边界，并将原reference中的误译‘参数量"
            "记忆’修正为‘参数化记忆’。"
        ),
    ),
    19: CaseDecision(
        action="accept",
        expected_question="CRAG方法中用于评估检索文档整体质量的组件是什么？",
        review_notes="接受：轻量级retrieval evaluator及其评估作用由单条引文直接支持。",
    ),
    20: CaseDecision(
        action="rewrite",
        expected_question="LightRAG采用了哪种检索框架来增强其信息检索能力？",
        new_reference=(
            "LightRAG采用双层检索框架：低层检索关注具体实体及其关系，高层检索"
            "关注更广泛的主题和概念。"
        ),
        review_notes="重写：只修正reference中‘更广泛的主题和主题’的重复翻译；question保持不变。",
    ),
    21: CaseDecision(
        action="accept",
        expected_question="GraphRAG如何支持对整个大型文本语料库的全局理解？",
        review_notes=(
            "接受：建图、社区层次与摘要、map-reduce问答由同一证据单元内的两段"
            "quote共同覆盖；single_hop保持。"
        ),
    ),
    22: CaseDecision(
        action="accept",
        expected_question="RAG研究的技术树中，涉及RAG的阶段主要包括哪些？",
        review_notes="接受：预训练、微调和推理三个阶段由单条引文直接支持。",
    ),
    23: CaseDecision(
        action="accept",
        expected_question="RQ-RAG模型如何改进查询以增强检索增强生成？",
        review_notes="接受：重写、分解和消除歧义三种query refinement操作均被直接引用。",
    ),
    24: CaseDecision(
        action="rewrite",
        expected_question="检索增强语言模型（RALMs）在哪些组件上得到了改进？",
        new_question=(
            "RAPTOR论文的相关工作部分指出，检索增强语言模型（RALM）主要在哪三个组件上得到改进？"
        ),
        review_notes=(
            "重写：只增加RAPTOR相关工作的出处锚点，避免将一个领域通用分类绑定到"
            "唯一exact evidence；reference保持不变。"
        ),
    ),
    25: CaseDecision(
        action="rewrite",
        expected_question="SELF-RAG如何利用反思标记（reflection tokens）来定制模型行为？",
        new_question=(
            "SELF-RAG在推理时如何利用reflection token预测来调节检索频率并按用户偏好定制模型行为？"
        ),
        new_reference=(
            "SELF-RAG用reflection token预测定义硬约束或软约束，并在段级波束搜索中把"
            "reflection token概率的加权线性和作为段得分；由此可按应用调节检索频率，并按"
            "用户偏好定制模型行为。"
        ),
        review_notes=(
            "重写：明确这是推理时机制，并修复原reference遗漏概率与segment score的含混表达。"
        ),
    ),
    26: CaseDecision(
        action="rewrite",
        expected_question="RankRAG方法在哪些基准测试上进行了评估？",
        new_question="RankRAG在多少个通用领域和多少个生物医学知识密集型RAG基准上进行了评估？",
        new_reference="在9个通用领域基准和5个生物医学知识密集型RAG基准上进行了评估。",
        review_notes=(
            "重写：原问‘哪些基准’通常要求具体名称，但quote只给出数量与领域；"
            "改为精确询问9个与5个。后续须删除与本题矛盾的原 case 56。"
        ),
    ),
    27: CaseDecision(
        action="accept",
        expected_question="HyDE模型如何通过假设文档嵌入实现密集检索？",
        review_notes=(
            "接受：生成假设文档、对比编码、使用向量搜索语料库三步由同一证据ID"
            "内的两段quote共同覆盖。"
        ),
    ),
    28: CaseDecision(
        action="rewrite",
        expected_question="RAG模型在哪些知识密集型任务上取得了最先进的结果？",
        new_question=(
            "Lewis等提出的原始RAG模型在Natural Questions、WebQuestions、CuratedTREC和"
            "TriviaQA上分别取得了怎样的结果？"
        ),
        new_reference=(
            "模型在开放域Natural Questions、WebQuestions和CuratedTREC上取得了当时的"
            "最先进结果；在TriviaQA上显著优于采用专门预训练目标的近期方法。"
        ),
        review_notes=(
            "重写：原问将四个基准都暗示为SOTA，但quote只对前三个称SOTA，"
            "对TriviaQA只称显著优于专门预训练方法。"
        ),
    ),
    29: CaseDecision(
        action="accept",
        expected_question="CRAG在哪些数据集上进行了实验？",
        review_notes=(
            "接受：PopQA、Biography、PubHealth和Arc-Challenge四个数据集均被单条引文直接列出。"
        ),
    ),
    30: CaseDecision(
        action="rewrite",
        expected_question="LightRAG如何实现高效的自适应检索？",
        new_question="LightRAG的增量更新机制如何降低接入新数据的成本并保持适应性？",
        new_reference=(
            "它无需重建整个索引，因此降低计算成本并加快适应；其增量更新算法还能"
            "及时整合新数据，使系统在动态环境中保持有效。"
        ),
        keep_evidence_quote_indexes=(1,),
        review_notes=(
            "重写：去掉与 case 20 重复的双层检索内容，只保留增量更新与适应性；"
            "evidence quote同步收窄为原第2段。"
        ),
    ),
    31: CaseDecision(
        action="rewrite",
        expected_question="GraphRAG与向量RAG在回答需要全局理解的查询时有何不同？",
        new_question=(
            "传统向量RAG与GraphRAG在处理全局sensemaking问题时分别依赖什么检索或汇总机制，"
            "为什么GraphRAG更适合这类问题？"
        ),
        new_reference=(
            "向量RAG将问题和文本编码为向量并按语义相似度返回邻近记录；GraphRAG从源数据"
            "构建图索引，分层检测社区并生成社区摘要，再聚合与全局问题相关的社区级信息。"
            "因此GraphRAG面向跨语料主题汇总，而向量RAG主要检索局部相似记录。"
        ),
        evidence_sources=(
            EvidenceSource(
                source_case_index=31,
                reference_context_indexes=(0,),
                context_evidence_id_indexes=(0,),
                evidence_id_indexes=(0,),
                document_id_indexes=(0,),
                evidence_quote_indexes=(0,),
            ),
            EvidenceSource(
                source_case_index=21,
                reference_context_indexes=(0,),
                context_evidence_id_indexes=(0,),
                evidence_id_indexes=(0,),
                document_id_indexes=(0,),
                evidence_quote_indexes=(0, 1),
            ),
        ),
        review_notes=(
            "重写并合并 case 31 RC1 与 case 21 RC1：补全向量相似度检索、GraphRAG分层"
            "社区摘要与全局聚合机制，替换原先缺少解释的能力断言。"
        ),
    ),
    32: CaseDecision(
        action="accept",
        expected_question="根据提供的证据，总结朴素RAG（Naive RAG）的主要步骤及其存在的局限性。",
        review_notes=(
            "接受：索引、检索、生成三阶段及precision/recall与生成质量局限均由同一"
            "reference context内的引文完整支持。"
        ),
    ),
    33: CaseDecision(
        action="accept",
        expected_question="根据论文内容，RQ-RAG框架如何构建训练数据集以支持查询细化？请总结其核心步骤。",
        review_notes=(
            "接受：特殊控制标记、查询重写/分解/消歧、检索与答案生成，以及数据场景"
            "均由单一证据页直接覆盖。"
        ),
    ),
    34: CaseDecision(
        action="accept",
        expected_question="请总结RAPTOR系统中用于构建树状结构的核心步骤，并说明其查询策略。",
        review_notes=(
            "接受：分块、嵌入、聚类、摘要迭代与tree traversal/collapsed tree两种查询"
            "策略均由同一证据单元支持。"
        ),
    ),
    35: CaseDecision(
        action="accept",
        expected_question="根据论文内容，总结SELF-RAG与RLHF在训练方法和推理控制方面的主要区别。",
        review_notes=(
            "接受：离线critic reflection token训练、较低训练成本及推理时可控生成均由直接引文支持。"
        ),
    ),
    36: CaseDecision(
        action="rewrite",
        expected_question=(
            "根据论文内容，总结当前RAG流水线中存在的两个主要问题，并说明RankRAG方法"
            "如何解决这些问题。"
        ),
        new_question=(
            "RankRAG论文指出，端到端更新检索器与LLM、以及使用额外中型reranker，各有什么"
            "限制？RankRAG怎样改变RAG流水线？"
        ),
        new_reference=(
            "端到端更新需要替代损失并因embedding更新而频繁重建索引；BERT/T5类reranker"
            "可能不足以刻画相关性且零样本泛化有限。RankRAG把上下文排序与答案生成统一进"
            "instruction-tuned LLM，推理时采用retrieve-rerank-generate并只保留top-k上下文。"
        ),
        evidence_sources=(
            EvidenceSource(
                source_case_index=36,
                reference_context_indexes=(0,),
                context_evidence_id_indexes=(0, 1),
                evidence_id_indexes=(0,),
                document_id_indexes=(0,),
                evidence_quote_indexes=(0, 1),
            ),
            EvidenceSource(
                source_case_index=45,
                reference_context_indexes=(1,),
                context_evidence_id_indexes=(1,),
                evidence_id_indexes=(1,),
                document_id_indexes=(1,),
                evidence_quote_indexes=(1, 2),
            ),
        ),
        review_notes=(
            "重写并合并 case 36 RC1 与 case 45 RC2：分别保留端到端更新和额外reranker"
            "的具体限制，再用RankRAG的统一训练与retrieve-rerank-generate流程闭环回答。"
        ),
    ),
    37: CaseDecision(
        action="accept",
        expected_question="根据论文内容，总结密集检索和指令跟随语言模型的研究进展，并说明它们之间的关系。",
        review_notes=(
            "接受：MIPS、instruction-following零样本泛化及task-aware dense encoder"
            "的联系均由同一上下文支持。"
        ),
    ),
    38: CaseDecision(
        action="accept",
        expected_question="请总结RAG-Sequence和RAG-Token两种模型在生成目标序列时的核心区别。",
        review_notes=(
            "接受：整序列共用一个检索文档与逐token可使用不同文档的区别由两段直接引文支持。"
        ),
    ),
    39: CaseDecision(
        action="accept",
        expected_question="根据论文内容，总结RAG方法面临的主要挑战以及现有研究如何应对这些挑战。",
        review_notes=(
            "接受：检索错误风险、相关工作策略及CRAG纠错定位由两个连续证据页完整支持；"
            "题型保持summary_reasoning。"
        ),
    ),
    40: CaseDecision(
        action="rewrite",
        expected_question=(
            "根据LightRAG的图增强实体和关系提取过程，总结其如何从原始文本构建知识图谱，"
            "并说明该过程的关键步骤。"
        ),
        new_question=(
            "LightRAG如何从文本chunk构建图索引？概括实体/关系识别、profiling和"
            "deduplication各自的作用。"
        ),
        new_reference=(
            "LightRAG先分块并用LLM识别实体与关系；profiling为节点和边生成文本key-value"
            "描述；deduplication合并跨segment重复的实体与关系，得到最终图。"
        ),
        evidence_sources=(
            EvidenceSource(
                source_case_index=40,
                reference_context_indexes=(0,),
                context_evidence_id_indexes=(0,),
                evidence_id_indexes=(0,),
                document_id_indexes=(0,),
                evidence_quote_indexes=(1, 2, 3),
            ),
            EvidenceSource(
                source_case_index=50,
                reference_context_indexes=(0,),
                context_evidence_id_indexes=(0,),
                evidence_id_indexes=(0,),
                document_id_indexes=(0,),
                evidence_quote_indexes=(0,),
            ),
        ),
        review_notes=(
            "重写并合并 case 40 RC1 与 case 50 RC1：明确识别、profiling、去重三步；"
            "两段来源共享同一page evidence ID，应用时按值去重。"
        ),
    ),
    41: CaseDecision(
        action="rewrite",
        expected_question="比较GraphRAG和传统RAG在检索源和索引构建方法上的主要区别是什么？",
        new_question=(
            "比较GraphRAG与RAG Survey（2312.10997）所述的朴素RAG在索引表示和检索单元上的差异。"
        ),
        new_reference=(
            "朴素向量RAG把文档分块、嵌入向量库并按查询相似度取top-k chunks；GraphRAG"
            "用LLM抽取知识图谱，再以社区检测和社区摘要形成分层图索引，查询面向实体关系"
            "与社区级主题信息。"
        ),
        evidence_sources=(
            EvidenceSource(
                source_case_index=41,
                reference_context_indexes=(0,),
                context_evidence_id_indexes=(0,),
                evidence_id_indexes=(0,),
                document_id_indexes=(0,),
                evidence_quote_indexes=(0,),
            ),
            EvidenceSource(
                source_case_index=32,
                reference_context_indexes=(0,),
                context_evidence_id_indexes=(0, 1),
                evidence_id_indexes=(0,),
                document_id_indexes=(0,),
                evidence_quote_indexes=(0, 1, 2),
            ),
        ),
        review_notes=(
            "重写并合并 case 41 RC1 与 case 32 RC1：将第二来源明确锚定到RAG Survey"
            "（2312.10997），并收窄为向量chunk索引/检索单元与GraphRAG分层图索引的差异。"
        ),
    ),
    42: CaseDecision(
        action="accept",
        expected_question=(
            "比较Naive RAG和Advanced RAG在检索增强生成中的挑战与改进策略，并说明RQ-RAG"
            "如何通过查询精炼进一步提升性能。"
        ),
        review_notes=(
            "接受：Naive/Advanced RAG局限与优化、RQ-RAG控制token和轨迹选择均由两个"
            "来源上下文直接覆盖。"
        ),
    ),
    43: CaseDecision(
        action="rewrite",
        expected_question=(
            "比较两种不同的检索增强生成（RAG）方法：一种使用ChatGPT自动标注构建训练数据，"
            "另一种使用RAPTOR的折叠树检索策略。它们在处理多跳问题时的优势和局限性分别是什么？"
        ),
        new_question=(
            "比较RQ-RAG与RAPTOR在支持多跳问答时，分别从查询侧和语料组织侧采用了什么策略？"
        ),
        new_reference=(
            "RQ-RAG用控制token重写、分解或消歧查询，并选择更优轨迹；RAPTOR从不同树层"
            "检索与问题粒度匹配的节点。前者改造查询，后者改造语料组织与跨层检索。"
        ),
        evidence_sources=(
            EvidenceSource(
                source_case_index=42,
                reference_context_indexes=(1,),
                context_evidence_id_indexes=(1,),
                evidence_id_indexes=(1,),
                document_id_indexes=(1,),
                evidence_quote_indexes=(3, 4),
            ),
            EvidenceSource(
                source_case_index=43,
                reference_context_indexes=(1,),
                context_evidence_id_indexes=(1,),
                evidence_id_indexes=(1,),
                document_id_indexes=(1,),
                evidence_quote_indexes=(4,),
            ),
        ),
        review_notes=(
            "重写并合并 case 42 RC2 与 case 43 RC2：去掉资源消耗和FAISS等旁支，"
            "聚焦RQ-RAG查询变换与RAPTOR跨树层检索对多跳问答的互补作用。"
        ),
    ),
    44: CaseDecision(
        action="rewrite",
        expected_question=(
            "比较两种文本处理方法：一种使用UMAP降维和GMM聚类（E1），另一种使用检索和"
            "相关性判断（E2）。它们分别如何解决处理大规模文本时的上下文长度限制问题？"
        ),
        new_question=("比较RAPTOR与SELF-RAG在控制送入生成模型的上下文规模和证据质量上的策略。"),
        new_reference=(
            "RAPTOR在cluster上下文超过摘要模型token阈值时递归聚类；SELF-RAG对检索"
            "passage预测ISREL和ISSUP，并从相关且有支持的候选中选择检索分数最高的续写。"
            "前者控制摘要上下文规模，后者控制证据质量。"
        ),
        evidence_sources=(
            EvidenceSource(
                source_case_index=44,
                reference_context_indexes=(0,),
                context_evidence_id_indexes=(0, 1),
                evidence_id_indexes=(0, 1),
                document_id_indexes=(0,),
                evidence_quote_indexes=(0,),
            ),
            EvidenceSource(
                source_case_index=55,
                reference_context_indexes=(0,),
                context_evidence_id_indexes=(0,),
                evidence_id_indexes=(0,),
                document_id_indexes=(0,),
                evidence_quote_context_indexes=(0,),
                evidence_id_field="context_evidence_ids",
            ),
        ),
        review_notes=(
            "重写并合并 case 44 RC1 与 case 55 RC1：分别保留RAPTOR递归聚类的token"
            "阈值控制和SELF-RAG按ISREL/ISSUP筛选证据的策略。case 55原为负样本，故将其"
            "context evidence ID显式提升为本题gold。"
        ),
    ),
    45: CaseDecision(
        action="rewrite",
        expected_question=(
            "比较两种RAG方法在检索上下文处理上的不同：一种方法（如2310.11511v1所述）"
            "如何检索段落，而另一种方法（如2407.02485v1所述）如何训练LLM以同时确定"
            "多个上下文的相关性？"
        ),
        new_question=("比较SELF-RAG与RankRAG在过滤候选证据时使用的相关性信号和保留策略。"),
        new_reference=(
            "SELF-RAG对每个passage及生成段预测ISREL、ISSUP和ISUSE，并用reflection "
            "scores排序；RankRAG训练LLM同时识别多个相关context，推理时对top-N计算"
            "相关性分数并重排为top-k后生成。"
        ),
        evidence_sources=(
            EvidenceSource(
                source_case_index=55,
                reference_context_indexes=(0,),
                context_evidence_id_indexes=(0,),
                evidence_id_indexes=(0,),
                document_id_indexes=(0,),
                evidence_quote_context_indexes=(0,),
                evidence_id_field="context_evidence_ids",
            ),
            EvidenceSource(
                source_case_index=45,
                reference_context_indexes=(1,),
                context_evidence_id_indexes=(1,),
                evidence_id_indexes=(1,),
                document_id_indexes=(1,),
                evidence_quote_indexes=(1, 2),
            ),
        ),
        review_notes=(
            "重写并合并 case 55 RC1 与 case 45 RC2：以对称句式比较SELF-RAG的"
            "critique相关性信号与RankRAG多上下文相关性评分及top-N到top-k保留策略。"
            "case 55 context ID被审计为正证据。"
        ),
    ),
    46: CaseDecision(
        action="rewrite",
        expected_question="比较RankRAG和HyDE两种检索增强生成方法在解决检索器容量限制方面的不同策略。",
        new_question="比较HyDE与RankRAG在提高查询和候选上下文匹配质量时采用的方式。",
        new_reference=(
            "HyDE先生成hypothetical document，再用无监督contrastive encoder编码，以"
            "document-document相似度搜索；RankRAG先检索top-N，再让instruction-tuned LLM"
            "计算相关性、重排并保留top-k用于生成。"
        ),
        evidence_sources=(
            EvidenceSource(
                source_case_index=27,
                reference_context_indexes=(0,),
                context_evidence_id_indexes=(0,),
                evidence_id_indexes=(0,),
                document_id_indexes=(0,),
                evidence_quote_indexes=(0, 1),
            ),
            EvidenceSource(
                source_case_index=45,
                reference_context_indexes=(1,),
                context_evidence_id_indexes=(1,),
                evidence_id_indexes=(1,),
                document_id_indexes=(1,),
                evidence_quote_indexes=(1, 2),
            ),
        ),
        review_notes=(
            "重写并合并 case 27 RC1 与 case 45 RC2：以对称句式比较HyDE文档级查询表示"
            "与RankRAG候选重排，并删除未被证据严格定义的‘检索器容量限制’。"
        ),
    ),
    47: CaseDecision(
        action="rewrite",
        expected_question="比较HyDE和RAG在检索方法上的主要区别，并说明它们各自如何处理训练数据的需求。",
        new_question="比较HyDE与RAG在构造检索查询表示及使用任务特定训练信号方面的差异。",
        new_reference=(
            "HyDE用InstructGPT把查询生成为hypothetical document并由Contriever编码，无需"
            "目标任务相关性标签或额外训练；RAG用学习到的query encoder检索，并在任务"
            "input-output pairs上以NLL联合微调query encoder与BART，document encoder保持固定。"
        ),
        evidence_sources=(
            EvidenceSource(
                source_case_index=47,
                reference_context_indexes=(0,),
                context_evidence_id_indexes=(0,),
                evidence_id_indexes=(0,),
                document_id_indexes=(0,),
                evidence_quote_indexes=(0,),
            ),
            EvidenceSource(
                source_case_index=47,
                reference_context_indexes=(1,),
                context_evidence_id_indexes=(1,),
                evidence_id_indexes=(1,),
                document_id_indexes=(1,),
                evidence_quote_indexes=(1, 2),
            ),
        ),
        review_notes=(
            "重写并复用 case 47 RC1/RC2：以对称句式比较查询表示与任务特定训练信号，"
            "并保留HyDE零样本设定及RAG任务NLL训练证据。"
        ),
    ),
    48: CaseDecision(
        action="accept",
        expected_question="比较RAG和CRAG在检索文档处理上的不同策略。",
        review_notes=(
            "接受：原始RAG直接拼接检索内容与CRAG三类置信动作及精炼/网络搜索处理"
            "均由两个文档证据直接支持。"
        ),
    ),
    49: CaseDecision(
        action="rewrite",
        expected_question="比较CRAG和LightRAG在提升检索增强生成鲁棒性方面的不同方法。",
        new_question=(
            "比较CRAG与LightRAG分别从检索后纠错和索引/检索架构两个层面提升证据质量的方式。"
        ),
        new_reference=(
            "CRAG用轻量评估器触发Correct、Incorrect或Ambiguous，并执行精炼、网络搜索"
            "或组合；LightRAG用图增强索引和local/global双层检索覆盖实体细节与高层主题。"
            "前者纠正不可靠结果，后者改善表示与检索覆盖。"
        ),
        evidence_sources=(
            EvidenceSource(
                source_case_index=49,
                reference_context_indexes=(0,),
                context_evidence_id_indexes=(0,),
                evidence_id_indexes=(0,),
                document_id_indexes=(0,),
                evidence_quote_indexes=(0,),
            ),
            EvidenceSource(
                source_case_index=49,
                reference_context_indexes=(1,),
                context_evidence_id_indexes=(1,),
                evidence_id_indexes=(1,),
                document_id_indexes=(1,),
                evidence_quote_indexes=(1,),
            ),
        ),
        review_notes=(
            "重写并复用 case 49 RC1/RC2：明确CRAG是检索后纠错，LightRAG是图增强"
            "索引与双层检索，避免将两者笼统归为同一种‘鲁棒性’机制。"
        ),
    ),
    50: CaseDecision(
        action="rewrite",
        expected_question="比较两种知识图谱构建方法中处理重复实体和关系的策略。",
        new_question="比较LightRAG与GraphRAG在知识图谱构建中处理重复实体和关系的策略。",
        review_notes=(
            "重写：只为原有双文档比较补充LightRAG与GraphRAG名称，消除E1/E2来源指代；"
            "reference与证据保持不变。"
        ),
    ),
    51: CaseDecision(
        action="rewrite",
        expected_question="GraphRAG在生成全局摘要时，使用了哪种具体的社区检测算法？",
        new_question=(
            "GraphRAG使用graspologic实现Leiden社区检测时，具体设置的resolution参数值是多少？"
        ),
        new_reference=(
            "论文说明使用graspologic实现分层Leiden社区检测，但未报告resolution参数值，"
            "因此无法回答。"
        ),
        review_notes=(
            "重写负样本：原问题可由语料中的Leiden直接回答；改问实现未披露的resolution"
            "参数，并保持unanswerable及空gold evidence IDs。"
        ),
    ),
    52: CaseDecision(
        action="rewrite",
        expected_question="在模块化RAG架构中，如何通过微调检索器来提升检索质量？",
        new_question=(
            "LLM-Embedder使用hard labels和LLM soft rewards微调检索器时，两类信号在训练"
            "损失中的具体权重比例是多少？"
        ),
        new_reference=("论文只说明两类监督信号，未报告其损失权重比例，因此无法回答。"),
        review_notes=(
            "重写负样本：原问题存在可回答的一般性方法描述；改问证据未报告的hard label"
            "与soft reward损失权重比例。"
        ),
    ),
    53: CaseDecision(
        action="rewrite",
        expected_question="在公式(2)中，期望E(x,y)∼D中的D代表什么？",
        new_question=(
            "RQ-RAG自动标注时将DuckDuckGo视为黑盒；它对候选网页采用的具体相关性评分函数是什么？"
        ),
        new_reference=(
            "论文明确把DuckDuckGo检索视为黑盒，未披露其内部相关性评分函数，因此无法回答。"
        ),
        review_notes=(
            "重写负样本：原公式上下文明确说明D是dataset；改问黑盒搜索内部未披露的相关性评分函数。"
        ),
    ),
    54: CaseDecision(
        action="accept",
        expected_question="RAPTOR模型在文本聚类时使用高斯混合模型（GMM）的准确率是多少？",
        review_notes=(
            "接受负样本：证据讨论GMM有效性和消融但未给出文本聚类‘准确率’，问题要求"
            "的数值不可由上下文推出。"
        ),
    ),
    55: CaseDecision(
        action="rewrite",
        expected_question="在SELF-RAG中，当检索被触发时，模型会生成多少个不同的critique token？",
        new_question=(
            "SELF-RAG对多个候选passage的ISREL、ISSUP、ISUSE加权得分完全相同时，"
            "推理算法用什么规则打破平局？"
        ),
        new_reference=(
            "论文定义了三类critique分数及加权排序，但未说明得分完全相同时的tie-break"
            "规则，因此无法回答。"
        ),
        review_notes=(
            "重写负样本：原上下文可数出critique token类型；改问算法未定义的完全同分tie-break规则。"
        ),
    ),
    56: CaseDecision(
        action="rewrite",
        expected_question="RankRAG方法在哪些数据集上进行了评估？",
        new_question=(
            "RankRAG在九个通用域知识密集型基准上，分别报告的单查询平均美元推理成本是多少？"
        ),
        new_reference=(
            "论文列出任务、指标、硬件与效率分析，但未报告各基准的单查询美元成本，因此无法回答。"
        ),
        review_notes=(
            "重写负样本：移除与answerable case 26直接矛盾的数据集问题；改问未报告的"
            "逐基准单查询美元成本。"
        ),
    ),
    57: CaseDecision(
        action="rewrite",
        expected_question="HyDE方法中，用于生成假设文档的语言模型具体是什么？",
        new_question=(
            "HyDE实验中，text-davinci-003为每个查询生成的hypothetical document平均包含多少token？"
        ),
        new_reference=("论文给出生成模型和温度，但未报告假设文档的平均token长度，因此无法回答。"),
        review_notes=(
            "重写负样本：原问题可由实验设置中的text-davinci-003回答；改问未披露的"
            "hypothetical document平均token数。"
        ),
    ),
    58: CaseDecision(
        action="rewrite",
        expected_question="RAG模型在训练过程中是否更新了文档编码器？",
        new_question=(
            "RAG固定document encoder和index而不周期更新，相比更新方案具体节省了百分之多少训练时间？"
        ),
        new_reference=(
            "论文只说明更新代价高并选择固定二者，未量化训练时间节省比例，因此无法回答。"
        ),
        review_notes=(
            "重写负样本：原上下文明确回答document encoder不更新；改问未量化的训练时间节省百分比。"
        ),
    ),
    59: CaseDecision(
        action="rewrite",
        expected_question="CRAG方法在推理时如何纠正检索到的文档？",
        new_question=(
            "CRAG在Incorrect动作触发网络搜索后，如果搜索无结果或请求失败，采用什么重试或回退策略？"
        ),
        new_reference=(
            "论文说明Incorrect时丢弃原文档并进行网络搜索，但未定义搜索失败时的重试或"
            "回退策略，因此无法回答。"
        ),
        review_notes=(
            "重写负样本：原问题可由CRAG三类动作与纠错流程直接回答；改问网络搜索失败时"
            "未定义的重试/回退行为。"
        ),
    ),
    60: CaseDecision(
        action="rewrite",
        expected_question="LightRAG在增量更新时，如何确保新旧图结构之间的语义一致性？",
        new_question=(
            "LightRAG增量更新若错误合并了同名但语义不同的实体，论文定义了什么冲突检测与回滚机制？"
        ),
        new_reference=(
            "论文说明以相同索引流程构图并合并新旧节点或边，但未定义错误实体合并的检测"
            "与回滚机制，因此无法回答。"
        ),
        review_notes=(
            "重写负样本：将含混的‘语义一致性’收窄为错误实体合并时未披露的冲突检测与回滚机制。"
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Pinned v1 benchmark (default: {DEFAULT_INPUT.relative_to(ROOT)}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Curated v2 benchmark (default: {DEFAULT_OUTPUT.relative_to(ROOT)}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file; the input file is never overwritten.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the audit summary without writing the v2 artifact.",
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must be different files")
    return args


def main() -> None:
    args = parse_args()
    source_bytes = _read_source(args.input)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            "source benchmark SHA-256 does not match the reviewed v1 artifact "
            f"({source_sha256} != {EXPECTED_SOURCE_SHA256})"
        )
    payload = json.loads(source_bytes)
    curated, summary = curate(payload, source_sha256=source_sha256)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run complete; no artifact was written.")
        return
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"output already exists: {args.output}; pass --overwrite to replace it"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(curated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Curated benchmark written to {args.output}")


def curate(payload: object, *, source_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the curated envelope and its deterministic audit summary."""

    _validate_decisions()
    source = _validate_source_payload(payload)
    source_cases = source["cases"]
    assert isinstance(source_cases, list)  # Narrowed by _validate_source_payload.

    curated_cases: list[dict[str, Any]] = []
    audit_decisions: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    for source_case_index, raw_case in enumerate(source_cases, start=1):
        case = _case_object(raw_case, source_case_index)
        decision = DECISIONS.get(source_case_index)
        if decision is None:
            curated_cases.append(deepcopy(case))
            continue
        _validate_case_identity(case, source_case_index, decision)
        action_counts[decision.action] += 1
        if decision.action == "drop":
            audit_decisions.append(
                _audit_record(
                    source_case_index=source_case_index,
                    output_case_index=None,
                    decision=decision,
                    curated_case=None,
                    changed_fields=("case_removed",),
                )
            )
            continue

        curated_case = _apply_decision(
            case,
            source_cases=source_cases,
            case_index=source_case_index,
            decision=decision,
        )
        curated_cases.append(curated_case)
        changed_fields = tuple(
            key
            for key in sorted(set(case) | set(curated_case))
            if case.get(key) != curated_case.get(key)
        )
        audit_decisions.append(
            _audit_record(
                source_case_index=source_case_index,
                output_case_index=len(curated_cases),
                decision=decision,
                curated_case=curated_case,
                changed_fields=changed_fields,
            )
        )

    if len(curated_cases) != EXPECTED_SOURCE_CASE_COUNT:
        raise RuntimeError(
            f"v2 must retain {EXPECTED_SOURCE_CASE_COUNT} cases, found {len(curated_cases)}"
        )

    summary: dict[str, Any] = {
        "policy_id": CURATION_POLICY_ID,
        "source_file_sha256": source_sha256,
        "source_case_count": len(source_cases),
        "output_case_count": len(curated_cases),
        "curated_source_case_index_range": {
            "first": min(DECISIONS),
            "last": max(DECISIONS),
        },
        "action_counts": {
            action: action_counts[action] for action in ("accept", "rewrite", "drop")
        },
        "reviewed_retained_count": sum(
            case.get("review_status") == "reviewed" for case in curated_cases
        ),
        "all_cases_reviewed": all(
            case.get("review_status") == "reviewed" for case in curated_cases
        ),
        "dropped_source_case_indexes": [
            index for index, decision in DECISIONS.items() if decision.action == "drop"
        ],
        "review_status_counts_before": _value_counts(source_cases, "review_status"),
        "review_status_counts_after": _value_counts(curated_cases, "review_status"),
    }
    curated = deepcopy(source)
    curated["cases"] = curated_cases
    curated["curation_audit"] = {
        "schema_version": "1",
        "policy_id": CURATION_POLICY_ID,
        "reviewed_on": "2026-07-21",
        "source": {
            "artifact": "rag-papers-benchmark-v1.json",
            "file_sha256": source_sha256,
            "corpus_content_hash": source["corpus_content_hash"],
        },
        "summary": summary,
        "decisions": audit_decisions,
    }
    return curated, summary


def _read_source(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"unable to read source benchmark {path}: {error}") from error


def _validate_source_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("source benchmark must be a JSON object envelope")
    if payload.get("schema_version") != "2":
        raise ValueError("source benchmark must use evaluation test-set schema version 2")
    if payload.get("corpus_content_hash") != EXPECTED_CORPUS_CONTENT_HASH:
        raise ValueError("source benchmark corpus_content_hash does not match the reviewed corpus")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_SOURCE_CASE_COUNT:
        raise ValueError(
            f"source benchmark must contain exactly {EXPECTED_SOURCE_CASE_COUNT} cases"
        )
    if "curation_audit" in payload:
        raise ValueError("source benchmark already contains a curation_audit")
    return payload


def _validate_decisions() -> None:
    if set(DECISIONS) != CURATED_SOURCE_CASE_INDEXES:
        missing = sorted(CURATED_SOURCE_CASE_INDEXES - set(DECISIONS))
        unexpected = sorted(set(DECISIONS) - CURATED_SOURCE_CASE_INDEXES)
        raise RuntimeError(
            f"curation decisions are incomplete: missing={missing}, unexpected={unexpected}"
        )
    dropped = [index for index, decision in DECISIONS.items() if decision.action == "drop"]
    if dropped:
        raise RuntimeError(f"v2 must retain all source cases; drop decisions found at {dropped}")
    for case_index, decision in DECISIONS.items():
        if not decision.expected_question.strip() or not decision.review_notes.strip():
            raise RuntimeError(f"case {case_index} has incomplete audit metadata")
        for field, value in (
            ("new_question", decision.new_question),
            ("new_reference", decision.new_reference),
        ):
            if value is not None and not value.strip():
                raise RuntimeError(f"case {case_index} has an empty {field}")
        if decision.action == "rewrite" and not (
            decision.new_question is not None or decision.new_reference is not None
        ):
            raise RuntimeError(f"rewrite case {case_index} does not change question or reference")
        if decision.action == "drop" and any(
            value is not None
            for value in (
                decision.new_question,
                decision.new_reference,
                decision.new_evidence_ids,
                decision.keep_evidence_quote_indexes,
            )
        ):
            raise RuntimeError(f"drop case {case_index} must not define replacement content")
        if (decision.expected_evidence_ids is None) != (decision.new_evidence_ids is None):
            raise RuntimeError(
                f"case {case_index} must define expected and replacement evidence IDs together"
            )
        if decision.evidence_sources and (
            decision.new_evidence_ids is not None
            or decision.keep_evidence_quote_indexes is not None
        ):
            raise RuntimeError(
                f"case {case_index} cannot combine evidence sources with direct evidence edits"
            )
        for source in decision.evidence_sources:
            _validate_evidence_source(case_index, source)


def _validate_evidence_source(case_index: int, source: EvidenceSource) -> None:
    if not 1 <= source.source_case_index <= EXPECTED_SOURCE_CASE_COUNT:
        raise RuntimeError(
            f"case {case_index} evidence source has invalid case {source.source_case_index}"
        )
    selectors = {
        "reference_context_indexes": source.reference_context_indexes,
        "context_evidence_id_indexes": source.context_evidence_id_indexes,
        "evidence_id_indexes": source.evidence_id_indexes,
        "document_id_indexes": source.document_id_indexes,
    }
    if not source.evidence_quote_indexes and not source.evidence_quote_context_indexes:
        raise RuntimeError(f"case {case_index} evidence source selects no quote")
    if not set(source.evidence_quote_context_indexes).issubset(source.reference_context_indexes):
        raise RuntimeError(
            f"case {case_index} quote context indexes must also be selected contexts"
        )
    selectors.update(
        {
            "evidence_quote_indexes": source.evidence_quote_indexes,
            "evidence_quote_context_indexes": source.evidence_quote_context_indexes,
        }
    )
    for field, indexes in selectors.items():
        if (
            field
            in {
                "reference_context_indexes",
                "context_evidence_id_indexes",
                "evidence_id_indexes",
                "document_id_indexes",
            }
            and not indexes
        ):
            raise RuntimeError(f"case {case_index} evidence source has no {field}")
        if any(index < 0 for index in indexes) or len(indexes) != len(set(indexes)):
            raise RuntimeError(f"case {case_index} evidence source has invalid {field}: {indexes}")


def _case_object(value: object, case_index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"source case {case_index} must be an object")
    return value


def _validate_case_identity(
    case: dict[str, Any],
    case_index: int,
    decision: CaseDecision,
) -> None:
    declared_index = case.get("case_index")
    if declared_index is not None and declared_index != case_index:
        raise ValueError(
            f"source case at position {case_index} declares case_index={declared_index!r}"
        )
    question = case.get("user_input")
    if question != decision.expected_question:
        raise ValueError(
            f"source case {case_index} question changed: "
            f"expected {decision.expected_question!r}, found {question!r}"
        )
    if case.get("review_status") != "unreviewed":
        raise ValueError(
            f"source case {case_index} must still be unreviewed before applying this policy"
        )
    if decision.expected_evidence_ids is not None:
        evidence_ids = case.get("evidence_ids")
        if evidence_ids != list(decision.expected_evidence_ids):
            raise ValueError(
                f"source case {case_index} evidence_ids changed: "
                f"expected {list(decision.expected_evidence_ids)!r}, found {evidence_ids!r}"
            )


def _apply_decision(
    case: dict[str, Any],
    *,
    source_cases: list[object],
    case_index: int,
    decision: CaseDecision,
) -> dict[str, Any]:
    curated = deepcopy(case)
    if decision.new_question is not None:
        curated["user_input"] = decision.new_question
    if decision.new_reference is not None:
        curated["reference"] = decision.new_reference
    if decision.new_evidence_ids is not None:
        curated["evidence_ids"] = list(decision.new_evidence_ids)
    if decision.evidence_sources:
        _rebuild_evidence(
            curated,
            source_cases=source_cases,
            case_index=case_index,
            sources=decision.evidence_sources,
        )
    if decision.keep_evidence_quote_indexes is not None:
        quotes = curated.get("evidence_quotes")
        if not isinstance(quotes, list) or not all(isinstance(quote, str) for quote in quotes):
            raise ValueError(f"source case {case_index} evidence_quotes must be a string list")
        try:
            curated["evidence_quotes"] = [
                quotes[index] for index in decision.keep_evidence_quote_indexes
            ]
        except IndexError as error:
            raise ValueError(
                f"source case {case_index} does not contain the selected evidence quote"
            ) from error
    curated["review_status"] = "reviewed"
    curated["review_notes"] = decision.review_notes
    _validate_curated_evidence(curated, case_index)
    return curated


def _rebuild_evidence(
    curated: dict[str, Any],
    *,
    source_cases: list[object],
    case_index: int,
    sources: tuple[EvidenceSource, ...],
) -> None:
    reference_contexts: list[str] = []
    context_evidence_ids: list[str] = []
    evidence_ids: list[str] = []
    evidence_quotes: list[str] = []
    document_ids: list[str] = []
    for source in sources:
        source_case = _case_object(
            source_cases[source.source_case_index - 1],
            source.source_case_index,
        )
        selected_contexts = _indexed_strings(
            source_case,
            "reference_contexts",
            source.reference_context_indexes,
            owner_case_index=case_index,
            source_case_index=source.source_case_index,
        )
        selected_context_ids = _indexed_strings(
            source_case,
            "context_evidence_ids",
            source.context_evidence_id_indexes,
            owner_case_index=case_index,
            source_case_index=source.source_case_index,
        )
        selected_evidence_ids = _indexed_strings(
            source_case,
            source.evidence_id_field,
            source.evidence_id_indexes,
            owner_case_index=case_index,
            source_case_index=source.source_case_index,
        )
        selected_documents = _indexed_strings(
            source_case,
            "document_ids",
            source.document_id_indexes,
            owner_case_index=case_index,
            source_case_index=source.source_case_index,
        )
        selected_quotes = _indexed_strings(
            source_case,
            "evidence_quotes",
            source.evidence_quote_indexes,
            owner_case_index=case_index,
            source_case_index=source.source_case_index,
        )
        selected_quotes.extend(
            _indexed_strings(
                source_case,
                "reference_contexts",
                source.evidence_quote_context_indexes,
                owner_case_index=case_index,
                source_case_index=source.source_case_index,
            )
        )
        if not set(selected_evidence_ids).issubset(selected_context_ids):
            raise ValueError(
                f"case {case_index} evidence source {source.source_case_index} selects gold IDs "
                "outside its selected context evidence IDs"
            )
        for quote in selected_quotes:
            if not any(quote in context for context in selected_contexts):
                raise ValueError(
                    f"case {case_index} evidence source {source.source_case_index} selects a "
                    "quote not contained in its selected reference contexts"
                )
        reference_contexts.extend(selected_contexts)
        context_evidence_ids.extend(selected_context_ids)
        evidence_ids.extend(selected_evidence_ids)
        evidence_quotes.extend(selected_quotes)
        document_ids.extend(selected_documents)

    curated["reference_contexts"] = _dedupe(reference_contexts)
    curated["context_evidence_ids"] = _dedupe(context_evidence_ids)
    curated["evidence_ids"] = _dedupe(evidence_ids)
    curated["evidence_quotes"] = _dedupe(evidence_quotes)
    curated["document_ids"] = _dedupe(document_ids)


def _indexed_strings(
    case: dict[str, Any],
    field: str,
    indexes: tuple[int, ...],
    *,
    owner_case_index: int,
    source_case_index: int,
) -> list[str]:
    values = case.get(field)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"source case {source_case_index} {field} must be a string list")
    try:
        return [values[index] for index in indexes]
    except IndexError as error:
        raise ValueError(
            f"case {owner_case_index} selects an out-of-range {field} index from source "
            f"case {source_case_index}"
        ) from error


def _validate_curated_evidence(case: dict[str, Any], case_index: int) -> None:
    fields = {
        field: _required_string_list(case, field, case_index)
        for field in (
            "reference_contexts",
            "context_evidence_ids",
            "evidence_ids",
            "evidence_quotes",
            "document_ids",
        )
    }
    answerable = case.get("answerable")
    if not isinstance(answerable, bool):
        raise ValueError(f"curated case {case_index} answerable must be boolean")
    if not fields["reference_contexts"] or not fields["context_evidence_ids"]:
        raise ValueError(f"curated case {case_index} requires contexts and context evidence IDs")
    if not fields["document_ids"]:
        raise ValueError(f"curated case {case_index} requires document IDs")
    if answerable:
        if not fields["evidence_ids"] or not fields["evidence_quotes"]:
            raise ValueError(f"answerable curated case {case_index} requires gold evidence")
        if not set(fields["evidence_ids"]).issubset(fields["context_evidence_ids"]):
            raise ValueError(
                f"curated case {case_index} gold evidence IDs must be covered by context IDs"
            )
        for quote in fields["evidence_quotes"]:
            if not any(quote in context for context in fields["reference_contexts"]):
                raise ValueError(
                    f"curated case {case_index} has an evidence quote outside its contexts"
                )
    elif fields["evidence_ids"]:
        raise ValueError(f"unanswerable curated case {case_index} must keep evidence_ids empty")
    if not answerable and case.get("question_type") != "unanswerable":
        raise ValueError(f"unanswerable curated case {case_index} changed question_type")


def _required_string_list(case: dict[str, Any], field: str, case_index: int) -> list[str]:
    value = case.get(field)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"curated case {case_index} {field} must be a string list")
    return value


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _audit_record(
    *,
    source_case_index: int,
    output_case_index: int | None,
    decision: CaseDecision,
    curated_case: dict[str, Any] | None,
    changed_fields: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "source_case_index": source_case_index,
        "output_case_index": output_case_index,
        "action": decision.action,
        "original_question": decision.expected_question,
        "curated_question": curated_case.get("user_input") if curated_case is not None else None,
        "changed_fields": list(changed_fields),
        "evidence_sources": [
            {
                "source_case_index": source.source_case_index,
                "reference_context_indexes": list(source.reference_context_indexes),
                "context_evidence_id_indexes": list(source.context_evidence_id_indexes),
                "evidence_id_field": source.evidence_id_field,
                "evidence_id_indexes": list(source.evidence_id_indexes),
                "document_id_indexes": list(source.document_id_indexes),
                "evidence_quote_indexes": list(source.evidence_quote_indexes),
                "evidence_quote_context_indexes": list(source.evidence_quote_context_indexes),
            }
            for source in decision.evidence_sources
        ],
        "review_notes": decision.review_notes,
    }


def _value_counts(cases: list[object], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in cases:
        case = value if isinstance(value, dict) else {}
        counts[str(case.get(field, "missing"))] += 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    main()
