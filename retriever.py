# retriever.py - 多路召回 + 重排序 + 置信度计算
import os
import pickle
import jieba
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import torch

from database import client
from model import Embedding_model
from variable import COLLECTION_NAME1
from logger_config import setup_logger

logger = setup_logger("retriever", log_file="retriever.log")

class Retriever:   #定义检索器总类，整合BM25稀疏检索、向量结果接收、RRF融合、Reranker重排、分数过滤整套混合检索能力。
    def __init__(  #类的构造方法，实例化 Retriever()对象时自动执行，所有配置、模型、索引全部在这里初始化。
        self,
        bm25_index_path: str = "bm25_index.pkl",  #初始化时加载之前离线构建好的BM25分词索引文件，用来做关键词检索
        rerank_model_name: str = "BAAI/bge-reranker-v2-m3",  #重排模型名称，固定使用业界通用中文最优模型bge-reranker-v2-m3
        device: Optional[str] = None,  #指定模型运行设备，传None时：代码内部会自动检测本机是否有可用GPU，有就自动用cuda，没有降级CPU运行
        vector_top_k: int = 20,      # 向量初选数量
        bm25_top_k: int = 20,        # BM25初选数量
        rerank_top_k: int = 5,       # 两路结果融合之后，送入CrossEncoder重排模型，打分排序后只保留分数最高的5条给到LLM作为参考上下文。
        confidence_threshold: float = 0.5,   # 置信度阈值
    ):
        """
        初始化检索器
        :param bm25_index_path: BM25索引文件路径（由build_index.py生成）
        :param rerank_model_name: 重排序模型名称（sentence-transformers CrossEncoder）
        :param device: 运行设备（cuda/cpu），默认自动检测
        :param vector_top_k: 向量检索初始候选数
        :param bm25_top_k: BM25初始候选数
        :param rerank_top_k: 重排序后最终返回数
        :param confidence_threshold: 置信度阈值，低于此值认为检索不可靠
        """
        self.vector_top_k = vector_top_k  #类构造函数的属性赋值，把外部传入的参数存为实例自有变量
        self.bm25_top_k = bm25_top_k
        self.rerank_top_k = rerank_top_k
        self.confidence_threshold = confidence_threshold

        # 1. 加载 BM25 索引
        self.bm25 = None
        self.bm25_chunks = None  # 保存对应的 Document 列表，用于获取元数据
        if os.path.exists(bm25_index_path):
            try:
                with open(bm25_index_path, "rb") as f: #二进制只读打开
                    self.bm25, self.bm25_chunks = pickle.load(f)
                logger.info(f"BM25 索引加载成功，共 {len(self.bm25_chunks)} 个文档")
            except Exception as e:
                logger.error(f"加载 BM25 索引失败: {e}", exc_info=True)
                self.bm25 = None
        else:
            logger.warning(f"BM25 索引文件 {bm25_index_path} 不存在，将仅使用向量检索")

        # 2. 加载重排序模型
        try:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.reranker = CrossEncoder(rerank_model_name, device=device)
            logger.info(f"重排序模型加载成功，使用设备: {device}")
        except Exception as e:
            logger.error(f"重排序模型加载失败: {e}", exc_info=True)
            self.reranker = None

    #向量检索
    def vector_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        向量检索
        :param query: 查询文本
        :param top_k: 返回数量，默认使用 self.vector_top_k
        :return: 列表，每个元素包含 text, source, chunk_id, vector_score
        """
        if top_k is None:
            top_k = self.vector_top_k

        try:
            query_vector = Embedding_model.embed_query(query)
            results = client.search(
                collection_name=COLLECTION_NAME1,
                data=[query_vector],
                limit=top_k,
                output_fields=["text", "source", "chunk_id"]
            )
            if not results or not results[0]:
                logger.warning(f"向量检索无结果: {query}")
                return []

            hits = []   #初始化空列表，用来存放标准化后的检索结果字典。
            for hit in results[0]:  #循环遍历当前 query 匹配到的每一条向量命中结果。
                entity = hit.get("entity", {})
                hits.append({
                    "text": entity.get("text", ""),
                    "source": entity.get("source", "unknown"),
                    "chunk_id": entity.get("chunk_id", -1),
                    "vector_score": hit.get("distance", 0.0)  # COSINE 相似度
})                                                      #统一格式化每条数据，存入hits列表：
                                                        # text：法条原文，缺失填空字符串
                                                        # source：文档来源文件名，缺失填unknown
                                                        # chunk_id：文本块编号，缺失填-1
                                                        # vector_score：Milvus返回的distance距离；COSINE模式下distance越小代表越相似
            logger.debug(f"向量检索返回 {len(hits)} 条结果")
            return hits
        except Exception as e:
            logger.error(f"向量检索失败: {e}", exc_info=True)
            return []

    #2. BM25 检索
    def bm25_search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        BM25 关键词检索
        :param query: 查询文本
        :param top_k: 返回数量
        :return: 列表，包含 text, source, chunk_id, bm25_score
        """
        if self.bm25 is None:
            logger.warning("BM25 索引未加载，跳过 BM25 检索")
            return []

        if top_k is None:
            top_k = self.bm25_top_k

        try:
            # 中文分词
            tokens = list(jieba.cut(query))  #使用jieba对用户问句做中文分词，把一句话拆成词语列表，BM25只能基于词语计算匹配度。
            scores = self.bm25.get_scores(tokens)
            # 获取 top_k 索引
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            hits = []
            for idx in sorted_indices:
                doc = self.bm25_chunks[idx]
                hits.append({
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", idx),  # 若未存 chunk_id，使用索引
                    "bm25_score": float(scores[idx])
                })
            logger.debug(f"BM25 检索返回 {len(hits)} 条结果")
            return hits
        except Exception as e:
            logger.error(f"BM25 检索失败: {e}", exc_info=True)
            return []

    #3. RRF 融合 ----------
    @staticmethod
    def rrf_fusion(
        vector_hits: List[Dict],
        bm25_hits: List[Dict],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        使用 RRF（Reciprocal Rank Fusion）融合两个排序列表
        :param vector_hits: 向量检索结果（需包含 vector_score）
        :param bm25_hits: BM25 检索结果（需包含 bm25_score）
        :param k: RRF 常数
        :return: 融合后的列表，按 RRF 分数降序，每个元素包含 text, source, chunk_id, rrf_score
        """
        # 先按分数排序（降序）
        vector_ranked = sorted(vector_hits, key=lambda x: x.get("vector_score", 0.0), reverse=True)
        bm25_ranked = sorted(bm25_hits, key=lambda x: x.get("bm25_score", 0.0), reverse=True)

        # 构建文本到元数据的映射（合并两个来源）
        text_to_meta = {}
        for item in vector_hits + bm25_hits:
            text = item.get("text", "")
            if text:
                text_to_meta[text] = {
                    "source": item.get("source", "unknown"),
                    "chunk_id": item.get("chunk_id", -1)
                }

        # 计算 RRF 分数
        rrf_scores = {}
        for rank, item in enumerate(vector_ranked, start=1):
            text = item.get("text", "")
            if text:
                rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (k + rank)
        for rank, item in enumerate(bm25_ranked, start=1):
            text = item.get("text", "")
            if text:
                rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (k + rank)

        # 排序并组装结果
        sorted_texts = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        fused_results = []
        for text, score in sorted_texts:
            meta = text_to_meta.get(text, {})
            fused_results.append({
                "text": text,
                "source": meta.get("source", "unknown"),
                "chunk_id": meta.get("chunk_id", -1),
                "rrf_score": score
            })
        return fused_results

    #重排序（CrossEncoder）
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        使用 CrossEncoder 对候选片段进行重排序
        :param query: 原始查询
        :param candidates: 候选列表，每个元素需包含 'text'
        :param top_k: 最终返回数量，默认 self.rerank_top_k
        :return: 重排序后的列表，每个元素新增 'rerank_score'
        """
        if self.reranker is None:
            logger.warning("重排序模型未加载，返回原候选（截取 top_k）")
            if top_k is None:
                top_k = self.rerank_top_k
            return candidates[:top_k]

        if top_k is None:
            top_k = self.rerank_top_k

        if not candidates:
            return []

        try:
            texts = [item.get("text", "") for item in candidates if item.get("text")]
            if not texts:
                return []

            pairs = [[query, text] for text in texts]
            scores = self.reranker.predict(pairs)  # 返回分数列表
            # 将分数添加到候选字典
            for item, score in zip(candidates, scores):
                item["rerank_score"] = float(score)
            # 按分数降序排序
            sorted_candidates = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return sorted_candidates[:top_k]
        except Exception as e:
            logger.error(f"重排序失败: {e}", exc_info=True)
            # 失败时返回原候选（截取 top_k）
            return candidates[:top_k]

    #完整检索流程（混合 + 重排序）
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        return_all_scores: bool = False
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        执行完整的检索流程：向量+BM25 -> RRF融合 -> 重排序
        :param query: 用户查询
        :param top_k: 最终返回数量
        :param return_all_scores: 是否在结果中包含所有中间分数（调试用）
        :return: (检索结果列表, 置信度)
                 检索结果每个元素包含 text, source, chunk_id, rerank_score（或 vector_score 如果无重排序）
                 置信度为最高得分（归一化到0~1）
        """
        if top_k is None:
            top_k = self.rerank_top_k

        # Step 1: 多路召回
        vector_hits = self.vector_search(query, top_k=self.vector_top_k)
        bm25_hits = self.bm25_search(query, top_k=self.bm25_top_k)

        if not vector_hits and not bm25_hits:
            logger.warning(f"未检索到任何结果，查询: {query}")
            return [], 0.0

        # Step 2: RRF 融合（如果两种检索都有结果）
        if vector_hits and bm25_hits:
            fused = self.rrf_fusion(vector_hits, bm25_hits)
        elif vector_hits:
            # 只有向量结果，直接使用并模拟 RRF 分数（按 vector_score 降序）
            fused = sorted(vector_hits, key=lambda x: x.get("vector_score", 0.0), reverse=True)
            for item in fused:
                item["rrf_score"] = item.get("vector_score", 0.0)
        else:
            # 只有 BM25 结果
            fused = sorted(bm25_hits, key=lambda x: x.get("bm25_score", 0.0), reverse=True)
            for item in fused:
                item["rrf_score"] = item.get("bm25_score", 0.0)

        # Step 3: 重排序
        if self.reranker is not None:
            reranked = self.rerank(query, fused, top_k=top_k)
        else:
            reranked = fused[:top_k]

        # 计算置信度（取最高 rerank_score，若没有则使用 rrf_score 或 vector_score）
        confidence = 0.0
        if reranked:
            # 优先使用 rerank_score，其次 rrf_score，最后 vector_score
            score_key = None
            if "rerank_score" in reranked[0]:
                score_key = "rerank_score"
            elif "rrf_score" in reranked[0]:
                score_key = "rrf_score"
            elif "vector_score" in reranked[0]:
                score_key = "vector_score"
            elif "bm25_score" in reranked[0]:
                score_key = "bm25_score"

            if score_key:
                # 归一化（简单处理：假设分数范围在 0~1，但可能超出，所以 clamp）
                max_score = max(item.get(score_key, 0.0) for item in reranked)
                confidence = min(max_score, 1.0)  # 限制最大值

        # 若用户不需要中间分数，可清理
        if not return_all_scores:
            for item in reranked:
                # 保留最终得分（rerank_score 或 rrf_score），删除其他临时字段
                final_score = item.get("rerank_score", item.get("rrf_score", 0.0))
                item["score"] = final_score
                # 删除中间字段（可选）
                item.pop("vector_score", None)
                item.pop("bm25_score", None)
                item.pop("rrf_score", None)
                item.pop("rerank_score", None)
        else:
            # 保留所有分数供调试
            for item in reranked:
                item["score"] = item.get("rerank_score", item.get("rrf_score", 0.0))

        return reranked, confidence  #返回清洗后的法条片段列表  本次检索最高置信分数

    #快速检索（仅返回文本列表，用于工具调用）
    def retrieve_texts(self, query: str, top_k: int = None) -> List[str]:
        """仅返回文本列表，用于简单场景"""
        results, _ = self.retrieve(query, top_k)
        return [item["text"] for item in results]

# 全局单例（可选）
_retriever_instance = None

def get_retriever() -> Retriever:   #定义一个对外获取检索器实例的全局函数，标注返回值类型是你写的 Retriever 类对象。
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance