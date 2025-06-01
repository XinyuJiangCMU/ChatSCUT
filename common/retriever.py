
import time
from pydantic.dataclasses import dataclass
from pydantic import Field
from langchain.embeddings import HuggingFaceEmbeddings
from vector_store import load_store_for_one_stage, load_store_for_two_stage


@dataclass
class SearchResult:
    time: float = 0.0

    knowledges: list = Field(default_factory=list)
    knowledge_scores: list = Field(default_factory=list)
    knowledge_names: list = Field(default_factory=list)
    knowledge_ids: list = Field(default_factory=list)

    knowledges_for_llm: str = ""
    names_or_abstracts: list = Field(default_factory=list)
    names_or_abstracts_scores: list = Field(default_factory=list)


class OneStageRetriever:
    def __init__(self, embedding_model: HuggingFaceEmbeddings):
        self.whole_store = load_store_for_one_stage(embedding_model=embedding_model)
        self.embedding_model=embedding_model

    def updatestore(self):
        print("已更新数据库")
        self.whole_store = load_store_for_one_stage(embedding_model=self.embedding_model)
        
    def search(
        self, query: str, topk_knowledge: int, knowledge_threshold=None
    ) -> SearchResult:
        t0 = time.process_time()

        # 搜索结果按照分数从大到小排序
        query_emb = self.embedding_model.embed_query(query)
        docs_with_scores = (
            self.whole_store.similarity_search_with_relevance_score_by_vector(
                embedding=query_emb,
                k=topk_knowledge,
                relevance_score_threshold=knowledge_threshold,
            )
        )
        if len(docs_with_scores) == 0:
            return SearchResult()

        t1 = time.process_time()

        return SearchResult(
            time=t1 - t0,
            knowledges=[d.metadata["knowledge"] for d, _ in docs_with_scores],
            knowledge_names=[d.metadata["file_name"] for d, _ in docs_with_scores],
            knowledge_ids=[d.metadata["id"] for d, _ in docs_with_scores],
            knowledge_scores=[s for _, s in docs_with_scores],
            # 输入到llm中的knowledges按照分数从小到大排序
            knowledges_for_llm="\n\n".join(
                [d.metadata["knowledge"] for d, _ in reversed(docs_with_scores)]
            ),
        )


class TwoStageRetriever:
    def __init__(self, embedding_model: HuggingFaceEmbeddings):
        self.names_store, self.abstracts_store, self.stores = load_store_for_two_stage(
            embedding_model=embedding_model
        )
        self.embedding_model=embedding_model
    def search_by_name(
        self,
        query: str,
        topk_name: int,
        topk_knowledge: int,
        knowledge_threshold=None,
    ) -> SearchResult:
        t0 = time.process_time()

        # 第一阶段，搜索结果按照分数从大到小排序
        query_emb = self.embedding_model.embed_query(query)
        names_with_scores = (
            self.names_store.similarity_search_with_relevance_score_by_vector(
                embedding=query_emb,
                k=topk_name,
            )
        )

        # 第二阶段
        docs_with_scores = []
        for name, _ in names_with_scores:
            docs_with_scores.extend(
                self.stores[
                    name.page_content
                ].similarity_search_with_relevance_score_by_vector(
                    embedding=query_emb,
                    k=topk_knowledge,
                    relevance_score_threshold=knowledge_threshold,
                )
            )
        if len(docs_with_scores) == 0:
            return SearchResult()

        t1 = time.process_time()

        # 搜索结果按照分数从大到小排序
        docs_with_scores = sorted(docs_with_scores, key=lambda x: -x[1])[
            0:topk_knowledge
        ]

        return SearchResult(
            time=t1 - t0,
            knowledges=[d.metadata["knowledge"] for d, _ in docs_with_scores],
            knowledge_names=[d.metadata["file_name"] for d, _ in docs_with_scores],
            knowledge_ids=[d.metadata["id"] for d, _ in docs_with_scores],
            knowledge_scores=[s for _, s in docs_with_scores],
            names_or_abstracts=[d.page_content for d, _ in names_with_scores],
            names_or_abstracts_scores=[s for _, s in names_with_scores],
            knowledges_for_llm="\n\n".join(
                [d.metadata["knowledge"] for d, _ in reversed(docs_with_scores)]
            ),
        )

    def search_by_abstract(
        self,
        query: str,
        topk_abstract: int,
        topk_knowledge: int,
        knowledge_threshold=None,
    ) -> SearchResult:
        t0 = time.process_time()

        # 第一阶段，搜索结果按照分数从大到小排序
        query_emb = self.embedding_model.embed_query(query)
        abstracts_with_scores = (
            self.abstracts_store.similarity_search_with_relevance_score_by_vector(
                embedding=query_emb,
                k=topk_abstract,
            )
        )

        # 第二阶段
        docs_with_scores = []
        for abstract, _ in abstracts_with_scores:
            docs_with_scores.extend(
                self.stores[
                    abstract.metadata["file_name"]
                ].similarity_search_with_relevance_score_by_vector(
                    embedding=query_emb,
                    k=topk_knowledge,
                    relevance_score_threshold=knowledge_threshold,
                )
            )
        if len(docs_with_scores) == 0:
            return SearchResult()

        t1 = time.process_time()

        # 搜索结果按照分数从大到小排序
        docs_with_scores = sorted(docs_with_scores, key=lambda x: -x[1])[
            0:topk_knowledge
        ]

        return SearchResult(
            time=t1 - t0,
            knowledges=[d.metadata["knowledge"] for d, _ in docs_with_scores],
            knowledge_names=[d.metadata["file_name"] for d, _ in docs_with_scores],
            knowledge_ids=[d.metadata["id"] for d, _ in docs_with_scores],
            knowledge_scores=[s for _, s in docs_with_scores],
            names_or_abstracts=[
                f"{d.metadata['file_name']}\n" + d.page_content
                for d, _ in abstracts_with_scores
            ],
            names_or_abstracts_scores=[s for _, s in abstracts_with_scores],
            knowledges_for_llm="\n\n".join(
                [d.metadata["knowledge"] for d, _ in reversed(docs_with_scores)]
            ),
        )
