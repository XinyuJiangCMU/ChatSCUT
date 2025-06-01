from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import json
from rich.progress import track
from custom_faiss import Faiss

from config import huggingface_embeddings_config, PROJ_TOP_DIR
from embedding import load_huggingface_embedding



def __new_relevance_score_fn(distance: float) -> float:
    # 将余弦相似度从[-1, 1]映射至[0, 1]中
    return 0.5 + 0.5 * distance


def __load_store(store_path: str, embedding_model: HuggingFaceEmbeddings) -> Faiss:
    #print("调用__load_store=======================================")

    return Faiss.load_local(
        folder_path=store_path,
        embeddings=embedding_model,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        relevance_score_fn=__new_relevance_score_fn,
        allow_dangerous_deserialization=True
    )


def load_store_for_one_stage(embedding_model: HuggingFaceEmbeddings) -> Faiss:
    #print("调用load_store_for_one_stage=======================================")
    embedding_name = embedding_model.model_name.split("/")[-1]
    store_path = os.path.join(PROJ_TOP_DIR, "vector_stores", embedding_name, "0_whole")
    #print("调用__load_store=======================================")
    whole_store = __load_store(
        store_path,
        embedding_model=embedding_model,
    )
    return whole_store


def load_store_for_two_stage(
    embedding_model: HuggingFaceEmbeddings,
):
    embedding_name = embedding_model.model_name.split("/")[-1]
    names_store_path = os.path.join(
        PROJ_TOP_DIR, "vector_stores", embedding_name, "1_names"
    )
    abstracts_store_path = os.path.join(
        PROJ_TOP_DIR, "vector_stores", embedding_name, "2_abstracts"
    )

    names_store = __load_store(
        store_path=names_store_path,
        embedding_model=embedding_model,
    )

    abstracts_store = __load_store(
        store_path=abstracts_store_path, embedding_model=embedding_model
    )

    names = os.listdir(os.path.join(PROJ_TOP_DIR, "vector_stores", embedding_name))
    names.remove("0_whole")
    names.remove("1_names")
    names.remove("2_abstracts")
    stores = {
        name: __load_store(
            os.path.join(PROJ_TOP_DIR, "vector_stores", embedding_name, name),
            embedding_model=embedding_model,
        )
        for name in names
    }

    return names_store, abstracts_store, stores


def build_vectorstores_per_name(
    json_dir: str,
    store_top_path: str,
    embedding_model: HuggingFaceEmbeddings,
) -> None:
    embedding_name = embedding_model.model_name.split("/")[-1]
    max_len = huggingface_embeddings_config[embedding_name]["max_len"]

    file_names = os.listdir(json_dir)

    store_names = []
    abstracts = []
    abstract_metadatas = []

    deduplicated_texts = []
    for file_name in track(
        file_names, description=f"{embedding_name} build_vectorstores_per_name"
    ):
        json_path = os.path.join(json_dir, file_name)
        with open(json_path, "r", errors='ignore') as f:
            #print(json_path) 
            data = json.load(fp=f)

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n", "。", "！", "？", "；", "，", ""],
            chunk_size=max_len,
            chunk_overlap=0,
            length_function=len,
        )
        texts_process = []
        metadatas = []

        for id, t in data["knowledges"].items():
            if t in deduplicated_texts:
                continue
            else:
                deduplicated_texts.append(t)
                if len(t) > max_len:
                    splits = text_splitter.split_text(text=t)
                    texts_process.extend(splits)
                    metadatas.extend(
                        [
                            {
                                "file_name": file_name.split(".")[0],
                                "id": id,
                                "knowledge": t,
                            }
                            for _ in splits
                        ]
                    )
                else:
                    texts_process.append(t)
                    metadatas.append(
                        {"file_name": file_name.split(".")[0], "id": id, "knowledge": t}
                    )
        if len(texts_process) > 0:
            store_name = file_name.split(".")[0]
            store_names.append(file_name.split(".")[0])

            abstracts.append(data["abstract"])
            abstract_metadatas.append({"file_name": file_name.split(".")[0]})

            txt_vecstore = Faiss.from_texts(
                texts=texts_process,
                embedding=embedding_model,
                # 距离度量使用余弦相似度
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                relevance_score_fn=__new_relevance_score_fn,
                metadatas=metadatas,
            )
            store_path = os.path.join(
                store_top_path,
                f"{embedding_name}/{store_name}",
            )
            txt_vecstore.save_local(store_path)

    names_vecstore = Faiss.from_texts(
        texts=store_names,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    names_vecstore.save_local(os.path.join(store_top_path, f"{embedding_name}/1_names"))

    abstracts_vecstore = Faiss.from_texts(
        texts=abstracts,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        metadatas=abstract_metadatas,
    )
    abstracts_vecstore.save_local(
        os.path.join(store_top_path, f"{embedding_name}/2_abstracts")
    )


def build_whole_vectorstore(
    json_dir: str, store_top_path: str, embedding_model: HuggingFaceEmbeddings
) -> None:
    embedding_name = embedding_model.model_name.split("/")[-1]
    max_len = huggingface_embeddings_config[embedding_name]["max_len"]

    file_names = os.listdir(json_dir)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "。", "！", "？", "；", "，", ""],
        chunk_size=max_len,
        chunk_overlap=0,
        length_function=len,
    )
    whole_texts = []
    metadatas = []
    deduplicated_texts = []

    vecstore_path = os.path.join(store_top_path, f"{embedding_name}/0_whole")

    # 检查向量库是否存在并尝试加载，添加了 allow_dangerous_deserialization=True
    if os.path.exists(vecstore_path):
        new_db = Faiss.load_local(vecstore_path, embedding_model, allow_dangerous_deserialization=True)
        print("Loaded existing vector store.")
    else:
        new_db = None
        print("No existing vector store found. A new one will be created.")

    # 处理新的文本
    for file_name in track(
        file_names, description=f"{embedding_name} build_whole_vectorstore"
    ):
        json_path = os.path.join(json_dir, file_name)
        with open(json_path, "r", errors='ignore') as f:
            data = json.load(f)
        for id, t in data["knowledges"].items():
            if t in deduplicated_texts:
                continue
            deduplicated_texts.append(t)
            if len(t) > max_len:
                splits = text_splitter.split_text(t)
                whole_texts.extend(splits)
                metadatas.extend(
                    [{"file_name": file_name.split(".")[0], "id": id, "knowledge": t} for _ in splits]
                )
            else:
                whole_texts.append(t)
                metadatas.append({"file_name": file_name.split(".")[0], "id": id, "knowledge": t})

    # 如果存在，合并新文本到现有库
    if new_db:
        temp_db =Faiss.from_texts(
                texts=whole_texts,
                embedding=embedding_model,
                # 距离度量使用余弦相似度
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                relevance_score_fn=__new_relevance_score_fn,
                metadatas=metadatas,
            )
        new_db.merge_from(temp_db)
    else:
        new_db = Faiss.from_texts(
                texts=whole_texts,
                embedding=embedding_model,
                # 距离度量使用余弦相似度
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                relevance_score_fn=__new_relevance_score_fn,
                metadatas=metadatas,
            )
    # 保存更新的索引
    new_db.save_local(vecstore_path)
    print("Vector store updated or created and saved.")


if __name__ == "__main__":
    from embedding import load_huggingface_embedding
    from config import huggingface_embeddings_config
    from config import PROJ_TOP_DIR

    json_dir = os.path.join(PROJ_TOP_DIR, "docs", "json")
    store_top_path = os.path.join(PROJ_TOP_DIR, "vector_stores")
    for k, _ in huggingface_embeddings_config.items():
        embedding_model = load_huggingface_embedding(k, device=0)
        build_vectorstores_per_name(
            json_dir=json_dir,
            store_top_path=store_top_path,
            embedding_model=embedding_model,
        )
        build_whole_vectorstore(
            json_dir=json_dir,
            store_top_path=store_top_path,
            embedding_model=embedding_model,
        )
