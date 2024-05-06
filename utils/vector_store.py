import abc
import pymilvus
import chromadb
import typing as t
import random

Embedding = t.List[float]

class SearchResult(t.TypedDict):
    id: str
    document: t.Dict
    score: t.Optional[float] = 0.0


class VectorStore(abc.ABC):
    def __init__(self, host: str, port: str, collection_name: str) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.collection_name = collection_name

    @abc.abstractmethod
    def insert(self, ids: t.List[str], documents: t.List[t.Dict], embeddings: t.List[Embedding]):
        pass

    @abc.abstractmethod
    def get(self, id: str):
        pass

    @abc.abstractmethod
    def search(self, embedding: Embedding, k: int = 5, metric_type: str = "COSINE") -> t.List[SearchResult]:
            pass

class MilvusVectorStore(VectorStore):
    def __init__(self, host: str, port: str, collection_name: str, timeout: int = 600, refresh_data: bool = False, refresh_index: bool = False) -> None:
        super().__init__(host, port, collection_name)
        self.client = pymilvus.MilvusClient(
            uri=f"http://{host}:{port}"
        )
        self.conn = pymilvus.connections.connect(
            host=host,
            port=port
        )
        self.collection = None
        self._setup_collection(collection_name, refresh_data, refresh_index)

    def _setup_collection(self, collection_name: str, refresh_data: bool, refresh_index: bool):
        fields = [
            pymilvus.FieldSchema(name="id", dtype=pymilvus.DataType.VARCHAR, is_primary=True, max_length=36),
            pymilvus.FieldSchema(name="embedding", dtype=pymilvus.DataType.FLOAT_VECTOR, dim=1536),
            pymilvus.FieldSchema(name="document", dtype=pymilvus.DataType.VARCHAR, max_length=65535),
        ]
        schema = pymilvus.CollectionSchema(fields, description="Document collection based on embeddings similarity")
        self.collection = pymilvus.Collection(collection_name, schema)
        self.collection.load()
        if refresh_data:
            self.collection.delete(expr="id > '0'")
        if refresh_index:
            self.collection.drop_index()
        self.collection.create_index(
            field_name="embedding", 
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 1024}
            }
        )

    def insert(self, ids: t.List[str], documents: t.List[t.Dict], embeddings: t.List[Embedding]):
        self.collection.insert(
            data=[
                {
                    "id": id,
                    "embedding": embedding,
                    "document": document
                }
                for id, document, embedding in zip(ids, documents, embeddings)
            ]
        )

    def search(self, embedding: Embedding, k: int = 5, metric_type: str = "COSINE") -> t.List[SearchResult]:
        result = self.collection.search(
            [embedding], 
            "embedding", 
            {"metric_type": metric_type}, 
            k, 
            output_fields=["id", "document"]
        )
        return [
            SearchResult(id=hit.entity.get("id"), score=hit.distance, document=hit.entity.get("document"))
            for hit in result[0]
        ]
    
    def get(self, id: str) -> SearchResult:
        results = self.collection.query(f"id == '{id}'", ["id", "document"])
        return SearchResult(id=results[0].get("id"), document=results[0].get("document")) if results else None
    
    @property
    def ids(self) -> t.List[str]:
        results = self.collection.query("", ["id"], limit=10000)
        return [hit.get("id") for hit in results]

    @property
    def documents(self) -> t.List[t.Dict]:
        results = self.collection.query("", ["document"], limit=1000)
        return [hit.get("document") for hit in results]


class ChromaDBVectorStore(VectorStore):
    def __init__(self, host: str, port: str, collection_name: str) -> None:
        super().__init__(host, port, collection_name)
        self.client = chromadb.HttpClient(host, port, ssl=False)
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={
                "hnsw:space": "cosine"
            }
        )

    def insert(self, ids: t.List[str], documents: t.List[t.Dict], embeddings: t.List[Embedding]):
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
        )

    def search(self, embedding: Embedding, k: int = 5, metric_type: str = "IP") -> t.List[SearchResult]:
        bulk_results = self.collection.search(
            query_embeddings=embedding,
            n_results=k, 
            include=["distances", "documents"]
        )
        embedding_result = zip(bulk_results['ids'][0], bulk_results['distances'][0], bulk_results['documents'][0])
        search_results = [
            SearchResult({
                "id": id,
                "score": distance,
                "document": document
            })
            for id, distance, document in embedding_result
        ]
        return search_results

    @property
    def ids(self, limit: int = 10000):
        return self.collection['ids'][:limit]
    
    @property
    def documents(self, limit: int = 10000):
        return self.collection['documents'][:limit]
    

def make_document_sampler(vector_store: VectorStore):
    used_ids = set()
    all_ids = list(enumerate(vector_store.ids))

    def sample(verbose: bool = False):
        trial = 0
        sampled_index, sampled_id = None, None
        while (sampled_id not in used_ids):
            trial += 1
            sampled_index, sampled_id = random.sample(all_ids, k=1)[0]
            if sampled_id is not None:
                used_ids.add(sampled_id)
        return sampled_index, sampled_id
        
    return sample
