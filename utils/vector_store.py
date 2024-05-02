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

    def insert(self, ids: t.List[str], documents: t.List[t.Dict], embeddings: t.List[Embedding]):
        pass

    def get(self, id: str):
        pass

    def search(self, embedding: Embedding, k: int = 5, metric_type: str = "Cosine") -> t.List[SearchResult]:
        pass

    @property
    def ids(self):
        pass

    @property
    def documents(self):
        pass


class MilvusVectorStore(VectorStore):
    def __init__(self, host: str, port: str, collection_name: str, timeout: int = 600, refresh_index: bool = False) -> None:
        super().__init__(host, port, collection_name)
        self.client = pymilvus.MilvusClient(
            uri=f"http://{host}:{port}"
        )
        self.conn = pymilvus.connections.connect(
            host=host,
            port=port
        )
        fields = [
            pymilvus.FieldSchema(name="id", dtype=pymilvus.DataType.VARCHAR, is_primary=True, max_length=36),
            pymilvus.FieldSchema(name="embedding", dtype=pymilvus.DataType.FLOAT_VECTOR, dim=1536),
            pymilvus.FieldSchema(name="document", dtype=pymilvus.DataType.VARCHAR, max_length=65535),
        ]
        schema = pymilvus.CollectionSchema(fields, description="Document collection based on embeddings similarity")
        self.collection = pymilvus.Collection(collection_name, schema)
        if refresh_index:
            self.collection.release()
            self.collection.drop_index()
        self.collection.create_index(
            field_name="embedding", 
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 1024}
            }
        )
        self.collection.load()

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
        bulk_results = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param={
                "metric_type": metric_type
            },
            limit=k,
            output_fields=["id", "document"]
        )
        embedding_result = bulk_results[0]
        search_results = [
            SearchResult({
                "id": neighbour.get("id"),
                "score": neighbour.get("distance"),
                "document": neighbour.get("document")
            })
            for neighbour in embedding_result
        ]
        return search_results
    
    def get(self, id: str) -> SearchResult:
        search_results = [
            SearchResult({
                "id": r.get("id"),
                "document": r.get("document")
            })
            for r in self.collection.query(expr=f"id == '{id}'", output_fields=["id", "document"], limit=1000)
        ]
        if len(search_results) > 0:
            return search_results[0]
        return None
    
    @property
    def ids(self):
        result = [
            r.get("id")
            for r in self.collection.query(expr="", output_fields=["id"], limit=1000)
        ]
        return result
    
    @property
    def documents(self):
        result = [
            r.get("document")
            for r in self.collection.query(expr="", output_fields=["document"], limit=1000)
        ]
        return result 


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
    def ids(self):
        return self.collection['ids']
    
    @property
    def documents(self):
        return self.collection['documents']
    

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
