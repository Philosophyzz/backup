from llama_index.core import SimpleDirectoryReader

# 加载文档
documents = SimpleDirectoryReader(r'D:\sxr\Llama\elearn\教材\1').load_data()

# 使文本生成node
from llama_index.core.node_parser import SimpleNodeParser

parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# 使用本地嵌入类大模型进行嵌入
from llama_index.embeddings import LocalEmbeddingModel

# 设置本地嵌入模型的路径
embedding_model_path = "D:\\models\\bge-large-zh-v1.5"
embedding_model = LocalEmbeddingModel(model_path=embedding_model_path)

# 使用嵌入模型创建向量存储索引
from llama_index.core import GPTVectorStoreIndex

index = GPTVectorStoreIndex(nodes, embedding_model=embedding_model)

# 持久化索引
index.storage_context.persist(persist_dir="./elearn_index")

# 从存储加载索引
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./elearn_index")
index = load_index_from_storage(storage_context)
