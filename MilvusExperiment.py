"""
MilvusExperiment-
Milvus数据库的第一次搭建及尝试调用
Author: wzpym
Date: 2025/6/14
"""
from glob import glob
from dashscope import TextEmbedding
import os
from pymilvus import MilvusClient
from tqdm import tqdm

# 检查数据源目录
data_dir = "D:/milvus/en/faq"
print(f"数据源目录是否存在: {os.path.exists(data_dir)}")
if os.path.exists(data_dir):
    print(f"目录中的文件: {os.listdir(data_dir)}")

text_lines = []
try:
    for file_path in glob("D:/milvus/en/faq/*.md", recursive=True):
        print(f"正在读取文件: {file_path}")
        with open(file_path, "r") as file:
            file_text = file.read()
        text_lines += file_text.split("# ")
    print(f"成功读取 {len(text_lines)} 行文本")
except Exception as e:
    print(f"读取文件时出错: {str(e)}")

# 检查是否有文本数据
if not text_lines:
    print("警告: 没有读取到任何文本数据")
    exit(1)

"""建立一个客户端，进行文本的embedding操作"""
api_key=os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    print("错误: 未设置 DASHSCOPE_API_KEY 环境变量")
    exit(1)

#定义一个embedding的方法
def get_embedding(text):
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v1,
        input=text,
        api_key=api_key
    )
    return resp.output["embeddings"][0]["embedding"]

# 使用示例
embedding = get_embedding("This is a test")
embedding_dim = len(embedding)
print(f"Embedding维度: {embedding_dim}")

# Connect to Milvus server
milvus_client = MilvusClient(
    uri="http://localhost:19530",
    user="root",
    password="Milvus123"
)

collection_name = "my_rag_collection"
if milvus_client.has_collection(collection_name):
    print(f"删除已存在的集合: {collection_name}")
    milvus_client.drop_collection(collection_name)

print(f"创建新集合: {collection_name}")
milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",
    consistency_level="Strong",
)

#插入数据
data = []
print("开始生成embeddings...")
for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
    try:
        data.append({"id": i, "vector": get_embedding(line), "text": line})
    except Exception as e:
        print(f"生成第 {i} 个embedding时出错: {str(e)}")

print(f"准备插入 {len(data)} 条数据")
if data:
    try:
        milvus_client.insert(collection_name=collection_name, data=data)
        print("数据插入完成")
        
        # 确保数据被持久化
        print("正在持久化数据...")
        milvus_client.flush(collection_name)
        print("数据持久化完成")
        
        # 验证数据是否成功插入
        stats = milvus_client.get_collection_stats(collection_name)
        print(f"集合统计信息: {stats}")
    except Exception as e:
        print(f"插入数据时出错: {str(e)}")
else:
    print("没有数据需要插入")