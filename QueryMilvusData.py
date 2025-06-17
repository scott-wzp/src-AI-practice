"""
QueryMilvusData-
查询Milvus数据库中的数据并使用千问大模型生成RAG响应
Author: wzpym
Date: 2025/6/14
"""
from pymilvus import MilvusClient
from dashscope import TextEmbedding, Generation
import os
import json

def get_embedding(text):
    """获取文本的向量表示"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未设置 DASHSCOPE_API_KEY 环境变量")
    
    resp = TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v1,
        input=text,
        api_key=api_key
    )
    return resp.output["embeddings"][0]["embedding"]

def get_rag_response(query_text, context):
    """使用千问大模型生成RAG响应"""
    try:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("未设置 DASHSCOPE_API_KEY 环境变量")

        # 构建提示词
        prompt = f"""基于以下参考信息回答问题。如果参考信息不足以回答问题，请说明无法回答。

参考信息：
{context}

问题：{query_text}

请基于参考信息给出详细回答："""

        # 调用千问大模型
        response = Generation.call(
            model='qwen-max',
            prompt=prompt,
            api_key=api_key,
            temperature=0.7,
            max_tokens=2000
        )

        if response.status_code == 200:
            return response.output.text
        else:
            return f"生成回答时出错: {response.message}"

    except Exception as e:
        return f"调用千问API时出错: {str(e)}"

def show_collection_info():
    """显示集合的字段信息"""
    try:
        # 连接到 Milvus
        milvus_client = MilvusClient(
            uri="http://localhost:19530",
            user="root",
            password="Milvus123"
        )
        
        # 获取所有集合
        collections = milvus_client.list_collections()
        print("\n所有集合列表:")
        for collection in collections:
            print(f"- {collection}")
        print("-" * 50)
        
        collection_name = "my_rag_collection"
        
        # 检查集合是否存在
        if not milvus_client.has_collection(collection_name):
            print(f"错误：集合 {collection_name} 不存在")
            return
        
        # 获取集合描述信息
        collection_info = milvus_client.describe_collection(collection_name)
        
        print(f"\n集合 '{collection_name}' 的字段信息:")
        print("-" * 50)
        
        # 打印集合的基本信息
        print(f"集合名称: {collection_name}")
        print(f"集合描述: {collection_info}")
        print("-" * 50)
            
        # 获取集合统计信息
        stats = milvus_client.get_collection_stats(collection_name)
        print(f"\n集合统计信息: {stats}")
            
    except Exception as e:
        print(f"获取集合信息时出错: {str(e)}")
        # 打印更详细的错误信息
        import traceback
        print(traceback.format_exc())

def search_by_text(query_text, top_k=5):
    """根据文本内容搜索相似数据"""
    try:
        # 连接到 Milvus
        milvus_client = MilvusClient(
            uri="http://localhost:19530",
            user="root",
            password="Milvus123"
        )
        
        collection_name = "my_rag_collection"
        
        # 检查集合是否存在
        if not milvus_client.has_collection(collection_name):
            print(f"错误：集合 {collection_name} 不存在")
            return
        
        # 获取集合统计信息
        stats = milvus_client.get_collection_stats(collection_name)
        print(f"集合统计信息: {stats}")
        
        # 将查询文本转换为向量
        query_vector = get_embedding(query_text)
        
        # 执行向量搜索
        results = milvus_client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field="vector",
            search_params={
                "metric_type": "IP",
                "params": {"nprobe": 10}
            },
            limit=top_k,
            output_fields=["text"]
        )
        
        # 打印搜索结果并返回查询到的结果
        context = []
        print(f"\n查询文本: {query_text}")
        print(f"找到 {len(results[0])} 个相似结果:")
        for i, hit in enumerate(results[0], 1):
            print(f"\n结果 {i}:")
            print(f"相似度得分: {hit.score}")
            print(f"文本内容: {json.dumps(hit.entity.get('text'))}")
            context.append('\n'.join(hit.entity.get('text')))
        
        # 使用千问大模型生成RAG响应
        if context:
            print("\n正在生成RAG响应...")
            rag_response = get_rag_response(query_text, '\n\n'.join(context))
            print("\nRAG响应:")
            print("-" * 50)
            print(rag_response)
            print("-" * 50)
        else:
            print("未找到相关上下文信息")
  
    except Exception as e:
        print(f"搜索过程中出错: {str(e)}")
    return context

def main():
    while True:
        print("\n请选择操作：")
        print("1. 查看集合字段信息")
        print("2. 搜索相似内容并获取RAG响应")
        print("3. 退出")
        
        choice = input("请输入选项（1-3）: ")
        
        if choice == "1":
            show_collection_info()
        elif choice == "2":
            query_text = input("请输入要搜索的内容: ")
            search_by_text(query_text)
        elif choice == "3":
            print("程序退出")
            break
        else:
            print("无效的选项，请重新选择")

if __name__ == "__main__":
    main()