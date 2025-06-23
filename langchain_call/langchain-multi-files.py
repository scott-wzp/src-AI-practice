#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pprint
import urllib.parse
import json5
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os


# 步骤 1：添加自定义工具
@register_tool('my_image_gen')
class MyImageGen(BaseTool):
    # `description` 用于告诉智能体该工具的功能。
    description = 'AI 绘画（图像生成）服务，输入文本描述，返回基于文本信息绘制的图像 URL。'
    # `parameters` 告诉智能体该工具有哪些输入参数。
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': '期望的图像内容的详细描述',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        # `params` 是由 LLM 智能体生成的参数。
        prompt = json5.loads(params)['prompt']
        prompt = urllib.parse.quote(prompt)
        return json5.dumps(
            {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'},
            ensure_ascii=False)

@register_tool('document_retriever')
class DocumentRetriever(BaseTool):
    description = '从文档库中检索相关文档内容。'
    parameters = [{
        'name': 'query',
        'type': 'string',
        'description': '检索查询文本',
        'required': True
    }]

    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever

    def call(self, params: str, **kwargs) -> str:
        query = json5.loads(params)['query']
        docs = self.retriever.get_relevant_documents(query)
        result = []
        for i, doc in enumerate(docs):
            result.append(f"文档片段 {i+1}:\n内容: {doc.page_content}\n元数据: {doc.metadata}")
        return json5.dumps({'retrieved_docs': result}, ensure_ascii=False)

# 步骤 2：配置您所使用的 LLM。
print("\n===== Checking Environment Variables =====")
print("DASHSCOPE_API_KEY present:", bool(os.getenv('DASHSCOPE_API_KEY')))
print("=======================================\n")

llm_cfg = {
    # 使用 DashScope 提供的模型服务：
    'model': 'qwen-max',
    'model_server': 'dashscope',
    'api_key': os.getenv('DASHSCOPE_API_KEY'),  # 从环境变量获取API Key
    'generate_cfg': {
        'top_p': 0.8
    }
}

'''
llm_cfg = {
    # 使用 DashScope 提供的模型服务：
    'model': 'deepseek-v3',
    'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'api_key': os.getenv('DASHSCOPE_API_KEY'),  # 从环境变量获取API Key
    'generate_cfg': {
        'top_p': 0.8
    }
}
'''
# 步骤 3：创建一个智能体。这里我们以 `Assistant` 智能体为例，它能够使用工具并读取文件。
system_instruction = '''你是一个乐于助人的AI助手。
在收到用户的请求后，你应该：
- 首先绘制一幅图像，得到图像的url，
- 然后运行代码`request.get`以下载该图像的url，
- 最后从给定的文档中选择一个图像操作进行图像处理。
用 `plt.show()` 展示图像。
你总是用中文回复用户。'''

# 获取文件夹下所有文件
file_dir = os.path.join('./', 'docs')
print("\n===== Checking Document Directory =====")
print(f"Document directory exists: {os.path.exists(file_dir)}")
print(f"Document directory path: {os.path.abspath(file_dir)}")
print("=====================================\n")

files = []
if os.path.exists(file_dir):
    # 遍历目录下的所有文件
    for file in os.listdir(file_dir):
        file_path = os.path.join(file_dir, file)
        if os.path.isfile(file_path):  # 确保是文件而不是目录
            files.append(file_path)
print('files=', files)

# 1. 加载并分割文档
documents = []
print("\n===== 开始加载文档 =====")
for file_path in files:
    if file_path.endswith('.pdf'):
        print(f"\n处理PDF文件: {file_path}")
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            print(f"成功加载文档，页数: {len(docs)}")
            print(f"第一页内容预览: {docs[0].page_content[:200] if docs else '无内容'}...")
            documents.extend(docs)
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {str(e)}")
    elif file_path.endswith('.txt'):
        print(f"\n处理TXT文件: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"文件内容预览: {content[:200]}...")
                documents.append(Document(page_content=content, metadata={"source": file_path}))
            print(f"成功加载TXT文件")
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {str(e)}")

print(f"\n总共加载的文档数量: {len(documents)}")
if documents:
    print(f"第一个文档预览: {documents[0].page_content[:200]}...")
print("===== 文档加载完成 =====\n")

# 2. 将文档分割成小块
print("\n===== 开始分割文档 =====")
text_splitter = CharacterTextSplitter(
    chunk_size=1000,  # 每个块的大小
    chunk_overlap=200  # 块之间的重叠部分
)
texts = text_splitter.split_documents(documents)
print(f"分割后的文本块数量: {len(texts)}")
if texts:
    print(f"第一个文本块预览: {texts[0].page_content[:200]}...")
print("===== 文档分割完成 =====\n")

# 3. 创建嵌入向量和向量存储
print("\n===== 创建向量存储 =====")
try:
    print("正在创建嵌入模型...")
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.getenv('DASHSCOPE_API_KEY')
    )
    print("成功创建嵌入模型")
    
    print("正在创建向量存储...")
    vectorstore = Chroma.from_documents(texts, embeddings)
    print("成功创建向量存储")
    
    # 验证向量存储
    print("\n验证向量存储:")
    print(f"向量存储中的文档数量: {len(vectorstore.get()['ids']) if hasattr(vectorstore, 'get') else '无法获取'}")
    print(f"向量存储类型: {type(vectorstore)}")
except Exception as e:
    print(f"创建向量存储时出错: {str(e)}")
    import traceback
    print(traceback.format_exc())
    raise
print("===== 向量存储创建完成 =====\n")

# 4. 创建检索器
retriever = vectorstore.as_retriever()
print("\n检索器创建完成")

# 5. 创建检索工具
retriever_tool = DocumentRetriever(retriever)

# 6. 创建带有检索器的 Assistant
tools = ['my_image_gen', 'code_interpreter', retriever_tool]
bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools,
                files=files)

# 添加调试信息
print("\n===== Debug Info =====")
print("retriever_tool:", retriever_tool)
print("retriever_tool.retriever:", getattr(retriever_tool, 'retriever', None))
print("bot.function_list:", getattr(bot, 'function_list', None))
print("bot.__dict__:", bot.__dict__)
print("=====================\n")

# 步骤 4：作为聊天机器人运行智能体。
print("\n===== 启动聊天机器人 =====")
messages = []  # 这里储存聊天历史。
query = "介绍下雇主责任险"
print(f"初始查询: {query}")
# 将用户请求添加到聊天历史。
messages.append({'role': 'user', 'content': query})
response = []
current_index = 0
for response in bot.run(messages=messages):
    print("\n[DEBUG] response内容：", response)
    if current_index == 0:
        import ipdb; ipdb.set_trace()
        # 尝试获取并打印召回的文档内容
        print("\n===== 召回的文档内容 =====")
        try:
            # 直接使用 retriever 进行检索
            print(f"执行检索，查询: {query}")
            print("\n===== 检索器信息 =====")
            print(f"retriever类型: {type(retriever)}")
            print(f"vectorstore类型: {type(vectorstore)}")
            print(f"vectorstore中的文档数量: {len(vectorstore.get()['ids']) if hasattr(vectorstore, 'get') else '无法获取'}")
            print("=====================\n")
            
            retrieved_docs = retriever.get_relevant_documents(query)
            print(f"检索到的文档数量: {len(retrieved_docs) if retrieved_docs else 0}")
            
            if retrieved_docs:
                print("\n===== 检索到的文档内容 =====")
                for i, doc in enumerate(retrieved_docs):
                    print(f"\n文档片段 {i+1}:")
                    print(f"内容: {doc.page_content[:200]}...")  # 只打印前200个字符
                    print(f"元数据: {doc.metadata}")
                print("===========================\n")
            else:
                print("没有召回任何文档内容")
                print("\n===== 调试向量存储 =====")
                try:
                    # 尝试直接使用向量存储进行相似度搜索
                    results = vectorstore.similarity_search(query, k=3)
                    print(f"直接使用向量存储搜索到的文档数量: {len(results)}")
                    if results:
                        print("\n直接搜索到的文档内容:")
                        for i, doc in enumerate(results):
                            print(f"\n文档 {i+1}:")
                            print(f"内容: {doc.page_content[:200]}...")
                            print(f"元数据: {doc.metadata}")
                except Exception as e:
                    print(f"直接搜索时出错: {str(e)}")
                print("========================\n")
        except Exception as e:
            print(f"检索文档时出错: {str(e)}")
            print("\n===== 错误详情 =====")
            import traceback
            print(traceback.format_exc())
            print("===================\n")
        #break

    current_response = response[0]['content'][current_index:]
    current_index = len(response[0]['content'])
    print(current_response, end='')
# 将机器人的回应添加到聊天历史。
#messages.extend(response)

import ipdb; ipdb.set_trace()

