"""
LangchainTools-
组合langchainTools做操作，比如做计算和查询天气
Author: wzpym
Date: 2025/6/23
"""
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.tools import Tool
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi  # 导入通义千问Tongyi模型
from langchain.prompts import StringPromptTemplate
from langchain.llms.base import BaseLLM
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
import dashscope
import os
import re

# 从环境变量获取 API Keys
# 设置方法：
# Windows: set DASHSCOPE_API_KEY=your_dashscope_api_key
# Windows: set SERPAPI_API_KEY=your_serpapi_api_key
# Linux/Mac: export DASHSCOPE_API_KEY=your_dashscope_api_key
# Linux/Mac: export SERPAPI_API_KEY=your_serpapi_api_key

dashscope_api_key = os.environ.get('DASHSCOPE_API_KEY')
serpapi_api_key = os.environ.get('SERPAPI_API_KEY')


if not dashscope_api_key:
    print("警告：未设置 DASHSCOPE_API_KEY 环境变量")
if not serpapi_api_key:
    print("警告：未设置 SERPAPI_API_KEY 环境变量")

dashscope.api_key = dashscope_api_key

AGENT_TMPL = """按照给定的格式回答以下问题。你可以使用下面这些工具：

{tools}

回答时需要遵循以下用---括起来的格式：

---
Question: 我需要回答的问题
Thought: 回答这个上述我需要做些什么
Action: "{tool_names}" 中的一个工具名
Action Input: 选择这个工具所需要的输入
Observation: 选择这个工具返回的结果
...（这个 思考/行动/行动输入/观察 可以重复N次）
Thought: 我现在知道最终答案
Final Answer: 原始输入问题的最终答案
---

现在开始回答，记得在给出最终答案前，需要按照指定格式进行一步一步的推理。

Question: {input}
{agent_scratchpad}
"""
class CustomPromptTemplate(StringPromptTemplate):
    template: str  # 标准模板
    tools: List[Tool]  # 可使用工具集合

    def format(self, **kwargs) -> str:
        """
        按照定义的 template，将需要的值都填写进去。
        Returns:
            str: 填充好后的 template。
        """
        # 取出中间步骤并进行执行
        intermediate_steps = kwargs.pop("intermediate_steps")
        print('intermediate_steps=', intermediate_steps)
        print('='*30)
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # 记录下当前想法 => 赋值给agent_scratchpad
        kwargs["agent_scratchpad"] = thoughts
        # 枚举所有可使用的工具名+工具描述
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # 枚举所有的工具名称
        kwargs["tool_names"] = ", ".join(
            [tool.name for tool in self.tools]
        )
        cur_prompt = self.template.format(**kwargs)
        #print(cur_prompt)
        return cur_prompt

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        解析 llm 的输出，根据输出文本找到需要执行的决策。
        """
        # 如果句子中包含 Final Answer 则代表已经完成
        if "Final Answer:" in llm_output:  
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # 需要进行 AgentAction
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"  # 解析 action_input 和 action
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Agent执行
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )

# 加载 Tongyi 模型
llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=dashscope_api_key)  # 使用通义千问qwen-turbo模型

# 加载工具，直接传递 API Key
tools = load_tools(
    ["serpapi", "llm-math"], 
    llm=llm,
    serpapi_api_key=serpapi_api_key
)

# 用户定义的模板
agent_prompt = CustomPromptTemplate(
    template=AGENT_TMPL,
    tools=tools,
    input_variables=["input", "intermediate_steps"],
)

# Agent返回结果解析
output_parser = CustomOutputParser()

# 最常用的Chain, 由LLM + PromptTemplate组成
llm_chain = LLMChain(llm=llm, prompt=agent_prompt)

# 定义的工具名称
tool_names = [tool.name for tool in tools]

# 定义Agent = llm_chain + output_parser + tools_names
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)

# 定义Agent执行器 = Agent + Tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)

if __name__ == "__main__":
    print("智能助手已启动！")
    print("可用工具：")
    print("- serpapi（网络搜索）")
    print("- llm-math（数学计算）")
    print("输入 'quit' 或按 Ctrl+C 退出")
    print("-" * 50)
    
    # 主过程：可以一直提问下去，直到退出
    while True:
        try:
            user_input = input("请输入您的问题：")
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
                
            response = agent_executor.invoke({"input": user_input})
            print(f"答案：{response['output']}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"发生错误：{str(e)}")
            print("-" * 50)
