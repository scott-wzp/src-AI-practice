"""
QAProcessor-处理问答信息

Author: wzpym
Date: 2025/6/7
"""
from PyPDF2 import PdfReader
from QASystem import QASystem

qa = QASystem()
class QAProcessor:
    def __init__(self):
        # 读取PDF文件
        #pdf_reader = PdfReader('浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf')
        # 提取文本和页码信息

        #text, page_numbers = qa.extract_text_with_page_numbers(pdf_reader)
        # 这里可以初始化模型或加载知识库
        self.knowledge_base = {
            "你好": "你好！我是智能问答助手。",
            "你是谁": "我是一个基于Python Flask构建的问答机器人。",
            "时间": self.get_current_time,
            "日期": self.get_current_date
        }


    def get_current_time(self):
        from datetime import datetime
        return f"当前时间是: {datetime.now().strftime('%H:%M:%S')}"

    def get_current_date(self):
        from datetime import datetime
        return f"今天是: {datetime.now().strftime('%Y-%m-%d')}"

    def process_question(self, question):
        # 简单的关键词匹配
        question = question.strip().lower()
        response = qa.prepare_question(question)
        if response:
            return response
        return {
            "answer": "抱歉，我暂时无法回答这个问题。请尝试其他问题。",
            "sources": []
        }
