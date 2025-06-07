"""
App-
主入口，便于提供访问
Author: wzpym
Date: 2025/6/7
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from QAProcessor import QAProcessor
from PyPDF2 import PdfReader
import os

# 初始化Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 创建问答处理器实例
qa_processor = QAProcessor()

@app.route('/')
def index():
    """渲染主页"""
    return render_template('index.html')

@app.route('/api/answer', methods=['POST'])
def answer_question():
    """
    问答接口
    接收JSON格式: {"question": "你的问题"}
    返回JSON格式: {
        "answer": "回答内容",
        "sources": [
            {
                "page": "页码",
                "content": "相关内容"
            }
        ],
        "status": "success/error"
    }
    """
    try:
        # 检查请求头
        if not request.is_json:
            return jsonify({"answer": "请求必须是JSON格式", "status": "error"}), 400

        # 获取JSON数据
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"answer": "无效的请求格式，需要包含question字段", "status": "error"}), 400

        question = data['question']
        
        # 检查PDF文件是否存在
        pdf_path = './浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf'
        if not os.path.exists(pdf_path):
            return jsonify({"answer": f"PDF文件不存在: {pdf_path}", "status": "error"}), 404

        try:
            result = qa_processor.process_question(question)
            return jsonify({
                "answer": result["answer"],
                "sources": result["sources"],
                "status": "success"
            })
        except Exception as e:
            print(f"处理问题时出错: {str(e)}")
            return jsonify({"answer": f"处理问题时出错: {str(e)}", "status": "error"}), 500

    except Exception as e:
        print(f"请求处理出错: {str(e)}")
        return jsonify({
            "answer": f"请求处理出错: {str(e)}",
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # 启动开发服务器
    app.run(host='0.0.0.0', port=5000, debug=True)
