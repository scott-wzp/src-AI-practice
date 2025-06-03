import jieba
import gensim
from gensim.models import Word2Vec
import re

def preprocess_text(text):
    # 移除标点符号和特殊字符
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 使用jieba进行分词
    words = jieba.lcut(text)
    return words

def train_word2vec(sentences, output_file):
    # 训练Word2Vec模型
    model = Word2Vec(sentences,
                    vector_size=100,  # 词向量维度
                    window=5,         # 上下文窗口大小
                    min_count=5,      # 词频阈值
                    workers=4,        # 并行线程数
                    sg=1,            # 使用Skip-gram模型
                    epochs=10)        # 训练轮数
    
    # 保存模型
    model.save(output_file)
    return model

def main():
    # 读取三国演义文本
    with open('three_kingdoms.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 预处理文本
    words = preprocess_text(text)
    
    # 将文本分割成句子（每100个词为一个句子）
    sentences = []
    sentence = []
    for word in words:
        sentence.append(word)
        if len(sentence) >= 100:
            sentences.append(sentence)
            sentence = []
    if sentence:  # 添加最后一个不完整的句子
        sentences.append(sentence)
    
    # 训练Word2Vec模型
    model = train_word2vec(sentences, 'three_kingdoms_word2vec.model')
    
    # 测试模型
    print("\n测试一些词语的相似词：")
    test_words = ['曹操', '刘备', '孙权', '关羽', '诸葛亮']
    for word in test_words:
        if word in model.wv:
            print(f"\n与'{word}'最相似的词：")
            similar_words = model.wv.most_similar(word, topn=5)
            for similar_word, similarity in similar_words:
                print(f"{similar_word}: {similarity:.4f}")
        else:
            print(f"\n'{word}'不在词表中")

if __name__ == "__main__":
    main() 