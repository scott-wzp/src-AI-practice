from gensim.models import Word2Vec
import numpy as np

def analyze_similar_words(model_path, target_word, top_n=20):
    # 加载模型
    model = Word2Vec.load(model_path)
    
    if target_word not in model.wv:
        print(f"词 '{target_word}' 不在词表中")
        return
    
    # 获取相似词
    similar_words = model.wv.most_similar(target_word, topn=top_n)
    
    print(f"\n与'{target_word}'最相似的{top_n}个词：")
    print("-" * 40)
    print("词语\t\t相似度")
    print("-" * 40)
    for word, similarity in similar_words:
        print(f"{word}\t\t{similarity:.4f}")
    
    # 分析相似词的类别
    print("\n相似词分析：")
    print("-" * 40)
    
    # 按相似度分组
    high_sim = [w for w, s in similar_words if s > 0.6]
    mid_sim = [w for w, s in similar_words if 0.5 <= s <= 0.6]
    low_sim = [w for w, s in similar_words if s < 0.5]
    
    if high_sim:
        print("\n高相似度词（相似度 > 0.6）：")
        print(", ".join(high_sim))
    
    if mid_sim:
        print("\n中等相似度词（0.5 <= 相似度 <= 0.6）：")
        print(", ".join(mid_sim))
    
    if low_sim:
        print("\n低相似度词（相似度 < 0.5）：")
        print(", ".join(low_sim))

def analogy(model_path, positive, negative, top_n=3):
    model = Word2Vec.load(model_path)
    print(f"\n类比分析：{' + '.join(positive)} - {' - '.join(negative)} ≈ ?")
    print("-" * 40)
    print("词语\t\t相似度")
    print("-" * 40)
    try:
        results = model.wv.most_similar(positive=positive, negative=negative, topn=top_n)
        for word, score in results:
            print(f"{word}\t\t{score:.4f}")
    except KeyError as e:
        print(f"词表中缺少：{e}")

if __name__ == "__main__":
    model_path = "three_kingdoms_word2vec.model"
    # 曹操+刘备-张飞的类比分析
    analogy(model_path, positive=["曹操", "刘备"], negative=["张飞"]) 