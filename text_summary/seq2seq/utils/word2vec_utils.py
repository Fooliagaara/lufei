from gensim.models.word2vec import Word2Vec


def load_embedding_matrix_from_model(wv_model_path):
    """
    从训练好的word2vec模型中加载词向量矩阵
    :param wv_model_path: 训练好的word2vec模型路径
    """
    # 加载训练好的word2vec模型
    wv_model = Word2Vec.load(wv_model_path)
    # 获取词向量矩阵
    embedding_matrix = wv_model.wv.vectors
    return embedding_matrix


def get_vocab_from_model(vocab_path, reverse_vocab_path):
    # 提取映射字典
    # vocab_path: word_to_id的⽂件存储路径
    # reverse_vocab_path: id_to_word的⽂件存储路径
    # 加载正向词典
    word_to_id, id_to_word = {}, {}
    with open(vocab_path, 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            w, v = line.strip('\n').split('\t')
            word_to_id[w] = int(v)

    # 加载反向词典
    with open(reverse_vocab_path, 'r', encoding='utf-8') as f2:
        for line in f2.readlines():
            v, w = line.strip('\n').split('\t')
            id_to_word[int(v)] = w

    return word_to_id, id_to_word


def save_vocab_as_txt(filename, word_to_id):
    # 保存字典
    # filename: ⽬标txt⽂件路径
    # word_to_id: 要保存的字典
    with open(filename, 'w', encoding='utf-8') as f:
        for k, v in word_to_id.items():
            f.write("{}\t{}\n".format(k, v))



            