import io
import numpy as np

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
        # print(tokens[0])
        # print(list(map(float, tokens[1:])))
    return data

def make_embed_pretrained():
    vocab_dict = {}
    cnt = 0
    with open('../data/vocab2.txt', 'r') as f:
        for line in f:
            line = line.rstrip()
            vocab_dict[line] = cnt
            cnt += 1
    # for key in vocab_dict:
    #     print(key)
    sorted_dict = sorted(vocab_dict.items(), key=lambda item: item[1], reverse=False)  # 结果是List
    # for key in sorted_dict:
    #     print(key[0])

    embed_dict = load_vectors('../data/wiki-news-300d-1M-subword.vec')
    
    with open('../data/embed_pre.txt', 'w') as f:
        # data_list = []
        # data_list.append(np.random.rand(300))
        # data_list.append(np.random.rand(300))
        f.write(str(list(np.random.rand(300))))
        f.write('\n')
        f.write(str(list(np.random.rand(300))))
        f.write('\n')
        for key in sorted_dict:
            key = key[0]
            print(key)
            if key == '<pad>' or key == '<unk>':
                continue
            if key not in embed_dict:
                f.write(str(list(np.random.rand(300))))
            else:
                f.write(str(embed_dict[key]))
            f.write('\n')


if __name__ == '__main__':
    make_embed_pretrained()






