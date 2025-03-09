import numpy as np

class Features(object):
    def __init__(self):
        self.dtype = np.float32

    def extract(self, str):
        bytes = [ord(c) for c in list(str)]
        h = np.bincount(bytes, minlength=256)

        # 构造特征向量：1 + 256 维
        h_norm = np.concatenate([
            [h.sum().astype(self.dtype)],
            h.astype(self.dtype).flatten() / h.sum().astype(self.dtype)  # 是做归一化处理,
            # 虽然 h.astype(self.dtype) 强制 h 变成了 float32，但是 h.sum() 是 int64 类型，在 NumPy 中，当 float32 除以 int64，结果会被提升为 float64
        ])
        # 这里要阐明的是h.sum()是为了得到字符串长度，为什么统计总数就可以得到字符串长度呢？
        # 因为在h中不是0就是每个字符出现的次数，这些次数加起来就是字符串长度

        return h_norm

#测试
if __name__ == '__main__':
    f = Features()
    t =f.extract('hello world')
    print(t.shape)
    print(t.dtype)
    print(f.extract('hello world'))