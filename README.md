# SWEM (Simple Word-Embedding-based Methods)
Python implementation of SWEM

[\[1805\.09843\] Baseline Needs More Love: On Simple Word\-Embedding\-Based Models and Associated Pooling Mechanisms](https://arxiv.org/abs/1805.09843)


## Usage

```py
from gensim.models import KeyedVectors

from swem import MeCabTokenizer
from swem import SWEM

w2v_path = "/path/to/word_embedding.bin"
w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
tokenizer = MeCabTokenizer("-O wakati")

swem = SWEM(w2v, tokenizer)
```

```py
In []: text = "吾輩は猫である。名前はまだ無い。"

# SWEM-aver
In []: swem.average_pooling(text)
Out[]:
array([ 2.31595367e-01,  5.31529129e-01, -6.28219426e-01, -7.73212969e-01,
        5.56734562e-01,  5.50618172e-01,  7.96405852e-01,  1.65987098e+00,
[...]

# SWEM-max
In []: swem.max_pooling(text)
Out[]:
array([ 1.4522485e+00,  2.1016493e+00,  9.1187710e-01,  6.2075871e-01,
        2.7146432e+00,  2.6316767e+00,  2.3899646e+00,  3.0643713e+00,
[...]

# SWEM-concat
In []: swem.concat_average_max_pooling(text)
Out[]:
array([ 2.31595367e-01,  5.31529129e-01, -6.28219426e-01, -7.73212969e-01,
        5.56734562e-01,  5.50618172e-01,  7.96405852e-01,  1.65987098e+00,
[...]

# SWEM-hier
In []: swem.hierarchical_pooling(text, n=2)
Out[]:
array([ 1.08240175e+00,  1.80855095e+00,  2.49545574e-02,  3.06840777e-01,
        1.25868618e+00,  1.97042620e+00,  1.59599078e+00,  2.99531865e+00,
[...]
```

```py
In []: swem.max_pooling(text).shape
Out[]: (200,)

In []: swem.concat_average_max_pooling(text).shape
Out[]: (400,)
```
