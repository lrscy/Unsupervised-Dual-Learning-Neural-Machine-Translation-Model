mkdir ~/codes/
mkdir ~/codes/data/
mkdir ~/codes/data/word2vec/
mkdir ~/codes/data/word2vec/chinese/
mkdir ~/codes/data/word2vec/english/
conda install tensorflow-gpu keras-gpu -y
cd ~/codes/data/word2vec/chinese/
curl -o cc.zh.300.vec.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz
gzip -d cc.zh.300.vec.gz &
cd ~/codes/data/word2vec/english/
curl -o cc.en.300.vec.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
gzip -d cc.en.300.vec.gz &
