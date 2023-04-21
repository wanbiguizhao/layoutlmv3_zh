# layoutlmv3_zh
layoutlmv3 在中文文档上的应用

# 安装环境
```
conda create -n lv3 python=3.10
conda activate lv3
## 下载transform库，方便本地调试
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
```


# 遇到的一些问题
1. funsd的数据集和xfunsd的数据集中，每一个图片的对应的对应的word个数不一样，xfunsd中汉字明显比较多，出现了多于512的情况，xfunsd对应的微软代码的处理情况有换行处理，需要看一下transform是否有相应的处理。
2. 