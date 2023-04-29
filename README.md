# layoutlmv3_zh
layoutlmv3 在中文文档上的应用

# 安装环境
```
conda create --name lv3 python=3.9 -y
conda activate lv3
pip install -r requirements.txt
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

```


# 遇到的一些问题
1. funsd的数据集和xfunsd的数据集中，每一个图片的对应的对应的word个数不一样，xfunsd中汉字明显比较多，出现了多于512的情况，xfunsd对应的微软代码的处理情况有换行处理，需要看一下transform是否有相应的处理。
2. 发现https://huggingface.co/datasets/ArneBinder/xfund/blob/main/xfund.py 提供的xfund的解析代码，不适用于汉字，里面的一个token是"账单寄送方式:'"还是需要进行额外的处理。


# 日志
## 2023-04-29 
publayout的数据集基于英文文档，目前没有中文文档，标注大概140份的文档，在中文的预训练模型上进行训练。

colab 训练命令
```
!/usr/local/envs/layoutlmv3/bin/python3 examples/object_detection/train_net.py   --config-file /content/layoutlmv3/examples/object_detection/cascade_layoutlmv3.yaml   --num-gpus 1 MODEL.WEIGHTS /content/layoutlmv3-base-chinese/pytorch_model.bin  OUTPUT_DIR output PUBLAYNET_DATA_DIR_TRAIN /content/drive/MyDrive/layoutlmv3/data/images PUBLAYNET_DATA_DIR_TEST /content/drive/MyDrive/layoutlmv3/data/images
```