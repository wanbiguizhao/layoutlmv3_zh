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

colab 16G的T4 GPU，目前只能训练batch_size 为2的情况，GPU占用13.1G，需要看一下混合精度训练和多卡多机训练。
AMP.enable 本身就是fp16混合精度计算了。

支持断点训练，现在的问题是，为什么现在的训练一半输出的是.pth 但是官方提供的确实一个bin文件，没有办法和publayout的模型统一起来。
``` bash
!/usr/local/envs/layoutlmv3/bin/python3 examples/object_detection/train_net.py  --resume --config-file /content/layoutlmv3/examples/object_detection/cascade_layoutlmv3.yaml   --num-gpus 1 MODEL.WEIGHTS /content/layoutlmv3-base-chinese/pytorch_model.bin   OUTPUT_DIR output PUBLAYNET_DATA_DIR_TRAIN /content/drive/MyDrive/layoutlmv3/data/images PUBLAYNET_DATA_DIR_TEST /content/drive/MyDrive/layoutlmv3/data/images

```

我觉的可能是跟代码有关系，把模型名称替换成model_final.pth就可以运行了。
把AMP.enable关了之后，GPU占用13.4G，可能是batch_size设置为2，所以减少的GPU显存300M有限。

双精度loss比单精度，loss收敛的快一些。

## 2023-04-30
auto-dl进行训练，训练命令
python object_detection/train_net.py --config-file object_detection/cascade_layoutlmv3.yaml   --num-gpus 1 OUTPUT_DIR output MODEL.WEIGHTS /root/mydata/layoutlmv3 PUBLAYNET_DATA_DIR_TRAIN /root/mydata/label-studio/data/train PUBLAYNET_DATA_DIR_TEST /root/mydata/label-studio/data/val 
24G的内存上，batch_size也只能设置为3，是采用中文的预训练模型还是基于应该的publayoutnet的训练模型，在训练的过程中，突然想到了，document layout analysis是完全基于视觉的模型，没有text embedding，所以训练一个小时候，替换成了publayout的预训练模型作为基础进行训练，目前看比完全基于中文的预训练模型，效果好的非常大。
| category   | AP     | category   | AP     | category   | AP      |
|:-----------|:-------|:-----------|:-------|:-----------|:--------|
| LU_Text    | 53.791 | RD_Text    | 52.405 | Table      | 100.000 |
| Text       | 76.165 | Title      | 75.259 | Title_info | 0.000   |
-------------------
最新的表格
[04/30 15:48:55 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 55.083 | 78.944 | 58.104 |  nan  | 42.471 | 64.024 |
[04/30 15:48:55 d2.evaluation.coco_evaluation]: Some metrics cannot be computed and is shown as NaN.
[04/30 15:48:55 d2.evaluation.coco_evaluation]: Per-category bbox AP: 
| category   | AP     | category   | AP     | category   | AP      |
|:-----------|:-------|:-----------|:-------|:-----------|:--------|
| LU_Text    | 20.000 | RD_Text    | nan    | Table      | 100.000 |
| Text       | 77.051 | Title      | 78.363 | Title_info | 0.000   |


python object_detection/train_net.py --config-file object_detection/cascade_layoutlmv3.yaml   --num-gpus 1 OUTPUT_DIR output MODEL.WEIGHTS /root/mydata/layoutlmv3 PUBLAYNET_DATA_DIR_TRAIN /root/mydata/label-studio/data/train PUBLAYNET_DATA_DIR_TEST /root/mydata/label-studio/data/val 


python object_detection/train_net.py --config-file object_detection/cascade_layoutlmv3.yaml   --num-gpus 1 OUTPUT_DIR /root/autodl-fs/output MODEL.WEIGHTS /root/autodl-fs/layoutlmv3-base-finetuned-publaynet/model_final.pth PUBLAYNET_DATA_DIR_TRAIN /root/mydata/data/train PUBLAYNET_DATA_DIR_TEST /root/mydata/data/val 

### 感想
目前大概标注了130+的图片，1000+的数据，最开始使用layoutlmv3-base-chinese，进行训练时，ap前1000step的训练AP不超过2，大模型的训练，
需要思考的模型，如果fintune的情况下，大模型训练需要多少样本才能达到比较好的效果？
500M的模型文件，需要多少GPU资源才能够很好的进行训练。

# 2023-05-01 
标题下面对于标题的解释信息，或者说副标题，预测中根本没有出现。肯定不是目标太小了，因为有时候可以被识别为text或者title。数据量也不是不够？
需要做的事情是，如何增加数据的权重？在数据集里面加倍一些数据。

预测错位的分析：
1. A，B两个区域，B预测的区域是A的子集，并且AB预测的类别一样，这种情况下，可以在工程上进行优化掉。
2. 出现了表格的情况，需要翻转90的情况。想到了，如果给模型输入一个表格，模型能给出一些语料吗？这个表格是干什么的？每一列是什么作用？

## 标注的数据
目前标注了大概650张图片，之前最好的map大概能达到60，之前的图片是337张图片AP-58，翻一倍的情况下，能提升多少ap？需要考虑的事情：
1. 多gpu训练。
2. 是否需要把图片和数据分开打散进行训练？不要，防止训练过程中，训练数据看到测试数据中的图片。

## 训练

把配置文件单独拿出来。
```
python object_detection/train_net.py --config-file /root/mydata/config/cascade_layoutlmv3.yaml   --num-gpus 1 OUTPUT_DIR /root/autodl-tmp/output MODEL.WEIGHTS /root/mydata/layoutlmv3-base-finetuned-publaynet/model_final.pth PUBLAYNET_DATA_DIR_TRAIN /root/mydata/data/train PUBLAYNET_DATA_DIR_TEST /root/mydata/data/val 
```

这次使用了A40进行训练,48G内存，发现batch_size设置为4，发现total_loss 下降的非常快，不到200个step的情况下目前已经到了0.77 ，以前要上万个step之后。
问题：batch_size造成的，还是数据集多的造成的？我认为是batch_size造成的。因为目前模型还没有看一遍数据。

iter: 199 的数据，目前看来挺好，但是title_info还是没有结果。
|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
|:------:|:------:|:------:|:-----:|:------:|:------:|
| 50.872 | 64.355 | 57.684 |  nan  | 46.606 | 48.020 |
[05/02 00:13:44 d2.evaluation.coco_evaluation]: Some metrics cannot be computed and is shown as NaN.
[05/02 00:13:44 d2.evaluation.coco_evaluation]: Per-category bbox AP: 
| category   | AP     | category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|:-----------|:-------|
| LU_Text    | 34.841 | RD_Text    | 60.246 | Table      | 44.094 |
| Text       | 88.649 | Title      | 77.405 | Title_info | 0.000  |
[05/02 00:13:44 d2.evaluation.testing]: copypaste: Task: bbox
[05/02 00:13:44 d2.evaluation.testing]: copypaste: AP,AP50,AP75,APs,APm,APl
[05/02 00:13:44 d2.evaluation.testing]: copypaste: 50.8724,64.3555,57.6841,nan,46.6062,48.0197

# 2023-05-02
## 训练结果
使用A40，大概训练了10个小时，loss降到了0.55左右，但是平均AP还是特别的低在55左右。