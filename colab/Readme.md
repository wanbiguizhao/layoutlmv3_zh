1. 为了方便在云上训练的一些代码
2. 批量推理代码
``` bash colab
!/usr/local/envs/layoutlmv3/bin/python run_object_detection.py --input-dir "/content/data" --config-file ../object_detection/cascade_layoutlmv3.yaml OUTPUT_DIR /tmp/ MODEL.WEIGHTS /content/layoutlmv3-base-govcn/model_final.pth
``` 