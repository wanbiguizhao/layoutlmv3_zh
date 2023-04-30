import os 
import random
import json
# 用于对数据分割的工具。
from sklearn.model_selection import train_test_split
PROJECT_DIR= os.path.dirname(__file__)
DATA_DIR=os.path.join(PROJECT_DIR,'Data')

def convert_label_studio_to_coco(export_data_dir):
    # 把label-studio导出数据的个格式，转化为coco数据格式，主要是json中路径的修改。
    image_dir=os.path.join(export_data_dir,"images")
    json_file_path=os.path.join(export_data_dir,"result.json")
    out_json_file_path=os.path.join(export_data_dir,"coco_result.json")
    with open(json_file_path,"r") as jf ,open(out_json_file_path,"w") as coco_jf:
        js_data=json.load(jf)
        for images_info in js_data["images"]:
            images_info["file_name"]=images_info["file_name"].replace("images/","")
        json.dump(js_data,coco_jf,ensure_ascii=False,indent=2)
    
    




if __name__ == "__main__":
    convert_label_studio_to_coco("tmp/label-studio/data")