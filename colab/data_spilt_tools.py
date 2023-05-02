from collections import defaultdict
import os 
import random
import json
# 用于对数据分割的工具。
from sklearn.model_selection import train_test_split
import shutil
PROJECT_DIR= os.path.dirname(__file__)
DATA_DIR=os.path.join(PROJECT_DIR,'Data')



def merge_tasks_json(export_data_dir):
    #标注的文件太多了，从多个项目进行标注，需要从多个项目中导出数据，需要将数据合并，每个导出的项目都是projec开头的
    pass 
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
def dump_json(data_info_list:list):
    for data_info in data_info_list:
        data,data_path=data_info[0],data_info[1]
        with open(data_path,'w') as df:
            json.dump(data,df,ensure_ascii=False,indent=2)

    
def makedir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
def make_train_val(export_data_dir,train_per=0.85):
    #对数据进行分割成train和val数据
    # 按照图片进行进行划分。
    assert  os.path.exists(export_data_dir)
    images_dir=os.path.join(export_data_dir,"images")
    TRAIN_DIR=os.path.join(export_data_dir,"train")
    VAL_DIR=os.path.join(export_data_dir,"val")
    makedir(TRAIN_DIR)
    makedir(VAL_DIR)

    coco_result_file_path=os.path.join(export_data_dir,"coco_result.json")
    with open(coco_result_file_path,"r") as coco_jf:
        coco_data=json.load(coco_jf)
        images_data=coco_data["images"]
        train_data,val_data={},{}
        train_data["categories"]=val_data["categories"]=coco_data["categories"]
        train_data["images"]=[]
        val_data["images"]=[]
        train_data["annotations"]=[]
        val_data["annotations"]=[]
        
        train_images_data,val_images_data=train_test_split(images_data,shuffle=True,train_size=train_per,random_state=42)
        train_data["images"]=train_images_data
        val_data["images"]=val_images_data

        annotations_data_key_imageid=defaultdict(list)
        annotations_data=coco_data["annotations"]
        for one_data in annotations_data:
            annotations_data_key_imageid[one_data["image_id"]].append(one_data)
        #def extend_annotations_data(origin_data):
        for  image_info in train_data["images"]:
            image_id=image_info["id"]
            file_name=image_info["file_name"]
            train_data["annotations"].extend(annotations_data_key_imageid[image_id])
            shutil.copyfile(f"{images_dir}/{file_name}",f"{TRAIN_DIR}/{file_name}")
        
        for  image_info in val_data["images"]:
            image_id=image_info["id"]
            file_name=image_info["file_name"]
            val_data["annotations"].extend(annotations_data_key_imageid[image_id])
            shutil.copyfile(f"{images_dir}/{file_name}",f"{VAL_DIR}/{file_name}")
        dump_json(
            [
                ( train_data,f"{export_data_dir}/train.json"),
                (val_data,f"{export_data_dir}/val.json")
            ]
        )




if __name__ == "__main__":
    data_dir="tmp/label-studio/data-0501"
    convert_label_studio_to_coco(data_dir)
    make_train_val(data_dir)