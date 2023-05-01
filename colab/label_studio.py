"""
1. 从label_studio中下载没有预测的数据
2. 对数据进行离线预测。
3. 准备可以导入label——studio的预测数据。
"""
import json 
import shutil
import os 
from label_studio_sdk import client
PROJECT_DIR=os.path.dirname(
                os.path.dirname(
                    os.path.realpath(__file__)
                )
        )
CURRENT_DIR=os.path.join(PROJECT_DIR,"colab")
LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', 'bf45ee964022f05fa2c4d025a719a9aedf4a8d2f')
label_studio_client=client.Client(url=LABEL_STUDIO_HOST,
                                  api_key=LABEL_STUDIO_API_KEY)
print(label_studio_client.check_connection())
project=label_studio_client.get_project(id=3)
tasks_data=project.get_tasks(
        filters= {
            "conjunction": "and",
            "items": [
                {
                    "filter": "filter:tasks:total_annotations",
                    "operator": "empty",
                    "value": 1,
                    "type": "Number"
                },
                {
                    "filter": "filter:tasks:total_predictions",
                    "operator": "empty",
                    "value": 1,
                    "type": "Number"
                }
            ]
        }
    )
with open(f"{CURRENT_DIR}/data/tasks.json","w") as taskf:
    json.dump(tasks_data,taskf)
shutil.rmtree(f"{CURRENT_DIR}/data/images/")
os.makedirs(f"{CURRENT_DIR}/data/images/")
for task in tasks_data:
    response=label_studio_client.make_request("GET",f'{task["data"]["image"]}')
    image_name=task["data"]["image"].split("/")[-1]

    with open(f"{CURRENT_DIR}/data/images/{image_name}", 'wb') as file:
        file.write(response.content)
    print(f'{task["data"]["image"]},{task["id"]}')

    

#print(tasks_data)
def load_from_label_studio():
    pass 