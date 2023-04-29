import os
import json
import random
import label_studio_sdk


from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, \
    get_single_tag_keys, DATA_UNDEFINED_NAME
from object_detection.ditod import DefaultPredictor
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from PIL import Image
import numpy as np
from detectron2.structures import Boxes, BoxMode, pairwise_iou
LABEL_STUDIO_HOST = os.getenv('LABEL_STUDIO_HOST', 'http://localhost:8080')
LABEL_STUDIO_API_KEY = os.getenv('LABEL_STUDIO_API_KEY', 'bf45ee964022f05fa2c4d025a719a9aedf4a8d2f')


class PublayoutnetModel(LabelStudioMLBase):
    """This simple Label Studio ML backend demonstrates training & inference steps with a simple scenario:
    on training: it gets the latest created annotation and stores it as "prediction model" artifact
    on inference: it returns the latest annotation as a pre-annotation for every incoming task

    When connected to Label Studio, this is a simple repeater model that repeats your last action on a new task
    """
    def __init__(self,cfg, config_file=None,
                 checkpoint_file=None,
                 image_dir=None,
                 labels_file=None, score_threshold=0.5, device='cpu', **kwargs):
        super(PublayoutnetModel, self).__init__(**kwargs)
        self.category_id_maping=[
                    {
                    "supercategory": "",
                    "id": 1,
                    "name": "Text"
                    },
                    {
                    "supercategory": "",
                    "id": 2,
                    "name": "Title"
                    },
                    {
                    "supercategory": "",
                    "id": 3,
                    "name": "List"
                    },
                    {
                    "supercategory": "",
                    "id": 4,
                    "name": "Table"
                    },
                    {
                    "supercategory": "",
                    "id": 5,
                    "name": "Figure"
                    }
        ]
        self.model=DefaultPredictor(cfg)
        self.parsed_label_config={
        "label": {
          "type": "RectangleLabels",
          "to_name": [
            "image"
          ],
          "inputs": [
            {
              "type": "Image",
              "value": "image"
            }
          ],
          "labels": [
            "Title",
            "Text",
            "Table",
            "List",
            "Figure"
          ],
          "labels_attrs": {
            "Title": {
              "value": "Title",
              "background": "green"
            },
            "Text": {
              "value": "Text",
              "background": "#FFA39E"
            },
            "Table": {
              "value": "Table",
              "background": "#0040ff"
            },
            "List": {
              "value": "List",
              "background": "#ff9500"
            },
            "Figure": {
              "value": "Figure",
              "background": "#AD8B00"
            }
          }
        }
      }
        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')
        print(self.labels_in_config)
    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        return image_url

    def parser_instance(self,instances,img_width,img_height):
        num_instance = len(instances)
        if num_instance == 0:
            return []
        boxes = instances.pred_boxes.tensor.numpy()
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        results=[]
        for k in range(num_instance):
            print(classes[k])
            output_label=self.category_id_maping[classes[k]]["name"]
            bbox=boxes[k]
            x, y, width, heigh = bbox[:4]
            result = {
                'from_name': "label",
                'to_name': "image",
                'type': 'rectanglelabels',
                "value":{
                    "origin_bbox": bbox[:4],
                    'rectanglelabels': [output_label],
                    'x': float(x) / img_width * 100,
                    'y': float(y) / img_height * 100,
                    'width': float(width) / img_width * 100,
                    'height': float(heigh) / img_height * 100
                },
                "score": scores[k],
            }
            results.append(
                result
            )
        avg_score=sum(scores)/max(1.0,len(scores)) 
        return [{
                'result': results,
                'score': avg_score
            }]

    def predict(self, tasks, **kwargs):
        """ This is where inference happens:
            model returns the list of predictions based on input list of tasks

            :param tasks: Label Studio tasks in JSON format
        """
        # self.train_output is a dict that stores the latest result returned by fit() method
        assert len(tasks) == 1
        task = tasks[0]
        
        image_url = self._get_image_url(task)
        image_path = self.get_local_path(image_url)
        original_image=Image.open(image_path)
        img_width,img_height=original_image.width,original_image.height
        image = np.asarray(original_image)
        res=self.model(image)
        output_prediction=self.parser_instance(res["instances"],img_width,img_height)
        print(f'Return output prediction: {json.dumps(output_prediction, indent=2)}')
        # 需要做标签映射
        # 
        return output_prediction

    def download_tasks(self, project):
        """
        Download all labeled tasks from project using the Label Studio SDK.
        Read more about SDK here https://labelstud.io/sdk/
        :param project: project ID
        :return:
        """
        ls = label_studio_sdk.Client(LABEL_STUDIO_HOST, LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=project)
        tasks = project.get_labeled_tasks()
        return tasks

    def fit(self, tasks, workdir=None, **kwargs):
        """
        This method is called each time an annotation is created or updated
        :param kwargs: contains "data" and "event" key, that could be used to retrieve project ID and annotation event type
                        (read more in https://labelstud.io/guide/webhook_reference.html#Annotation-Created)
        :return: dictionary with trained model artefacts that could be used further in code with self.train_output
        """
        if 'data' not in kwargs:
            raise KeyError(f'Project is not identified. Go to Project Settings -> Webhooks, and ensure you have "Send Payload" enabled')
        data = kwargs['data']
        project = data['project']['id']
        tasks = self.download_tasks(project)
        if len(tasks) > 0:
            print(f'{len(tasks)} labeled tasks downloaded for project {project}')
            prediction_example = tasks[-1]['annotations'][0]['result']
            print(f'We\'ll return this as dummy prediction example for every new task:\n{json.dumps(prediction_example, indent=2)}')
            return {
                'prediction_example': prediction_example,
                'also you can put': 'any artefact here'
            }
        else:
            print('No labeled tasks found: make some annotations...')
            return {}
