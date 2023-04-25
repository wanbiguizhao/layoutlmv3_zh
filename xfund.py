from datasets import load_dataset
from transformers import AutoProcessor, AutoModel,AutoConfig,AutoTokenizer
funsd_dataset=load_dataset("nielsr/funsd-layoutlmv3", split="train")
# dataset = load_dataset("json",data_files={"train":"./tmp/data/zh.xfund.train.json"},split='train')
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, RobertaConverter
SLOW_TO_FAST_CONVERTERS.update({"LayoutLMv3Tokenizer": RobertaConverter})

config = AutoConfig.from_pretrained("./tmp/layoutlmv3-base-chinese",num_labels=7,)
tokenizer = AutoTokenizer.from_pretrained("./tmp/layoutlmv3-base-chinese",use_fast=True,)
processor = AutoProcessor.from_pretrained("./tmp/layoutlmv3-base-chinese", config=config,apply_ocr=False,tokenizer=tokenizer)
model = AutoModel.from_pretrained("./tmp/layoutlmv3-base-chinese",config=config)
# print(dataset)


# -*- coding: utf-8 -*-

import json
import os

from PIL import Image

import datasets

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)

def normalize_bbox(bbox, size):
    width, height = size
    def clip(min_num, num, max_num):
        return min(max(num, min_num), max_num)

    x0, y0, x1, y1 = bbox
    x0 = clip(0, int((x0 / width) * 1000), 1000)
    y0 = clip(0, int((y0 / height) * 1000), 1000)
    x1 = clip(0, int((x1 / width) * 1000), 1000)
    y1 = clip(0, int((y1 / height) * 1000), 1000)
    assert x1 >= x0
    assert y1 >= y0
    return [x0, y0, x1, y1]

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{xu-etal-2022-xfund,
    title = "{XFUND}: A Benchmark Dataset for Multilingual Visually Rich Form Understanding",
    author = "Xu, Yiheng  and
      Lv, Tengchao  and
      Cui, Lei  and
      Wang, Guoxin  and
      Lu, Yijuan  and
      Florencio, Dinei  and
      Zhang, Cha  and
      Wei, Furu",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.253",
    doi = "10.18653/v1/2022.findings-acl.253",
    pages = "3214--3224",
    abstract = "Multimodal pre-training with text, layout, and image has achieved SOTA performance for visually rich document understanding tasks recently, which demonstrates the great potential for joint learning across different modalities. However, the existed research work has focused only on the English domain while neglecting the importance of multilingual generalization. In this paper, we introduce a human-annotated multilingual form understanding benchmark dataset named XFUND, which includes form understanding samples in 7 languages (Chinese, Japanese, Spanish, French, Italian, German, Portuguese). Meanwhile, we present LayoutXLM, a multimodal pre-trained model for multilingual document understanding, which aims to bridge the language barriers for visually rich document understanding. Experimental results show that the LayoutXLM model has significantly outperformed the existing SOTA cross-lingual pre-trained models on the XFUND dataset. The XFUND dataset and the pre-trained LayoutXLM model have been publicly available at https://aka.ms/layoutxlm.",
}
"""

_DESCRIPTION = """\
https://github.com/doc-analysis/XFUND
"""


_LANG = ["de", "es", "fr", "it", "ja", "pt", "zh"]
_URL = "https://github.com/doc-analysis/XFUND/releases/download/v1.0"


class XFund(datasets.GeneratorBasedBuilder):
    """XFund dataset."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name=f"{lang}", version=datasets.Version("1.0.0"), description=f"XFUND {lang} dataset") for lang in _LANG
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "HEADER", "QUESTION", "ANSWER"]
                        )
                    ),
                    "image": datasets.features.Image(),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/doc-analysis/XFUND",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        lang = self.config.name
        fileinfos = dl_manager.download_and_extract({
            "train_image": f"{_URL}/{lang}.train.zip",
            "train_annotation": f"{_URL}/{lang}.train.json",
            "valid_image": f"{_URL}/{lang}.val.zip",
            "valid_annotation": f"{_URL}/{lang}.val.json",
        })
        logger.info(f"file infos: {fileinfos}")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"image_path": fileinfos['train_image'], "annotation_path": fileinfos["train_annotation"]}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"image_path": fileinfos["valid_image"], "annotation_path": fileinfos["valid_annotation"]}
            ),
        ]

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def _generate_examples(self, image_path, annotation_path):
        logger.info("⏳ Generating examples from = %s %s", image_path, annotation_path)
        with open(annotation_path) as fi:
            ann_infos = json.load(fi)
            document_list = ann_infos["documents"]
        for guid, doc in enumerate(document_list):
            tokens, bboxes, tags = list(), list(), list()
            image_file = os.path.join(image_path, doc["img"]["fname"])
            # cannot load image when submit code to huggingface
            # image, size = load_image(image_file)
            # assert size[0] == doc["img"]["width"]
            # assert size[1] == doc["img"]["height"]
            size = [doc["img"]["width"], doc["img"]["height"]]

            for item in doc["document"]:
                cur_line_bboxes = list()
                text, label = item["text"], item["label"]
                bbox = normalize_bbox(item["box"], size)
                if len(text) == 0:
                    continue
                tokens.append(text)
                bboxes.append(bbox)
                tags.append(label.upper() if label != "other" else "O")

            yield guid, {"id": doc["id"], "tokens": tokens, "bboxes": bboxes, "tags": tags, "image": Image.open(image_file)}

#xfund=XFund("zh_xfunds",data_files="tmp/data/zh.train.json",data_dir="tmp/data/")
test=XFund(name="zh")
xfund_ds=test.download_and_prepare()
xfund_ds=test.as_dataset()
# print(len(xfund_ds["train"]))
# print([len(xfund_ds["test"][i]["tokens"]) for i in range(len(xfund_ds["test"]))])
print(processor.backend_tokenizer)
#xfund._generate_examples("tmp/data/images","tmp/data/zh.train.json")
