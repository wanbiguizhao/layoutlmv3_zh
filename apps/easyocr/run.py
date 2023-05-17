import os
import easyocr 
from tqdm import tqdm
from glob import glob
import cv2 as cv
PROJECT_DIR= os.path.dirname(
        os.path.dirname(os.path.realpath( __file__))
    
)


def run():
    reader=easyocr.Reader(["ch_tra"],gpu=True)
    BASE_IMAGE_DIR="tmp/project_ocrSentences"
    #DST_IMAGE_DIR="tmp/ocrSentences_resize"
    #width_list=[]
    for image_path in tqdm(
        sorted(glob(os.path.join(PROJECT_DIR,BASE_IMAGE_DIR,"*","*.png"),recursive=True))[:100]
        ):
        result=reader.readtext(image_path)
        print(image_path,result)

if __name__=="__main__":
    run()