import numpy as np
import cv2
from bs4 import BeautifulSoup
import re

def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)

def remove_num(text):
    NUM_RE = re.compile("[^0-9.]")
    return ''.join(NUM_RE.findall(text))

def img_box(img_f, xml_f, reverse=1):

    # 이미지 파일 
    image = cv2.imread(img_f)
    
    # 이미지 반전
    if reverse == 1:
        image = ~image  # 비트 반전

    # xml 파일
    fp = open(xml_f,"r")

    soup = BeautifulSoup(fp, "html.parser")

    h = float(soup.height.text)
    w = float(soup.width.text)

    for songElement in soup.findAll('object'):

        # box draw
        cv2.rectangle(image, (int(w * float(songElement.xmin.string)), int(h * float(songElement.ymin.string))), 
                      (int(w * float(songElement.xmax.string)), int(h * float(songElement.ymax.string))), (0, 255, 0), 2)
        
        # class text draw
        # 가져올 tag 가 name 이라서 beautifulsoup 로 잘 처리가 안되네요. 
        name = remove_num(remove_tags(str(songElement)))
        cv2.putText(image, "{}".format(name), (int(w * float(songElement.xmin.string)), int(h * float(songElement.ymin.string))-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return image

if __name__ == '__main__':
    image = img_box("data/lg-46690-aug-gutenberg1939-.png","data/lg-46690-aug-gutenberg1939-.xml",1)
    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
