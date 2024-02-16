import sys
import os

from face_alignment import mtcnn
# import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime
mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(112, 112))
import cv2
from uuid import uuid1

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result



def get_aligned_face(image_path, rgb_pil_image=None):
    bboxes, faces, landmarks, yaws = mtcnn_model.align_multi(image_path)
    
    # bboxes = [b[0:4] for b in bboxes]
    return bboxes, faces, landmarks, yaws



if __name__ == "__main__":
    path_to_img = "/home/rmarefat/projects/face/DATA/Dotin_images/reza/reza_01.jpg"
    face = get_aligned_face(path_to_img)

    cv2.imwrite("./ttt.jpg", face)