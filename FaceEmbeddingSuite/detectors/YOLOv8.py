from ultralytics import YOLO
from .face_alignment.mtcnn_pytorch.src.align_trans import warp_and_crop_face, get_reference_facial_points
from uuid import uuid1
import cv2
import os
import gdown

class YOLOLandmark:
    def __init__(self, device):

        os.makedirs(os.path.join(os.getcwd(), "weights"), exist_ok=True)
        self._ckpt_path = os.path.join(os.getcwd(), "weights", "yolov8n_face_general.pt")
        ckpt_url = "https://drive.google.com/uc?id=10-iUQGoAkTaeahs-jC0W05G4qaspTF3N"

        if not(os.path.isfile(self._ckpt_path)):
            gdown.download(ckpt_url, self._ckpt_path, quiet=False)

        self._device = device
        assert os.path.isfile(self._ckpt_path), "Please provide a valid ckpt path"
        self._model = YOLO(self._ckpt_path)

    def _get_faces(sel, image, box, landmark):
        
        converted_landmarks = []
        for key in landmark:
            if len(key.xy.tolist()[0]) == 0:
                continue

            converted_landmarks.append([
                [int(key.xy.tolist()[0][0][0]), int(key.xy.tolist()[0][0][1])],
                [int(key.xy.tolist()[0][1][0]), int(key.xy.tolist()[0][1][1])],
                [int(key.xy.tolist()[0][2][0]), int(key.xy.tolist()[0][2][1])],
                [int(key.xy.tolist()[0][3][0]), int(key.xy.tolist()[0][3][1])],
                [int(key.xy.tolist()[0][4][0]), int(key.xy.tolist()[0][4][1])]
            ])
        

        reference_pts = get_reference_facial_points(default_square=True)

        
        
        try:
            face = warp_and_crop_face(image, converted_landmarks[0], reference_pts=reference_pts, align_type='smilarity')
        except:
            print("Error happend in YOLOLandmark")
            return None

        return face

    def _process_bboxes_n_landmakes(self, image, bounding_boxes, landmarks, find_largest_box=False):
        largest_area = 0
        largest_box = None
        result = {
            "xyxy_boxes": [],
            "faces": [],
            "landmarks":[]
        }
        for box, landmark in zip(bounding_boxes, landmarks):
                landmark_1 = (int(landmark.xy.tolist()[0][0][0]), int(landmark.xy.tolist()[0][0][1]))
                landmark_2 = (int(landmark.xy.tolist()[0][1][0]), int(landmark.xy.tolist()[0][1][1]))
                landmark_3 = (int(landmark.xy.tolist()[0][2][0]), int(landmark.xy.tolist()[0][2][1]))
                landmark_4 = (int(landmark.xy.tolist()[0][3][0]), int(landmark.xy.tolist()[0][3][1]))
                landmark_5 = (int(landmark.xy.tolist()[0][4][0]), int(landmark.xy.tolist()[0][4][1]))

                top_left_x = int(box.xyxy.tolist()[0][0])
                top_left_y = int(box.xyxy.tolist()[0][1])
                bottom_right_x = int(box.xyxy.tolist()[0][2])
                bottom_right_y = int(box.xyxy.tolist()[0][3])

                face = self._get_faces(image, box, landmark)


                if find_largest_box:
                    area = (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)

                    if area > largest_area:
                        largest_area = area
                    
                        result.__setitem__("biggest_xyxy", [top_left_x, top_left_y, bottom_right_x, bottom_right_y])
                        result.__setitem__("biggest_landmarks", [landmark_1, landmark_2, landmark_3, landmark_4, landmark_5])
                        result.__setitem__("biggest_face", face)

                result["xyxy_boxes"].append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
                result["landmarks"].append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])
                result["faces"].append(face)

        return result

    def detect(self, image, find_largest_box=False):
        results = self._model.predict(image, device=self._device)
        boxes = results[0].boxes
        
        landmarks = results[0].keypoints


        if len(boxes) == 0:

            return {
                "bbox": [],
                "landmarks": [],
                "faces": [],
            }
        else:
            result = self._process_bboxes_n_landmakes(image, boxes, landmarks, find_largest_box)

            return result

if __name__ == "__main__":
    face_det = YOLOLandmark("/home/mehran/rezamarefat/YOLO_Landmark/yolov8n__face_general.pt")

    img_path = "/home/mehran/rezamarefat/YOLO_Landmark/face_image.jpg"
    res = face_det.detect(img_path)
    

    # The following is just for testing
    img = cv2.imread(img_path)
    for key, value in res.items():
        tlx = value["bbox"][0]
        tly = value["bbox"][1]
        brx = value["bbox"][2]
        bry = value["bbox"][3]
        cv2.rectangle(img, (tlx, tly), (brx, bry), (0,0,255), 1)
        point_1 = value["landmarks"][0]
        point_2 = value["landmarks"][1]
        point_3 = value["landmarks"][2]
        point_4 = value["landmarks"][3]
        point_5 = value["landmarks"][4]
        cv2.circle(img, point_1, 2, (0,255, 0), -1)
        cv2.circle(img, point_2, 2, (0,255, 0), -1)
        cv2.circle(img, point_3, 2, (0,255, 0), -1)
        cv2.circle(img, point_4, 2, (0,255, 0), -1)
        cv2.circle(img, point_5, 2, (0,255, 0), -1)

    cv2.imwrite("vis.jpg", img)