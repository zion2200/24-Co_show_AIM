import os
import cv2
import numpy as np
import tensorflow as tf
from face.demo.SSRNET_model import SSR_net

class FaceDemo:
    def __init__(self):
        tf.keras.backend.clear_session()
        
        # 모델 파라미터 설정
        self.img_size = 64
        self.stage_num = [3, 3, 3]
        self.lambda_local = 1
        self.lambda_d = 1
        
        # 모델 경로 설정
        self.weight_file = "face/pre-trained/megaface_asian/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"
        self.weight_file_gender = "face/pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"
        
        # 모델 초기화
        self.model = self._init_model(self.weight_file)
        self.model_gender = self._init_model(self.weight_file_gender)
        
        # 얼굴 감지기 초기화
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def _init_model(self, weight_file):
        model = SSR_net(self.img_size, self.stage_num, self.lambda_local, self.lambda_d)()
        model.load_weights(weight_file, by_name=True, skip_mismatch=True)
        return model
    
    def process_image(self, image):
        """
        이미지를 처리하여 나이와 성별을 예측합니다.
        
        Args:
            image: numpy array, BGR 형식의 이미지
            
        Returns:
            list of tuples: [(age, gender_str), ...] 형식의 결과
        """
        results = []
        img_h, img_w, _ = image.shape
        ad = 0.5
        
        # 그레이스케일 변환
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 감지
        detected = self.face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(detected) > 0:
            faces = np.empty((len(detected), self.img_size, self.img_size, 3))
            
            # 감지된 각 얼굴에 대해 처리
            for i, (x, y, w, h) in enumerate(detected):
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                xw1 = max(int(x1 - ad * w), 0)
                yw1 = max(int(y1 - ad * h), 0)
                xw2 = min(int(x2 + ad * w), img_w - 1)
                yw2 = min(int(y2 + ad * h), img_h - 1)
                
                # 얼굴 영역 추출 및 전처리
                faces[i] = cv2.resize(image[yw1:yw2 + 1, xw1:xw2 + 1, :], (self.img_size, self.img_size))
                faces[i] = cv2.normalize(faces[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            
            try:
                # 나이와 성별 예측
                predicted_ages = self.model.predict(faces)
                predicted_genders = self.model_gender.predict(faces)
                
                # 결과 저장
                for i in range(len(detected)):
                    age = int(predicted_ages[i])
                    gender = 'male' if predicted_genders[i] >= 0.5 else 'female'
                                # 나이 범위 결정
                    if age < 20:
                        age = "u20"
                    elif age < 30:
                        age = "20s"
                    elif age < 40:
                        age = "30s"
                    elif age < 50:
                        age = "40s"
                    else:
                        age = "over50"
                        
                    results.append((age, gender))
            except Exception as e:
                print(f"Prediction error: {e}")
                return image, []
        
        return image, results