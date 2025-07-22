import os
import cv2
import numpy as np
from face.demo.SSRNET_model import SSR_net, SSR_net_general
import tensorflow as tf


class FaceDemo:
    def __init__(self, model_age, model_gender, img_size=64, ad=0.5, skip_frame=5):
        self.model_age = model_age
        self.model_gender = model_gender
        self.img_size = img_size
        self.ad = ad
        self.skip_frame = skip_frame

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def draw_label(self, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def process_image(self, input_img):
        """
        전달된 이미지를 처리하고, 첫 번째 얼굴의 나이 및 성별을 반환합니다.
        """
        img_h, img_w, _ = input_img.shape
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        # 얼굴 감지
        detected = self.face_cascade.detectMultiScale(
            gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        result = None  # 첫 번째 결과만 저장

        if len(detected) > 0:
            # 첫 번째 얼굴만 처리
            x, y, w, h = detected[0]
            x1, y1, x2, y2 = x, y, x + w, y + h
            xw1 = max(int(x1 - self.ad * w), 0)
            yw1 = max(int(y1 - self.ad * h), 0)
            xw2 = min(int(x2 + self.ad * w), img_w - 1)
            yw2 = min(int(y2 + self.ad * h), img_h - 1)

            # 얼굴 영역을 정규화 및 크기 조정
            face_img = input_img[yw1:yw2 + 1, xw1:xw2 + 1, :]
            face_resized = cv2.resize(face_img, (self.img_size, self.img_size))
            face_normalized = cv2.normalize(face_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            face_input = np.expand_dims(face_normalized, axis=0)  # 배치 차원 추가

            # 모델 예측
            try:
                predicted_age = self.model_age.predict(face_input)[0]
                predicted_gender = self.model_gender.predict(face_input)[0]

                # 나이 범위 결정
                age = int(predicted_age)
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

                # 성별 결정
                gender = 'male' if predicted_gender >= 0.5 else 'female'

                # 결과 저장
                result = (age, gender)

                # 결과 표시
                label = f"{age}, {gender}"
                self.draw_label(input_img, (x, y), label)

            except Exception as e:
                print(f"Prediction error: {e}")

        return result


    def run(self):
        """
        웹캠을 통해 실시간으로 프레임을 처리하며, 결과를 출력합니다.
        """
        tf.keras.backend.clear_session()

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        os.makedirs('./img', exist_ok=True)

        img_idx = 0

        while True:
            ret, input_img = cap.read()
            if not ret:
                break

            img_idx += 1

            if img_idx == 1 or img_idx % self.skip_frame == 0:
                # 프레임 처리
                processed_img, results = self.process_image(input_img)
                print(f"Frame {img_idx} Results: {results}")

                # 처리된 프레임 저장
                cv2.imwrite(f'img/{img_idx}.png', processed_img)

            # 화면 표시
            cv2.imshow("Processed Frame", input_img)

            # 종료 조건
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
