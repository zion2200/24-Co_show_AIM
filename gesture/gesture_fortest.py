import cv2
import mediapipe as mp
import numpy as np
import time

class GestureRecognition:
    def __init__(self, model_path, confidence_threshold=1300, recognize_delay=1):
        # Mediapipe Hands 초기화
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Gesture 정의
        self.gesture = {
            0: "0", 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
            8: '8', 9: '9', 10: '10'
        }

        # 모델 로드
        self.knn, self.train_data = self._initialize_model(model_path)
        
        # 신뢰도 및 딜레이 설정
        self.confidence_threshold = confidence_threshold
        self.recognize_delay = recognize_delay
        
        # 시간 및 이전 인덱스 초기화
        self.start_time = time.time()
        self.prev_index = -1

    def _initialize_model(self, file_path):
        """KNN 모델 초기화 및 훈련 데이터 로드"""
        try:
            file = np.genfromtxt(file_path, delimiter=',')
            angle_file = file[:, :18]
            locate_file = file[:, 18:-1]
            label_file = file[:, -1]

            angle = angle_file.astype(np.float32)
            locate = locate_file.astype(np.float32)
            label = label_file.astype(np.float32)

            if angle.ndim == 1:
                angle = angle[:, np.newaxis]
            if locate.ndim == 1:
                locate = locate[:, np.newaxis]

            knn = cv2.ml.KNearest_create()
            train_data = np.hstack([angle, locate])
            knn.train(train_data, cv2.ml.ROW_SAMPLE, label)

            return knn, train_data
        except Exception as e:
            raise ValueError(f"Error loading model data: {e}")

    def process_frame(self, frame):
        """단일 프레임 처리"""
        if frame is None:
            return "0"
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                try:
                    # 손 랜드마크 그리기
                    self.mp_drawing.draw_landmarks(frame, res, self.mp_hands.HAND_CONNECTIONS)
                    
                    # 각도와 상대 좌표 계산
                    angle, relative_landmarks_flat = self._calculate_angles_and_relative_landmarks(res)

                    # 데이터 준비
                    data = np.hstack([angle, relative_landmarks_flat]).astype(np.float32)
                    data = np.array([data], dtype=np.float32)

                    if data.shape[1] != self.train_data.shape[1]:
                        print("Error: Feature dimension mismatch!")
                        return "0"

                    # KNN 예측
                    ret, results, neighbours, dist = self.knn.findNearest(data, 3)
                    index = int(results[0][0])

                    # 신뢰도 확인
                    avg_distance = np.mean(dist)
                    if avg_distance > self.confidence_threshold:
                        return "0"
                    else:
                        return self.gesture.get(index, "0")
                    
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    return "0"
        
        return "0"