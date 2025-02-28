import cv2
import mediapipe as mp
import numpy as np
import time

class GestureRecognition:
    def __init__(self, model, confidence_threshold=1300, recognize_delay=1):
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
        self.knn, self.train_data = model
        
        # 신뢰도 및 딜레이 설정
        self.confidence_threshold = confidence_threshold
        self.recognize_delay = recognize_delay
        
        # 시간 및 이전 인덱스 초기화
        self.start_time = time.time()
        self.prev_index = -1
        
    def _initialize_model(file_path):
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

    def _calculate_wrist_to_middle_finger_tip(self, landmarks):
        """손목에서 중지 끝까지의 벡터 계산"""
        wrist = landmarks.landmark[0]  # 손목
        middle_finger_tip = landmarks.landmark[12]  # 중지 끝

        wrist_to_middle_finger_tip = np.array([middle_finger_tip.x - wrist.x, middle_finger_tip.y - wrist.y])
        return wrist_to_middle_finger_tip

    def _calculate_angles_and_relative_landmarks(self, landmarks):
        """각도 및 상대 좌표 계산"""
        joint = np.zeros((21, 3))
        for j, lm in enumerate(landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z]

        # 벡터 및 각도 계산
        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
        v = v2 - v1
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        compare_v1 = v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :]
        compare_v2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
        angle = np.degrees(np.arccos(np.einsum('nt,nt->n', compare_v1, compare_v2)))

        # 상대 좌표 계산
        relative_landmarks = joint[:, :2] - joint[0, :2]
        distances = np.linalg.norm(relative_landmarks, axis=1)
        max_distance = np.max(distances)

        relative_landmarks_normalized = (
            relative_landmarks / max_distance if max_distance > 0 else relative_landmarks
        )
        return angle, relative_landmarks_normalized.flatten()

    def process_frame(self, frame):
        """단일 프레임 처리"""
        if frame is None:
            return frame, None
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img_rgb)
        gesture_result = None

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                try:
                    # 손 랜드마크 그리기
                    self.mp_drawing.draw_landmarks(frame, res, self.mp_hands.HAND_CONNECTIONS)
                    
                    wrist_to_middle_finger_tip = self._calculate_wrist_to_middle_finger_tip(res)
                    angle, relative_landmarks_flat = self._calculate_angles_and_relative_landmarks(res)

                    # 데이터 준비
                    data = np.hstack([wrist_to_middle_finger_tip[:2], angle, relative_landmarks_flat]).astype(np.float32)
                    data = np.array([data], dtype=np.float32)

                    if data.shape[1] != self.train_data.shape[1]:
                        print("Error: Feature dimension mismatch!")
                        continue

                    # KNN 예측
                    ret, results, neighbours, dist = self.knn.findNearest(data, 3)
                    index = int(results[0][0])

                    # 신뢰도 확인
                    avg_distance = np.mean(dist)
                    if avg_distance > self.confidence_threshold:
                        cv2.putText(frame, 'Undefined', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        gesture_result = 'Undefined'
                    else:
                        if index != self.prev_index:
                            self.start_time = time.time()
                            self.prev_index = index
                        else:
                            if time.time() - self.start_time > self.recognize_delay:
                                gesture_result = self.gesture.get(index, 'Undefined')
                                self.start_time = time.time()

                    
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    
        if gesture_result != "Undefined":
            return gesture_result