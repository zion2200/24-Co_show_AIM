import cv2
import mediapipe as mp
import numpy as np
import screeninfo

class CalibratedEyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 모니터 정보
        try:
            screen = screeninfo.get_monitors()[0]
            self.SCREEN_W = screen.width
            self.SCREEN_H = screen.height
        except:
            # 기본값 설정
            self.SCREEN_W = 1920
            self.SCREEN_H = 1080

        # 시선 좌표 저장 변수
        self.current_gaze_position = (self.SCREEN_W // 2, self.SCREEN_H // 2)
        
        # 랜드마크 인덱스
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

        # 캘리브레이션 관련 변수
        self.calibration_complete = False
        self.base_position = None
        self.base_head_pose = None
        
        # 스무딩 관련 변수
        self.smoothing_factor = 0.1
        self.prev_pos = None

        self.weights = {
            'head': {
                'vertical': 0.3,
                'horizontal': 0.4
            },
            'eye': {
                'vertical': 1.0,
                'horizontal': 1.0
            }
        }

    def set_calibration_data(self, iris_pos, head_pose):
        """외부에서 캘리브레이션 데이터 설정"""
        self.base_position = iris_pos
        self.base_head_pose = head_pose
        self.calibration_complete = True

    def get_head_pose(self, landmarks, image_shape):
        """얼굴의 방향과 위치를 추정"""
        face_3d = []
        face_2d = []
        
        ih, iw, _ = image_shape
        
        for idx in self.FACE_OVAL:
            lm = landmarks.landmark[idx]
            x, y = int(lm.x * iw), int(lm.y * ih)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z * 3000])

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * iw
        cam_matrix = np.array([
            [focal_length, 0, iw/2],
            [0, focal_length, ih/2],
            [0, 0, 1]
        ])

        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(
            face_3d, face_2d, cam_matrix, dist_matrix)

        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        return angles

    def get_iris_position(self, landmarks, image_shape):
        """양쪽 눈 홍채 중심점의 평균 계산"""
        ih, iw, _ = image_shape
        left_iris_points = np.array([[landmarks.landmark[idx].x * iw, landmarks.landmark[idx].y * ih]
                                     for idx in self.LEFT_IRIS])
        right_iris_points = np.array([[landmarks.landmark[idx].x * iw, landmarks.landmark[idx].y * ih]
                                      for idx in self.RIGHT_IRIS])
        
        left_iris_center = np.mean(left_iris_points, axis=0)
        right_iris_center = np.mean(right_iris_points, axis=0)
        return (left_iris_center + right_iris_center) / 2

    def map_to_screen(self, current_pos):
        """홍채 위치를 화면 좌표로 변환"""
        if self.base_position is None:
            return self.SCREEN_W // 2, self.SCREEN_H // 2

        delta_x = (current_pos[0] - self.base_position[0]) * self.weights['eye']['horizontal']
        delta_y = (current_pos[1] - self.base_position[1]) * self.weights['eye']['vertical']

        screen_x = int(self.SCREEN_W/2 + delta_x * self.SCREEN_W * 0.1)
        screen_y = int(self.SCREEN_H/2 + delta_y * self.SCREEN_H * 0.1)

        screen_x = max(0, min(screen_x, self.SCREEN_W - 1))
        screen_y = max(0, min(screen_y, self.SCREEN_H - 1))

        return screen_x, screen_y

    def smooth_position(self, new_pos):
        """위치 스무딩"""
        if self.prev_pos is None:
            self.prev_pos = new_pos
            return new_pos

        smoothed_x = int(self.prev_pos[0] * (1 - self.smoothing_factor) + new_pos[0] * self.smoothing_factor)
        smoothed_y = int(self.prev_pos[1] * (1 - self.smoothing_factor) + new_pos[1] * self.smoothing_factor)
        
        self.prev_pos = (smoothed_x, smoothed_y)
        return smoothed_x, smoothed_y

    def get_current_gaze_position(self):
        """현재 시선 위치를 반환하는 메서드"""
        if self.current_gaze_position:
            track = True
            x, y = self.current_gaze_position
            # 화면을 벗어나면 종료 신호 보내기
            if x < 0 or y < 0 or x > self.SCREEN_W or y > self.SCREEN_H:
                return True, -1, -1
            return track, x, y
        return False, 0, 0

    def process_frame(self, frame):
        """프레임 처리"""
        if frame is None or len(frame) == 0:
            return
            
        rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            current_pos = self.get_iris_position(face_landmarks, frame.shape)
            head_pose = self.get_head_pose(face_landmarks, frame.shape)

            if self.calibration_complete:
                compensated_pos = self.compensate_head_pose(current_pos, head_pose)
                screen_x, screen_y = self.map_to_screen(compensated_pos)
                screen_x, screen_y = self.smooth_position((screen_x, screen_y))
                self.current_gaze_position = (screen_x, screen_y)

            return current_pos, head_pose
        return None, None

    def compensate_head_pose(self, iris_pos, current_angles):
        """머리 방향에 따른 눈 위치 보정"""
        if self.base_head_pose is None:
            return iris_pos

        current_angles_np = np.array(current_angles)
        base_head_pose_np = np.array(self.base_head_pose)
        angle_diff = current_angles_np - base_head_pose_np

        vertical_compensation = angle_diff[0] * self.weights['head']['vertical']
        horizontal_compensation = angle_diff[1] * self.weights['head']['horizontal']

        compensated_x = iris_pos[0] - horizontal_compensation
        compensated_y = iris_pos[1] - vertical_compensation

        return np.array([compensated_x, compensated_y])