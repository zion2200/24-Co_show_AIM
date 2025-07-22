# 팀 이름
TEAM_NAME = "AI GO"
import cv2
import os
from gesture.gesture_fortest import GestureRecognition
from face.demo.face_fortest import FaceDemo

def do_face(image_file_path: str) -> tuple:
    # FaceDemo 인스턴스 생성
    face_demo = FaceDemo()
    
    # 이미지 로드
    image = cv2.imread(image_file_path)
    if image is None:
        print(f"Error: Could not load image from {image_file_path}")
        return (0, "couldn't find image")
        
    # 이미지 크기 조정 (640x480)
    resized_image = cv2.resize(image, (640, 480))
    
    # FaceDemo에 현재 프레임 전달 및 결과 처리
    _, results = face_demo.process_image(resized_image)
    
    # 결과가 있다면 첫 번째 얼굴의 결과를 반환
    if results:
        return results[0]  # (age, gender) 튜플 반환
    
    return (0, "couldn't find face")  # 얼굴이 감지되지 않은 경우

def do_gesture(image_file_path: str) -> str:
    gesture_model_path = 'gesture/train_0.txt'
    # Create an instance of GestureRecognition first
    gesture_recognizer = GestureRecognition(gesture_model_path)
    
    image = cv2.imread(image_file_path)
    if image is None:
        return "0"
    
    result = gesture_recognizer.process_frame(image)
    return result  # 문자열로 된 제스처 결과만 반환

def do_voice(audio_file_path: str) -> str:
    """
    음성 인식 함수.

    주어진 경로의 오디오 파일에 있는 음성을 인식하고,
    해당하는 글자를 반환합니다.

    파라미터:
        audio_file_path (str): 오디오 파일의 경로 (wav파일)

    리턴값:
        str: 각 음성에 해당하는 글자. (ex. "주민등록"을 인식했으면 "주민등록")
    """
    result = "주민등록" # 예시
    return result