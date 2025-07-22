# 팀 이름
TEAM_NAME = "AI GO"
import cv2
from gesture.gesture_fortest import GestureRecognition
from face.demo.face_fortest import FaceDemo

def do_face(frame) -> tuple:
    # FaceDemo 인스턴스 생성
    face_demo = FaceDemo()
    
    # 이미지 크기 조정 (640x480)
    resized_frame = cv2.resize(frame, (640, 480))
    
    # FaceDemo에 현재 프레임 전달 및 결과 처리
    annotated_frame, results = face_demo.process_image(resized_frame)
    
    # 결과가 있다면 첫 번째 얼굴의 결과를 반환
    if results:
        return annotated_frame, results[0]  # (프레임, (age, gender)) 튜플 반환
    
    return resized_frame, (0, "unknown")  # 얼굴이 감지되지 않은 경우

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

def webcam_face_detection():
    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    
    # 웹캠 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Press 'q' to quit")
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # 프레임 처리
            processed_frame, (age, gender) = do_face(frame)
            
            # 결과 텍스트 추가
            result_text = f"Age: {age}, Gender: {gender}"
            cv2.putText(processed_frame, result_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 결과 표시
            cv2.imshow('Face Detection', processed_frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # 자원 해제
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_face_detection()