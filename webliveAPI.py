import os
import configparser
import cv2
from face.demo.SSRNET_model import SSR_net, SSR_net_general
from gaze.eyetracking import CalibratedEyeTracker
from gesture.gesture_forlive import GestureRecognition
from face.demo.face_forlive import FaceDemo  # FaceDemo 클래스가 정의된 파일

# AI.ini 파일을 경로에서 불러와 읽습니다.
OUTPUT_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AI.ini')
output = configparser.ConfigParser()
ret = output.read(OUTPUT_FILE_PATH)

# 만약 실패하면 프로그램을 종료합니다.
if not ret:
    print(f"{OUTPUT_FILE_PATH} 파일이 존재하지 않거나, 읽을 수 없습니다.")
    exit(0)

# AI모델 로딩 작업을 수행합니다.
weight_file = "face/pre-trained/megaface_asian/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"  # 나이 예측 모델
weight_file_gender = "face/pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"  # 성별 예측 모델

# FaceDemo 초기화
# Initialize models
stage_num = [3, 3, 3]
lambda_local = 1
lambda_d = 1

img_size = 64  # 모델 입력 이미지 크기
ad = 0.5       # 얼굴 여백 비율

model_age = SSR_net(img_size, stage_num, lambda_local, lambda_d)()
model_age.load_weights(weight_file)

model_gender = SSR_net(img_size, stage_num, lambda_local, lambda_d)()
model_gender.load_weights(weight_file_gender, by_name=True, skip_mismatch=True)

face_model = FaceDemo(
    model_age=model_age,
    model_gender=model_gender,
    img_size=img_size,
    ad=ad,
    skip_frame=1,  # Sample_API에서는 프레임 건너뛰기를 하지 않음
)

print('모델 로딩 완료')

# AI모델 로딩이 완료되면, AI.ini의 loaded 항목을 1로 변경합니다.
output.set('MODEL', 'loaded', '1')
with open(OUTPUT_FILE_PATH, 'w') as ini:
    output.write(ini)

# 웹캠 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit(1)

def do_face(frame):
    # FaceDemo에 현재 프레임 전달 및 결과 처리
    processed_frame, results = face_model.process_image(frame)
    for age, gender in results:
        return age, gender
    return None

def do_gesture(frame):
    gesture_recognizer = GestureRecognition(model_path="C:/APPLICATOR/AIM/FFinal/gesture/train_0.txt")
    processed_image, gesture = gesture_recognizer.process_frame(frame)
    if gesture != "Undefined":
        return gesture
    return None

def do_voice(audio_file_path):
    # 음성 인식 예제 (실제 구현 필요)
    return "주민등록"

# 나중에 사용될지 모르겠어서 남겨둠 (시선 추적)
# def eyetracking():
#     tracker = CalibratedEyeTracker(show_window=True)
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         success, frame = cap.read()
#         if not success:
#             print("카메라를 찾을 수 없습니다.")
#             break
#         tracker.process_frame(frame)
#         gaze_position = tracker.get_current_gaze_position()
#         print(f"Current gaze position: {gaze_position}")
#         if cv2.waitKey(5) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()

print("웹캠 활성화 완료. 명령어를 입력하세요 (face, gesture, voice, quit).")

# 메인 루프
while True:
    # 웹캠에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 사용자 입력 받기
    user_input = input("명령어를 입력하세요: ").strip()

    if user_input == 'face' and output['FACE']['recognized'] == '0':
        face_result = do_face(frame)
        if face_result:
            age, gender = face_result
            print(f"얼굴이 인식되었습니다. 나이: {age}, 성별: {gender}")
            output.set('FACE', 'recognized', '1')
            output.set('FACE', 'age', str(age))
            output.set('FACE', 'sex', gender)
            with open(OUTPUT_FILE_PATH, 'w') as ini:
                output.write(ini)
        else:
            print("얼굴이 인식되지 않았습니다.")

    elif user_input == 'gesture' and output['GESTURE']['recognized'] == '0':
        gesture_result = do_gesture(frame)
        if gesture_result:
            print(f"제스쳐가 인식되었습니다: {gesture_result}")
            output.set('GESTURE', 'recognized', '1')
            output.set('GESTURE', 'value', gesture_result)
            with open(OUTPUT_FILE_PATH, 'w') as ini:
                output.write(ini)
        else:
            print("제스쳐가 인식되지 않았습니다.")

    elif user_input == 'voice' and output['VOICE']['recognized'] == '0':
        voice_result = do_voice("dummy_audio_path")
        if voice_result:
            print(f"음성이 인식되었습니다: {voice_result}")
            output.set('VOICE', 'recognized', '1')
            output.set('VOICE', 'value', voice_result)
            with open(OUTPUT_FILE_PATH, 'w') as ini:
                output.write(ini)
        else:
            print("음성이 인식되지 않았습니다.")

    # 미사용 코드 유지
    # elif user_input == 'eye' and output['EYE']['recognized'] == '0':
    #     print("시선 추적을 실행해야 합니다.")

    elif user_input == 'quit':
        print("프로그램을 종료합니다.")
        break

    # 프레임 출력 (원한다면 주석 해제)
    # cv2.imshow("Webcam", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
