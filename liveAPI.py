

"""

This API is an example.

You can change it to suit your situation.

Input to Python uses standard input, and output from Python uses file writing.


"""

import numpy as np
import os
import configparser
import cv2
import time
import multiprocessing
from face.demo.SSRNET_model import SSR_net, SSR_net_general
from face.demo.face_forlive import FaceDemo
from gesture.gesture_forlive import GestureRecognition
from process import KoreanSimilarity
import time
import faster_whisper
import sounddevice as sd
from scipy.io.wavfile import write
from eye.eyetracking import CalibratedEyeTracker
from multiprocessing import Process,Queue

def perform_calibration(frame, eyetracker):
    """캘리브레이션 수행"""
    # 캘리브레이션 윈도우 생성
    cv2.namedWindow('Calibration', cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('Calibration', 0, 0)
    cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # 캘리브레이션 화면 생성
    screen_h, screen_w = eyetracker.SCREEN_H, eyetracker.SCREEN_W
    calibration_window = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    
    # 중앙에 점 표시
    center_x, center_y = screen_w // 2, screen_h // 2
    cv2.circle(calibration_window, (center_x, center_y), 20, (0, 255, 0), -1)
    
    # 안내 텍스트 표시
    text = "Look at the green dot (3 seconds)"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (screen_w - text_size[0]) // 2
    text_y = center_y - 50
    cv2.putText(calibration_window, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Calibration', calibration_window)
    
    # 3초 동안 데이터 수집
    start_time = time.time()
    iris_positions = []
    head_poses = []
    
    while time.time() - start_time < 3:
        if frame is not None:
            iris_pos, head_pose = eyetracker.process_frame(frame)
            if iris_pos is not None and head_pose is not None:
                iris_positions.append(iris_pos)
                head_poses.append(head_pose)
    
    # 창 닫기
    cv2.destroyWindow('Calibration')
    
    # 평균값 계산
    if iris_positions and head_poses:
        avg_iris_pos = np.mean(iris_positions, axis=0)
        avg_head_pose = np.mean(head_poses, axis=0)
        return avg_iris_pos, avg_head_pose
    return None, None

def get_frames(capture:int, q:Queue, event:multiprocessing.Event):
    capture=cv2.VideoCapture(capture)
    try :
        while True:
            if event.is_set():
                capture.release()
                print('quiting')
                q.put([])
                break
            try :         
                ret, frame = capture.read()
            except Exception as e:
                capture.release
                return
            if not ret:
                break
            if q.qsize()>1:
                try:
                    q.get()  # discard previous (unprocessed) frame
                except q.empty():
                    pass
            q.put(frame)
    except KeyboardInterrupt:
        print('here')
        capture.release()
        return
    print('done')
    return
    
if __name__=='__main__':
    try :
        OUTPUT_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AI.ini')
        output = configparser.ConfigParser()     
        # AI모델 로딩 작업을 수행합니다.
        weight_file = "face/pre-trained/megaface_asian/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5" # 나이 예측 모델
        weight_file_gender = "face/pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5" # 성별 예측 모델


        # FaceDemo 초기화 ######
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
        # self, weight_file, weight_file_gender, img_size=64, ad=0.5, skip_frame=5
        face_model = FaceDemo(
            model_age=model_age,
            model_gender=model_gender,
            img_size=img_size,
            ad=ad,
            skip_frame=1,  # Sample_API에서는 프레임 건너뛰기를 하지 않음
        )
        ################
        # gesture 모델 초기화 #############
        gesture_model_path = 'gesture/train_0.txt'
        gesture_model = GestureRecognition._initialize_model(gesture_model_path)
        gesture_recognizer = GestureRecognition(gesture_model)
        #########################
        cap = 0
        q=Queue()
        #voice_transcriber = get_transcriber(1)
        event = multiprocessing.Event()
        p1=Process(target=get_frames,args=[cap,q,event])
        p1.start()

        eyetracker = CalibratedEyeTracker()


        print('')
        print('Models Loaded')


        ccc = configparser.ConfigParser()
        ccc.read(OUTPUT_FILE_PATH)
        ccc.set('MODEL', 'loaded', '1')
        dc=False
        while not dc:
            try:
                with open(OUTPUT_FILE_PATH, 'w') as ini:
                    ccc.write(ini)
                    dc=True
            except Exception:
                continue

        while True:

            ret = output.read(OUTPUT_FILE_PATH)
            
            if ret == []:
                break

            #try :
            #    user_input = output['INPUT']['value']
            #except Exception as e:
                
            user_input = input()
            if user_input=='0':
                continue

            if user_input == 'face' and output['FACE']['recognized'] == '0':
                # 얼굴 인식을 수행합니다.
                done = False
                while not done:
                    frame = q.get()  # 큐에서 프레임 가져오기
                    # 이미지를 640x480으로 크기 조정
                    resized_image = cv2.resize(frame, (640, 480))

                    # FaceDemo에 현재 프레임 전달 및 결과 처리
                    result = face_model.process_image(resized_image)  # 결과 반환
                    print(result)
                    if result:
                        done = True

                if result:  # 결과가 존재하면 처리
                    age, gender = result  # result는 (age, gender) 형태로 반환
                    print(f"Face has been recognized.")
                    
                    # AI.ini 파일에 결과 기록
                    output.set('FACE', 'recognized', '1')
                    output.set('FACE', 'age', age)
                    output.set('FACE', 'sex', gender)
                    
                    # 파일에 쓰기
                    done = False
                    while not done:
                        try:
                            with open(OUTPUT_FILE_PATH, 'w') as ini:
                                output.write(ini)
                                done = True
                        except Exception as e:
                            pass
                else:
                    print("Face not recognized")

            if user_input == 'voice' and output['VOICE']['recognized'] == '0':

                while True:
                    recon=False
                    while not recon:
                        try:
                            output.read(OUTPUT_FILE_PATH)
                            rec=output['VOICE']['recognized']
                            recon=True
                        except Exception as e:
                            continue
                    if rec== '0':
                    #if True:
                        # Do voice recognition
                        transcription = voice_transcriber.transcribe()
                        if transcription:
                            # If recognized
                            print("Voice has been recognized.")
                            print(transcription)
                            output.set('VOICE', 'recognized', '1')
                            output.set('VOICE', 'value', transcription)
                            
                            done=False
                            while not done:
                                try :
                                    with open(OUTPUT_FILE_PATH, 'w') as ini:
                                        output.write(ini)
                                        done=True
                                except Exception as e:
                                    pass

                            if transcription == '모바일':
                                print('Exit Voice mode.')
                                break
                        else :
                            print("Voice not recognized")
                    time.sleep(1)
                    
            if user_input == 'gesture' and output['GESTURE']['recognized'] == '0':    
                result = []            
                while True:
                    recon=False
                    while not recon:
                        try:
                            output.read(OUTPUT_FILE_PATH)
                            rec= output['GESTURE']['recognized']
                            recon=True
                        except Exception as e:
                            continue
                    if True:
                        # Do gesture recognition
                        fin = 0
                        if q.empty():
                            continue
                        frame=q.get()
                        res = gesture_recognizer.process_frame(frame)
                        if res:
                            result.append(str(res))
                            print(res, len(result))

                            # If recognized
                            print("Gesture has been recognized.")
                            output.set('GESTURE', 'recognized', str(len(result)))
                            output.set('GESTURE', 'value', ','.join([result[i] for i in range(len(result))]))
                            done=False
                            while not done:
                                try :
                                    with open(OUTPUT_FILE_PATH, 'w') as ini:
                                        output.write(ini)
                                        done=True
                                except Exception as e:
                                    pass
                            

                            if result[-1] == '10':
                                print('Exit Gesture mode.')
                                break
                        else:
                            print('Gesture not recognized')
                    time.sleep(1)

            if user_input == 'eye' and output['EYE']['recognized'] == '0':
                frame = q.get()
                # 캘리브레이션 수행
                iris_pos, head_pose = perform_calibration(frame, eyetracker)
                if iris_pos is not None and head_pose is not None:
                    eyetracker.set_calibration_data(iris_pos, head_pose)
                
                while True:
                    rec = False
                    recon = '0'
                    while not rec:
                        try:
                            output.read(OUTPUT_FILE_PATH)
                            recon = output['EYE']['recognized'] 
                            rec = True
                        except AttributeError:
                            pass
                    
                    frame = q.get()
                    eyetracker.process_frame(frame)
                    track, x, y = eyetracker.get_current_gaze_position()
                    if track:
                        print("Gaze has been tracked.")
                        output.set('EYE', 'recognized', '1')
                        output.set('EYE', 'x', str(x))
                        output.set('EYE', 'y', str(y))
                        print(x, y)
                        done = False
                        while not done:
                            try:
                                with open(OUTPUT_FILE_PATH, 'w') as ini:
                                    output.write(ini)
                                    done = True
                            except Exception:
                                continue
                    elif x <= -1:
                        print("Exit eye mode")
                        break
                    else:
                        print('eye not seen')

                    time.sleep(0.01)
            if user_input == 'quit':
                break
    except KeyboardInterrupt :
        print('interupted')
        pass
    event.set()
    #voice_transcriber.stop()
    print('exiting')
    while True:
        if q.empty():
            continue
        a=q.get()
        try :
            if any(a):
                continue
            else:
                p1.terminate()
                break
        except Exception as e:
            print(e)
            continue
      
    
    print("Program is terminated.")

