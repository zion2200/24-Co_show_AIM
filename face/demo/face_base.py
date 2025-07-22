import os
import cv2
import numpy as np
import timeit
from SSRNET_model import SSR_net, SSR_net_general
import tensorflow as tf

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

def draw_results(detected, input_img, faces, ad, img_size, img_w, img_h, model, model_gender, time_detection, time_network, time_plot):
    for i, (x, y, w, h) in enumerate(detected):
        x1, y1, x2, y2 = x, y, x + w, y + h
        xw1 = max(int(x1 - ad * w), 0)
        yw1 = max(int(y1 - ad * h), 0)
        xw2 = min(int(x2 + ad * w), img_w - 1)
        yw2 = min(int(y2 + ad * h), img_h - 1)

        faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
        faces[i, :, :, :] = cv2.normalize(faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(input_img, (xw1, yw1), (xw2, yw2), (0, 0, 255), 2)

    start_time = timeit.default_timer()
    if len(detected) > 0:
        try:
            predicted_ages = model.predict(faces)
            predicted_genders = model_gender.predict(faces)
        except Exception as e:
            print(f"Prediction error: {e}")
            return input_img, time_network, time_plot

    for i, (x, y, w, h) in enumerate(detected):
        gender_str = 'male' if predicted_genders[i] >= 0.5 else 'female'
        label = "{},{}".format(int(predicted_ages[i]), gender_str)
        draw_label(input_img, (x, y), label)

    elapsed_time = timeit.default_timer() - start_time
    time_network += elapsed_time

    start_time = timeit.default_timer()
    cv2.imshow("result", input_img)
    elapsed_time = timeit.default_timer() - start_time
    time_plot += elapsed_time

    return input_img, time_network, time_plot

def main():
    tf.keras.backend.clear_session()
    
    # 경로 설정 (필요에 맞게 수정)
    weight_file = "face/pre-trained/megaface_asian/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"
    weight_file_gender = "face/pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"

    # 얼굴 인식 모델 로드
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # 다른 모델 사용
    os.makedirs('./img', exist_ok=True)

    img_size = 64
    stage_num = [3, 3, 3]
    lambda_local = 1
    lambda_d = 1

    # 모델 로드
    model = SSR_net(img_size, stage_num, lambda_local, lambda_d)()
    model.load_weights(weight_file)

    model_gender = SSR_net(img_size, stage_num, lambda_local, lambda_d)()
    model_gender.load_weights(weight_file_gender, by_name=True, skip_mismatch=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 해상도 낮추기
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    img_idx = 0
    detected = []
    time_detection = 0
    time_network = 0
    time_plot = 0
    skip_frame = 5
    ad = 0.5

    while True:
        ret, input_img = cap.read()
        if not ret:
            break

        img_idx += 1
        img_h, img_w, _ = input_img.shape

        if img_idx == 1 or img_idx % skip_frame == 0:
            time_detection = 0
            time_network = 0
            time_plot = 0

            gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            start_time = timeit.default_timer()
            detected = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            elapsed_time = timeit.default_timer() - start_time
            time_detection += elapsed_time
            faces = np.empty((len(detected), img_size, img_size, 3))

            input_img, time_network, time_plot = draw_results(
                detected, input_img, faces, ad, img_size, img_w, img_h, model, model_gender, time_detection, time_network, time_plot
            )
            cv2.imwrite(f'img/{img_idx}.png', input_img)
        else:
            input_img, time_network, time_plot = draw_results(
                detected, input_img, faces, ad, img_size, img_w, img_h, model, model_gender, time_detection, time_network, time_plot
            )

        print(f'avefps_time_detection: {1 / time_detection:.2f}')
        print(f'avefps_time_network: {skip_frame / time_network:.2f}')
        print(f'avefps_time_plot: {skip_frame / time_plot:.2f}')
        print(f'gender" {model_gender}')
        print('===============================')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()