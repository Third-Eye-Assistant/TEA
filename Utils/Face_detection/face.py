import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hiding logs
import tensorflow as tf
import cv2
import mediapipe as mp
import time
import numpy as np

def face_model():
    res = []
    cap = cv2.VideoCapture(0)
    pTime = 0
    model = tf.keras.models.load_model('Utils/Face_detection/face_final.h5')

    mp_face_detection = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils
    faceDetection = mp_face_detection.FaceDetection()

    while True:
        success, img = cap.read()

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)

        if results.detections:
            for id, detection in enumerate(results.detections):
                # mp_draw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                #print(img.shape)
                final_img = cv2.resize(img,(224,224))
                final_img = np.expand_dims(final_img,axis=0) # need 4th dimension
                final_img = final_img/255 # normalizing             

                prediction = model.predict(final_img)
                pred_gen = model.predict_generator(final_img)
                pred = np.argmax(prediction[0])
                print(np.round(np.amax(pred_gen[0])*100,2))
                mylist = ["Adi","Subham"]
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                #cv2.rectangle(img, bbox, (255, 0, 255), 2)
                cv2.putText(img, f"{mylist[pred]}", (bbox[0],bbox[1]-25 ), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
                res.append(mylist[pred])


        
        new_list = list(set(res))
        
        cv2.imshow("Output", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return new_list
