import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pytesseract
import asyncio
import cv2
import tensorflow as tf

tf.config.experimental.enable_mlir_graph_optimization()






class live_text:
    vid_camera_id=None
    flag=0
    min_value=3
    stopper=None
    readable_data_store={0:""}
    id_data_store={0:""}
    poster_data_store={0:""}
    length_store={}
    data=None
    model=None
    max_key=None

    predicted_text=None


    color = (20, 20, 20)

    # for i in range(stopper):
    #     length_store[i]=0
        

    predictions_dictonary_occurance={
        "Readable":0,
        "RoadSign":0,
        "Id":0,
        "Posters":0,
        "NoImage":0,
    }

    predictions_dictonary={
        0:"Readable",
        1 :"RoadSign",
        2 :"Id",
        3 :"Posters"
    }

    def __init__(self,vid_camera_id=0,stopper=10):
        self.vid_camera_id=vid_camera_id
        self.stopper=stopper
        self.model_path = "Utils/TextDetection/model/keras_model.h5"
        # np.set_printoptions(suppress=True)
        self.model = tf.keras.models.load_model(self.model_path)
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    

    
    def  reset(self):
    
        self.min_value=3
        self.predictions_dictonary_occurance={
            "Readable":0,
            "RoadSign":0,
            "Id":0,
            "Posters":0,
            "NoImage":0,
        }
    
        self.readable_data_store={0:""}
        self.id_data_store={0:""}
        self.poster_data_store={0:""}
        self.length_store={}

        return

        
        

    def getPrediction(self, img):
        imgS = cv2.resize(img, (224, 224))
        image_array = np.asarray(imgS)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        self.data[0] = normalized_image_array
        prediction = self.model.predict(self.data)
        indexVal = np.argmax(prediction)
        return indexVal

    def validate_image(self,img):
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        # print(laplacian_var)
        if laplacian_var < 5:
            return False
        else:
            return True
    def classify(self,image):
        index=self.getPrediction(image)
        prediction_name=self.predictions_dictonary[index]
        return index,prediction_name
    
    def mapper(self,h):
        h = h.reshape((4,2))
        
        hnew = np.zeros((4,2),dtype = np.float32)
        add = h.sum(1)
        
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]
        diff = np.diff(h,axis = 1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]
        return hnew
    async def get_prespectiveImage(self,approx):
        pts=np.float32([[0,0],[800,0],[800,800],[0,800]])
        op=cv2.getPerspectiveTransform(approx,pts)
        self.dst=cv2.warpPerspective(self.image,op,(800,800))

        return self.dst
    async def clear_text(self,text1):
        text2=text1.replace(','," ")
        text3=text2.replace('\n'," ")
        text4=text3.replace('\\'," ")
        text5=text4.replace('  '," ")
      

        return text5

    async def detect_text(self,image):
        # if self.image_process_flag == 0 :self.image_preprocess()

        self.text=pytesseract.image_to_string(image)
        self.text=await self.clear_text(self.text) 
        self.text_detection_flag=1

      

        if (len(self.text.split(" ")) < 3):
            print('No useful Image Found')
            return "No Text"
        else:
            return self.text
        

    def image_preprocess(self,img):
        self.image=img

        if(self.validate_image(img) != True):
            print('Please take image again')
            return
        orig=img.copy()
        gray=cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
        blurred=cv2.GaussianBlur(gray,(5,5),0)
        edge=cv2.Canny(blurred,30,50)
        contours,hierarchy=cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        contours=sorted(contours,key=cv2.contourArea,reverse=True)
        for c in contours:
            p=cv2.arcLength(c,True)
            approx=cv2.approxPolyDP(c,0.02*p,True)

            if (len(approx) == 4):
                self.target=approx
                break
        approx=self.mapper(self.target)

        return approx
    
    async def get_readable_text(self,text,length,img,isId=False,isPoster=False):
    
        if (isId==False and length > self.min_value and self.validate_image(img) and length not in self.readable_data_store):
            self.readable_data_store[length]=text
            self.min_value=length
        elif (isId==True and length > self.min_value and self.validate_image(img) and length not in self.id_data_store):
            self.id_data_store[length]=text
            self.min_value=length
        elif (isPoster==True and length > self.min_value and self.validate_image(img) and length not in self.poster_data_store):
            self.poster_data_store[length]=text
            self.min_value=length
    
        # print('exit')
        return
            

        
    def get_id_text(self):
        pass
    
    async def live_image(self):
        vid = cv2.VideoCapture(self.vid_camera_id)  
        
        if (vid.isOpened()==False):
            print('please check the webcam')
            return
        while vid.isOpened():
            _,img=vid.read()
            img=cv2.resize(img,(800,800))
            approx=self.image_preprocess(img)
            start_point = (int(approx[0][0]), int(approx[0][1]))
            end_point = (int(approx[2][0]), int(approx[2][1]))
            img=cv2.rectangle(img,start_point,end_point,self.color,3)
            dst=await self.get_prespectiveImage(approx)
            text=await self.detect_text(dst)
            id_text=await self.detect_text(img)


            __,prediction2=self.classify(dst)
            __,prediction1=self.classify(img)

            length=len(text.split(" "))
            id_text_length=len(id_text.split(" "))

            print((self.flag-self.stopper)," times remain")
            # print(id_text,text," times remain")

            self.flag+=1

            if (prediction2 =="Readable" and length >5):
                self.predictions_dictonary_occurance["Readable"]+=1
                # print("Readable")
                await  self.get_readable_text(text,length,dst) 
                
            elif (prediction2=="Id" and length > 2 or (prediction1=="Id" and id_text_length > 3 )):
                self.predictions_dictonary_occurance["Id"]+=1
                if (prediction1==prediction2):
                    await  self.get_readable_text(text,length,dst,True) 
                else:
                    await  self.get_readable_text(id_text,length,dst,True) 
                # print("identification card",prediction1,prediction2)

            elif (prediction2 == "Posters" and length > 5 or (prediction1 == "Posters" and id_text_length > 3) ):
                await  self.get_readable_text(id_text,length,dst,isPoster=True)
                self.predictions_dictonary_occurance["Posters"]+=1
                # print('Posters')
            elif (prediction2 == "RoadSign"):
                self.predictions_dictonary_occurance["RoadSign"]+=1
                # print("Road Sign detetcted")

            else:
                # print(self.predictions_dictonary_occurance["NoImage"])
                self.predictions_dictonary_occurance["NoImage"]+=1
                

                # print("no readable file")
                # print(text)


            if (self.flag == self.stopper or (cv2.waitKey(1) & 0xFF == ord('q'))):    
                break
            
            img2=cv2.resize(img,(400,400))
            cv2.imshow("Frame",img2)

            
           
        
        vid.release()
        cv2.destroyAllWindows()


        self.max_key = max(self.predictions_dictonary_occurance, key=self.predictions_dictonary_occurance.get)

        # print(self.predictions_dictonary_occurance)
        # print(self.readable_data_store)
        # print(self.id_data_store)
        # print(self.poster_data_store)
        # print(self.max_key)

        if(self.max_key =="Readable"):
            temp_store=max(self.readable_data_store.keys())
            self.predicted_text=self.readable_data_store[temp_store]
           
        elif(self.max_key =="RoadSign"):
            self.predicted_text="Road Sign Detected"
        elif(self.max_key =="Id"):
            temp_store=max(self.id_data_store.keys())
            self.predicted_text=self.id_data_store[temp_store]

        elif(self.max_key =="Posters"):
            temp_store=max(self.poster_data_store.keys())
            self.predicted_text=self.poster_data_store[temp_store]

        elif(self.max_key =="NoImage"):
            self.predicted_text="No Readable Image Found"
            

        
        
        return self.max_key,self.predicted_text

        


    def run(self):
        self.reset()
        return asyncio.run(self.live_image())
       



if __name__ == '__main__':
    lt=live_text(0,10)
    predicted_object,text_=lt.run()
    print(text_,predicted_object)
