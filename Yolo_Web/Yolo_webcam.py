from ultralytics import YOLO
import cv2
import cvzone
import math


#  for webcam

# cap = cv2.VideoCapture(0)# it will work on 0 nor one since i have only one web cam that can be default one
# cap.set(3,1280)
# cap.set(4,720)

#  for video

cap = cv2.VideoCapture("v1.mp4")



model = YOLO('../Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success,img = cap.read()
    res=model(img,stream=True)
    for r in res:
        boxes=r.boxes
        for box in boxes:
# for CV
#             x1,y1,x2,y2=box.xyxy[0] # box.xywh also can be used
#             x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
#             # print(x1,y1,x2,y2)
#             cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3) # so first is image , box point x any y coordimnates and then the color , then width

# for cvzone
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)

            w, h = x2-x1,y2-y1
            # bbox=int(x1),int(y1),int(w),int(h)
            cvzone.cornerRect(img,(x1,y1,w,h))

            conf=math.ceil((box.conf[0]*100))/100
    #         craete a rectangle and type on that
    #  making the confidence level at the screen not out of the ecsreen
    #         cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1)))

            cls=int(box.cls[0])
            cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1)),scale=1,thickness=1)


    cv2.imshow("image",img)
    cv2.waitKey(1)

#