##EDITED BY ABHISHEK 20/12/2021 : 1:47 AM 

'''NOTES : Threshold Value should be 0.5 else it doesnt give results'''

import numpy as np
import cv2
import timeit
import datetime
import math
from line import get_points
from TFLite_detection_webcam import get_bbox
from TFLite_detection_webcam import VideoStream

import TFLite_detection_webcam as TF

#for SERVER SIDE
import json                    
import requests
#convert img to JSON object
import base64
import argparse
import os
import sys
import time
from threading import Thread
import importlib.util

global pt3_pxl,ymin,ymax
#API endpoint
#api = 'https://tdispeeddetection.free.beeceptor.com/success'  
api = 'https://tdinightday.free.beeceptor.com/success'


##

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

speed_limit = int(input('Enter The Speed Limit: '))
distance =int(input('Enter distance between 2 lines in Meters(for better results use 10 Meters): '))



global start_time,start_time1,later,later1,starttime,endtime

#get values from TFLite Model ( Quantization Model)



def show_angle(speed_limit):
    if speed_limit !=0:
        show_direction = cv2.imread("PromptAngleinfo.JPG")
        #cv2.imshow("Angle Help",show_direction)
        k = cv2.waitKey(1) & 0xff
        cv2.waitKey(50)
        Angle = int(input("Enter apporximate Angle with road :")) 

        return Angle
#Prompts user with demo image for choosing right angle.
Angle = show_angle(speed_limit) #get Angle input 
Angle = math.radians(Angle)
Angle = math.cos(Angle)




# Play until the user decides to stop    ## SENDING IMAGES TO SERVER FOR PROCESSING THERE>>>>>>>>>>>>>>>>>>>>>
#for sending data to server
# def send(img):
#     retval, buffer = cv2.imencode(".jpg", img)
#     img = base64.b64encode(buffer).decode('utf-8')
#     data = json.dumps({"image1": img, "id" : "2345AB"})
#     response = requests.post(api, data=data, timeout=5, headers = {'Content-type': 'application/json', 'Accept': 'text/plain'})
#     try:
#        data = response.json()     
#        print(data)                
#     except requests.exceptions.RequestException:
#        print(response.text)


#make line
vs = VideoStream()
vs.stream = cv2.VideoCapture("test.mp4")
frame = vs.read()

frame_ = frame.copy()
ymin,xmin,ymax,xmax,object_name,label,scores,frame,frame_rate_calc,boxes,max_cx_,max_cy_= get_bbox(vs)

line1, line2 = get_points(frame_)



#for line 1
l1_x1,l1_x2,l1_y1,l1_y2 = line1[0][0],line1[1][0],line1[0][1],line1[1][1]

#for line2
l2_x1,l2_x2,l2_y1,l2_y2 = line2[0][0],line2[1][0],line2[0][1],line2[1][1]

#last check point for reference of centroid tracking
'''find dist between frst 2 lines '''
starttrack = l1_y1
midtrack = l2_y1
lasttrack = int((midtrack-starttrack))
if lasttrack < 100 :
    lasttrack = (int(midtrack-starttrack)*3)+l2_y1
else:
    lasttrack = (int(midtrack-starttrack)*2)+l2_y1

print("start",starttrack)
print("last",lasttrack)
print("mid",midtrack)








## Function to Auto Calculate the detection range
'''takes input from users - speed_limit,gets FPS,distance,//pt2 and pt1 from line.py 
auto calibrates the last reference line on frame, in order to get min of 2 images for detection, code calibrates for ANY SPEED RANGE and any 
actual distance marked in Meters --- physically on ground, if distance is less say 2 Meters and you want to detect high speed of 120 KMph,code auto
calculates the new reference line, provided it doesnt fall outside the frame height'''

fps = frame_rate_calc
print('fps : ', fps)

def max_images(speed_limit,fps,distance,midtrack,starttrack): #midtrack is last line Y pt and starttrack is first line Y pt. First line from TOP.
  
  time2coverdistance = (distance/(speed_limit*0.277))
  max_img = (time2coverdistance*fps)
  if max_img <2.0:
    #cal distance which will ensure we get atleast 2 images of vehicle d = s*t
    max_dstnc = (speed_limit*0.277)*1/fps*2
    delta = (max_dstnc-distance)
    pxl_mtr = ((midtrack-starttrack)/distance)
    pt3 = delta*pxl_mtr
    pt3_pxl = round(midtrack+pt3)
    # print("max_dstnc",max_dstnc)
    # print("distance",distance)
    # print("delta",delta)
    # print("pxl_mtr",pxl_mtr)
    # print("pt3",pt3)
    # print("pt3_pxl",pt3_pxl)
  else:
    pt3_pxl = midtrack+100
    print("pt3",pt3_pxl)
    return pt3_pxl




#defining time variables: WARNING : DONT CHANGE THESE AT ALL>>>>>>>
start_time= datetime.datetime.now()

later= datetime.datetime.now()


# Play until the user decides to stop

# Initialize video stream
vs = VideoStream(resolution=(imW,imH),framerate=30).start()
#time.sleep(1)


flag = True
pt3_pxl = max_images(speed_limit,fps,distance,midtrack,starttrack)

selectedY =[]
selectedX=[]
hit_time=[]

#for crossing second line
selectedY_ =[]
selectedX_= []
hit_time_ =[]
#frequence
# count = 0
frequency = cv2.getTickFrequency()
flag3 = -1
while True:
    #ymin,xmin,ymax,xmax,object_name,label,scores,frame,frame_rate_calc,boxes = get_bbox(vs)
    ymin,xmin,ymax,xmax,object_name,label,scores,frame,frame_rate_calc,boxes,cx,cy = get_bbox(vs)
    fps = frame_rate_calc
    
    
    #print("FPS",fps)
    
    w = int(xmax - xmin)
    h = int(ymax-ymin)
    #cx =int((w/2) + xmin)
    #cy = int((h/2)+ymin)
    if flag3 > 0:
        roi = frame[ymin-50:ymin + h, xmin:xmin+ w]

        cv2.imshow(f'lane_{flag3}', roi)
        flag3 += 1
        if flag3 == 3:
            flag3 = -1

    
    
    cnt_in =0
    cnt_out =0
    # if cy> starttrack and flag is True and cy< midtrack:
    if cy> starttrack and cy< midtrack:

        selectedY.append(cy)
        selectedX.append(cx)
        t1 = cv2.getTickCount()
        start_time = datetime.datetime.now()
        hit_time.append(start_time)

        #++++++++++++++++++++++++++++++++++++++++

        print("HIT",hit_time)
        print("selectedY",selectedY)
        # print("selectedX",selectedX)
        #print(start_time)
        # final_tracking = list(zip(selectedX,selectedY,hit_time))
        # print("FINAL TRACKING",final_tracking)

        # flag = False #Alex
        cnt_in += 1
    # elif  cy > midtrack and cy < (midtrack+100) and flag is False:
    elif  cy > midtrack and cy < (midtrack+50):
        t2 = cv2.getTickCount()
        later = datetime.datetime.now()
        selectedY_.append(cy)
        selectedX_.append(cx)
        hit_time_.append(later)

        print("selectedY_",selectedY_)
        #print("selectedX_",selectedX_)
        print("hit_time_",hit_time_)
        time_calc = t2 - t1
        x = time_calc / frequency
        print('x : ', x)
        print('Time_calc : ', time_calc)

        seconds = (later - start_time).total_seconds()
        frame_crossed1 = x * fps
        speed_insta = (distance/frame_crossed1)*fps*3.6
        
        speed = speed_insta*Angle
        print("SPEED",speed)
        print("frame_crossed1",frame_crossed1)
        print("Time taken",seconds)
        # flag = True
        cnt_out+=1

        missed_obj = (cnt_in - cnt_out)

        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++",missed_obj)
        if seconds <= 0.2:
            print("diff 0")
        else:
            # if flag is True:
            if 1==1:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, str(int(speed)), (xmin, ymin), font, 2, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(frame, str(int(speed)), (xmin, ymin), font, 2, (255, 255, 255), 4, cv2.LINE_AA)
                # if not os.path.exists(path):
                #     os.makedirs(path)
                if int(speed) > speed_limit and cy <= (midtrack +100):
                    roi = frame[ymin-50:ymin + h, xmin:xmin+ w]

                    cv2.imshow("Lane_1", roi)
                    flag3 = 1


                    # write_name = 'corners_found' + str(cnt1) + '.jpg'
                    # cv2.imwrite(write_name, roi)
                    # cv2.imwrite(os.path.join(path, 'carimage_l2_' + str(cnt1)) + '.jpg', roi)
                    
                    flag = True
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, str(int(speed)), (xmin, ymin), font, 2, (255, 255, 255), 8, cv2.LINE_AA)
                    # # print("ROI : ", roi)
                    # path = '/home/lionel/Desktop/TDI_speed/speedDetection/'
                    # cnt1 = cnt1
                    # if int(speed) > 20:
                    #     photo1 = cv2.imwrite(os.path.join(path, 'image1' + str(cnt1)) + '.jpg', roi)
                    #     photo1 = cv2.imwrite(os.path.join(path, 'image1'  + str(cnt1)) + '.jpg', roi)
                    #     photo1 = cv2.imwrite(os.path.join(path, 'image1'  + str(cnt1)) + '.jpg', roi)

                        

                
                
    cv2.line(frame, (l1_x1, l1_y1), (l1_x2,l1_y2), (0, 255, 0), 1)
    cv2.line(frame, (l2_x1, l2_y1), (l2_x2,l2_y2), (0, 0, 255), 1)
    cv2.line(frame, (l2_x1, pt3_pxl), (l2_x2,pt3_pxl), (0, 255, 255), 1)

        #cv2.line(frame,(l1_x1,pt3_pxl),(l1_x2,pt3_pxl),(125,200,127),3)
    
    cv2.imshow("Robust",frame )

    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()

