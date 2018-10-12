# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:54:09 2018

@author: user
"""

import time
import paho.mqtt.client as paho
import cv2   
import numpy as np
import imutils
import time
from multiprocessing import Queue    #使用多核心的模組 Queue
from collections import deque
import threading, time
import os
import requests
from PIL import Image
from scipy import ndimage

def thread_video(q):
    print('T1 start\n')
    
    #capturing video through webcam
    cap=cv2.VideoCapture(0)

    while(1):
        
        
        ret, img = cap.read()
    	    
    	#converting frame(img i.e BGR) to HSV (hue-saturation-value)
        if ret == True: 
            #img = imutils.resize(img, width=1000)            #定義視窗長寬
            hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        else:
            continue
        
        
        '''
    	#definig the range of red color
        red_lower=np.array([0,200,150],np.uint8)
        red_upper=np.array([60,255,255],np.uint8)
      
        '''  
    	#definig the range of red color
        red_lower=np.array([136,87,111],np.uint8)
        red_upper=np.array([180,255,255],np.uint8)
        
    	#defining the Range of Blue color
        blue_lower=np.array([99,115,150],np.uint8)
        blue_upper=np.array([110,255,255],np.uint8)
    	
    	#defining the Range of yellow color
        yellow_lower=np.array([22,60,200],np.uint8)
        yellow_upper=np.array([60,255,255],np.uint8)
    
    	#finding the range of red,blue and yellow color in the image
        red=cv2.inRange(hsv, red_lower, red_upper)
        blue=cv2.inRange(hsv,blue_lower,blue_upper)
        yellow=cv2.inRange(hsv,yellow_lower,yellow_upper)
    	
    	#Morphological transformation, Dilation  	
        kernal = np.ones((5 ,5), "uint8")
       
        red=cv2.dilate(red, kernal)
        res=cv2.bitwise_and(img, img, mask = red)
    
        blue=cv2.dilate(blue,kernal)
        res1=cv2.bitwise_and(img, img, mask = blue)
    
        yellow=cv2.dilate(yellow,kernal)
        res2=cv2.bitwise_and(img, img, mask = yellow)    
    
    
    	#Tracking the Red Color
        (_,contours,hierarchy)=cv2.findContours(red,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    	
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            #print(area)
            if(area>1000):
                ((x, y), radius) = cv2.minEnclosingCircle(contour)	
                cv2.circle(img, (int(x), int(y)), int(radius),(0,0,255),2)
                #cv2.putText(img,"RED color",(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
              			
    	#Tracking the Blue Color
        (_,contours,hierarchy)=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area>300):
                x,y,w,h = cv2.boundingRect(contour)	
                img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(img,"Blue color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))
    
    	#Tracking the yellow Color
        (_,contours,hierarchy)=cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
        	area = cv2.contourArea(contour)
        	if(area>300):
        		x,y,w,h = cv2.boundingRect(contour)	
        		img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        		cv2.putText(img,"yellow  color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))  
                

            
        	#cv2.imshow("Redcolour",red)
        cv2.imshow("Color Tracking",img)
    
        	#cv2.imshow("red",res) 	
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
        
        
    print('T1 finish')
def t2_start() :
    fsize = os.path.getsize("D:\Desktop\machine_data"+"\\"+"output..png")
    fsize = fsize/float(1024)
    while(fsize>40):
           thread2 = threading.Thread(target=T2_job, args=(q,))
           thread2.start() 
           break
     
def T2_job(q):
    broker="broker.mqttdashboard.com"
    print('T2 start') 
    
    def on_connect(client, userdata, flags, rc):
        print("Connected with result code "+str(rc))
        client.subscribe("house/")

    def on_message(client, userdata, msg):
        print("message topic " ,str(msg.topic))
        print("message received " ,str(msg.payload.decode("utf-8")))
        
        
    client = paho.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, 1883, 60)
    client.loop_forever()
    
    print('T2 finish')
    
    
    
'''
def t3_start() :
    thread3 = threading.Thread(target=T3_job, args=(q,))
    thread3.start() 

    

def T3_job(q):
    print('T3 start') 
    text_b=q.get()
    
    if (text_a!=text_b):
        global text_a
        text_a=text_b
        my_data = {'key': text_a}
        r = requests.post('http://2fb191a0.ngrok.io/python1', data = my_data)


        
    #print(r.status_code)
    
    print('T3 finish')
 
'''


def main():
    #thread1 = threading.Thread(target=thread_video, args=(q,))
    #thread1.start()
    thread2 = threading.Thread(target=T2_job, args=(q,))
    thread2.start()
    print('all done')
    
    
if __name__=='__main__':
    q = Queue() # 開一個 Queue 物件
    main()
   

