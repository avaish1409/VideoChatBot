#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports

#audio to text
import speech_recognition as sr 
import pyttsx3 

# text to emotion
from text2emotion import get_emotion

# Chatbot
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

# display reply
import threading 
# import pyglet

# user emotion
from fer import FER
import matplotlib.pyplot as plt 
import cv2
import os

# display gif
import imageio

# hide logging info
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# extras - for video
import tensorflow as tf


# In[2]:


# simple video-chat-bot

class vcbot():
    def __init__(self):
        # flag - false if vcbot is diplaying reply (speaking something)
        global flag
        flag = True
        self.r = sr.Recognizer() 

        # Chatbot
        self.chatbot = ChatBot(
            'Charlie'
        )

        self.trainer = ChatterBotCorpusTrainer(self.chatbot)

        self.trainer.train(
            "./resources/my_export.json"
        )
        # speak
        self.engine = pyttsx3.init() 
        self.engine.setProperty('rate', 145)
        logger.info("VCbot initialized")
        
    def getUserEmotion(self):
        # detector for facial emotion
        detector = FER(mtcnn=True)
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cv2.imwrite('temp.jpeg', frame)
        img = plt.imread('temp.jpeg')
        res = detector.detect_emotions(img)
        os.remove('temp.jpeg')
        if len(res)==0:
            logger.info("No face detected")
            return 'neutral'
        res_emotion = res[0]['emotions']
        return max(res_emotion, key=res_emotion.get)
    
    def audioToText(self):
        # convert user audio to text (language = english)
        try: 

            # use the microphone as source for input. 
            with sr.Microphone() as source2: 

                # wait for a second to let the recognizer 
                # adjust the energy threshold based on 
                # the surrounding noise level 
                self.r.adjust_for_ambient_noise(source2, duration=0.2) 

                #listens for the user's input 
                audio2 = self.r.listen(source2) 

                # Using ggogle to recognize audio 
                MyText = self.r.recognize_google(audio2) 
                MyText = MyText.lower() 

                return MyText

        except sr.RequestError as e: 
            print("Could not request results; {0}".format(e)) 
            logger.warning("Aud2text: Could not request results; {0}".format(e))
            return 'error'

        except sr.UnknownValueError: 
            print("unknown error occured") 
            logger.warning("Aud2text: unknown error occured")
            return 'error'
        
    def getTextEmotion(self, t):
        # derive emotion from any text input
        res = get_emotion(t)
        emotion = max(res, key = res.get)
        if res[emotion] == 0:
            return 'neutral'
        return emotion.lower()
    
    def getChatReply(self, q):
        # chatterbot reply for given text input
        return str(self.chatbot.get_response(q))
    
    def vid(self, lock):
        # use gif for displaying reply to user
        gif = imageio.mimread('resources/boy-talk.gif')
        nums = len(gif)
        imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in gif]
        i = 0
        global flag
        while True:
            lock.acquire()
            if flag:
                # no movement if not speaking
                cv2.imshow("gif", imgs[0])
            else:
                # gif enabled for speaking
                cv2.imshow("gif", imgs[i])
            lock.release()
            pressed = cv2.waitKey(25)&0xFF
            if pressed == ord('q'):
                # quit
                logger.info("user requested to quit!")
                break
            if pressed == ord('r'):
                # just to check if gif is used in dynamic sense (check by pressing 'r')
                lock.acquire()
                flag = not flag
                lock.release()
            i = (i+1)%nums
        cv2.destroyAllWindows()


    def SpeakText(self, command): 
        #Initialize the engine to speak
        self.engine.say(command) 
        self.engine.runAndWait()     
        
    def ensemble(self, lock):
        # combined together: video-emotion, audio-to-text, text-emotion, emotion validation, chat-reply, speak-reply
        txt = ''
        while txt != 'exit':
            video_emotion = self.getUserEmotion()
            # print(video_emotion)
            logger.info("Video Emotion: " + video_emotion)
            self.SpeakText('Your Turn')
            txt = self.audioToText()
            # print('aud2txt: ', txt)
            logger.info("Audio To Text: " + txt)
            txt_emotion = self.getTextEmotion(txt)
            # print('textEmotion: ', txt_emotion)
            logger.info("Text Emotion: " + txt_emotion)
            txt_inference = ''
            if txt_emotion != video_emotion and video_emotion != 'neutral':
                txt_inference = 'I am '+video_emotion
                logger.info("Added Text: " + txt_inference)
            cbot_reply = self.getChatReply(txt+txt_inference)
            # print('cbot: ', cbot_reply)
            logger.info("Chatbot Reply: " + cbot_reply)
            lock.acquire()
            global flag
            flag = False
            lock.release()
            self.SpeakText(cbot_reply)
            logger.info("Speaking something")
            lock.acquire()
            flag = True
            lock.release()

    def run(self):
        #thread lock for critical section
        lock = threading.Lock()

        # creating thread 
        # thread 1: display gif
        t1 = threading.Thread(target = self.vid, args=(lock,)) 
        # thread2: ensemble
        t2 = threading.Thread(target = self.ensemble, args=(lock,)) 

        # starting thread 1 
        t1.start() 
        # starting thread 2 
        t2.start() 

        # wait until thread 1 is completely executed 
        t1.join()
        # wait until thread 2 is completely executed 
        t2.join() 

        # both threads completely executed 
        print("Done!")
        logger.info("Successfully completed execution, terminating vcbot!")
        return
    
    


# In[3]:


# vcbot().run()

