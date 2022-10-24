from asyncio.windows_events import NULL
from deepface import DeepFace
import cv2

        
def Face_analyzer(ImageFrame):
    
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  
    
    frame = ImageFrame
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    Analyzer=""
    try:
        Analyzer = DeepFace.analyze(frame , actions = ['emotion'])
        result = Analyzer['dominant_emotion']
    except:
        result = ""
        pass
    
    faces = faceCascade.detectMultiScale(gray , 1.1 , 4)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,255,0) , 2)
    
    if(result != ""):
        cv2.putText(frame , 
                    result , (50,50) , 
                    cv2.FONT_HERSHEY_COMPLEX , 
                    3 , (0,0,255) , 2 , cv2.LINE_4);
    
    if(Analyzer):
        return frame , Analyzer
    return frame , ""
            

    
    
    
    