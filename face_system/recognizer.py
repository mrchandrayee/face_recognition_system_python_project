import cv2
from .config import SIMILARITY_THRESHOLD

class FaceRecognizer:
    def __init__(self,database,threshold=SIMILARITY_THRESHOLD):
        self.db=database; self.threshold=threshold
        self.app=database.initialize_app()

    def process_frame(self,frame):
        faces=self.app.get(frame)
        results=[]
        for face in faces:
            match=self.db.match(face.embedding,self.threshold)
            x1,y1,x2,y2=map(int,face.bbox)
            results.append({"bbox":(x1,y1,x2,y2),**match})
        return results
