import cv2

def draw_results(frame,results,show_score=True):
    for item in results:
        x1,y1,x2,y2=item["bbox"]
        known=item["is_match"]
        color=(50,210,70) if known else (30,50,230)
        label=item["name"]
        if show_score: label+=f" | {item['score']:.2f}"
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        ty=max(24,y1-9)
        cv2.rectangle(frame,(x1,ty-23),(x1+max(130,len(label)*9),ty+3),color,-1)
        cv2.putText(frame,label,(x1+4,ty-5),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA)
    return frame
