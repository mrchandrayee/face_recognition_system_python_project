import argparse, cv2, time
from face_system.database import FaceDatabase
from face_system.recognizer import FaceRecognizer
from face_system.visualizer import draw_results

def main():
    ap=argparse.ArgumentParser(description="Real-time face recognition")
    ap.add_argument("--source",default="0",help="Camera index or video file path")
    ap.add_argument("--threshold",type=float,default=0.45)
    ap.add_argument("--no-display",action="store_true")
    ap.add_argument("--reload",action="store_true",help="Force re-index registered faces")
    ap.add_argument("--output",default="",help="Optional annotated output video path")
    args=ap.parse_args()
    db=FaceDatabase(); db.load_database(force_reload=args.reload)
    recognizer=FaceRecognizer(db,args.threshold)
    source=int(args.source) if args.source.isdigit() else args.source
    cap=cv2.VideoCapture(source)
    if not cap.isOpened(): raise SystemExit(f"Could not open source: {args.source}")
    writer=None; fps=cap.get(cv2.CAP_PROP_FPS) or 25
    print("Press Q to quit. Add one clear face image per person in registered_faces/<person_name>.jpg")
    while True:
        ok,frame=cap.read()
        if not ok: break
        start=time.time()
        results=recognizer.process_frame(frame)
        draw_results(frame,results)
        elapsed=time.time()-start
        cv2.putText(frame,f"FPS: {1/max(elapsed,1e-6):.1f} | Faces: {len(results)}",(12,28),
                    cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
        if args.output and writer is None:
            h,w=frame.shape[:2]
            writer=cv2.VideoWriter(args.output,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
        if writer: writer.write(frame)
        if not args.no_display:
            cv2.imshow("Face Recognition",frame)
            if cv2.waitKey(1)&0xFF in (ord("q"),27): break
    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

if __name__=="__main__": main()
