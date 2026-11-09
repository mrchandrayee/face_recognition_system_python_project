import argparse, shutil
from pathlib import Path
from face_system.database import FaceDatabase

def main():
    ap=argparse.ArgumentParser(description="Register a face image")
    ap.add_argument("name",help="Identity name (used as folder/file stem)")
    ap.add_argument("image",help="Path to one clear face image")
    args=ap.parse_args()
    src=Path(args.image)
    if not src.exists(): raise SystemExit(f"Image not found: {src}")
    dest=Path("registered_faces"); dest.mkdir(exist_ok=True)
    safe="".join(c for c in args.name.strip() if c.isalnum() or c in "-_ ")
    if not safe: raise SystemExit("Please provide a valid name.")
    target=dest/(safe.replace(" ","_")+src.suffix.lower())
    shutil.copy2(src,target)
    FaceDatabase().load_database(force_reload=True)
    print(f"Registered: {target}")
if __name__=="__main__": main()
