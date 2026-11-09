import os, pickle
from pathlib import Path
import cv2
import numpy as np
from .config import FACE_REG_DIR, CACHE_FILE, MODEL_NAME, PROVIDERS, DET_SIZE, SIMILARITY_THRESHOLD

class FaceDatabase:
    """Registered face embeddings; each image contributes its largest detected face."""
    def __init__(self, reg_dir=FACE_REG_DIR, cache_file=CACHE_FILE):
        self.reg_dir=Path(reg_dir)
        self.cache_file=Path(cache_file)
        self.records=[]
        self.app=None

    def initialize_app(self, app=None):
        if app is not None:
            self.app=app
        elif self.app is None:
            try:
                from insightface.app import FaceAnalysis
            except ImportError as exc:
                raise RuntimeError("Install InsightFace and ONNX Runtime: pip install insightface onnxruntime") from exc
            self.app=FaceAnalysis(name=MODEL_NAME, providers=PROVIDERS)
            self.app.prepare(ctx_id=-1, det_size=DET_SIZE)
        return self.app

    def _embedding(self, face):
        emb=np.asarray(face.embedding,dtype=np.float32)
        return emb/(np.linalg.norm(emb)+1e-10)

    def load_database(self, app=None, force_reload=False):
        self.initialize_app(app)
        self.reg_dir.mkdir(parents=True,exist_ok=True)
        if not force_reload and self.cache_file.exists():
            try:
                with self.cache_file.open("rb") as f:
                    cached=pickle.load(f)
                if isinstance(cached,list) and all("embedding" in r for r in cached):
                    self.records=cached
                    print(f"[Faces] Loaded {len(self.records)} registered faces from cache.")
                    return self.records
            except Exception as exc:
                print(f"[Faces] Cache load failed ({exc}); re-indexing directory.")
        self.records=[]
        valid_exts={".png",".jpg",".jpeg",".bmp",".webp"}
        files=[f for f in self.reg_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        print(f"[Faces] Indexing {len(files)} image files from {self.reg_dir}...")
        for file_path in sorted(files):
            img=cv2.imread(str(file_path))
            if img is None:
                print(f"[Faces] Warning: could not read {file_path}")
                continue
            faces=self.app.get(img)
            if not faces:
                print(f"[Faces] Warning: no face detected in {file_path.name}")
                continue
            best=max(faces,key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            x1,y1,x2,y2=map(int,best.bbox)
            x1,y1=max(0,x1),max(0,y1)
            crop=img[y1:max(y1+1,y2),x1:max(x1+1,x2)]
            crop=cv2.resize(crop,(120,120)) if crop.size else None
            self.records.append({"name":file_path.stem,"embedding":self._embedding(best),
                                 "crop":crop,"source":file_path.name})
        try:
            with self.cache_file.open("wb") as f: pickle.dump(self.records,f)
            print(f"[Faces] Saved cache: {len(self.records)} identities.")
        except Exception as exc: print(f"[Faces] Failed to save cache: {exc}")
        return self.records

    def match(self, query_embedding, threshold=SIMILARITY_THRESHOLD):
        """Cosine similarity match. Returns name, score, is_match and registered crop."""
        if not self.records or query_embedding is None:
            return {"name":"Unknown","score":0.0,"is_match":False,"reg_crop":None}
        q=np.asarray(query_embedding,dtype=np.float32)
        q=q/(np.linalg.norm(q)+1e-10)
        best_score=-1.0; best_record=None
        for rec in self.records:
            emb=np.asarray(rec["embedding"],dtype=np.float32)
            score=float(np.dot(q,emb)/(np.linalg.norm(emb)+1e-10))
            if score>best_score:
                best_score=score; best_record=rec
        is_match=best_score>=threshold
        return {"name":best_record["name"] if is_match else "Unknown",
                "score":max(0.0,best_score),"is_match":is_match,
                "reg_crop":best_record.get("crop") if best_record else None}
