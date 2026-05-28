from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager
import re
import os
import time

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres-db")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant-vector")

# Global değişkenler tanımlanıyor
embedding_model = None
qdrant_client = None

def init_systems():
    """Veritabanı şemalarını ve ilk kuralları hazırlar."""
    print("Veritabanlarının hazır olması için 3 saniye bekleniyor...")
    time.sleep(3) 
    
    # --- 1. POSTGRESQL SEMBOLİK KATMAN ---
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST, 
            database="nlp_linguistics", 
            user="nlp_user", 
            password="nlp_strong_password"
        )
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lexicon (
                id SERIAL PRIMARY KEY, 
                root VARCHAR(100) NOT NULL UNIQUE, 
                pos_tag VARCHAR(20) NOT NULL
            );
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS morphotactics (
                id SERIAL PRIMARY KEY, 
                current_state VARCHAR(30) NOT NULL, 
                suffix_surface VARCHAR(20) NOT NULL,
                next_state VARCHAR(30) NOT NULL, 
                is_accept_state BOOLEAN DEFAULT TRUE,
                CONSTRAINT unique_transition UNIQUE (current_state, suffix_surface)
            );
        """)
        
        cursor.execute("""
            INSERT INTO lexicon (root, pos_tag) VALUES 
            ('kitap', 'NOUN'), 
            ('kalem', 'NOUN'), 
            ('göz', 'NOUN') 
            ON CONFLICT (root) DO NOTHING;
        """)
        
        cursor.execute("""
            INSERT INTO morphotactics (current_state, suffix_surface, next_state, is_accept_state) VALUES
            ('NOUN_ROOT', 'lar', 'PLURAL_STATE', true), 
            ('NOUN_ROOT', 'ler', 'PLURAL_STATE', true),
            ('NOUN_ROOT', 'ı', 'ACCUSATIVE_STATE', true), 
            ('NOUN_ROOT', 'i', 'ACCUSATIVE_STATE', true),
            ('PLURAL_STATE', 'ı', 'ACCUSATIVE_STATE', true), 
            ('PLURAL_STATE', 'i', 'ACCUSATIVE_STATE', true)
            ON CONFLICT (current_state, suffix_surface) DO NOTHING;
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("-> PostgreSQL Sembolik Katman Kuralları Hazır.")
    except Exception as e:
        print(f"-> PostgreSQL başlatılamadı: {e}")

    # --- 2. QDRANT NÖRAL VEKTÖR KATMANI ---
    try:
        global qdrant_client
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=6333)
        collection_name = "academic_turkish_semantic"
        collections = qdrant_client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if not exists:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            print("-> Qdrant Nöral Vektör Koleksiyonu Oluşturuldu.")
        else:
            print("-> Qdrant Koleksiyonu Hazır.")
    except Exception as e:
        print(f"-> Qdrant başlatılamadı: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI başlamadan önce model ve veritabanı kurulumlarının tamamlanmasını garanti eder."""
    print("MİMARİ BAŞLATILIYOR: Model yükleme ve DB entegrasyonu...")
    global embedding_model
    embedding_model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
    init_systems()
    print("MİMARİ HAZIR: Uygulama istekleri kabul etmeye başlayabilir.")
    yield
    # Kapanışta yapılacak bir şey varsa buraya yazılabilir.

# Uygulamayı lifespan mimarisi ile kuruyoruz
app = FastAPI(title="Hybrid Neuro-Symbolic Turkish NLP Engine", lifespan=lifespan)

class AcademicFSAParser:
    def db_get_root(self, potential_root: str) -> dict:
        try:
            conn = psycopg2.connect(host=POSTGRES_HOST, database="nlp_linguistics", user="nlp_user", password="nlp_strong_password", cursor_factory=RealDictCursor)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM lexicon WHERE root = %s", (potential_root,))
            res = cursor.fetchone()
            cursor.close()
            conn.close()
            return res
        except: 
            return None

    def db_check_transition(self, current_state: str, suffix: str) -> dict:
        try:
            conn = psycopg2.connect(host=POSTGRES_HOST, database="nlp_linguistics", user="nlp_user", password="nlp_strong_password", cursor_factory=RealDictCursor)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM morphotactics WHERE current_state = %s AND suffix_surface = %s", (current_state, suffix))
            res = cursor.fetchone()
            cursor.close()
            conn.close()
            return res
        except: 
            return None

    def parse_word(self, word: str) -> dict:
        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        current_stem = clean_word
        validated_root = None
        while len(current_stem) > 1:
            validated_root = self.db_get_root(current_stem)
            if validated_root: 
                break
            current_stem = current_stem[:-1]
            
        if not validated_root:
            return {"word": word, "status": "REJECTED", "reason": "Root Unknown"}
            
        remaining_suffixes = clean_word[len(current_stem):]
        if not remaining_suffixes:
            return {"word": word, "status": "ACCEPTED", "root": validated_root["root"], "automated_path": [f"{validated_root['pos_tag']}_ROOT"]}
        
        current_state = f"{validated_root['pos_tag']}_ROOT"
        path = [current_state]
        chunks = []
        if remaining_suffixes.startswith(("lar", "ler")):
            chunks.append(remaining_suffixes[:3])
            if remaining_suffixes[3:]: chunks.append(remaining_suffixes[3:])
        else:
            if remaining_suffixes: chunks.append(remaining_suffixes)

        for suffix in chunks:
            transition = self.db_check_transition(current_state, suffix)
            if transition:
                current_state = transition["next_state"]
                path.append(f"--({suffix})--> {current_state}")
            else:
                return {"word": word, "status": "REJECTED", "reason": f"Violation after {current_state}"}
                
        return {"word": word, "status": "ACCEPTED", "root": validated_root["root"], "final_state": current_state, "automated_path": path}

parser = AcademicFSAParser()

class ProcessRequest(BaseModel):
    text: str
    doc_id: int

class SearchRequest(BaseModel):
    query: str

@app.post("/analyze-and-index")
def analyze_and_index(request: ProcessRequest):
    if embedding_model is None or qdrant_client is None:
        raise HTTPException(status_code=503, detail="Model veya Vektör veritabanı henüz hazır değil.")
    
    words = request.text.split()
    fsa_results = [parser.parse_word(w) for w in words]
    rejected_count = sum(1 for r in fsa_results if r["status"] == "REJECTED")
    
    vector = embedding_model.encode(request.text).tolist()
    
    qdrant_client.upsert(
        collection_name="academic_turkish_semantic",
        points=[
            PointStruct(
                id=request.doc_id,
                vector=vector,
                payload={
                    "original_text": request.text,
                    "linguistic_score": {"total_words": len(words), "rejected_words": rejected_count}
                }
            )
        ]
    )
    return {
        "status": "PROCESSED_AND_INDEXED",
        "symbolic_fsa_analysis": fsa_results,
        "neural_vector_status": "Success"
    }

@app.post("/semantic-search")
def semantic_search(request: SearchRequest):
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Model hazır değil.")
    query_vector = embedding_model.encode(request.query).tolist()
    search_result = qdrant_client.search(
        collection_name="academic_turkish_semantic",
        query_vector=query_vector,
        limit=2
    )
    return {
        "query": request.query, 
        "matches": [{"doc_id": hit.id, "score": hit.score, "text": hit.payload.get("original_text")} for hit in search_result]
    }

@app.get("/health")
def health():
    return {"status": "ready"}
