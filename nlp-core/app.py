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

# Global bileşen tanımlamaları
embedding_model = None
qdrant_client = None

def init_systems():
    """
    Sistem ilk açıldığında ilişkisel ve vektörel veritabanlarını hazırlar.
    Genişletilmiş akademik Türkçe kök ve morfotaktik kural matrisini yükler.
    """
    print("Veritabanlarının hazır olması için bekliyor (3 saniye)...")
    time.sleep(3) 
    
    # --- 1. POSTGRESQL SEMBOLİK DİLBİLİM KATMANI ---
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST, 
            database="nlp_linguistics", 
            user="nlp_user", 
            password="nlp_strong_password"
        )
        cursor = conn.cursor()
        
        # Kelime Kökü ve Türü Sözlüğü (Lexicon)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lexicon (
                id SERIAL PRIMARY KEY, 
                root VARCHAR(100) NOT NULL UNIQUE, 
                pos_tag VARCHAR(20) NOT NULL
            );
        """)
        
        # Sonlu Durumlu Otomat Durum Geçiş Matrisi (Morphotactics)
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
        
        # Genişletilmiş Akademik Sözlük Verileri (İsim ve Fiil Kökleri)
        cursor.execute("""
            INSERT INTO lexicon (root, pos_tag) VALUES 
            ('kitap', 'NOUN'), 
            ('kalem', 'NOUN'), 
            ('göz', 'NOUN'),
            ('oku', 'VERB'),
            ('gel', 'VERB'),
            ('yaz', 'VERB')
            ON CONFLICT (root) DO NOTHING;
        """)
        
        # Genişletilmiş FSA Geçiş Matrisi (Türkçe Morfotaktik Kuralları)
        cursor.execute("""
            INSERT INTO morphotactics (current_state, suffix_surface, next_state, is_accept_state) VALUES
            -- İSİM ÇEKİM KURALLARI
            ('NOUN_ROOT', 'lar', 'PLURAL_STATE', true), 
            ('NOUN_ROOT', 'ler', 'PLURAL_STATE', true),
            ('NOUN_ROOT', 'ı', 'ACCUSATIVE_STATE', true), 
            ('NOUN_ROOT', 'i', 'ACCUSATIVE_STATE', true),
            ('PLURAL_STATE', 'ı', 'ACCUSATIVE_STATE', true), 
            ('PLURAL_STATE', 'i', 'ACCUSATIVE_STATE', true),

            -- FİİL ÇEKİM KURALLARI (Fiil Kökü -> Olumsuzluk / Zaman / Şahıs Hiyerarşisi)
            -- Kökten Olumsuzluk Durumuna Geçiş (Kabul durumu değil, zaman eki bekliyor)
            ('VERB_ROOT', 'ma', 'NEGATION_STATE', false),
            ('VERB_ROOT', 'me', 'NEGATION_STATE', false),
            
            -- Kökten Doğrudan Şimdiki Zaman Durumuna Geçiş
            ('VERB_ROOT', 'yor', 'PRESENT_TENSE_STATE', true),
            
            -- Olumsuz Şimdiki Zaman Kombinasyonları (Ünlü daralması gözetilerek tasarlanan pratik durumlar)
            ('VERB_ROOT', 'mıyor', 'PRESENT_TENSE_STATE', true),
            ('VERB_ROOT', 'miyor', 'PRESENT_TENSE_STATE', true),
            ('NEGATION_STATE', 'yor', 'PRESENT_TENSE_STATE', true),

            -- Şimdiki Zamandan Şahıs Eklerine Geçiş Durumları (Kabul Durumları)
            ('PRESENT_TENSE_STATE', 'um', 'PERSON_1SG_STATE', true),   -- okuyor-um
            ('PRESENT_TENSE_STATE', 'sun', 'PERSON_2SG_STATE', true),  -- okuyor-sun
            ('PRESENT_TENSE_STATE', 'lar', 'PERSON_3PL_STATE', true),  -- okuyor-lar
            ('PRESENT_TENSE_STATE', 'ler', 'PERSON_3PL_STATE', true)   -- geliyor-ler
            ON CONFLICT (current_state, suffix_surface) DO NOTHING;
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("-> PostgreSQL Sembolik Katman Kuralları Hazır.")
    except Exception as e:
        print(f"-> PostgreSQL ilklendirme hatası: {e}")

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
        print(f"-> Qdrant ilklendirme hatası: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI'nin yaşam döngüsünü kontrol eder; modeller yüklenmeden HTTP istek hattını açmaz."""
    print("MİMARİ BAŞLATILIYOR: BERT Modeli yükleniyor ve DB optimizasyonları yapılıyor...")
    global embedding_model
    embedding_model = SentenceTransformer("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
    init_systems()
    print("MİMARİ HAZIR: Çekirdek servis tüm isteklere açık.")
    yield

# Uygulama güvenli asenkron lifespan altyapısı ile başlatılıyor
app = FastAPI(title="Hybrid Neuro-Symbolic Turkish NLP Engine", lifespan=lifespan)


class AcademicFSAParser:
    """
    Türkçe dil kurallarını veritabanından dinamik olarak okuyup işleten,
    soldan sağa doğru ek eriten esnek Sonlu Durumlu Otomat (FSA) motoru.
    """
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
        
        # 1. Adım: En uzun kökü bul (Sözlük Tarama Katmanı)
        current_stem = clean_word
        validated_root = None
        while len(current_stem) > 1:
            validated_root = self.db_get_root(current_stem)
            if validated_root: 
                break
            current_stem = current_stem[:-1]
            
        if not validated_root:
            return {"word": word, "status": "REJECTED", "reason": "Kök Sözlükte Bulunamadı (OOV)"}
            
        remaining_suffixes = clean_word[len(current_stem):]
        current_state = f"{validated_root['pos_tag']}_ROOT"
        path = [current_state]
        
        # Kelime sadece kökten ibaretse doğrudan kabul durumuna bakılır
        if not remaining_suffixes:
            return {
                "word": word, 
                "status": "ACCEPTED", 
                "root": validated_root["root"], 
                "pos": validated_root["pos_tag"],
                "automated_path": path
            }
        
        # 2. Adım: Dinamik Ek Çözümleme Katmanı (FSA İlerleme Motoru)
        while len(remaining_suffixes) > 0:
            matched_suffix = None
            
            # En uzun uyumlu eki bulmak için sağdan sola doğru daralan pencere analizi
            for i in range(len(remaining_suffixes), 0, -1):
                potential_suffix = remaining_suffixes[:i]
                transition = self.db_check_transition(current_state, potential_suffix)
                
                if transition:
                    matched_suffix = potential_suffix
                    current_state = transition["next_state"]
                    path.append(f"--({matched_suffix})--> {current_state}")
                    break
            
            if matched_suffix:
                # Eşleşen ek dizilimden düşürülür, kalanı için döngü sürer
                remaining_suffixes = remaining_suffixes[len(matched_suffix):]
            else:
                # İzin verilen kuralların dışına çıkıldıysa morfotaktik ihlal verilir
                return {
                    "word": word, 
                    "status": "REJECTED", 
                    "reason": f"Morfotaktik İhlal: '{current_state}' durumundan sonra gelen '{remaining_suffixes}' eki geçersizdir."
                }
                
        return {
            "word": word, 
            "status": "ACCEPTED", 
            "root": validated_root["root"], 
            "pos": validated_root["pos_tag"],
            "final_state": current_state, 
            "automated_path": path
        }

parser = AcademicFSAParser()

# --- PYDANTIC MODEL TANIMLAMALARI ---
class ProcessRequest(BaseModel):
    text: str
    doc_id: int

class SearchRequest(BaseModel):
    query: str

# --- ENDPOINT KONTROLLERİ ---

@app.post("/analyze-and-index")
def analyze_and_index(request: ProcessRequest):
    if embedding_model is None or qdrant_client is None:
        raise HTTPException(status_code=503, detail="Yapay zeka modelleri veya veritabanları henüz hazır değil.")
    
    words = request.text.split()
    fsa_results = [parser.parse_word(w) for w in words]
    rejected_count = sum(1 for r in fsa_results if r["status"] == "REJECTED")
    
    # Nöral Alan: BERT modeli ile cümle düzeyinde semantik embedding üretimi
    vector = embedding_model.encode(request.text).tolist()
    
    # Qdrant Vektör Veritabanına kayıt yükleme (Payload kısmına sembolik süzgeç skorları gömülür)
    qdrant_client.upsert(
        collection_name="academic_turkish_semantic",
        points=[
            PointStruct(
                id=request.doc_id,
                vector=vector,
                payload={
                    "original_text": request.text,
                    "linguistic_score": {
                        "total_words": len(words),
                        "rejected_words": rejected_count
                    }
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
        raise HTTPException(status_code=503, detail="Gömülü dil modeli yüklenemedi.")
        
    query_vector = embedding_model.encode(request.query).tolist()
    
    search_result = qdrant_client.search(
        collection_name="academic_turkish_semantic",
        query_vector=query_vector,
        limit=2
    )
    
    return {
        "query": request.query, 
        "matches": [
            {
                "doc_id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("original_text")
            } for hit in search_result
        ]
    }

@app.get("/health")
def health():
    return {"status": "ready", "engine": "neuro_symbolic_fsa_v2"}
