import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#python .\backend\local_tensor_loder.py 로 실행

# 현재 파일의 디렉토리 절대 경로를 구합니다 (backend/ 위치)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "chroma_db")


BASE_DATA_DIR = os.path.join(CURRENT_DIR, "storage", "data")
BASE_QUIZ_DIR = os.path.join(CURRENT_DIR, "storage", "quiz")

def create_tensor_db():
    print("1. 기존 텐서 공간(DB) 초기화 중...")
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    all_chunks = []

    # ==========================================
    # [A 파트] 모든 자격증 개념 데이터 자동 로딩 및 태깅
    # ==========================================
    print("\n2. [개념 데이터] 로딩 및 자동 메타데이터 부여 중...")
    if os.path.exists(BASE_DATA_DIR):
        # data/ 하위의 모든 자격증(.md) 파일을 가져옴
        loader = DirectoryLoader(
            BASE_DATA_DIR, 
            glob="**/*.md", 
            loader_cls=TextLoader, 
            loader_kwargs={'encoding': 'utf-8'}
        )
        concept_documents = loader.load()
        
        if concept_documents:
            # 법 조문과 IT 지식 블록을 모두 안정적으로 소화하는 700자 분할 세팅
            concept_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=700,
                chunk_overlap=50
            )
            concept_chunks = concept_splitter.split_documents(concept_documents)
            
            for chunk in concept_chunks:
                # 🚀 파일의 절대 경로를 분석해서 cert와 subject를 자동 추출하는 마법의 로직일세!
                file_path = chunk.metadata.get("source", "")
                rel_path = os.path.relpath(file_path, BASE_DATA_DIR)
                path_parts = rel_path.split(os.sep)
                
                # 예: EIP/topic1.md -> cert="EIP", subject="EIP"
                # 예: LREA_1/civil_law/civil_law_1.md -> cert="LREA_1", subject="civil_law"
                cert = path_parts[0] if len(path_parts) > 0 else "UNKNOWN"
                subject = path_parts[1] if len(path_parts) > 2 else cert
                
                chunk.metadata["doc_type"] = "concept"
                chunk.metadata["cert"] = cert        # 자격증 종류 꼬리표
                chunk.metadata["subject"] = subject  # 세부 과목 종류 꼬리표
                
            all_chunks.extend(concept_chunks)
            print(f" - 통합 개념 텐서 조각: {len(concept_chunks)}개 생성 완료")
    else:
        print(f"경고: {BASE_DATA_DIR} 폴더를 찾을 수 없습니다. 개념 처리를 건너뜁니다.")

    # ==========================================
    # [B 파트] 모든 자격증 퀴즈 데이터 자동 로딩 및 태깅
    # ==========================================
    print("\n3. [퀴즈 데이터] 글로벌 통합 로딩 및 자동 메타데이터 부여 중...")
    if os.path.exists(BASE_QUIZ_DIR):
        quiz_loader = DirectoryLoader(
            BASE_QUIZ_DIR, 
            glob="**/*.md", 
            loader_cls=TextLoader, 
            loader_kwargs={'encoding': 'utf-8'}
        )
        quiz_documents = quiz_loader.load()
        
        if quiz_documents:
            # 기출문제는 문제 단위가 쪼개지지 않도록 기존 규칙 엄격하게 유지!
            quiz_splitter = CharacterTextSplitter(
                separator="\n---\n", 
                chunk_size=1000,
                chunk_overlap=0
            )
            quiz_chunks = quiz_splitter.split_documents(quiz_documents)
            
            for chunk in quiz_chunks:
                file_path = chunk.metadata.get("source", "")
                rel_path = os.path.relpath(file_path, BASE_QUIZ_DIR)
                path_parts = rel_path.split(os.sep)
                
                cert = path_parts[0] if len(path_parts) > 0 else "UNKNOWN"
                subject = path_parts[1] if len(path_parts) > 2 else cert
                
                chunk.metadata["doc_type"] = "quiz"
                chunk.metadata["cert"] = cert
                chunk.metadata["subject"] = subject
                
            all_chunks.extend(quiz_chunks)
            print(f" - 통합 퀴즈 텐서 조각: {len(quiz_chunks)}개 생성 완료")
    else:
        print(f"알림: {BASE_QUIZ_DIR} 폴더가 아직 없어 퀴즈 처리를 건너뜁니다.")

    # ==========================================
    # [C 파트] 글로벌 병합 적재 (기존 구조 유지)
    # ==========================================
    if not all_chunks:
        print("\n오류: 로드할 텐서 조각이 전혀 없습니다. 폴더와 파일을 확인해주세요.")
        return

    print(f"\n4. 총 {len(all_chunks)}개의 텐서 조각을 통합 DB에 적재 중...")
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", encode_kwargs={'normalize_embeddings': True})
    
    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"\n5. 완료! 텐서 DB가 '{DB_DIR}' 폴더에 성공적으로 재구축되었습니다.")

if __name__ == "__main__":
    create_tensor_db()