import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 환경 변수 로드 (.env 파일에서 OPENAI_API_KEY 추출)
load_dotenv()

# 동적 경로 텐서 앵커링 (현재 파일의 위치를 영점으로 설정)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "chroma_db")

class AITutorEngine:
    def __init__(self):
        """AI 튜터 시스템의 핵심 엔진을 초기화하는 생성자"""
        print("[시스템] 텐서 파이프라인 초기화 중...")
        
        # 1. 로컬 임베딩 모델 로드 (한국어 다차원 텐서 매핑기)
        self.embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        
        # 2. 로컬 벡터 텐서 DB 연결
        self.vector_db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
        
        # 3. LLM (대형 언어 모델) 초기화 
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        
        print("[시스템] 튜터 엔진 가동 준비 완료!\n")

    def get_relevant_tensor(self, query: str, k: int = 3) -> str:
        """
        [독립 모듈] 지식 공간에서 텐서를 인출하는 함수
        향후 벡터 DB를 지식 그래프(Graph DB)로 교체하더라도, 이 함수 내부만 수정하면 됨.
        """
        # 사용자 질문과 가장 유사도가 높은 k개의 텐서 블록 검색
        docs = self.vector_db.similarity_search(query, k=k)
        
        # 검색된 텐서 블록들을 하나의 문자열 시퀀스로 병합
        context_tensor = "\n\n".join([doc.page_content for doc in docs])
        return context_tensor

    def generate_response(self, query: str) -> str:
        """사용자의 질문을 받아 최종 답변 텐서를 생성하는 메인 로직"""
        
        # 1. 지식 인출 (Retrieval)
        context = self.get_relevant_tensor(query)
        
        # 2. 프롬프트 텐서 조립
        # 시스템에 튜터로서의 페르소나와 제약 조건을 부여
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 정보처리기사 자격증 합격을 돕는 친절하고 똑똑한 AI 튜터다. 
주어진 [참고 지식]만을 바탕으로 학생의 질문에 답변해라. 
설명은 이해하기 쉽게 예시를 들어주고, 만약 [참고 지식]에 없는 내용이라면 솔직하게 모른다고 대답해라.

[참고 지식]
{context}"""),
            ("human", "{question}")
        ])
        
        # 3. 텐서 체인 구성 (Prompt -> LLM -> String Output)
        chain = prompt | self.llm | StrOutputParser()
        
        # 4. 연산 실행 및 결과 반환
        return chain.invoke({"context": context, "question": query})

# --- 실행 테스트 블록 ---
if __name__ == "__main__":
    tutor = AITutorEngine()
    
    print("="*50)
    print(" 정보처리기사 AI 튜터가 작동을 시작했습니다.")
    print("질문을 입력해주세요. (종료하려면 'q' 입력)")
    print("="*50)
    
    while True:
        user_input = input("\n학생 🙋‍♂️: ")
        
        if user_input.lower() == 'q':
            print("튜터 : 학습을 종료합니다. 수고하셨습니다!")
            break
            
        print("튜터 : (텐서 검색 및 답변 생성 중...)")
        
        try:
            answer = tutor.generate_response(user_input)
            print(f"\n튜터 : {answer}")
        except Exception as e:
            print(f"\n[오류] 텐서 연산 중 문제가 발생했습니다: {e}")