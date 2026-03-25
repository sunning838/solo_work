import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# 환경 변수 로드
load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(CURRENT_DIR, "chroma_db")

class AITutorEngine:
    def __init__(self):
        print("[시스템] 메모리 텐서가 탑재된 파이프라인 초기화 중...")
        self.embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        self.vector_db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
        
        # 최신 텐서 모델 유지
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        print("[시스템] 튜터 엔진 가동 준비 완료!\n")

    def get_relevant_tensor(self, query: str, k: int = 3) -> str:
        """지식 공간에서 텐서를 인출하는 함수"""
        docs = self.vector_db.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])

    def generate_response(self, query: str, chat_history: list) -> str:
        """대화 기록(Memory Tensor)을 포함하여 최종 답변을 생성하는 메인 로직"""
        
        context = self.get_relevant_tensor(query)
        
        # 프롬프트 텐서에 'MessagesPlaceholder'를 추가하여 기억력을 이식!
        prompt = ChatPromptTemplate.from_messages([
            ("system", """너는 정보처리기사 자격증 합격을 돕는 친절하고 똑똑한 AI 튜터다. 
주어진 [참고 지식]과 이전 대화 맥락을 바탕으로 학생의 질문에 답변해라. 
설명은 이해하기 쉽게 예시를 들어주고, 지식에 없는 내용이라면 모른다고 대답해라.

[참고 지식]
{context}"""),
            # 이곳에 이전까지의 대화 텐서들이 통째로 삽입됨
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        # 연산 실행 시 chat_history 변수도 함께 넘겨줌
        return chain.invoke({
            "context": context, 
            "chat_history": chat_history, 
            "question": query
        })

# --- 실행 테스트 블록 ---
if __name__ == "__main__":
    tutor = AITutorEngine()
    
    # 세션 동안의 대화 텐서를 저장할 빈 리스트 생성 (메모리 모듈)
    session_chat_history = []
    
    print("="*50)
    print(" [기억력 탑재 완료] 정보처리기사 AI 튜터가 작동을 시작했습니다.")
    print("질문을 입력해주세요. (종료하려면 'q' 입력)")
    print("="*50)
    
    while True:
        user_input = input("\n학생 : ")
        
        if user_input.lower() == 'q':
            print("튜터 : 학습을 종료합니다. 수고하셨습니다!")
            break
            
        print("튜터 : (텐서 맥락 분석 및 답변 생성 중...)")
        
        try:
            # 질문과 함께 누적된 대화 기록을 튜터 엔진에 전달
            answer = tutor.generate_response(user_input, session_chat_history)
            print(f"\n튜터 : {answer}")
            
            # 답변이 무사히 생성되면, 현재 턴의 문답을 기억 텐서에 업데이트
            session_chat_history.append(HumanMessage(content=user_input))
            session_chat_history.append(AIMessage(content=answer))
            
        except Exception as e:
            print(f"\n[오류] 텐서 연산 중 문제가 발생했습니다: {e}")