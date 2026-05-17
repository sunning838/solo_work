import os
import requests
import xml.etree.ElementTree as ET
import time 

API_KEY = "rhddlswndrotk" 
TARGET_LAW = "민법"

SAVE_DIR = os.path.join("backend", "storage", "data", "LREA_1", "civil_law")
os.makedirs(SAVE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def crawl_law_data():
    print(f"🚀 [시스템] 1단계: 국가법령정보센터에서 '{TARGET_LAW}'의 고유 번호(MST)를 검색합니다...")
    
    search_url = "https://www.law.go.kr/DRF/lawSearch.do"
    search_params = {
        "OC": API_KEY,
        "target": "law",
        "type": "XML",
        "query": TARGET_LAW
    }
    
    try:
        search_response = requests.get(search_url, params=search_params, headers=HEADERS)
        search_response.raise_for_status()
        search_root = ET.fromstring(search_response.content)
        
        mst = None
        for law in search_root.findall('.//law'):
            if law.findtext('법령명한글') == TARGET_LAW:
                mst = law.findtext('법령일련번호')
                break
                
        if not mst:
            print(f"⚠️ '{TARGET_LAW}' 검색 실패")
            return
            
        print(f"✅ MST 획득: [{mst}]\n🚀 [시스템] 2단계: 본문 데이터를 다운로드하여 조립합니다...")
        time.sleep(1)

        service_url = "https://www.law.go.kr/DRF/lawService.do"
        service_params = {
            "OC": API_KEY,
            "target": "law",
            "type": "XML",
            "MST": mst
        }
        
        service_response = requests.get(service_url, params=service_params, headers=HEADERS)
        service_response.raise_for_status()
        service_root = ET.fromstring(service_response.content)
        
        articles = service_root.findall('.//조문단위')
        
        if not articles:
            print("⚠️ 본문 데이터를 찾지 못했습니다.")
            return

        # 🚀 일단 테스트로 처음 10개의 조문만 변환해보세!
        success_count = 0
        for article in articles[:10]: 
            # 1. 🚀 [수정] 조문번호를 태그에서 직접 추출!
            article_num = article.findtext('조문번호')
            
            # 조문번호가 없는 껍데기 태그(예: 장, 절 제목)는 건너뛰기
            if not article_num:
                continue

            title = article.findtext('조문제목', '')
            
            # 2. 🚀 [수정] 본문 + 항(①) + 호(1.) 내용까지 싹쓸이!
            content_lines = []
            
            jo_content = article.findtext('조문내용')
            if jo_content: content_lines.append(jo_content.strip())
            
            for hang in article.findall('.//항내용'):
                if hang.text: content_lines.append(hang.text.strip())
                
            for ho in article.findall('.//호내용'):
                if ho.text: content_lines.append(ho.text.strip())
                
            full_content = "\n".join(content_lines)

            # 내용이 정상적으로 모였다면 파일 생성
            if full_content:
                filename = f"civil_law_{article_num}.md"
                filepath = os.path.join(SAVE_DIR, filename)
                
                title_text = title if title else f"제{article_num}조"
                md_content = f"## {TARGET_LAW} {title_text}\n\n**내용:**\n{full_content}\n"
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                    
                print(f"✅ 생성 완료: {filename} (내용길이: {len(full_content)}자)")
                success_count += 1
                
        print(f"\n🎉 [성공] 총 {success_count}개의 마크다운 파일이 '{SAVE_DIR}'에 완벽하게 저장되었습니다!")

    except Exception as e:
        print(f"❌ 크롤링 중 오류 발생: {e}")

if __name__ == "__main__":
    crawl_law_data()