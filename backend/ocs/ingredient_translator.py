"""
한글 약품명/성분명 → 영문 성분명 변환 헬퍼
HIRA 매핑 테이블 활용
"""
import pymysql
import os
import re
from dotenv import load_dotenv
from typing import List, Set
from drug_dictionary import get_english_name, KOREAN_TO_ENGLISH

load_dotenv()

def get_db_conn():
    return pymysql.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', ''),
        db=os.getenv('DB_NAME', 'drug'),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

def normalize_korean_ingredient(name: str) -> str:
    """한글 성분명 정규화"""
    suffixes = [
        '고체분산체', '염산염', '나트륨', '칼륨', '수화물', '무수물',
        '타르타르산염', '말레산염', '장용정', '연질캡슐', '정', '캡슐',
        '주사액', '주', '서방정', '이초산염', '질산염', '황산염',
        '붕산염', '인산염', '구연산염', '아세트산염', '염화물',
        '베실산염', '토실산염', '메실산염', '서방캡슐', '액', '과립',
        '시럽', '산', '염', '정제', '필름코팅정', '이수화물', '일수화물', 
        '시트르산염', '구연산염', '탄산염', '말레인산염', '브롬화수소산염',
        '숙신산염', '옥살산염', '푸마르산염', '주석산염', '인산염',
        '나프로실산염', '비스무트', '알긴산염', '황산', '질산',
        '초산', '젖산염', '아스코르브산염', '글루콘산염', '살리실산염',
        '벤조산염', '프로피온산염', '스테아린산염', '라우릴황산염',
        '무수', '수화물', '삼수화물', '육수화물', '주사', '정주',
        '크림', '점안액', '연고', '로션', '겔', '패취', '산제', '캡슐제',
        '칼슘', '마그네슘', '트로메테타민', '트로메타민', '프로판디올', '푸로에이트',
        '수출용'
    ]
    
    cleaned = name.strip()
    
    # 긴 것부터 제거
    for suffix in sorted(suffixes, key=len, reverse=True):
        cleaned = cleaned.replace(suffix, '')
    
    return cleaned.strip()

def extract_korean_ingredients_from_name(drug_name: str) -> List[str]:
    """약품명에서 한글 성분명 추출"""
    candidates = []
    
    # 1. 괄호 안 내용 추출
    match = re.search(r'\(([^)]+)\)', drug_name)
    if match:
        inner = match.group(1)
        # 쉼표나 슬래시로 분리된 복합제
        parts = re.split(r'[,/]', inner)
        for part in parts:
            cleaned = normalize_korean_ingredient(part)
            if len(cleaned) >= 2:
                candidates.append(cleaned)
    
    # 2. 괄호 제거한 이름
    no_paren = re.sub(r'\([^)]*\)', '', drug_name)
    cleaned = normalize_korean_ingredient(no_paren)
    if len(cleaned) >= 2:
        candidates.append(cleaned)
    
    return list(set(candidates))

def korean_to_english_ingredients(korean_names: List[str], cursor=None) -> Set[str]:
    """
    한글 성분명 → 영문 성분명 변환 (HIRA 매핑 활용)
    
    Args:
        korean_names: 한글 성분명 리스트
        cursor: DB 커서 (없으면 새로 생성)
        
    Returns:
        Set[str]: 영문 성분명 집합
    """
    if not korean_names:
        return set()
    
    english_names = set()
    close_conn = False
    
    if cursor is None:
        conn = get_db_conn()
        cursor = conn.cursor()
        close_conn = True
    
    try:
        for kor_name in korean_names:
            # 0. 먼저 사전에서 검색 (가장 빠르고 정확)
            eng_from_dict = get_english_name(kor_name)
            if eng_from_dict:
                english_names.add(eng_from_dict.lower())
                print(f"[Dict] {kor_name} → {eng_from_dict}")
            
            # 1. HIRA 매핑에서 한글→영문 검색 (테이블 존재 시에만)
            try:
                cursor.execute("SHOW TABLES LIKE 'hira_drug_ingredient_map'")
                if cursor.fetchone():
                    cursor.execute("""
                        SELECT DISTINCT gnl_nm FROM hira_drug_ingredient_map
                        WHERE gnl_nm_cd LIKE %s OR gnl_nm LIKE %s
                        LIMIT 10
                    """, (f"%{kor_name}%", f"%{kor_name}%"))
                    
                    results = cursor.fetchall()
                    for row in results:
                        eng_name = row.get('gnl_nm')
                        if eng_name and eng_name.strip():
                            # 영문만 추출
                            eng_only = re.sub(r'\([^)]*\)', '', eng_name).strip()
                            if eng_only:
                                english_names.add(eng_only.lower())
            except Exception as e:
                print(f"⚠️ hira_drug_ingredient_map 조회 실패: {e}")
            
            # 2. HIRA 성분 효능 테이블에서도 검색 (테이블 존재 시에만)
            try:
                cursor.execute("SHOW TABLES LIKE 'hira_ingredient_effects'")
                if cursor.fetchone():
                    cursor.execute("""
                        SELECT DISTINCT gnl_nm FROM hira_ingredient_effects
                        WHERE gnl_nm LIKE %s
                        LIMIT 10
                    """, (f"%{kor_name}%",))
                    
                    results2 = cursor.fetchall()
                    for row in results2:
                        eng_name = row.get('gnl_nm')
                        if eng_name and eng_name.strip():
                            english_names.add(eng_name.strip().lower())
            except Exception as e:
                print(f"⚠️ hira_ingredient_effects 조회 실패: {e}")
    
    finally:
        if close_conn:
            cursor.close()
            conn.close()
    
    return english_names

def get_all_ingredient_names(drug_name: str, cursor=None) -> dict:
    """
    약품명에서 모든 형태의 성분명 추출
    
    Returns:
        {
            'korean': ['심바스타틴', ...],
            'english': ['simvastatin', ...],
            'all': ['심바스타틴', 'simvastatin', ...]
        }
    """
    # 한글 성분명 추출
    korean_ingredients = extract_korean_ingredients_from_name(drug_name)
    
    # 영문 성분명 변환
    english_ingredients = korean_to_english_ingredients(korean_ingredients, cursor)
    
    # 통합
    all_names = set(korean_ingredients) | english_ingredients
    
    return {
        'korean': korean_ingredients,
        'english': list(english_ingredients),
        'all': list(all_names)
    }

# 테스트
if __name__ == "__main__":
    test_drugs = [
        "뉴바스틴정(심바스타틴)",
        "글로트라정(이트라코나졸고체분산체)",
        "타이레놀정(아세트아미노펜)"
    ]
    
    conn = get_db_conn()
    cursor = conn.cursor()
    
    print("="*80)
    print("한글→영문 성분명 변환 테스트")
    print("="*80)
    
    for drug in test_drugs:
        print(f"\n약품: {drug}")
        result = get_all_ingredient_names(drug, cursor)
        print(f"  한글: {result['korean']}")
        print(f"  영문: {result['english']}")
        print(f"  통합: {result['all']}")
    
    cursor.close()
    conn.close()
    
    print("\n" + "="*80)
    print("완료!")
