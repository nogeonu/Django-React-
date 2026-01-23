"""
DDInter 약물 상호작용 체크 헬퍼 함수
main.py에서 import해서 사용
"""
from ingredient_translator import get_all_ingredient_names

def check_ddinter_interactions(cursor, drug_a, drug_b, item_a, item_b, ai_service):
    """
    DDInter 데이터베이스에서 상호작용 검색 (영문 성분명 변환 활용)
    
    Returns:
        dict or None: 상호작용 발견 시 interaction dict, 없으면 None
    """
    
    # 약품명 → 영문 성분명 변환 (HIRA 매핑 활용)
    ingredients_a = get_all_ingredient_names(drug_a['name_kor'], cursor)
    ingredients_b = get_all_ingredient_names(drug_b['name_kor'], cursor)
    
    # 검색용 키워드 (영문 우선, 한글 보조)
    search_a = ingredients_a['english'] + ingredients_a['korean']
    search_b = ingredients_b['english'] + ingredients_b['korean']
    
    if not search_a or not search_b:
        return None
    
    try:
        found_interaction = None
        # DDInter에서 검색 (영문 성분명으로)
        for name_a in search_a[:5]:  # 상위 5개만
            for name_b in search_b[:5]:
                cursor.execute("""
                    SELECT drug_a, drug_b, level 
                    FROM ddinter_interactions
                    WHERE (drug_a LIKE %s AND drug_b LIKE %s)
                       OR (drug_a LIKE %s AND drug_b LIKE %s)
                    LIMIT 1
                """, (f"%{name_a}%", f"%{name_b}%", f"%{name_b}%", f"%{name_a}%"))
                
                result = cursor.fetchone()
                
                if not result:
                    continue  # 다음 조합 시도
                else:
                    found_interaction = result
                    break # 상호작용을 찾았으므로 내부 루프 종료
            if found_interaction:
                break # 상호작용을 찾았으므로 외부 루프 종료
        
        if not found_interaction:
            return None
        
        level = found_interaction.get('level', 'Unknown')
        
        # Level → Severity 매핑
        severity_map = {
            "Major": "CRITICAL",
            "Moderate": "HIGH",
            "Minor": "MEDIUM",
            "Unknown": "MEDIUM"
        }
        severity = severity_map.get(level, "MEDIUM")
        
        # AI 분석 (있으면)
        ai_analysis = None
        if ai_service:
            try:
                ai_result = ai_service.analyze_interaction(
                    drug_a_name=drug_a['name_kor'],
                    drug_a_ingr=found_interaction['drug_a'],
                    drug_b_name=drug_b['name_kor'],
                    drug_b_ingr=found_interaction['drug_b'],
                    reason_from_db=f"DDInter database: {level} level interaction"
                )
                
                confidence_map = {"CRITICAL": 95, "HIGH": 85, "MEDIUM": 75}
                ai_analysis = {
                    "confidence": confidence_map.get(severity, 70),
                    "summary": ai_result.get("summary", f"{level} 수준 상호작용"),
                    "mechanism": ai_result.get("mechanism", ""),
                    "recommendation": ai_result.get("clinical_recommendation", "")
                }
            except Exception as e:
                print(f"AI analysis failed for DDInter: {e}")
                ai_analysis = None
        
        # 상호작용 객체 반환 - 조원 형식: 약물명 + AI 분석
        drug_a_display = drug_a['name_kor']
        drug_b_display = drug_b['name_kor']
        
        # 기본 메시지: 약물명
        base_msg = f"{drug_a_display} + {drug_b_display}:"
        
        # AI 분석이 있으면 AI 분석 메시지 추가
        if ai_analysis and ai_analysis.get('summary'):
            base_msg = f"{base_msg}\nAI 분석: {ai_analysis['summary']}"
        else:
            base_msg = f"{base_msg} {level} level (International DDI DB)"
        
        return {
            "item_seq_a": item_a,
            "drug_name_a": drug_a['name_kor'],
            "item_seq_b": item_b,
            "drug_name_b": drug_b['name_kor'],
            "interaction_type": f"DDInter ({level})",
            "severity": severity,
            "warning_message": base_msg,
            "prohbt_content": f"DDInter 국제 약물상호작용 데이터베이스에서 {level} 등급으로 분류된 상호작용입니다.",
            "ai_analysis": ai_analysis
        }
        
    except Exception as e:
        print(f"DDInter check error: {e}")
        return None
