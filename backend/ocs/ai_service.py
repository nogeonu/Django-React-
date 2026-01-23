
import os
import json
from typing import Optional, Dict, Any

# OpenAI 라이브러리가 없어도 코드가 깨지지 않도록 처리
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from dotenv import load_dotenv

# 환경 변수 로드 (.env에서 OPENAI_API_KEY 확인)
load_dotenv()

class DrugInteractionAI:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        self.model = "gpt-4o-mini"  # Cost-efficient model

        if self.api_key and OpenAI:
            try:
                self.client = OpenAI(api_key=self.api_key)
                print("✅ AI Service: OpenAI Client init success.")
            except Exception as e:
                print(f"⚠️ AI Service: Failed to init client. {e}")
        else:
            print("ℹ️ AI Service: Running in offline mode (No API Key or module).")

    def analyze_interaction(
        self, 
        drug_a_name: str, 
        drug_a_ingr: str, 
        drug_b_name: str, 
        drug_b_ingr: str, 
        reason_from_db: str
    ) -> Dict[str, Any]:
        """
        약물 상호작용에 대한 심층 분석을 제공합니다.
        AI가 사용 불가능하면 DB 내용을 기반으로 기본 응답을 반환합니다.
        """
        
        # 1. 기본 응답 (Fallback) - DB 정보를 활용한 상세 메시지
        # reason_from_db에서 핵심 정보 추출
        summary = "병용 금기 (DUR 경고)"
        if reason_from_db:
            # contraindication에서 핵심 정보 추출 (예: "횡문근융해증", "세로토닌성증후군" 등)
            if "|" in reason_from_db:
                main_risk = reason_from_db.split("|")[0].strip()
                if main_risk:
                    summary = f"{drug_a_name}과 {drug_b_name}을 병용할 경우 {main_risk}의 위험이 증가합니다."
            elif len(reason_from_db) > 10:
                # 긴 설명이 있으면 요약
                summary = f"{drug_a_name}과 {drug_b_name}을 병용할 경우 {reason_from_db[:100]}의 위험이 있습니다."
            else:
                summary = f"{drug_a_name}과 {drug_b_name}을 병용할 경우 {reason_from_db}의 위험이 있습니다."
        else:
            summary = f"{drug_a_name}과 {drug_b_name}을 병용할 경우 심각한 약물 상호작용이 발생할 수 있습니다."
        
        base_response = {
            "analysis_type": "BASIC_DB",
            "summary": summary,
            "mechanism": reason_from_db[:200] if reason_from_db else "상세 약리기전 정보 없음 (AI 미연동)",
            "clinical_recommendation": "해당 약물의 동시 처방을 피하거나 전문의와 상담하십시오.",
            "raw_reason": reason_from_db
        }

        # 2. AI 사용 불가 시 바로 반환
        if not self.client:
            return base_response

        # 3. AI 분석 요청 (Prompt Engineering)
        try:
            prompt = f"""
            You are an expert Clinical Pharmacist using a CDSS (Clinical Decision Support System).
            Analyze the drug interaction between the following two drugs:

            Drug A: {drug_a_name} (Ingredient: {drug_a_ingr})
            Drug B: {drug_b_name} (Ingredient: {drug_b_ingr})
            Reported Issue (MFDS DB): {reason_from_db}

            Please provide a structured analysis in KOREAN (한국어):
            1. SUMMARY: One sentence summary of the risk.
            2. MECHANISM: Pharmacological explanation of why this interaction occurs (e.g., CYP450 inhibition, QT prolongation).
            3. RECOMMENDATION: Clinical advice for the doctor (e.g., maintain gap, choose alternative).

            Output Format (JSON):
            {{
                "summary": "...",
                "mechanism": "...",
                "recommendation": "..."
            }}
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful clinical pharmacist assistant. Answer in Korean."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3  # 사실 기반이므로 낮게 설정
            )
            
            content = response.choices[0].message.content
            ai_data = json.loads(content)

            return {
                "analysis_type": "AI_GENERATED",
                "summary": ai_data.get("summary", base_response["summary"]),
                "mechanism": ai_data.get("mechanism", base_response["mechanism"]),
                "clinical_recommendation": ai_data.get("recommendation", base_response["clinical_recommendation"]),
                "raw_reason": reason_from_db
            }

        except Exception as e:
            print(f"⚠️ AI Analysis Failed: {e}")
            # 실패 시 기본 응답 반환
            return base_response

    def detect_interactions(self, drug_list: list) -> list:
        """
        약물 리스트 전체를 검토하여 DB에 없는 잠재적 상호작용/중복을 AI가 직접 탐지합니다.
        
        Args:
            drug_list: [{'name_kor': '드럭이름', ...}, ...]
            
        Returns:
            List of interactions (dict)
        """
        if not self.client or len(drug_list) < 2:
            return []

        # 프롬프트 구성
        drug_texts = []
        for d in drug_list:
            name = d.get("name_kor", "Unknown")
            # 성분 정보 등 추가 정보가 있으면 좋지만 이름만으로도 GPT는 잘함
            drug_texts.append(f"- {name}")
            
        drugs_str = "\n".join(drug_texts)
        
        prompt = f"""
        You are an expert Clinical Pharmacist.
        Review the following prescription list for any potential Drug-Drug Interactions (DDI) or Therapeutic Duplications.
        
        Prescription List:
        {drugs_str}
        
        Instructions:
        1. Identify ANY significant interactions (Pharmacokinetic/Dynamic) or Duplications (Same ingredient/class).
        2. Specifically check for interaction with Alcohol (if listed as "술", "소주", "Alcohol", etc).
        3. Only report interactions that are 'Medium' severity or higher.
        4. If SAFE, return an empty list [].
        
        Output Format (JSON Array of Objects):
        [
            {{
                "drug_a": "Exact Name from list",
                "drug_b": "Exact Name from list",
                "type": "Interaction" (or "Duplication" or "Contraindication"),
                "severity": "CRITICAL" or "HIGH" or "MEDIUM",
                "summary": "Short explanation in Korean (한글)",
                "recommendation": "Advice in Korean (한글)"
            }}
        ]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a clinical pharmacist. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}, 
                temperature=0.0 # 분석은 정확해야 함
            )
            
            # JSON format correction (sometimes API wraps in dict)
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # 'interactions' 키로 감싸져 있을 수 있음
            result = []
            if isinstance(data, list):
                result = data
            elif isinstance(data, dict):
                # 키 이름이 무엇이든 리스트를 찾아냄
                for k, v in data.items():
                    if isinstance(v, list):
                        result = v
                        break
            
            return result

        except Exception as e:
            print(f"⚠️ AI Detection Failed: {e}")
            return []

# 싱글톤 인스턴스 (필요 시 import해서 사용)
ai_service = DrugInteractionAI()
