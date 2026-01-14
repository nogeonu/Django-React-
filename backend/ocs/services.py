"""
OCS 서비스 로직
"""
from django.utils import timezone
from .models import Order, DrugInteractionCheck, AllergyCheck, OrderStatusHistory
from patients.models import Patient
import logging

logger = logging.getLogger(__name__)


# 약물 상호작용 데이터 (임시: 하드코딩)
# TODO: 실제 운영 환경에서는 DB 모델(Drug, DrugInteraction) 또는 외부 API 사용 필요
# 참고: ocs/drug_api_integration.md 파일 참조
DRUG_INTERACTIONS = {
    'warfarin': {
        'aspirin': {'severity': 'severe', 'description': '출혈 위험 증가'},
        'ibuprofen': {'severity': 'moderate', 'description': '출혈 위험 증가'},
    },
    'digoxin': {
        'furosemide': {'severity': 'moderate', 'description': '디곡신 독성 위험'},
    },
    # 더 많은 상호작용 데이터 추가 가능
    # 실제 운영 시에는 Drug, DrugInteraction 모델 사용 권장
}


def check_drug_interactions(order):
    """약물 상호작용 체크"""
    if order.order_type != 'prescription':
        return None
    
    medications = order.order_data.get('medications', [])
    if not medications:
        return None
    
    # 약물명 추출
    drug_names = [med.get('name', '').lower() for med in medications if med.get('name')]
    
    interactions = []
    checked_drugs = []
    
    for i, drug1 in enumerate(drug_names):
        checked_drugs.append(drug1)
        for drug2 in drug_names[i+1:]:
            # 양방향 체크
            if drug1 in DRUG_INTERACTIONS and drug2 in DRUG_INTERACTIONS[drug1]:
                interaction = DRUG_INTERACTIONS[drug1][drug2]
                interactions.append({
                    'drug1': drug1,
                    'drug2': drug2,
                    'severity': interaction['severity'],
                    'description': interaction['description']
                })
            elif drug2 in DRUG_INTERACTIONS and drug1 in DRUG_INTERACTIONS[drug2]:
                interaction = DRUG_INTERACTIONS[drug2][drug1]
                interactions.append({
                    'drug1': drug2,
                    'drug2': drug1,
                    'severity': interaction['severity'],
                    'description': interaction['description']
                })
    
    if interactions:
        # 가장 심각한 상호작용의 심각도 결정
        severities = [inter['severity'] for inter in interactions]
        if 'severe' in severities:
            severity = 'severe'
        elif 'moderate' in severities:
            severity = 'moderate'
        else:
            severity = 'mild'
        
        return DrugInteractionCheck.objects.create(
            order=order,
            checked_drugs=checked_drugs,
            interactions=interactions,
            severity=severity
        )
    
    return None


def check_allergies(order):
    """알레르기 체크"""
    patient = order.patient
    
    # 환자 알레르기 정보 가져오기
    patient_allergies = []
    if patient.allergies:
        # 알레르기 정보를 파싱 (예: "페니실린, 아스피린" 형식)
        allergies_text = patient.allergies.strip()
        if allergies_text:
            patient_allergies = [a.strip().lower() for a in allergies_text.split(',')]
    
    if not patient_allergies:
        # 알레르기 없으면 체크 불필요
        return None
    
    # 주문 항목 추출
    order_items = []
    warnings = []
    has_risk = False
    
    if order.order_type == 'prescription':
        medications = order.order_data.get('medications', [])
        for med in medications:
            med_name = med.get('name', '').lower()
            order_items.append(med_name)
            
            # 알레르기 체크
            for allergy in patient_allergies:
                if allergy in med_name or med_name in allergy:
                    warnings.append({
                        'item': med_name,
                        'allergy': allergy,
                        'description': f'{med_name}은(는) 환자의 알레르기 항목({allergy})과 관련이 있습니다.'
                    })
                    has_risk = True
    
    elif order.order_type == 'lab_test':
        test_items = order.order_data.get('test_items', [])
        for test in test_items:
            test_name = test.get('name', '').lower()
            order_items.append(test_name)
            
            # 검사 항목 중 알레르기 관련 체크 (예: 조영제 검사)
            for allergy in patient_allergies:
                if 'contrast' in test_name and allergy in ['iodine', 'iodinated']:
                    warnings.append({
                        'item': test_name,
                        'allergy': allergy,
                        'description': f'{test_name} 검사는 조영제를 사용하며, 환자가 요오드 알레르기가 있습니다.'
                    })
                    has_risk = True
    
    if warnings or has_risk:
        return AllergyCheck.objects.create(
            order=order,
            patient_allergies=patient_allergies,
            order_items=order_items,
            warnings=warnings,
            has_allergy_risk=has_risk
        )
    
    return None


def validate_order(order):
    """주문 검증 (약물 상호작용 + 알레르기)"""
    validation_passed = True
    validation_notes = []
    
    # 약물 상호작용 체크
    if order.order_type == 'prescription':
        interaction_check = check_drug_interactions(order)
        if interaction_check and interaction_check.severity == 'severe':
            validation_passed = False
            validation_notes.append(f"심각한 약물 상호작용이 발견되었습니다: {interaction_check.interactions}")
        elif interaction_check:
            validation_notes.append(f"약물 상호작용 경고: {interaction_check.interactions}")
    
    # 알레르기 체크
    allergy_check = check_allergies(order)
    if allergy_check and allergy_check.has_allergy_risk:
        validation_passed = False
        validation_notes.append(f"알레르기 위험이 발견되었습니다: {allergy_check.warnings}")
    
    order.validation_passed = validation_passed
    order.validation_notes = '\n'.join(validation_notes)
    order.save()
    
    return validation_passed, validation_notes


def update_order_status(order, new_status, changed_by=None, notes=''):
    """주문 상태 업데이트 및 이력 기록"""
    old_status = order.status
    order.status = new_status
    
    # 완료일 설정
    if new_status == 'completed' and not order.completed_at:
        order.completed_at = timezone.now()
    
    order.save()
    
    # 상태 이력 기록
    OrderStatusHistory.objects.create(
        order=order,
        status=new_status,
        changed_by=changed_by,
        notes=notes or f"상태 변경: {old_status} → {new_status}"
    )
    
    logger.info(f"Order {order.id} status changed: {old_status} → {new_status} by {changed_by}")
