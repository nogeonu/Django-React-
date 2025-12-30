import requests
from requests.auth import HTTPBasicAuth

# Orthanc 설정
ORTHANC_URL = "http://34.64.185.63:8042"
ORTHANC_USER = "orthanc"
ORTHANC_PASSWORD = "orthanc"

# 환자 검색
response = requests.get(
    f"{ORTHANC_URL}/patients",
    auth=HTTPBasicAuth(ORTHANC_USER, ORTHANC_PASSWORD)
)

patients = response.json()
print(f"Total patients: {len(patients)}")

# P2025020 환자 찾기
for patient_id in patients:
    patient_info = requests.get(
        f"{ORTHANC_URL}/patients/{patient_id}",
        auth=HTTPBasicAuth(ORTHANC_USER, ORTHANC_PASSWORD)
    ).json()
    
    dicom_patient_id = patient_info.get('MainDicomTags', {}).get('PatientID', '')
    
    if 'P2025020' in dicom_patient_id or 'p2025020' in dicom_patient_id.lower():
        print(f"\n=== Found Patient: {dicom_patient_id} ===")
        print(f"Orthanc ID: {patient_id}")
        
        # Studies 확인
        for study_id in patient_info.get('Studies', []):
            study_info = requests.get(
                f"{ORTHANC_URL}/studies/{study_id}",
                auth=HTTPBasicAuth(ORTHANC_USER, ORTHANC_PASSWORD)
            ).json()
            
            print(f"\nStudy: {study_id}")
            
            # Series 확인
            for series_id in study_info.get('Series', []):
                series_info = requests.get(
                    f"{ORTHANC_URL}/series/{series_id}",
                    auth=HTTPBasicAuth(ORTHANC_USER, ORTHANC_PASSWORD)
                ).json()
                
                series_desc = series_info.get('MainDicomTags', {}).get('SeriesDescription', 'N/A')
                body_part = series_info.get('MainDicomTags', {}).get('BodyPartExamined', 'N/A')
                instances = len(series_info.get('Instances', []))
                
                print(f"  Series: {series_id}")
                print(f"    SeriesDescription: {series_desc}")
                print(f"    BodyPartExamined: {body_part}")
                print(f"    Instances: {instances}")
                
                # 첫 번째 인스턴스의 상세 정보 확인
                if series_info.get('Instances'):
                    instance_id = series_info['Instances'][0]
                    instance_info = requests.get(
                        f"{ORTHANC_URL}/instances/{instance_id}",
                        auth=HTTPBasicAuth(ORTHANC_USER, ORTHANC_PASSWORD)
                    ).json()
                    
                    tags = instance_info.get('MainDicomTags', {})
                    print(f"    Instance Tags:")
                    print(f"      ViewPosition: {tags.get('ViewPosition', 'N/A')}")
                    print(f"      ImageLaterality: {tags.get('ImageLaterality', 'N/A')}")
                    print(f"      ViewCodeSequence: {tags.get('ViewCodeSequence', 'N/A')}")
