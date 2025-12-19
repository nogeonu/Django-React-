#!/usr/bin/env python3
"""
MRI 뷰어 API 테스트 스크립트
백엔드 서버가 실행 중이어야 합니다.
"""

import requests
import json
from pathlib import Path

API_BASE_URL = "http://localhost:5000/api/mri"

def test_patient_list():
    """환자 목록 조회 테스트"""
    print("=" * 60)
    print("1. 환자 목록 조회 테스트")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/patients/")
        data = response.json()
        
        if response.status_code == 200 and data.get('success'):
            print("✓ 성공!")
            print(f"환자 수: {len(data.get('patients', []))}")
            for patient in data.get('patients', []):
                print(f"  - {patient['patient_id']}")
        else:
            print(f"✗ 실패: {data}")
    except Exception as e:
        print(f"✗ 에러: {e}")
    print()

def test_patient_info(patient_id="ISPY2_100899"):
    """환자 정보 조회 테스트"""
    print("=" * 60)
    print(f"2. 환자 정보 조회 테스트 ({patient_id})")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/patients/{patient_id}/")
        data = response.json()
        
        if response.status_code == 200 and data.get('success'):
            print("✓ 성공!")
            print(f"환자 ID: {data['patient_id']}")
            print(f"나이: {data['patient_info']['clinical_data']['age']}세")
            print(f"종양 유형: {data['patient_info']['primary_lesion']['tumor_subtype']}")
            print(f"스캐너: {data['patient_info']['imaging_data']['scanner_manufacturer']} {data['patient_info']['imaging_data']['scanner_model']}")
            print(f"시퀀스 수: {len(data['series'])}")
            print(f"슬라이스 수: {data['num_slices']}")
            print(f"세그멘테이션: {'있음' if data['has_segmentation'] else '없음'}")
        else:
            print(f"✗ 실패: {data}")
    except Exception as e:
        print(f"✗ 에러: {e}")
    print()

def test_slice_image(patient_id="ISPY2_100899", series=0, slice_idx=50):
    """슬라이스 이미지 조회 테스트"""
    print("=" * 60)
    print(f"3. 슬라이스 이미지 조회 테스트 (시퀀스: {series}, 슬라이스: {slice_idx})")
    print("=" * 60)
    
    try:
        url = f"{API_BASE_URL}/patients/{patient_id}/slice/"
        params = {
            'series': series,
            'slice': slice_idx,
            'axis': 'axial',
            'segmentation': 'false'
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code == 200 and data.get('success'):
            print("✓ 성공!")
            print(f"슬라이스 인덱스: {data['slice_index']}")
            print(f"시퀀스 인덱스: {data['series_index']}")
            print(f"축: {data['axis']}")
            print(f"이미지 크기: {data['shape']}")
            print(f"이미지 데이터 길이: {len(data['image'])} bytes")
        else:
            print(f"✗ 실패: {data}")
    except Exception as e:
        print(f"✗ 에러: {e}")
    print()

def test_slice_with_segmentation(patient_id="ISPY2_100899", series=0, slice_idx=100):
    """세그멘테이션 포함 슬라이스 이미지 조회 테스트"""
    print("=" * 60)
    print(f"4. 세그멘테이션 오버레이 테스트 (시퀀스: {series}, 슬라이스: {slice_idx})")
    print("=" * 60)
    
    try:
        url = f"{API_BASE_URL}/patients/{patient_id}/slice/"
        params = {
            'series': series,
            'slice': slice_idx,
            'axis': 'axial',
            'segmentation': 'true'
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code == 200 and data.get('success'):
            print("✓ 성공!")
            print(f"슬라이스 인덱스: {data['slice_index']}")
            print(f"세그멘테이션 오버레이: 활성화")
            print(f"이미지 데이터 길이: {len(data['image'])} bytes")
        else:
            print(f"✗ 실패: {data}")
    except Exception as e:
        print(f"✗ 에러: {e}")
    print()

def test_volume_info(patient_id="ISPY2_100899"):
    """볼륨 정보 조회 테스트"""
    print("=" * 60)
    print(f"5. 볼륨 정보 조회 테스트 ({patient_id})")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/patients/{patient_id}/volume/")
        data = response.json()
        
        if response.status_code == 200 and data.get('success'):
            print("✓ 성공!")
            print(f"환자 ID: {data['patient_id']}")
            print(f"시퀀스 목록:")
            for series in data['series']:
                print(f"  [{series['index']}] {series['filename']}")
                print(f"      크기: {series['shape']}, 슬라이스: {series['num_slices']}")
        else:
            print(f"✗ 실패: {data}")
    except Exception as e:
        print(f"✗ 에러: {e}")
    print()

def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "MRI 뷰어 API 테스트" + " " * 23 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    print("백엔드 서버: http://localhost:5000")
    print()
    
    # 모든 테스트 실행
    test_patient_list()
    test_patient_info()
    test_slice_image()
    test_slice_with_segmentation()
    test_volume_info()
    
    print("=" * 60)
    print("테스트 완료!")
    print("=" * 60)
    print()

if __name__ == "__main__":
    main()

