# CDSS (차세대 지능형 의료지원 시스템) 프로젝트 기획서

## 1. 프로젝트 개요

### 1.1 프로젝트 목표
- **프로젝트명:** CDSS (Clinical Decision Support System) - 차세대 지능형 의료지원 시스템
- **아키텍처:** MSA (Microservices Architecture)
- **목적:** 의료진과 환자를 위한 통합 의료 정보 시스템 구축
- **핵심 기능:** AI 기반 진단 지원, 역할별 맞춤형 인터페이스, 실시간 의료 데이터 관리

### 1.2 시스템 범위
- **병원 홈페이지**: 일반 사용자 및 환자를 위한 공개 웹사이트
- **의료진 플랫폼 (CDSS)**: 의료진 전용 내부 시스템
- 5가지 사용자 역할별 독립적인 인터페이스 제공
- 마이크로서비스 기반 서버 아키텍처
- AI 딥러닝 모델을 활용한 의료 이미지 분석
- 실시간 데이터 동기화 및 공유
- 병원 홈페이지에서 의료진 플랫폼으로의 통합 접근

### 1.3 시스템 구조 개요
```
병원 홈페이지 (Public) - 환자 전용
    ├─ 환자 회원가입/로그인
    ├─ 환자 인터페이스
    │   ├─ 본인 정보 조회
    │   ├─ 진료 기록 조회
    │   ├─ 의료 이미지 조회
    │   ├─ 예약 관리
    │   └─ 챗봇 문의
    └─ 의료진 로그인 버튼 → 의료진 플랫폼 이동

의료진 플랫폼 (CDSS) - 의료진 전용 내부 시스템
    ├─ 의사 인터페이스
    ├─ 간호사 인터페이스
    ├─ 방사선과 인터페이스
    └─ 영상의학과 인터페이스
```

---

## 2. 사용자 역할 정의

### 2.0 접근 경로

**병원 홈페이지 (Public)**
- **환자**: 회원가입/로그인 → 본인 정보 조회 및 서비스 이용
- **일반 사용자**: 공개 정보 조회, 챗봇 문의
- **의료진**: 로그인 불가 (의료진 플랫폼으로 이동)

**의료진 플랫폼 (CDSS - 내부 시스템)**
- **의료진 전용**: 의사, 간호사, 원무과, 방사선과, 영상의학과만 접근 가능
- **환자 접근 불가**: 환자는 병원 홈페이지에서만 서비스 이용

**총 6가지 역할:** 의사, 간호사, 원무과, 환자, 방사선과, 영상의학과

### 2.1 의사 (Doctor)
**역할:** 진단 및 치료 결정을 내리는 주 의료진

**진료과 구분:**
1. **폐-호흡기내과 (Pulmonology)**
   - 폐 질환 진단 및 치료
   - 폐 CT 이미지 분석
   - 호흡기 관련 검사 결과 확인

2. **머리-신경외과 (Neurosurgery)**
   - 뇌 및 신경계 질환 진단 및 치료
   - 뇌 CT/MRI 이미지 분석
   - 신경외과 수술 관련 기록

3. **유방-유방외과 (Breast Surgery)**
   - 유방 질환 진단 및 치료
   - **의료 프로세스:**
     - **1차 검진: 유방 초음파** (선별 검사 및 정밀 검사)
       - 초음파 이미지: **2D 딥러닝 모델** 사용
       - 초기 병변 탐지 및 정밀 진단
     - **2차 검진: 조직 검사 (병리 이미지)** (확진)
       - 병리 이미지: **2D 딥러닝 모델** 사용
       - 현미경 이미지 분석
       - 최종 암 확진
     - **3차 검진: 유방 MRI** (암 확진 후 수술 전 범위 평가, 다발성 암 확인)
       - MRI 이미지: **3D 딥러닝 모델** 사용
       - DICOM 파일 형식 (Orthanc 서버 활용)
       - 볼륨 데이터 분석 (세그멘테이션 및 분류)
   - 유방암 세그멘테이션 및 분류 결과 활용

4. **췌장-소화기내과 (Gastroenterology)**
   - 췌장 및 소화기 질환 진단 및 치료
   - 복부 CT/MRI 이미지 분석
   - 소화기 내시경 결과 확인

5. **피부-피부과 (Dermatology)**
   - 피부 질환 진단 및 치료
   - 피부 이미지 분석 (피부암 여부 판별)
   - 피부 병변 탐지 및 분류
   - Flutter 모바일 앱을 통한 피부 이미지 촬영 및 분석
   - **환자 1차 스크리닝 서비스**: 환자가 병원 방문 전 본인 모바일로 피부암 여부를 1차 확인할 수 있는 서비스 제공

**주요 권한:**
- 환자 정보 조회 및 수정
- 담당 진료과 관련 의료 이미지 조회 및 AI 분석 결과 확인
- 진단서 작성 및 처방전 발급
- 환자 진료 기록 작성 및 관리
- 예약 관리 및 스케줄 확인
- AI 분석 결과 기반 진단 지원 받기
- 진료과별 특화된 AI 분석 도구 활용

**접근 가능한 데이터:**
- 담당 환자 정보
- 담당 진료과 관련 의료 이미지 및 분석 결과
- 진료 기록 및 처방 이력
- AI 분석 리포트
- 진료과별 통계 및 대시보드

---

### 2.2 간호사 (Nurse)
**역할:** 환자 간호 및 의료진 보조 업무 담당

**주요 권한:**
- 환자 기본 정보 조회
- 환자 상태 모니터링 및 기록
- 간호 기록 작성
- 활력 징후 입력 (혈압, 맥박, 체온 등)
- 투약 관리 및 스케줄 확인
- 의료 이미지 업로드 보조
- 환자 알림 및 안내

**접근 가능한 데이터:**
- 담당 환자 정보
- 환자 상태 기록
- 간호 기록
- 투약 스케줄
- 기본 의료 이미지 (AI 분석 결과 제한적 접근)

---

### 2.3 원무과 (Administrative Staff)
**역할:** 환자 접수, 예약 관리, 수납 및 행정 업무 담당

**주요 권한:**
- 환자 접수 및 등록
- 예약 관리 (생성, 조회, 수정, 취소)
- 예약 스케줄 확인 및 관리
- 수납 처리 (진료비 계산 및 수납)
- 보험 처리 (건강보험, 의료급여 등)
- 환자 정보 관리 (기본 정보 수정)
- 진료비 조회 및 영수증 발급
- 예약 통계 및 현황 조회

**접근 가능한 데이터:**
- 환자 기본 정보 (등록, 수정)
- 예약 정보 (전체 조회 및 관리)
- 수납 정보
- 보험 정보
- 진료비 정보
- 예약 통계

**접근 불가능한 데이터:**
- 진료 기록 상세 내용 (의사 전용)
- 의료 이미지 (제한적 접근)
- AI 분석 결과 (제한적 접근)

---

### 2.4 환자 (Patient)
**역할:** 자신의 의료 정보를 조회하는 사용자

**주요 권한:**
- 본인 의료 정보 조회
- 본인 의료 이미지 및 AI 분석 결과 확인
- 예약 조회 및 신청
- 진료 기록 조회
- 처방전 조회 및 다운로드
- 건강 상태 모니터링
- **Flutter 모바일 앱을 통한 피부암 1차 스크리닝**
  - 병원 방문 전 본인 모바일로 피부 이미지 촬영
  - AI 기반 피부암 여부 1차 판별
  - 결과에 따른 병원 방문 권장 여부 확인
  - 스크리닝 결과를 병원 시스템에 전송하여 예약 연계

**접근 가능한 데이터:**
- 본인 환자 정보
- 본인 의료 이미지 및 분석 결과
- 본인 진료 기록
- 본인 예약 정보
- 본인 피부암 스크리닝 결과 (Flutter 앱)

---

### 2.5 방사선과 (Radiology Technician)
**역할:** 의료 이미지 촬영 및 업로드 담당

**주요 권한:**
- 의료 이미지 촬영
- 촬영한 의료 이미지 업로드
- 이미지 메타데이터 입력 (촬영 정보)
- 촬영한 이미지 조회 및 관리
- 이미지 품질 검사
- 촬영 스케줄 확인

**접근 가능한 데이터:**
- 촬영한 의료 이미지
- 촬영 스케줄 정보
- 기본 환자 정보 (촬영 목적)

---

### 2.6 영상의학과 (Radiology Doctor)
**역할:** 의료 이미지 판독 및 분석 전문가

**주요 권한:**
- 모든 환자의 의료 이미지 조회
- AI 분석 실행 및 결과 확인
- 영상 판독서 작성
- 3D 시각화 생성 및 분석
- 판독 이력 관리
- 판독 품질 관리

**접근 가능한 데이터:**
- 모든 환자 의료 이미지
- AI 분석 결과 및 리포트
- 영상 판독 기록
- 이미지 메타데이터

---

## 3. 데이터베이스 테이블 설계

### 3.0 기존 테이블 (유지 및 활용)

본 프로젝트는 기존 프로젝트의 데이터베이스 테이블을 그대로 유지하면서, 새로운 테이블을 추가하는 방식으로 진행합니다.

#### 3.0.1 기존 테이블 목록

**사용자 관련 테이블**
- ✅ **auth_user** (의료진 사용자)
  - Django 기본 User 모델 확장
  - `doctor_id` (의사 코드), `department` (진료과) 필드 포함
  - 기존 구조 유지

- ✅ **patient_user** (환자 사용자)
  - 커스텀 사용자 모델 (AbstractBaseUser 기반)
  - `account_id`, `email`, `name`, `patient_id` 필드 포함
  - 기존 구조 유지

**환자 관련 테이블**
- ✅ **patients_patient** (환자 정보)
  - 환자 기본 정보 (이름, 생년월일, 성별, 연락처 등)
  - `patient_id`, `name`, `birth_date`, `gender`, `age`, `blood_type` 등
  - `user_account_id` (FK → patient_user)
  - 기존 구조 유지

**진료 관련 테이블**
- ✅ **patients_appointment** (예약 정보)
  - 예약 관리 (환자, 의사, 시간, 상태 등)
  - `patient_id` (FK → patients_patient)
  - `doctor_id` (FK → auth_user)
  - `doctor_code`, `doctor_department` 필드 포함
  - 기존 구조 유지

- ✅ **medical_record** (진료 기록)
  - 진료 기록 정보
  - `patient_fk_id` (FK → patients_patient)
  - `doctor_fk_id` (FK → auth_user)
  - `department`, `status`, `notes` 등
  - 기존 구조 유지

**폐암 검사 관련 테이블**
- ✅ **lung_record** (폐암 검사 기록)
  - 폐암 검사 데이터 (흡연, 증상 등)
  - `patient_fk_id` (FK → patients_patient)
  - 기존 구조 유지

- ✅ **lung_result** (폐암 검사 결과)
  - 폐암 검사 결과 (예측, 위험 점수)
  - `lung_record_id` (FK → lung_record)
  - 기존 구조 유지

**의료 이미지 관련 테이블**
- ✅ **medical_images_medicalimage** (의료 이미지)
  - 의료 이미지 정보
  - `patient_id`, `image_type`, `image_file` 등
  - 기존 구조 유지 (필드 확장 가능)

- ✅ **medical_images_aianalysisresult** (AI 분석 결과)
  - AI 분석 결과 정보
  - `image` (FK → medical_images_medicalimage)
  - `analysis_type`, `results`, `confidence` 등
  - 기존 구조 유지 (필드 확장 가능)

#### 3.0.2 기존 테이블 확장 전략

기존 테이블은 구조를 유지하되, 필요시 필드를 추가하여 확장합니다:

- **medical_images_medicalimage**: 
  - `taken_by`, `uploaded_by`, `requested_by` 필드 추가
  - `department`, `body_part`, `image_source` 필드 추가
  - `radiology_notes`, `status` 필드 추가

- **medical_record**:
  - `department` 필드 확인 및 확장 (필요시)
  - `related_images` (ManyToMany) 필드 추가 가능

- **patients_appointment**:
  - `department` 필드 확인 (이미 `doctor_department` 있음)
  - 필요시 추가 필드 확장

---

### 3.1 사용자 및 인증 관련 테이블 (새로 추가)

> **참고**: 기존 `auth_user` 테이블은 유지하며, 아래 테이블들은 의료진 역할별 상세 정보를 저장하는 새로운 테이블입니다.

#### 3.1.1 User (기본 사용자 테이블 - Django Auth 확장)
**기존 테이블 활용**: `auth_user` 테이블은 그대로 사용하며, 아래 역할별 테이블과 연결됩니다.
```sql
- id (PK)
- username (고유)
- email
- password (암호화)
- role (의사/간호사/원무과/환자/방사선과/영상의학과)
- is_active
- is_staff
- date_joined
- last_login
```

#### 3.1.2 Doctor (의사 정보)
```sql
- id (PK)
- user_id (FK → User)
- doctor_id (고유, 예: D2025001)
- name
- department (진료과 선택)
  - PULMONOLOGY (폐-호흡기내과)
  - NEUROSURGERY (머리-신경외과)
  - BREAST_SURGERY (유방-유방외과)
  - GASTROENTEROLOGY (췌장-소화기내과)
  - DERMATOLOGY (피부-피부과)
- license_number (의사 면허번호)
- specialization (전문분야 상세)
- phone
- email
- created_at
- updated_at
```

#### 3.1.3 Nurse (간호사 정보)
```sql
- id (PK)
- user_id (FK → User)
- nurse_id (고유, 예: N2025001)
- name
- department (근무과)
- license_number (간호사 면허번호)
- phone
- email
- created_at
- updated_at
```

#### 3.1.4 AdministrativeStaff (원무과 직원)
```sql
- id (PK)
- user_id (FK → User)
- staff_id (고유, 예: AS2025001)
- name
- department (원무과)
- position (직급/직책)
- phone
- email
- created_at
- updated_at
```

#### 3.1.5 RadiologyTechnician (방사선과 기사)
```sql
- id (PK)
- user_id (FK → User)
- technician_id (고유, 예: RT2025001)
- name
- department (방사선과)
- license_number (방사선사 면허번호)
- phone
- email
- created_at
- updated_at
```

#### 3.1.6 RadiologyDoctor (영상의학과 의사)
```sql
- id (PK)
- user_id (FK → User)
- doctor_id (고유, 예: RD2025001)
- name
- department (영상의학과)
- license_number (의사 면허번호)
- specialization (전문분야)
- phone
- email
- created_at
- updated_at
```

#### 3.1.7 Patient (환자 정보) - 기존 모델 활용
```sql
- id (PK)
- patient_id (고유)
- user_id (FK → User, nullable)
- name
- birth_date
- gender
- age
- phone
- blood_type
- address
- emergency_contact
- medical_history
- allergies
- created_at
- updated_at
```

---

### 3.2 의료 이미지 관련 테이블

#### 3.2.1 MedicalImage (의료 이미지) - 기존 모델 확장
**기존 테이블 활용**: `medical_images_medicalimage` 테이블을 유지하며, 필요시 필드를 추가합니다.

**기존 필드:**
```sql
- id (PK)
- patient_id (환자 ID, CharField)  # 기존 필드
- image_type (MRI/CT/X-ray/초음파 등)  # 기존 필드
- image_file (파일 경로)  # 기존 필드
- description  # 기존 필드
- taken_date  # 기존 필드
- doctor_notes  # 기존 필드
- created_at  # 기존 필드
```

**새로 추가할 필드:**
```sql
- taken_by (FK → RadiologyTechnician, 촬영 담당자 - 방사선과 기사)
- uploaded_by (FK → RadiologyTechnician, 업로드 담당자)
- requested_by (FK → Doctor, 촬영 의뢰 의사)
- department (진료과)
  - PULMONOLOGY (폐-호흡기내과)
  - NEUROSURGERY (머리-신경외과)
  - BREAST_SURGERY (유방-유방외과)
  - GASTROENTEROLOGY (췌장-소화기내과)
  - DERMATOLOGY (피부-피부과)
- body_part (촬영 부위)
- image_source (이미지 출처)
  - CAMERA (카메라 촬영)
  - UPLOAD (파일 업로드)
  - MOBILE_APP (Flutter 모바일 앱)
- radiology_notes (영상의학과 판독 소견)
- status (촬영완료/판독대기/판독중/판독완료)
- updated_at
```

#### 3.2.2 AIAnalysisResult (AI 분석 결과) - 기존 모델 활용
**기존 테이블 활용**: `medical_images_aianalysisresult` 테이블을 유지하며, 필요시 필드를 추가합니다.

**기존 필드:**
```sql
- id (PK)
- image (FK → MedicalImage)
- analysis_type (분석 유형)
  - BREAST_ULTRASOUND (유방 초음파)  # 새로 추가 (2D 딥러닝 모델)
  - BREAST_ULTRASOUND_ANALYSIS (유방 초음파 분석)  # 새로 추가
  - BREAST_MRI (유방 MRI)  # 기존 (3D 딥러닝 모델, DICOM)
  - BREAST_MRI_SEGMENTATION (유방 MRI 세그멘테이션)  # 기존 (3D 볼륨 데이터)
  - BREAST_MRI_CLASSIFICATION (유방 MRI 종양분석)  # 기존 (3D 볼륨 데이터)
  - LUNG_CT (폐 CT)  # 기존
  - XRAY (X-ray)  # 기존
  - OTHER (기타)  # 기존
- results (JSON)
- confidence
- findings
- recommendations
- model_version
- analysis_date
```

**새로 추가할 분석 타입:**
```sql
- analysis_type 확장:
  - BRAIN_CT_MRI_ANALYSIS (뇌 CT/MRI 분석)
  - PANCREAS_CT_MRI_ANALYSIS (췌장 CT/MRI 분석)
  - SKIN_CANCER_CLASSIFICATION (피부암 분류 분석)
  - SKIN_LESION_DETECTION (피부 병변 탐지)
- analyzed_by (FK → User, 분석 실행자)  # 새로 추가
```

#### 3.2.3 RadiologyReport (영상 판독서)
```sql
- id (PK)
- image_id (FK → MedicalImage)
- patient_id (FK → Patient)
- radiologist_id (FK → RadiologyDoctor, 판독 의사)
- report_content (판독 내용)
- findings (발견사항)
- conclusion (결론)
- recommendations (권고사항)
- report_date
- created_at
- updated_at
```

---

### 3.3 진료 관련 테이블

#### 3.3.1 MedicalRecord (진료 기록) - 기존 모델 확장
**기존 테이블 활용**: `medical_record` 테이블을 유지하며, 필요시 필드를 추가합니다.

```sql
- id (PK)
- patient_fk_id (FK → patients_patient)  # 기존 필드
- doctor_fk_id (FK → auth_user)  # 기존 필드
- name  # 기존 필드
- department (진료과)  # 기존 필드 (확장 가능)
  - PULMONOLOGY (폐-호흡기내과)
  - NEUROSURGERY (머리-신경외과)
  - BREAST_SURGERY (유방-유방외과)
  - GASTROENTEROLOGY (췌장-소화기내과)
  - DERMATOLOGY (피부-피부과)
- status  # 기존 필드
- notes  # 기존 필드
- reception_start_time  # 기존 필드
- treatment_end_time  # 기존 필드
- is_treatment_completed  # 기존 필드
- visit_date  # 새로 추가 가능
- diagnosis (진단명)  # 새로 추가 가능
- symptoms (증상)  # 새로 추가 가능
- treatment (치료내용)  # 새로 추가 가능
- prescription (처방전)  # 새로 추가 가능 (또는 별도 Prescription 테이블)
- doctor_notes  # 새로 추가 가능
- related_images (ManyToMany → MedicalImage, 관련 의료 이미지)  # 새로 추가 가능
- created_at  # 새로 추가 가능
```

#### 3.3.2 Appointment (예약) - 기존 모델 활용
**기존 테이블 활용**: `patients_appointment` 테이블을 그대로 사용합니다.

```sql
- id (PK, UUID)  # 기존 필드
- patient_id (FK → patients_patient)  # 기존 필드
- doctor_id (FK → auth_user)  # 기존 필드
- doctor_department (진료과)  # 기존 필드
  - PULMONOLOGY (폐-호흡기내과)
  - NEUROSURGERY (머리-신경외과)
  - BREAST_SURGERY (유방-유방외과)
  - GASTROENTEROLOGY (췌장-소화기내과)
  - DERMATOLOGY (피부-피부과)
- title  # 기존 필드
- type (일반/검진/회의 등)  # 기존 필드
- start_time  # 기존 필드
- end_time  # 기존 필드
- status (예약됨/완료/취소)  # 기존 필드
- memo  # 기존 필드
- created_by_id (FK → auth_user)  # 기존 필드
- patient_identifier  # 기존 필드
- patient_name  # 기존 필드
- patient_gender  # 기존 필드
- patient_age  # 기존 필드
- doctor_code  # 기존 필드
- doctor_username  # 기존 필드
- doctor_name  # 기존 필드
- created_at  # 기존 필드
- updated_at  # 기존 필드
```

#### 3.3.3 Prescription (처방전) - 새로 추가
**새 테이블**: 처방전 정보를 저장하는 새로운 테이블입니다.

```sql
- id (PK)
- medical_record_id (FK → medical_record)
- patient_id (FK → patients_patient)
- doctor_id (FK → Doctor)
- prescription_date
- medications (JSON, 약물 목록)
- dosage (용법)
- duration (복용 기간)
- notes
- created_at
```

---

### 3.4 간호 관련 테이블

#### 3.4.1 NursingRecord (간호 기록) - 새로 추가
**새 테이블**: 간호 기록을 저장하는 새로운 테이블입니다.

```sql
- id (PK)
- patient_id (FK → patients_patient)
- nurse_id (FK → Nurse)
- record_date
- vital_signs (JSON: 혈압/맥박/체온 등)
- symptoms (증상)
- nursing_notes (간호 소견)
- medications_administered (투약 내역)
- created_at
- updated_at
```

#### 3.4.2 PatientStatus (환자 상태) - 새로 추가
```sql
- id (PK)
- patient_id (FK → Patient)
- status (입원/외래/퇴원 등)
- room_number (병실 번호)
- admission_date
- discharge_date
- current_condition (현재 상태)
- updated_by (FK → User)
- updated_at
```

---

### 3.5 시스템 관리 테이블 (새로 추가)

#### 3.5.1 ActivityLog (활동 로그) - 새로 추가
```sql
- id (PK)
- user_id (FK → User)
- action_type (조회/생성/수정/삭제)
- resource_type (환자/이미지/기록 등)
- resource_id
- description
- ip_address
- created_at
```

#### 3.5.2 Notification (알림) - 새로 추가
```sql
- id (PK)
- user_id (FK → User)
- notification_type (예약/분석완료/알림 등)
- title
- message
- is_read
- related_resource_type
- related_resource_id
- created_at
```

#### 3.5.3 ChatRoom (채팅방) - 새로 추가
```sql
- id (PK)
- room_type (일대일/그룹/진료상담 등)
- name (그룹 채팅방 이름)
- created_by (FK → User)
- related_patient_id (FK → Patient, nullable, 진료 상담 시)
- related_appointment_id (FK → Appointment, nullable)
- created_at
- updated_at
```

#### 3.5.4 ChatRoomMember (채팅방 멤버) - 새로 추가
```sql
- id (PK)
- room_id (FK → ChatRoom)
- user_id (FK → User)
- role (참여자/관리자)
- joined_at
- last_read_at
```

#### 3.5.5 ChatMessage (채팅 메시지) - 새로 추가
```sql
- id (PK)
- room_id (FK → ChatRoom)
- sender_id (FK → User)
- message_type (텍스트/이미지/파일/의료이미지링크 등)
- content (메시지 내용)
- file_url (첨부 파일 URL, nullable)
- related_image_id (FK → MedicalImage, nullable)
- related_record_id (FK → MedicalRecord, nullable)
- is_read
- created_at
- updated_at
```

#### 3.5.6 ChatBotConversation (챗봇 대화) - 새로 추가
```sql
- id (PK)
- user_id (FK → User)
- bot_type (영상판독/병원홈페이지/일반문의)
- session_id (대화 세션 ID)
- context (JSON, 대화 컨텍스트)
- created_at
- updated_at
```

#### 3.5.7 ChatBotMessage (챗봇 메시지) - 새로 추가
```sql
- id (PK)
- conversation_id (FK → ChatBotConversation)
- message_type (user/bot)
- content (메시지 내용)
- intent (의도 분석 결과, nullable)
- response_type (텍스트/버튼/이미지/링크)
- metadata (JSON, 추가 정보)
- created_at
```

---

## 4. 역할별 홈페이지 기능 정의

### 4.1 의사 홈페이지

#### 4.1.1 대시보드
- 오늘의 예약 목록 (진료과별 필터링)
- 담당 환자 현황
- 최근 진료 기록
- AI 분석 대기 중인 이미지 (진료과별)
- 진료과별 통계 (환자 수, 진료 건수 등)

#### 4.1.2 환자 관리
- 환자 검색 및 조회
- 환자 상세 정보
- 진료 기록 작성
- 처방전 발급
- 진료과별 환자 목록 필터링

#### 4.1.3 의료 이미지
- 환자별 이미지 조회
- 진료과별 특화 이미지 필터링
  - 폐-호흡기내과: 폐 CT 이미지
  - 머리-신경외과: 뇌 CT/MRI 이미지
  - 유방-유방외과: 유방 MRI 이미지
  - 췌장-소화기내과: 복부 CT/MRI 이미지
  - 피부-피부과: 피부 이미지 (카메라 촬영, 모바일 앱)
- AI 분석 결과 확인
- 영상 판독서 확인
- 이미지 다운로드

#### 4.1.4 예약 관리
- 예약 조회 및 관리
- 예약 생성 및 수정
- 스케줄 캘린더
- 진료과별 예약 필터링

#### 4.1.5 AI 진단 지원
- AI 분석 결과 기반 진단 제안
- 위험도 평가
- 권장 검사 항목
- 진료과별 특화 AI 분석 도구
  - 폐-호흡기내과: 폐 결절 탐지 분석
  - 머리-신경외과: 뇌종양 탐지 분석
  - 유방-유방외과:
    - **1차 검진: 유방 초음파 분석 (2D 딥러닝 모델)**
      - 초기 병변 탐지 및 정밀 진단
      - 2D 이미지 처리
    - **2차 검진: 조직 검사 (병리 이미지) 분석 (2D 딥러닝 모델)**
      - 현미경 이미지 분석
      - 최종 암 확진
      - 2D 병리 이미지 처리
    - **3차 검진: 유방 MRI 분석 (3D 딥러닝 모델, DICOM 파일)**
      - **사용 시점:** 암 확진 후 수술 전 정밀 평가
      - **목적:** 암의 크기 및 범위 평가, 다발성 암 확인, 수술 계획 수립
      - **데이터 형식:** DICOM 파일 (Orthanc 서버 활용)
      - **처리 방식:** 3D 볼륨 데이터 세그멘테이션 및 분류
  - 췌장-소화기내과: 췌장 병변 탐지 분석
  - 피부-피부과: 피부암 여부 분류, 피부 병변 탐지 및 분류

#### 4.1.6 채팅 및 소통
- 의료진 간 실시간 채팅
- 환자와의 진료 상담 채팅
- 의료 이미지 공유
- 진료 기록 공유
- 그룹 채팅 (진료팀)

---

### 4.2 간호사 홈페이지

#### 4.2.1 대시보드
- 오늘의 예약 목록
- 담당 환자 현황
- 투약 스케줄
- 환자 상태 알림

#### 4.2.2 환자 관리
- 환자 기본 정보 조회
- 환자 상태 기록
- 간호 기록 작성
- 활력 징후 입력

#### 4.2.3 의료 이미지
- 기본 이미지 조회 (제한적)
- 촬영 스케줄 확인

#### 4.2.4 채팅 및 소통
- 의료진 간 실시간 채팅
- 환자와의 소통
- 의사에게 문의 전달

---

### 4.3 원무과 홈페이지

#### 4.3.1 대시보드
- 오늘의 예약 현황
- 접수 대기 목록
- 수납 현황
- 예약 통계

#### 4.3.2 환자 접수 및 등록
- 신규 환자 등록
- 환자 정보 조회 및 수정
- 환자 검색

#### 4.3.3 예약 관리
- 예약 생성
- 예약 조회 및 수정
- 예약 취소
- 예약 스케줄 관리
- 예약 통계 및 현황

#### 4.3.4 수납 관리
- 진료비 계산
- 수납 처리
- 영수증 발급
- 수납 이력 조회

#### 4.3.5 보험 처리
- 건강보험 처리
- 의료급여 처리
- 보험 정보 조회

#### 4.3.6 통계 및 현황
- 일일 예약 통계
- 수납 통계
- 환자 현황

---

### 4.4 환자 홈페이지

#### 4.3.1 대시보드
- 건강 요약 정보
- 최근 진료 기록
- 예약 정보
- 알림 및 공지사항

#### 4.3.2 내 진료 기록
- 진료 기록 조회
- 처방전 조회 및 다운로드
- 진단서 조회

#### 4.3.3 내 의료 이미지
- 본인 의료 이미지 조회
- AI 분석 결과 확인
- 3D 시각화 확인
- 이미지 다운로드

#### 4.3.4 예약 관리
- 예약 조회
- 예약 신청
- 예약 취소

#### 4.3.5 건강 정보
- 건강 상태 모니터링
- 복용 중인 약물
- 알레르기 정보

#### 4.3.6 피부암 1차 스크리닝 (Flutter 모바일 앱)
- **병원 방문 전 자가 스크리닝**
  - Flutter 모바일 앱 다운로드 및 설치
  - 본인 모바일로 피부 이미지 촬영
  - AI 기반 피부암 여부 1차 판별
  - 스크리닝 결과 확인
    - 정상/의심/위험 단계별 결과 제공
    - 병원 방문 권장 여부 안내
  - 스크리닝 결과를 병원 시스템에 전송
  - 결과에 따른 피부과 예약 연계
- **스크리닝 이력 관리**
  - 과거 스크리닝 결과 조회
  - 스크리닝 결과 추이 확인
- **병원 방문 후 연계**
  - 스크리닝 결과를 의사가 확인
  - 정밀 진단 및 치료 계획 수립

#### 4.3.7 채팅 및 상담
- 담당 의사와 실시간 채팅
- 진료 상담
- 증상 문의
- 챗봇을 통한 기본 문의 (병원 홈페이지)

---

### 4.5 방사선과 홈페이지

#### 4.4.1 대시보드
- 오늘의 촬영 스케줄
- 촬영 대기 목록
- 촬영 완료 목록
- 업로드 대기 목록

#### 4.4.2 이미지 촬영 및 업로드
- 촬영 스케줄 확인
- 의료 이미지 업로드
- 촬영 정보 입력 (환자 ID, 촬영 부위, 촬영 조건 등)
- 이미지 메타데이터 관리
- 이미지 품질 검사

#### 4.4.3 촬영 관리
- 촬영 이력 조회
- 업로드 이력 조회
- 촬영 통계

---

### 4.6 영상의학과 홈페이지

#### 4.5.1 대시보드
- 판독 대기 목록
- AI 분석 대기 목록
- 오늘의 판독 현황
- 판독 통계

#### 4.5.2 이미지 조회
- 모든 의료 이미지 조회
- 환자별 이미지 조회
- 촬영 정보 확인

#### 4.5.3 AI 분석
- AI 분석 실행
- 분석 결과 확인
- 분석 리포트 생성
- 3D 시각화 생성

#### 4.5.4 영상 판독
- 판독서 작성
- 판독 이력 관리
- 판독 품질 관리
- 판독 통계

#### 4.5.5 환자 조회
- 환자별 이미지 조회
- 판독 이력 확인

#### 4.5.6 채팅 및 챗봇
- 의료진 간 실시간 채팅
- 판독 관련 협의
- **판독 챗봇**: 
  -  제공
  - DICOM 표준 질의응답
  - 판독서 작성 도움

---

## 5. 시스템 아키텍처 설계

### 5.1 전체 시스템 구조

**CDSS (차세대 지능형 의료지원 시스템(MSA)) - 시스템 아키텍처 다이어그램**

```
                    ┌─────────────────────┐
                    │    외부기관          │
                    │ (다른 병원, 보험사)   │
                    └──────────┬──────────┘
                               │ FHIR Protocol
                               ↓
                    ┌─────────────────────┐
                    │   FHIR Server       │
                    │ (의료정보 표준 교환)  │
                    └───┬───────────┬─────┘
                        │           │
        ┌───────────────┘           └───────────────┐
        ↓                                           ↓
┌───────────────┐                          ┌───────────────┐
│   EMR/EHR     │                          │      OCS       │
│ 전자의무기록   │                          │ 처방전달시스템  │
└───────┬───────┘                          └───────┬───────┘
        │                                           │
        │                                           │
        └───────────────┬───────────────────────────┘
                        │
                        ↓
        ┌───────────────────────────────────────┐
        │      Django Framework                 │
        │  (CDSS 핵심 애플리케이션 서버)          │
        │  - API Gateway                        │
        │  - 비즈니스 로직 처리                   │
        │  - 사용자 인증/인가                     │
        │  - 병원 홈페이지 + 의료진 플랫폼 통합   │
        └───┬───────────┬───────────┬───────────┘
            │           │           │
    ┌───────┘           │           └───────┐
    │                   │                   │
    ↓                   ↓                   ↓
┌───────────┐    ┌───────────┐    ┌───────────┐
│   RIS     │    │   LIS     │    │ AI(Mosec) │
│영상정보시스템│  │검사정보시스템│  │ AI 분석엔진│
└─────┬─────┘    └─────┬─────┘    └─────┬─────┘
      │                │                 │
      └────────────────┼─────────────────┘
                       │
                       ↓
            ┌─────────────────────┐
            │     Database        │
            │  (MySQL/MariaDB)    │
            │  환자정보, 의료기록   │
            └─────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Nginx (HTTPS)                            │
│  (리버스 프록시, SSL/TLS, WebSocket 프록시)                  │
└───┬───────────────┬───────────────┬─────────────────────────┘
    │               │               │
    ↓               ↓               ↓
┌───────────┐  ┌───────────┐  ┌───────────┐
│  Patient  │  │   Doctor  │  │   Nurse   │  │  Admin   │
│  (환자)    │  │  (의사)    │  │ (간호사)  │  │ (원무과) │
│병원홈페이지│  │의료진플랫폼│  │의료진플랫폼│  │의료진플랫폼│
└───────────┘  └───────────┘  └───────────┘
                       │
                       ↓
        ┌───────────────────────────────┐
        │  Radiology Tech │ Radiology Doc│
        │   (방사선과)     │ (영상의학과)  │
        │  의료진 플랫폼   │ 의료진 플랫폼 │
        └───────────────────────────────┘

        ┌───────────────────────────────┐
        │   Orthanc (PACS/DICOM)        │
        │   DICOM 서버                   │
        │   의료 영상 저장 및 관리        │
        └───────────────┬───────────────┘
                        │
                        ↓
        ┌───────────────────────────────┐
        │   채팅 서비스                  │
        │   (Django Channels)            │
        │   - WebSocket 실시간 채팅      │
        └───────────────────────────────┘

        ┌───────────────────────────────┐
        │   챗봇 서비스                  │
        │   (NLP + 규칙 기반)            │
        │   - 영상 판독 챗봇             │
        │   - 병원 홈페이지 챗봇         │
        └───────────────────────────────┘
```

**데이터 흐름 설명:**

1. **사용자 접근 경로:**
   - 환자: Nginx → 병원 홈페이지 (Public) → Django Framework
   - 의료진: Nginx → 의료진 플랫폼 (CDSS) → Django Framework

2. **의료 정보 시스템 연동:**
   - Django Framework ↔ EMR/EHR (전자의무기록)
   - Django Framework ↔ OCS (처방전달시스템)
   - Django Framework ↔ Orthanc (DICOM 서버)
   - Django Framework ↔ RIS (영상정보시스템)
   - Django Framework ↔ LIS (검사정보시스템)

3. **외부 시스템 연동:**
   - 외부기관 ↔ FHIR Server ↔ EMR/EHR, OCS
   - FHIR Server를 통한 표준 의료 정보 교환

4. **AI 분석 흐름:**
   - 의료 이미지: Orthanc → Django → AI(Mosec) → 분석 결과 → Database
   - 처방/검사: OCS/LIS → AI(Mosec) → 분석 결과 → Database

5. **데이터 저장:**
   - 모든 의료 정보: Database (MySQL/MariaDB)
   - DICOM 파일: Orthanc 서버
   - 정적 파일: Nginx 직접 서빙

6. **실시간 통신:**
   - 채팅: Django Channels (WebSocket)
   - 챗봇: NLP 서비스 (규칙 기반 + 자연어 처리)

### 5.2 시스템 구성 요소 상세

#### 5.2.1 외부 시스템 연동
- **FHIR Server**: HL7 FHIR 표준을 통한 의료 정보 교환
  - **구현 방식 선택지:**
     **Django 내장 FHIR API** (권장)
       - Django 애플리케이션에 FHIR 엔드포인트 추가
       - `django-fhir` 또는 `fhir.resources` 라이브러리 사용
       - 별도 서버 불필요, Django와 통합 관리
  - **외부기관**: 다른 병원, 보험사, 정부 기관 등과의 데이터 교환

#### 5.2.2 핵심 의료 정보 시스템

- **EMR/EHR (Electronic Medical Record/Electronic Health Record)**
  - 전자의무기록 시스템
  - 환자 진료 기록 통합 관리
  - Django Framework와 양방향 데이터 동기화

**OCS (Order Communication System) - 처방전달시스템**

- **구현 방식:** Django 앱으로 구현 (backend/ocs/)
- **주요 기능:**
  - 처방/검사/영상촬영 주문 생성
  - 주문을 약국/검사실/영상의학과로 자동 전달
  - 주문 상태 추적 및 관리
  - 주문 완료 알림
- **연동 시스템:**
  - EMR/EHR (진료 기록 연동)
  - RIS (영상 촬영 의뢰)
  - LIS (검사 주문)
  - Database (주문 정보 저장)

**RIS (Radiology Information System) - 영상정보시스템**

- **구현 방식:** Django 앱으로 구현 (backend/ris/)
- **주요 기능:**
  - 촬영 스케줄 관리 (예약, 조회, 수정)
  - 촬영 의뢰 수신 및 처리
  - 판독서 작성 및 관리
  - 촬영/판독 상태 추적
- **연동 시스템:**
  - Orthanc (DICOM 서버) - RESTful API 연동
  - OCS (촬영 의뢰 수신)
  - Database (스케줄, 판독서 저장)
  - Django Framework (의료진 인터페이스)

**LIS (Laboratory Information System) - 검사정보시스템**

- **구현 방식:** Django 앱으로 구현 (backend/lis/)
- **주요 기능:**
  - 검사 주문 관리
  - 검사 결과 입력 및 조회
  - 검사 항목별 결과 관리
  - 이상 결과 알림
- **연동 시스템:**
  - OCS (검사 주문 수신)
  - AI(Mosec) (검사 결과 AI 분석, 선택 사항)
  - Database (검사 주문, 결과 저장)
  - Django Framework (의료진 인터페이스)

- **Orthanc (DICOM 서버)**
  - 오픈소스 DICOM 서버
  - DICOM 표준 기반 영상 저장 및 관리
  - 의료 이미지 아카이빙 및 검색
  - RESTful API 제공
  - RIS와 연동하여 영상 정보 관리
  - Django Framework와 API 연동

#### 5.2.3 AI 분석 시스템
- **AI(MonAI)**
  - MonAI 프레임워크 기반 딥러닝 모델 실행
  - 의료 이미지 분석 (세그멘테이션, 분류)
  - 진료과별 특화 모델 지원
  - OCS, LIS, Database와 연동

#### 5.2.4 웹 서버 및 프레임워크
- **Nginx**
  - 리버스 프록시 서버
  - SSL/TLS 암호화 (HTTPS)
  - 정적 파일 서빙
  - 로드 밸런싱
  - 요청 라우팅

- **Django Framework**
  - 핵심 애플리케이션 서버
  - RESTful API 제공
  - 비즈니스 로직 처리
  - 사용자 인증/인가
  - EMR, Orthanc (DICOM), RIS, LIS와 연동

#### 5.2.5 데이터베이스
- **MySQL/MariaDB**
  - 환자 정보
  - 의료 기록
  - 이미지 메타데이터
  - 사용자 정보
  - 시스템 설정

#### 5.2.6 사용자 인터페이스

**병원 홈페이지 (Public)**
- **일반 사용자**: 병원 소개, 진료 안내, 예약 문의, 챗봇 문의
- **환자**: 회원가입/로그인 → 본인 정보 조회, 진료 기록 조회, 의료 이미지 조회, 예약 관리, 챗봇 문의
- **의료진 로그인 버튼**: 의료진 플랫폼(CDSS)으로 이동 (의료진 계정으로 로그인 불가)

**의료진 플랫폼 (CDSS - 내부 시스템)**
- **의료진 전용**: 환자 접근 불가
- **Doctor (의사)**: 진료과별 맞춤형 인터페이스
- **Nurse (간호사)**: 환자 간호 관리 인터페이스
- **Administrative Staff (원무과)**: 환자 접수, 예약, 수납 관리 인터페이스
- **Radiology Technician (방사선과)**: 촬영 및 업로드 인터페이스
- **Radiology Doctor (영상의학과)**: 판독 인터페이스

---

## 6. 마이크로서비스 아키텍처 설계

### 5.1 서비스 구성

#### 5.1.1 API Gateway 서비스
- **역할:** 모든 요청의 진입점
- **기능:**
  - 라우팅
  - 인증/인가
  - 요청 로깅
  - Rate Limiting

#### 5.1.2 사용자 관리 서비스 (User Service)
- **역할:** 사용자 인증 및 권한 관리
- **기능:**
  - 로그인/로그아웃
  - 사용자 정보 관리
  - 역할별 권한 관리
  - JWT 토큰 발급

#### 5.1.3 환자 관리 서비스 (Patient Service)
- **역할:** 환자 정보 관리
- **기능:**
  - 환자 CRUD
  - 환자 검색
  - 환자 상태 관리

#### 5.1.4 의료 이미지 서비스 (Medical Image Service)
- **역할:** 의료 이미지 관리
- **기능:**
  - 이미지 업로드/다운로드
  - 이미지 메타데이터 관리
  - 이미지 조회 및 검색

#### 5.1.5 AI 분석 서비스 (AI Analysis Service)
- **역할:** 딥러닝 모델 실행
- **기능:**
  - 세그멘테이션 분석
  - 분류 분석
  - 분석 결과 저장
  - 모델 관리

#### 5.1.6 진료 관리 서비스 (Medical Record Service)
- **역할:** 진료 기록 관리
- **기능:**
  - 진료 기록 CRUD
  - 처방전 관리
  - 진단서 관리

#### 5.1.7 예약 관리 서비스 (Appointment Service)
- **역할:** 예약 관리
- **기능:**
  - 예약 CRUD
  - 스케줄 관리
  - 예약 알림

#### 5.1.8 방사선과 서비스 (Radiology Technician Service)
- **역할:** 촬영 및 이미지 업로드 관리
- **기능:**
  - 촬영 스케줄 관리
  - 이미지 업로드
  - 촬영 정보 관리
  - 촬영 이력 관리

#### 5.1.9 영상 판독 서비스 (Radiology Doctor Service)
- **역할:** 영상 판독 관리
- **기능:**
  - 판독서 작성
  - 판독 이력 관리
  - 판독 통계

#### 5.1.10 알림 서비스 (Notification Service)
- **역할:** 알림 관리
- **기능:**
  - 알림 생성
  - 알림 전송
  - 알림 조회

#### 5.1.11 3D 시각화 서비스 (Visualization Service)
- **역할:** 3D 시각화 생성
- **기능:**
  - 3D 볼륨 생성
  - 시각화 HTML 생성
  - 시각화 캐싱

---

### 5.2 데이터베이스 분리 전략

#### 5.2.1 서비스별 데이터베이스
- **User DB:** 사용자 및 인증 정보
- **Patient DB:** 환자 정보
- **Medical Image DB:** 의료 이미지 메타데이터
- **Medical Record DB:** 진료 기록
- **Appointment DB:** 예약 정보
- **Radiology DB:** 영상 판독 정보

#### 5.2.2 데이터 동기화
- 이벤트 기반 아키텍처 (Event Bus)
- 메시지 큐 (RabbitMQ/Kafka)
- API 간 통신 (REST/gRPC)

---

## 6. 프론트엔드 구조

### 6.1 공통 컴포넌트
- 기존 React 디자인 시스템 재사용
- 공통 UI 컴포넌트
- 공통 유틸리티 함수

### 6.2 프론트엔드 구조

#### 6.2.1 병원 홈페이지 (Public)

**공개 페이지**
- **경로:** `/` (루트)
- **기능:**
  - 병원 소개
  - 진료 안내
  - 예약 문의
  - 챗봇 (일반 문의, 예약 안내)
  - 의료진 로그인 버튼 → `/login/medical` (의료진 플랫폼으로 이동)

**환자 인터페이스**
- **경로:** `/patient/*` (환자 로그인 후)
- **기능:**
  - 환자 회원가입/로그인
  - 본인 정보 조회 (`/patient/profile`)
  - 진료 기록 조회 (`/patient/records`)
  - 의료 이미지 조회 (`/patient/images`)
  - 예약 관리 (`/patient/appointments`)
  - 챗봇 문의 (`/patient/chatbot`)
- **접근 제어:** 환자 본인만 접근 가능

#### 6.2.2 의료진 플랫폼 (CDSS - 내부 시스템)

**인증 및 접근 제어**
- **의료진 로그인:** `/login/medical`
- **인증 후 리다이렉트:** 역할별 대시보드로 자동 이동
- **접근 제어:** 의료진만 접근 가능 (환자 접근 불가)
- **역할 기반 라우팅 가드**

**역할별 앱 분리**

#### 6.2.3 의사 앱 (`/doctor/*`)
- 경로: `/doctor/dashboard`, `/doctor/patients`, `/doctor/images` 등
- 라우팅: 역할 기반 접근 제어

#### 6.2.4 간호사 앱 (`/nurse/*`)
- 경로: `/nurse/dashboard`, `/nurse/patients`, `/nurse/records` 등

#### 6.2.5 원무과 앱 (`/admin/*`)
- 경로: `/admin/dashboard`, `/admin/appointments`, `/admin/patients`, `/admin/payment` 등

#### 6.2.6 방사선과 앱 (`/radiology-tech/*`)
- 경로: `/radiology-tech/dashboard`, `/radiology-tech/upload`, `/radiology-tech/schedule` 등

#### 6.2.7 영상의학과 앱 (`/radiology-doctor/*`)
- 경로: `/radiology-doctor/dashboard`, `/radiology-doctor/images`, `/radiology-doctor/analysis` 등

---

## 7. API 엔드포인트 설계

### 7.1 인증 API
```
POST /api/auth/login
POST /api/auth/logout
POST /api/auth/refresh
GET  /api/auth/me
```

### 7.2 환자 API
```
GET    /api/patients/
GET    /api/patients/{id}/
POST   /api/patients/
PUT    /api/patients/{id}/
DELETE /api/patients/{id}/
GET    /api/patients/{id}/records/
```

### 7.3 의료 이미지 API
```
GET    /api/medical-images/
GET    /api/medical-images/{id}/
POST   /api/medical-images/
GET    /api/medical-images/?department={진료과} (진료과별 필터링)
GET    /api/medical-images/{id}/analyze/
POST   /api/medical-images/{id}/analyze/
GET    /api/medical-images/generate_3d_visualization/
```

### 7.4 진료 기록 API
```
GET    /api/medical-records/
GET    /api/medical-records/{id}/
POST   /api/medical-records/
PUT    /api/medical-records/{id}/
```

### 7.5 예약 API
```
GET    /api/appointments/
GET    /api/appointments/{id}/
POST   /api/appointments/
PUT    /api/appointments/{id}/
DELETE /api/appointments/{id}/
GET    /api/appointments/?department={진료과} (진료과별 필터링)
```

### 7.6 방사선과 API
```
GET    /api/radiology-tech/schedules/
GET    /api/radiology-tech/schedules/{id}/
POST   /api/radiology-tech/images/upload/
GET    /api/radiology-tech/images/
GET    /api/radiology-tech/images/{id}/
```

### 7.7 영상 판독 API
```
GET    /api/radiology-doctor/reports/
GET    /api/radiology-doctor/reports/{id}/
POST   /api/radiology-doctor/reports/
PUT    /api/radiology-doctor/reports/{id}/
GET    /api/radiology-doctor/images/
GET    /api/radiology-doctor/images/{id}/analyze/
```

### 7.8 채팅 API
```
GET    /api/chat/rooms/ (채팅방 목록)
GET    /api/chat/rooms/{id}/ (채팅방 상세)
POST   /api/chat/rooms/ (채팅방 생성)
GET    /api/chat/rooms/{id}/messages/ (메시지 목록)
POST   /api/chat/rooms/{id}/messages/ (메시지 전송)
PUT    /api/chat/messages/{id}/read/ (메시지 읽음 처리)
WS     /ws/chat/{room_id}/ (WebSocket 실시간 채팅)
```

### 7.9 챗봇 API
```
POST   /api/chatbot/conversations/ (대화 시작)
POST   /api/chatbot/conversations/{id}/messages/ (메시지 전송)
GET    /api/chatbot/conversations/{id}/messages/ (대화 기록)
POST   /api/chatbot/radiology/ask/ (영상 판독 챗봇 질문)
POST   /api/chatbot/hospital/ask/ (병원 홈페이지 챗봇 질문)
WS     /ws/chatbot/{conversation_id}/ (WebSocket 실시간 챗봇)
```

---

## 8. 보안 및 권한 관리

### 8.1 인증 방식
- JWT (JSON Web Token)
- 토큰 기반 인증
- Refresh Token 구현

### 8.2 권한 관리
- 역할 기반 접근 제어 (RBAC)
- 리소스별 권한 체크
- API 레벨 권한 검증

### 8.3 데이터 접근 제어
- 환자 데이터: 본인 또는 담당 의료진만 접근
- 의료 이미지: 역할별 접근 권한 분리
- 진료 기록: 담당 의사 및 환자 본인만 접근

### 8.4 시스템 접근 제어
- **병원 홈페이지**: 
  - 공개 접근 가능 (일반 사용자)
  - 환자 회원가입/로그인 (환자 계정만 가능)
  - 의료진 계정으로 로그인 불가
  - 의료진 로그인 버튼 → 의료진 플랫폼으로 이동
- **의료진 플랫폼**: 
  - 의료진 로그인 필수 (의료진 계정만 가능)
  - 환자 접근 불가
  - IP 화이트리스트 (선택 사항)
  - VPN 접근 (선택 사항)
  - 역할 기반 접근 제어

---

## 9. 기술 스택

### 9.1 프론트엔드
- **프레임워크:** React + TypeScript
- **상태 관리:** TanStack Query
- **UI 라이브러리:** shadcn/ui (기존 디자인 재사용)
- **라우팅:** React Router
- **3D 시각화:** Plotly
- **모바일 앱:** Flutter (피부암 1차 스크리닝 서비스)
  - 환자 본인 모바일로 피부 이미지 촬영
  - AI 기반 피부암 여부 1차 판별
  - 병원 방문 전 자가 스크리닝
  - 스크리닝 결과를 병원 시스템에 전송
  - 결과에 따른 예약 연계
  - 피부 이미지 촬영
  - 이미지 업로드 및 AI 분석
  - 분석 결과 확인

### 9.2 백엔드
- **프레임워크:** Django + Django REST Framework
- **인증:** Django JWT
- **데이터베이스:** MySQL
- **서버:** Gunicorn + Nginx
- **실시간 통신:** Django Channels (WebSocket)
- **챗봇:** 
  - 자연어 처리: spaCy 또는 Transformers (BERT)
  - 의도 분석: Rasa 또는 커스텀 모델
  - 응답 생성: 규칙 기반 또는 LLM (선택 사항)

### 9.3 딥러닝 서비스
- **프레임워크:** Mosec
- **모델:** PyTorch
- **모델:** UNet (세그멘테이션), ResNet50 (분류)

### 9.4 인프라
- **클라우드:** GCP VM
- **배포 주소:** http://34.42.223.43/
- **컨테이너:** Docker (예정)
- **메시지 큐:** RabbitMQ (예정)
- **모니터링:** (추가 예정)

---

## 10. GCP 배포 전략

### 10.1 배포 환경 개요

#### 10.1.1 서버 정보
- **배포 서버:** GCP VM 인스턴스
- **서버 주소:** http://34.42.223.43/
- **운영 체제:** Ubuntu 22.04 LTS
- **웹 서버:** Nginx
- **애플리케이션 서버:** Gunicorn (Django)
- **데이터베이스:** MySQL/MariaDB
- **AI 서비스:** Mosec (별도 포트)

#### 10.1.2 배포 방식
- **버전 관리:** GitHub 저장소 활용
- **배포 자동화:** GitHub Actions 또는 수동 배포
- **컨테이너화:** Docker (선택 사항, 향후 확장 시)

---

## 11. 네트워크 및 통신 프로토콜

### 10.1 통신 프로토콜

#### 10.1.1 HTTP/HTTPS
- **용도:** 웹 애플리케이션 통신
- **포트:** 80 (HTTP), 443 (HTTPS)
- **인증:** JWT 토큰 기반
- **암호화:** TLS 1.3

#### 10.1.2 FHIR (Fast Healthcare Interoperability Resources)
- **용도:** 외부 시스템과의 의료 정보 교환
- **버전:** FHIR R4
- **표준:** HL7 FHIR
- **구현 방식:**
  - **옵션 1: Django 내장 FHIR API** (권장)
    - Django REST Framework에 FHIR 엔드포인트 추가
    - `fhir.resources` Python 라이브러리 사용
    - 엔드포인트 예: `/fhir/Patient/{id}`, `/fhir/Observation`
    - 별도 서버 불필요, Django와 통합
  - **옵션 2: 별도 FHIR 서버**
    - HAPI FHIR (Java), Firely Server (.NET), IBM FHIR Server
    - 별도 VM 필요, Django와 API로 통신
  - **옵션 3: 클라우드 FHIR 서비스**
    - Google Cloud Healthcare API
    - AWS HealthLake
- **리소스 타입:**
  - Patient (환자 정보)
  - Observation (검사 결과)
  - DiagnosticReport (진단 리포트)
  - ImagingStudy (영상 연구)

#### 10.1.3 DICOM (Digital Imaging and Communications in Medicine)
- **용도:** 의료 영상 교환
- **표준:** DICOM 3.0
- **통신:** DICOM C-STORE, C-FIND, C-MOVE
- **Orthanc 연동:** 
  - Orthanc DICOM 서버와 통신
  - RESTful API를 통한 이미지 조회 및 저장
  - DICOM 파일 메타데이터 추출

#### 10.1.4 HL7 (Health Level Seven)
- **용도:** 의료 정보 시스템 간 데이터 교환
- **버전:** HL7 v2.x, HL7 v3
- **메시지 타입:**
  - ADT (Admission, Discharge, Transfer)
  - ORU (Observation Result)
  - ORM (Order Message)

### 10.2 API 통신

#### 10.2.1 RESTful API
- **프로토콜:** HTTP/HTTPS
- **데이터 형식:** JSON
- **인증:** JWT Bearer Token
- **버전 관리:** `/api/v1/`, `/api/v2/`

#### 10.2.2 gRPC (선택 사항)
- **용도:** 마이크로서비스 간 고성능 통신
- **프로토콜:** HTTP/2
- **데이터 형식:** Protocol Buffers

### 10.3 데이터 동기화

#### 10.3.1 실시간 동기화
- **WebSocket:** 
  - 실시간 알림 전송
  - 실시간 채팅 (Django Channels)
  - 챗봇 실시간 대화
- **Server-Sent Events (SSE):** 서버 푸시 알림

#### 10.3.2 배치 동기화
- **스케줄 작업:** Celery (선택 사항)
- **주기:** 시간별/일별 배치 처리

---

## 12. 개발 단계

### 11.1 Phase 1: 기반 구축 (1-2주)
- 데이터베이스 스키마 설계 및 구현
- 사용자 인증 시스템 구축
- 기본 API 엔드포인트 개발
- 역할별 라우팅 설정

### 11.2 Phase 2: 역할별 홈페이지 개발 (2-3주)
- 병원 홈페이지 개발 (공개 페이지)
- 의료진 플랫폼 접근 통합
- 의사 홈페이지 개발
- 간호사 홈페이지 개발
- 환자 홈페이지 개발
- 방사선과 홈페이지 개발
- 영상의학과 홈페이지 개발

### 11.3 Phase 3: 핵심 기능 구현 (2-3주)
- 의료 이미지 관리 기능
- AI 분석 연동
- 진료 기록 관리
- 예약 시스템

### 11.4 Phase 4: 고급 기능 (1-2주)
- 3D 시각화 통합
- 알림 시스템
- 리포트 생성
- 통계 대시보드

### 11.5 Phase 5: 테스트 및 최적화 (1주)
- 통합 테스트
- 성능 최적화
- 보안 검증
- 사용자 테스트

---

## 13. 데이터 흐름도

### 13.1 시스템 접근 흐름

**병원 홈페이지 (Public) - 환자 접근**
```
일반 사용자
    ↓
병원 홈페이지 (Public) - https://hospital.com
    ├─ 병원 소개, 진료 안내
    ├─ 챗봇 문의 (일반 문의, 예약 안내)
    └─ 예약 문의

환자
    ↓
병원 홈페이지 (Public) - https://hospital.com
    ↓
환자 회원가입/로그인
    ↓
환자 인터페이스
    ├─ 본인 정보 조회 (/patient/profile)
    ├─ 진료 기록 조회 (/patient/records)
    ├─ 의료 이미지 조회 (/patient/images)
    ├─ 예약 관리 (/patient/appointments)
    └─ 챗봇 문의 (/patient/chatbot)
```

**의료진 플랫폼 (CDSS) - 의료진 접근**
```
의료진
    ↓
병원 홈페이지 (Public) - https://hospital.com
    ↓
의료진 로그인 버튼 클릭
    ↓
의료진 로그인 페이지 - /login/medical
    ↓
인증 성공 (의료진 계정만 가능)
    ↓
의료진 플랫폼 (CDSS) - 내부 시스템
    ├─ 의사 대시보드 (/doctor/dashboard)
    ├─ 간호사 대시보드 (/nurse/dashboard)
    ├─ 원무과 대시보드 (/admin/dashboard)
    ├─ 방사선과 대시보드 (/radiology-tech/dashboard)
    └─ 영상의학과 대시보드 (/radiology-doctor/dashboard)
```

### 13.2 의료 이미지 분석 흐름
```
의사 (진료과별) → 촬영 의뢰
  - 폐-호흡기내과: 폐 CT 의뢰
  - 머리-신경외과: 뇌 CT/MRI 의뢰
  - 유방-유방외과: 유방 MRI 의뢰
  - 췌장-소화기내과: 복부 CT/MRI 의뢰
  - 피부-피부과: 피부 이미지 촬영 의뢰 (카메라/Flutter 모바일 앱)
    ↓
방사선과/피부과 → 의료 이미지 촬영 (진료과별)
  - 일반 영상: 방사선과 촬영
  - 피부 이미지: 피부과에서 직접 촬영 또는 Flutter 모바일 앱
    ↓
방사선과/피부과 → 이미지 업로드 및 촬영 정보 입력 (진료과 태그)
    ↓
영상의학과/피부과 → 이미지 확인 및 AI 분석 실행
    ↓
AI 분석 서비스 → 진료과별 특화 분석
  - 유방-유방외과: 세그멘테이션 → 분류 분석
  - 폐-호흡기내과: 폐 결절 탐지 분석
  - 머리-신경외과: 뇌종양 탐지 분석
  - 췌장-소화기내과: 췌장 병변 탐지 분석
  - 피부-피부과: 피부암 여부 분류, 피부 병변 탐지 및 분류
    ↓
영상의학과/피부과 → 판독서 작성 (피부과는 의사가 직접 판독)
    ↓
의사 (해당 진료과) → 판독서 및 분석 결과 확인
    ↓
의사 → 진단 및 치료 결정
    ↓
환자 → 본인 결과 확인 (Flutter 모바일 앱에서도 확인 가능)
```

### 13.3 진료 프로세스 흐름
```
환자 → 예약 신청
    ↓
원무과 → 예약 접수 및 확인
    ↓
의사 → 진료 진행
    ↓
의사 → 진료 기록 작성
    ↓
의사 → 처방전 발급 (OCS를 통해 약국으로 전달)
    ↓
간호사 → 투약 및 간호 기록
    ↓
원무과 → 수납 처리
    ↓
환자 → 진료 기록 및 처방전 확인
```

### 13.4 OCS (처방전달시스템) 전달 흐름
```
의사 (진료 중)
    ↓
처방/검사/영상 주문 입력
    ↓
OCS 시스템
    ├─ 주문 정보 저장 (Database)
    ├─ 주문 검증 (약물 상호작용, 알레르기 체크 등)
    └─ 주문 전달
        ├─ 처방전 → 약국 시스템
        ├─ 검사 주문 → LIS (검사실)
        └─ 영상 의뢰 → RIS (방사선과)
    ↓
각 부서에서 주문 처리
    ↓
상태 업데이트
    ├─ 약국: 조제중 → 조제완료
    ├─ 검사실: 검사중 → 검사완료
    └─ 방사선과: 촬영중 → 촬영완료
    ↓
결과 전달
    ├─ 약국: 조제 완료 알림
    ├─ 검사실: 검사 결과 전달
    └─ 방사선과: 판독서 전달
    ↓
의사가 결과 확인
```

**OCS 전달 형태:**
- **데이터 형태**: JSON 형식의 구조화된 주문 정보
- **전달 방식**: RESTful API 또는 메시지 큐 (RabbitMQ/Kafka)
- **실시간 알림**: WebSocket을 통한 상태 업데이트
- **상태 추적**: 주문 생성 → 처리중 → 완료까지 단계별 추적
- **자동 연동**: 약국, 검사실, 영상의학과와 자동 연동

**주문 타입별 전달 내용:**

1. **처방전 주문 (Prescription Order)**
   - 환자 정보, 의사 정보
   - 약물 목록 (약물명, 용량, 용법, 기간)
   - 주문 상태 (대기중 → 조제중 → 조제완료)

2. **검사 주문 (Lab Test Order)**
   - 환자 정보, 의사 정보
   - 검사 항목 목록 (검사명, 우선순위, 특이사항)
   - 주문 상태 (대기중 → 검사중 → 검사완료)
   - 검사 결과 자동 전달

3. **영상 촬영 의뢰 (Imaging Order)**
   - 환자 정보, 의사 정보
   - 촬영 정보 (촬영 타입, 부위, 조영제 사용 여부)
   - 주문 상태 (대기중 → 촬영중 → 촬영완료 → 판독완료)
   - 판독서 자동 전달

---

## 14. 주요 고려사항

### 14.1 보안
- 개인정보 보호 (의료법 준수)
- 데이터 암호화
- 접근 로그 관리
- 정기적인 보안 점검

### 14.2 성능
- 이미지 로딩 최적화
- AI 분석 비동기 처리
- 데이터베이스 쿼리 최적화
- 캐싱 전략

### 14.3 확장성
- 마이크로서비스 독립 배포
- 수평 확장 가능한 구조
- 모니터링 및 로깅 시스템

### 14.4 사용성
- 직관적인 UI/UX
- 역할별 맞춤형 인터페이스
- 모바일 반응형 디자인
- 접근성 고려

---

## 15. 향후 확장 계획

### 15.1 추가 기능
- 고급 챗봇 기능 (LLM 기반)
- 원격 진료 기능
- 모바일 앱 개발
- 웨어러블 기기 연동
- 음성 채팅 지원
- 화상 상담 기능

### 15.2 AI 기능 확장
- 추가 딥러닝 모델 통합
- 예측 분석 기능
- 자동 리포트 생성
- 진단 제안 시스템 고도화

### 15.3 시스템 통합
- EMR/EHR 시스템 연동
- Orthanc DICOM 서버 연동
- LIS 시스템 연동
- 외부 의료기관 연동

---

## 16. 참고 자료

### 16.1 기존 시스템 활용
- React 홈페이지 디자인 재사용
- 기존 모델 구조 참고
- 기존 API 패턴 활용

### 16.2 새로운 개발 항목
- 역할별 독립 홈페이지
- 마이크로서비스 구조
- 새로운 데이터베이스 테이블
- 역할별 권한 관리 시스템

---

## 17. 프로젝트 일정 (예상)

| 단계 | 기간 | 주요 작업 |
|------|------|----------|
| 기획 및 설계 | 1주 | 요구사항 분석, DB 설계, API 설계 |
| 기반 구축 | 2주 | 인증 시스템, 기본 API, DB 구축 |
| 역할별 홈페이지 | 3주 | 4가지 역할별 프론트엔드 개발 |
| 핵심 기능 | 3주 | 이미지 관리, AI 연동, 진료 기록 |
| 고급 기능 | 2주 | 3D 시각화, 알림, 리포트 |
| 테스트 및 배포 | 1주 | 통합 테스트, 최적화, 배포 |

**총 예상 기간: 12주 (약 3개월)**

---

## 18. 팀 구성 및 역할 분담 (예시)

### 18.1 백엔드 팀
- API 서버 개발
- 데이터베이스 설계 및 구현
- 마이크로서비스 구조 설계

### 18.2 프론트엔드 팀
- 역할별 홈페이지 개발
- UI/UX 디자인
- 사용자 인터페이스 구현

### 18.3 AI/ML 팀
- 딥러닝 모델 통합
- AI 서비스 개발
- 모델 성능 최적화

### 18.4 인프라 팀
- 서버 구축 및 관리
- 배포 자동화
- 모니터링 시스템 구축

---

이 문서는 프로젝트 기획 단계의 가이드라인입니다. 실제 개발 시 세부 사항은 추가로 보완하겠습니다.

