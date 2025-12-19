# EventEye - 병원 관리 시스템

Django + React를 사용한 병원 환자 관리 및 의료 이미지 분석 시스템입니다.

## 프로젝트 구조

```
EventEye/
├── backend/                 # Django 백엔드
│   ├── eventeye_backend/   # Django 프로젝트 설정
│   ├── hospital/           # 병원 관리 앱
│   ├── requirements.txt    # Python 의존성
│   └── manage.py          # Django 관리 스크립트
├── frontend/              # React 프론트엔드
│   ├── src/
│   │   ├── components/    # UI 컴포넌트
│   │   ├── pages/        # 페이지 컴포넌트
│   │   ├── lib/          # 유틸리티
│   │   └── hooks/        # 커스텀 훅
│   ├── package.json      # Node.js 의존성
│   └── vite.config.ts    # Vite 설정
└── README.md
```

## 주요 기능

- **환자 관리**: 환자 등록, 조회, 수정, 삭제
- **검사 관리**: 의료 검사 등록 및 관리
- **의료 이미지**: MRI, CT, X-RAY 등 이미지 업로드 및 관리
- **MRI 뷰어**: 3D 의료 영상 시각화 및 세그멘테이션 오버레이 (NEW! 🎉)
  - NIfTI 파일(.nii.gz) 지원
  - 마우스 휠로 슬라이스 탐색
  - 3가지 단면 방향 (Axial, Sagittal, Coronal)
  - 실시간 세그멘테이션 오버레이
- **AI 분석**: 의료 이미지 AI 분석 및 결과 제공
- **대시보드**: 통계 및 현황 모니터링

## 기술 스택

### 백엔드
- Django 4.2.7
- Django REST Framework
- PostgreSQL
- Python 3.8+

### 프론트엔드
- React 18
- TypeScript
- Vite
- Tailwind CSS
- Radix UI
- React Query

## 설치 및 실행

### 백엔드 설정

1. Python 가상환경 생성 및 활성화
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. 의존성 설치
```bash
pip install -r requirements.txt
```

3. 환경변수 설정
```bash
cp env.example .env
# .env 파일을 편집하여 데이터베이스 설정
```

4. 데이터베이스 마이그레이션
```bash
# 마이그레이션 파일 생성
python manage.py makemigrations

# 로컬 sqlite3로 마이그레이션
python manage.py migrate

# MySQL(hospital_db)로 마이그레이션
# settings.py의 DATABASES.hospital_db 설정을 사용합니다
python manage.py migrate --database=hospital_db
```

5. 슈퍼유저 생성
```bash
python manage.py createsuperuser
```

6. 서버 실행
```bash
python manage.py runserver
```

### 프론트엔드 설정

1. 의존성 설치
```bash
cd frontend
npm install
```

2. 개발 서버 실행
```bash
npm run dev
```

## API 엔드포인트

### 환자 관리
- `GET /api/patients/` - 환자 목록 조회
- `POST /api/patients/` - 환자 등록
- `GET /api/patients/{id}/` - 환자 상세 조회
- `PUT /api/patients/{id}/` - 환자 정보 수정
- `DELETE /api/patients/{id}/` - 환자 삭제

### 검사 관리
- `GET /api/examinations/` - 검사 목록 조회
- `POST /api/examinations/` - 검사 등록
- `GET /api/examinations/{id}/` - 검사 상세 조회

### 의료 이미지
- `GET /api/medical-images/` - 이미지 목록 조회
- `POST /api/medical-images/` - 이미지 업로드
- `POST /api/medical-images/{id}/analyze/` - AI 분석 실행

### AI 분석 결과
- `GET /api/ai-analysis/` - 분석 결과 조회

## 개발 가이드

### 데이터베이스 모델

- **Patient**: 환자 정보
- **Examination**: 검사 정보
- **MedicalImage**: 의료 이미지
- **AIAnalysisResult**: AI 분석 결과

### 프론트엔드 컴포넌트

- **Dashboard**: 대시보드 페이지
- **Patients**: 환자 관리 페이지
- **MedicalImages**: 의료 이미지 관리 페이지
- **Sidebar**: 네비게이션 사이드바

## 라이선스

MIT License

---

## 데이터베이스 설정 가이드 (MySQL)

백엔드 `backend/eventeye/settings.py`에 MySQL 연결이 구성되어 있습니다.

예시 설정:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    },
    'hospital_db': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'hospital_db',
        'USER': 'acorn',
        'PASSWORD': 'acorn1234',
        'HOST': '34.42.223.43',
        'PORT': '3306',
        'OPTIONS': {'charset': 'utf8mb4'},
    },
}
```

연결 테스트(로컬 또는 VM):

```bash
mysql -h 34.42.223.43 -u acorn -pacorn1234 -D hospital_db -e "SHOW TABLES;"
```

마이그레이션 실행(테이블 생성):

```bash
cd backend
python manage.py makemigrations
python manage.py migrate --database=hospital_db
```

