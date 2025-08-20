# 병원 환자관리 시스템

## 개요

병원 환자관리 시스템은 의료진이 환자 정보를 효율적으로 관리하고 MRI, CT 등의 의료 이미지를 AI로 분석할 수 있는 웹 애플리케이션입니다. 환자 등록, 의료 이미지 업로드, AI 분석 결과 관리 등의 기능을 제공합니다.

## 사용자 선호도

선호하는 의사소통 스타일: 간단하고 일상적인 언어를 사용합니다.

## 시스템 아키텍처

### 프론트엔드 아키텍처
- **프레임워크**: React with TypeScript, Vite 사용
- **UI 라이브러리**: Radix UI + shadcn/ui 디자인 시스템
- **스타일링**: Tailwind CSS with CSS 변수를 사용한 테마 및 반응형 디자인
- **상태 관리**: TanStack Query (React Query)를 사용한 서버 상태 관리 및 캐싱
- **라우팅**: Wouter 경량 클라이언트 사이드 라우팅
- **폼 처리**: React Hook Form + Zod 검증을 사용한 타입 안전 폼 관리

### 백엔드 아키텍처
- **런타임**: Node.js + Express.js 서버 프레임워크
- **언어**: TypeScript with ES modules
- **API 설계**: RESTful 엔드포인트 with JSON 응답
- **오류 처리**: 중앙집중식 오류 미들웨어
- **요청 로깅**: API 요청 추적 및 성능 모니터링을 위한 커스텀 미들웨어

### 데이터 저장소
- **데이터베이스**: PostgreSQL with Drizzle ORM
- **연결**: Neon Database 서버리스 PostgreSQL
- **스키마 관리**: Drizzle Kit을 사용한 데이터베이스 마이그레이션
- **개발 저장소**: 개발 및 테스트를 위한 인메모리 저장소 구현

### 주요 기능 구현
- **환자 관리**: 환자 등록, 조회, 수정 기능
- **의료 이미지 업로드**: MRI, CT 등 의료 이미지 업로드 및 저장
- **AI 분석**: 의료 이미지에 대한 AI 분석 (YOLO 모델 준비)
- **검사 관리**: 검사 등록 및 결과 관리
- **대시보드**: 환자 현황 및 시스템 통계 표시

### 성능 최적화
- **이미지 처리**: 클라이언트 사이드 이미지 압축 및 base64 인코딩
- **캐싱**: React Query를 사용한 지능형 데이터 캐싱
- **번들 최적화**: Vite를 사용한 빠른 개발 및 최적화된 프로덕션 빌드

## 외부 의존성

### 핵심 프레임워크 의존성
- **@tanstack/react-query**: 서버 상태 관리 및 데이터 페칭
- **wouter**: 단일 페이지 애플리케이션을 위한 경량 라우팅 라이브러리
- **react-hook-form** + **@hookform/resolvers**: 검증이 포함된 폼 처리
- **zod**: 런타임 타입 검증 및 스키마 정의

### UI 및 스타일링 의존성
- **@radix-ui/***: 접근 가능한 UI 프리미티브 세트
- **tailwindcss**: 빠른 스타일링을 위한 유틸리티 우선 CSS 프레임워크
- **class-variance-authority**: 컴포넌트의 타입 안전 변형 스타일링
- **lucide-react**: React 컴포넌트가 포함된 현대적인 아이콘 라이브러리

### 데이터베이스 및 백엔드 의존성
- **drizzle-orm** + **drizzle-kit**: 자동 마이그레이션이 포함된 타입 안전 ORM
- **@neondatabase/serverless**: Neon Database용 서버리스 PostgreSQL 클라이언트
- **express**: Node.js용 웹 애플리케이션 프레임워크

### 개발 및 빌드 도구
- **vite**: 빠른 빌드 도구 및 개발 서버
- **typescript**: JavaScript를 위한 정적 타입 검사
- **esbuild**: 프로덕션 빌드를 위한 빠른 JavaScript 번들러

### 이미지 처리 및 유틸리티
- **date-fns**: JavaScript용 현대적인 날짜 유틸리티 라이브러리

## 데이터베이스 스키마

### 주요 테이블
1. **patients**: 환자 기본 정보 (이름, 연락처, 병력 등)
2. **medical_images**: 의료 이미지 파일 정보 (MRI, CT 등)
3. **examinations**: 검사 정보 및 상태
4. **ai_analysis_results**: AI 분석 결과 저장

### 향후 AI 모델 통합
- YOLO 모델을 사용한 의료 이미지 분석 준비
- 분석 결과의 신뢰도 및 권장사항 표시
- 모델 버전 관리 및 결과 추적