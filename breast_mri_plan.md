# 유방 MRI 종양 분류 플랫폼 (계획 발표)

---

## 1. 표지
- **프로젝트명**: 유방 MRI 종양 분류 플랫폼
- **팀**: EventEye AI 팀
- **발표 목적**: 딥러닝 기반 의료 이미지 분석 서비스 계획

---

## 2. 배경과 필요성
- 유방암 조기 발견의 중요성 → 신속한 판독 보조 도구 필요
- MRI는 민감도가 높지만 판독 시간과 피로도가 큼
- 의료진이 웹에서 즉시 결과를 확인할 수 있는 **AI 보조 플랫폼** 제공

---

## 3. 목표 및 범위
1. **딥러닝 모델**로 양성/악성 종양 분류
2. **Django + React** 통합 플랫폼
3. 분석 이력 저장, 재확인 기능 제공
4. 계획 발표이므로 모델 성능/파라미터는 향후 보고 예정

---

## 4. 데이터셋 계획
- **TCIA Breast Diagnostic Dataset**
- 클래스 균형: 양성 12,012 / 악성 12,012
- 분할: Train 70% / Val 15% / Test 15%
- 양/악성 균형으로 학습 안정성 확보

---

## 5. 데이터 처리 파이프라인
1. DICOM → PNG 변환
2. Resize (224×224) & ImageNet 정규화
3. Train 세트만 증강 (Flip/Rotation/Affine/ColorJitter)
4. Val/Test는 전처리만 적용

---

## 6. 모델 설계
- 백본: **ResNet50 (pretrained)**
- 출력층: 2-class Linear (Benign/Malignant)
- 손실함수: Cross Entropy
- 옵티마이저: AdamW + Cosine Warmup
- 전략: EarlyStopping, GradClip, Mixed Precision

---

## 7. 학습 파이프라인
1. 데이터로더 구성 (증강/전처리 포함)
2. 최대 50 Epoch 학습
3. Validation F1-Score 모니터링
4. 베스트 모델 체크포인트(`best_breast_mri_model.pth`) 저장

---

## 8. 서비스 아키텍처 개요
```
[React UI] → [Django API] → [mosec (breast_ai_service)] → [PyTorch 모델]
    │                                      │
    └─ 이미지 저장 (/media) ◄──────────────┘
```
- Django: 이미지 업로드/분석 API, DB 저장
- mosec: 병렬 추론, base64/URL 입력 처리

---

## 9. 백엔드 상세
- 이미지 업로드: `/api/medical-images/` (FormData)
- 파일 저장 경로: `/srv/django-react/app/backend/media/medical_images/YYYY/MM/DD/파일.jpg`
- 분석 요청: `/api/medical-images/{id}/analyze/`
- 분석 결과 저장: `AIAnalysisResult` (신뢰도, 발견, 권장 등)

---

## 10. 프론트엔드 화면 구상
- 의료진 로그인 → 환자 선택 → 이미지 카드
- “분석” 버튼 → 진행 상태/결과 표시
- 카드에 신뢰도, 예방 조치, 재분석 버튼 표시
- 분석 이력 재확인 가능

---

## 11. 딥러닝 서빙(mosec)
- 폴더: `backend/breast_ai_service`
- PyTorch 모델 로드 + 전처리 + 추론
- Worker 2개로 병렬 처리 (`server.append_worker(..., num=2)`)
- REST 엔드포인트: `POST /inference`

---

## 12. 배포 전략 (GCP)
- GitHub Actions → GCP VM 동기화
- Django: gunicorn + nginx + PostgreSQL(또는 MySQL)
- mosec: systemd 서비스 (`breast-ai-service`)
- 이미지/모델 파일: VM 디스크 내 `backend/media/`, `backend/breast_ai_service/ml_model/`

---

## 13. 일정 계획 (예시)
| 단계 | 기간 | 주요 작업 |
|---|---|---|
| Week 1 | 데이터 준비 | DICOM 변환/정리 |
| Week 2 | 모델 학습 | ResNet50 훈련 및 튜닝 |
| Week 3 | Django API | 이미지 저장, 분석 API 완성 |
| Week 4 | 프론트/UI | 의료진 화면 구성, QA |
| Week 5 | 배포/발표 | GCP 배포, 발표자료 완성 |

---

## 14. 기대 효과
- 의료진이 AI 결과를 즉시 참고 → 판독 시간 단축
- 분석 이력 저장으로 추적 가능
- 기존 ML(Flask) 대비 **딥러닝 서빙 구조**로 고성능 확보

---

## 15. 차별점 & 확장성
1. 균형 데이터셋 기반 높은 신뢰도
2. mosec으로 병렬 추론/확장 용이
3. 향후 Grad-CAM, BI-RADS 다중 클래스, PACS 연동 가능

---

## 16. 향후 과제
- Explainability(Grad-CAM) 도입
- BI-RADS 등급/다중 병변 분류
- PACS 연동, 자동 업로드 파이프라인
- GPU 및 모델 최적화(ONNX/TensorRT)

---

## 17. 요약 & Q&A
- 딥러닝 모델 + Django/React + mosec 통합 구조 제안
- 이미지 업로드 → 분석 → 결과 저장 흐름 확립
- 질의응답 및 피드백 환영

---

