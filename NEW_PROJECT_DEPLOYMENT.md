# 🚀 새 프로젝트 배포하기 (빠른 가이드)

새 프로젝트를 GCP VM에 자동 배포하는 방법입니다.

## 📌 핵심 요약

**같은 VM, 같은 SSH 키 사용** - 프로젝트만 다름!

## ⚡ 빠른 설정 (5분 안에)

### 1. GitHub 저장소 생성
```bash
# 새 저장소 생성 후 연결
cd /path/to/new-project
git init
git remote add origin https://github.com/YOUR_USERNAME/NEW_REPO.git
git branch -M main
```

### 2. 워크플로 파일 추가
프로젝트 루트에 `.github/workflows/deploy.yml` 파일 생성 (이 프로젝트의 파일 복사해서 사용)

### 3. GitHub Secrets 등록
새 저장소 → Settings → Secrets에서:
- `SSH_HOST`: `34.42.223.43` (동일)
- `SSH_USER`: `shrjsdn908` (동일)
- `SSH_PORT`: `22` (동일)
- `SSH_KEY`: **기존 키 그대로** (동일)
- `APP_DIR`: `/srv/new-project/app` ⚠️ **프로젝트마다 다름!**

### 4. VM 설정 (간단)
```bash
# VM 접속
ssh shrjsdn908@34.42.223.43

# 새 디렉토리만 생성
sudo mkdir -p /srv/new-project/app
sudo chown -R shrjsdn908:shrjsdn908 /srv/new-project/app
```

### 5. 푸시하면 완료!
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

---

## 🔑 핵심 포인트

| 항목 | 이전 프로젝트 | 새 프로젝트 | 변경 여부 |
|------|--------------|------------|----------|
| SSH 키 | ~/.ssh/gcp_key | ~/.ssh/gcp_key | ❌ 안바뀜 |
| GitHub Secrets | 등록됨 | **새로 등록** | ⚠️ 새 저장소마다 |
| VM IP | 34.42.223.43 | 34.42.223.43 | ❌ 안바뀜 |
| APP_DIR | /srv/django-react/app | /srv/new-project/app | ✅ 바뀜 |
| 포트 | 80 | 81 (또는 서브도메인) | ✅ 다르게 |

---

## 🎯 두 가지 방식

### 방식 A: 같은 VM, 다른 포트
- 기존: http://34.42.223.43 (포트 80)
- 새: http://34.42.223.43:81 (포트 81)

### 방식 B: 새로운 VM 생성
- 완전히 독립적인 프로젝트
- SSH 키 새로 만들어야 함
- VM IP 다름

---

## 📝 체크리스트

새 프로젝트 배포 전 확인:

- [ ] GitHub 저장소 생성 완료
- [ ] `.github/workflows/deploy.yml` 파일 추가
- [ ] GitHub Secrets 5개 등록
- [ ] VM 디렉토리 생성 (`mkdir -p /srv/new-project/app`)
- [ ] Nginx 설정 (필요시)
- [ ] Gunicorn 서비스 (필요시)

---

## 💡 팁

1. **SSH 키 재사용**: 같은 사용자, 같은 VM이면 SSH 키는 계속 사용 가능
2. **워크플로 파일**: 이 프로젝트 파일 복사 후 경로만 수정
3. **디렉토리 구조**: `/srv/{project-name}/app` 형식 추천
4. **포트 관리**: 프로젝트가 많아지면 포트 매핑 관리

---

**핵심**: SSH 키와 VM은 재사용, GitHub Secrets와 APP_DIR만 새로 설정! 🚀

