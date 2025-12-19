# GCP 인스턴스 상태 확인 방법

## 🔍 확인해야 할 사항

### 1. GCP 콘솔에서 VM 상태 확인

**GCP 콘솔 접속:**
```
https://console.cloud.google.com/compute/instances
```

**확인 사항:**
- ✅ VM 인스턴스 이름: `koyang-2510`
- ✅ 상태: **실행 중** (녹색)
- ✅ 외부 IP: `34.42.223.43`
- ✅ 영역: `us-central1-a`

### 2. VM이 중지되어 있는 경우

**원인:**
- 자동 종료 설정
- 비용 절감을 위한 수동 중지
- GCP 크레딧 소진

**해결:**
1. GCP 콘솔에서 VM 선택
2. **시작** 버튼 클릭
3. 2-3분 대기
4. 상태가 "실행 중"으로 변경 확인

### 3. 방화벽 규칙 확인

**GCP 콘솔:**
```
VPC 네트워크 → 방화벽 → 방화벽 규칙
```

**필요한 규칙:**
- ✅ SSH (포트 22) - 0.0.0.0/0
- ✅ HTTP (포트 80) - 0.0.0.0/0
- ✅ HTTPS (포트 443) - 0.0.0.0/0
- ✅ Django (포트 5000) - 0.0.0.0/0

### 4. SSH 키 확인

**Compute Engine → 메타데이터 → SSH 키**

GitHub Actions가 사용하는 SSH 공개키가 등록되어 있어야 합니다.

## 🐛 GitHub Actions SSH 타임아웃 원인

### 가능한 원인:

1. **VM이 중지됨** ⭐ (가장 흔함)
   - GCP 콘솔에서 VM 상태 확인
   - 중지되어 있으면 시작

2. **외부 IP가 변경됨**
   - VM 재시작 시 IP 변경 가능
   - GitHub Secrets의 `SSH_HOST` 업데이트 필요

3. **SSH 키가 제거됨**
   - GCP 메타데이터에 SSH 키 재등록 필요

4. **방화벽에서 SSH 차단**
   - 포트 22 허용 규칙 확인

5. **GitHub Actions IP 차단**
   - GCP 방화벽에서 특정 IP만 허용하도록 설정된 경우

## ✅ 해결 순서

### 1단계: VM 상태 확인
```
GCP 콘솔 → Compute Engine → VM 인스턴스
→ koyang-2510 상태 확인
```

### 2단계: VM이 중지되어 있으면
```
VM 선택 → 시작 버튼 클릭 → 2-3분 대기
```

### 3단계: 외부 IP 확인
```
실행 중인 VM의 외부 IP 확인
→ 34.42.223.43이 맞는지 확인
→ 다르면 GitHub Secrets 업데이트
```

### 4단계: 웹사이트 접속 테스트
```
http://34.42.223.43
```
접속되면 VM은 정상!

### 5단계: SSH 키 확인
```
GCP 콘솔 → Compute Engine → 메타데이터 → SSH 키
→ GitHub Actions용 SSH 공개키 등록 확인
```

## 🔧 SSH 키 재등록 방법

### 1. 로컬에서 SSH 키 생성
```bash
ssh-keygen -t rsa -b 4096 -C "github-actions" -f ~/.ssh/gcp_github_actions
```

### 2. 공개키 복사
```bash
cat ~/.ssh/gcp_github_actions.pub
```

### 3. GCP에 공개키 등록
```
GCP 콘솔 → Compute Engine → 메타데이터 → SSH 키
→ 항목 추가 → 공개키 붙여넣기
```

### 4. GitHub Secrets에 개인키 등록
```bash
cat ~/.ssh/gcp_github_actions
```

복사한 내용을:
```
GitHub → Settings → Secrets and variables → Actions
→ SSH_KEY 업데이트
```

## 📊 체크리스트

- [ ] GCP 콘솔에서 VM 상태 확인
- [ ] VM이 "실행 중" 상태인지 확인
- [ ] 외부 IP가 34.42.223.43인지 확인
- [ ] 웹사이트 http://34.42.223.43 접속 테스트
- [ ] 방화벽 규칙에서 SSH(22) 허용 확인
- [ ] GCP 메타데이터에 SSH 키 등록 확인
- [ ] GitHub Secrets의 SSH_HOST, SSH_KEY 확인

## 💡 가장 가능성 높은 원인

**VM이 중지되어 있을 가능성이 높습니다!**

GCP 콘솔에서 VM 상태를 확인하고, 중지되어 있으면 시작하세요.

