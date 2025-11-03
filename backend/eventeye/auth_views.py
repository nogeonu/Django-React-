from django.contrib.auth import authenticate, login as django_login, logout as django_logout, get_user_model
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt


def get_role_from_user(user):
    if user.is_superuser:
        return "superuser"
    if user.is_staff:
        return "medical_staff"  # 의료진 (is_staff=1)
    return "admin_staff"  # 원무과 (is_staff=0)


@require_http_methods(["GET"])
def me(request):
    if not request.user.is_authenticated:
        return JsonResponse({"detail": "Unauthorized"}, status=401)
    user = request.user
    return JsonResponse({
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "first_name": getattr(user, 'first_name', ''),
        "last_name": getattr(user, 'last_name', ''),
        "role": get_role_from_user(user),
    })


@csrf_exempt
@require_http_methods(["POST"])  # CSRF는 개발 편의를 위해 면제. 운영 시 CSRF 처리 필요
def login(request):
    import json
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        payload = {}
    username = payload.get("username")
    password = payload.get("password")

    if not username or not password:
        return JsonResponse({"detail": "username and password are required"}, status=400)

    user = authenticate(request, username=username, password=password)
    if user is None:
        return JsonResponse({"detail": "Invalid credentials"}, status=401)

    actual_role = get_role_from_user(user)

    django_login(request, user)
    return JsonResponse({
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "first_name": getattr(user, 'first_name', ''),
        "last_name": getattr(user, 'last_name', ''),
        "role": actual_role,
    })


@csrf_exempt
@require_http_methods(["POST"])  # Session 기반 로그아웃
def logout(request):
    if not request.user.is_authenticated:
        return JsonResponse({"detail": "Already logged out"})
    django_logout(request)
    return JsonResponse({"detail": "Logged out"})


@csrf_exempt
@require_http_methods(["POST"])  # 간단 회원가입: 의료진 또는 원무과 선택 허용
def register(request):
    import json
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except Exception:
        payload = {}

    username = payload.get("username")
    password = payload.get("password")
    email = payload.get("email")
    first_name = payload.get("first_name")
    last_name = payload.get("last_name")
    role = payload.get("role", "medical_staff")  # 'medical_staff' | 'admin_staff'
    
    print(f"[회원가입] 받은 데이터: username={username}, role={role}, email={email}")

    if not username or not password:
        return JsonResponse({"detail": "username and password are required"}, status=400)

    if role not in ("medical_staff", "admin_staff"):
        print(f"[회원가입] 잘못된 role: {role}")
        return JsonResponse({"detail": "Only medical_staff or admin_staff can self-register"}, status=403)

    User = get_user_model()
    # 중복 확인
    try:
        if User.objects.filter(username=username).exists():
            print(f"[회원가입] 중복된 사용자명: {username}")
            return JsonResponse({"detail": "Username already exists"}, status=400)
    except Exception as e:
        print(f"[회원가입] 중복 확인 실패: {e}")

    # 사용자 생성
    print(f"[회원가입] 사용자 생성 시도: {username}")
    try:
        user = User.objects.create_user(
            username=username,
            password=password,
            email=email,
        )
        if first_name:
            user.first_name = first_name
        if last_name:
            user.last_name = last_name
        if role == "medical_staff":
            user.is_staff = True
        user.save()
        print(f"[회원가입] 사용자 생성 성공: ID={user.id}")
    except Exception as e:
        print(f"[회원가입] 사용자 생성 실패: {e}")
        return JsonResponse({"detail": f"Failed to save user: {str(e)}"}, status=500)

    return JsonResponse({
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": get_role_from_user(user),
    }, status=201)
