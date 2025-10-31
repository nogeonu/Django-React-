from django.contrib.auth import authenticate, login as django_login, logout as django_logout, get_user_model
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt


def get_role_from_user(user):
    if user.is_superuser:
        return "superuser"
    if user.is_staff:
        return "admin_staff"  # 원무과
    return "medical_staff"  # 의료진


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

    if not username or not password:
        return JsonResponse({"detail": "username and password are required"}, status=400)

    if role not in ("medical_staff", "admin_staff"):
        return JsonResponse({"detail": "Only medical_staff or admin_staff can self-register"}, status=403)

    User = get_user_model()
    # 두 DB 모두 중복 확인
    try:
        exists_default = User.objects.filter(username=username).exists()
    except Exception:
        exists_default = False
    try:
        exists_hosp = User.objects.db_manager('hospital_db').filter(username=username).exists()
    except Exception:
        exists_hosp = False
    if exists_default or exists_hosp:
        return JsonResponse({"detail": "Username already exists"}, status=400)

    # 1) 기본 DB 생성
    user = User.objects.create_user(username=username, password=password, email=email)
    if first_name:
        user.first_name = first_name
    if last_name:
        user.last_name = last_name
    if role == "admin_staff":
        user.is_staff = True
    user.save()

    # 2) hospital_db 생성(실패 시 롤백)
    try:
        user_h = User.objects.db_manager('hospital_db').create_user(
            username=username,
            password=password,
            email=email,
        )
        if first_name:
            user_h.first_name = first_name
        if last_name:
            user_h.last_name = last_name
        if role == "admin_staff":
            user_h.is_staff = True
        user_h.save(using='hospital_db')
    except Exception as e:
        try:
            user.delete()
        except Exception:
            pass
        return JsonResponse({"detail": f"Failed to save user in hospital_db: {str(e)}"}, status=500)

    return JsonResponse({
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": get_role_from_user(user),
    }, status=201)
