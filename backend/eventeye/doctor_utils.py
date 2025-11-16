from typing import Optional

from django.db import connection, IntegrityError, transaction
from django.utils import timezone

DEPARTMENT_ADMIN = "원무과"
DEPARTMENT_RESPIRATORY = "호흡기내과"
DEPARTMENT_SURGERY = "외과"
ALLOWED_DEPARTMENTS = {
  DEPARTMENT_ADMIN,
  DEPARTMENT_RESPIRATORY,
  DEPARTMENT_SURGERY,
}

# 하위 호환성을 위한 영어-한글 매핑
DEPARTMENT_MAPPING = {
    "admin": DEPARTMENT_ADMIN,
    "respiratory": DEPARTMENT_RESPIRATORY,
    "surgery": DEPARTMENT_SURGERY,
}

def normalize_department(dept: str) -> str:
    """영어 진료과 코드를 한글로 변환 (하위 호환성)"""
    return DEPARTMENT_MAPPING.get(dept, dept)


def get_doctor_id(user_id: int) -> Optional[str]:
    """Fetch doctor_id for a given auth_user primary key."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT doctor_id FROM auth_user WHERE id = %s", [user_id])
        row = cursor.fetchone()
        return row[0] if row else None


def get_department(user_id: int) -> Optional[str]:
    with connection.cursor() as cursor:
        cursor.execute("SELECT department FROM auth_user WHERE id = %s", [user_id])
        row = cursor.fetchone()
        return row[0] if row else None


def set_department(user_id: int, department: str) -> None:
    # 영어 코드가 들어오면 한글로 변환 (하위 호환성)
    normalized_dept = normalize_department(department)
    if normalized_dept not in ALLOWED_DEPARTMENTS:
        raise ValueError(f"Invalid department: {department}")
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE auth_user SET department = %s WHERE id = %s",
            [normalized_dept, user_id],
        )


def _next_doctor_sequence(prefix: str) -> int:
    """Determine the next numeric sequence for the given prefix (year)."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT doctor_id FROM auth_user WHERE doctor_id LIKE %s ORDER BY doctor_id DESC LIMIT 1",
            [f"{prefix}%"],
        )
        row = cursor.fetchone()
    if not row or not row[0]:
        return 1
    suffix = row[0][len(prefix):]
    if suffix.isdigit():
        return int(suffix) + 1
    return 1


def ensure_doctor_id(user_id: int, force: bool = False) -> Optional[str]:
    """Ensure the specified user has a doctor_id assigned (medical staff only)."""
    department = get_department(user_id)
    if department in (None, DEPARTMENT_ADMIN):
        return None

    current = get_doctor_id(user_id)
    if current and not force:
        return current

    prefix = f"D{timezone.now().year}"
    attempt = 0

    while True:
        seq = _next_doctor_sequence(prefix) + attempt
        candidate = f"{prefix}{seq:03d}"
        try:
            with transaction.atomic():
                with connection.cursor() as cursor:
                    cursor.execute(
                        "UPDATE auth_user SET doctor_id = %s WHERE id = %s",
                        [candidate, user_id],
                    )
            return candidate
        except IntegrityError:
            attempt += 1
            continue
