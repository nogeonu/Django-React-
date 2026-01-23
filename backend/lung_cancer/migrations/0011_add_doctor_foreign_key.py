from django.db import migrations

# 1단계: doctor_fk_id 컬럼 추가 (NULL 허용)
ADD_DOCTOR_COLUMN = """
ALTER TABLE medical_record
    ADD COLUMN doctor_fk_id INT NULL;
"""

# 2단계: 기존 데이터 백필 (department 기준으로 임시 매핑)
# 주의: 이 예시에서는 기존 데이터에 NULL을 유지합니다.
# 실제 운영환경에서는 department에 맞는 의사를 매핑하는 로직을 추가해야 합니다.
POPULATE_DOCTOR_COLUMN = """
-- 기존 데이터는 doctor_fk_id를 NULL로 유지
-- 실제 운영시에는 부서별로 기본 의사를 지정하거나
-- 데이터 마이그레이션 스크립트를 별도로 작성해야 합니다.
"""

# 3단계: FK 제약조건 추가 (NOT NULL로 변경하지 않음, 이전 데이터 호환성 유지)
FINALIZE_DOCTOR_COLUMN = """
ALTER TABLE medical_record
    ADD CONSTRAINT medical_record_doctor_fk
        FOREIGN KEY (doctor_fk_id) REFERENCES auth_user(id)
        ON DELETE RESTRICT ON UPDATE CASCADE;
"""

# Rollback SQL
DROP_DOCTOR_COLUMN = """
ALTER TABLE medical_record
    DROP FOREIGN KEY medical_record_doctor_fk;
ALTER TABLE medical_record
    DROP COLUMN doctor_fk_id;
"""

REVERT_DOCTOR_FINALIZE = """
ALTER TABLE medical_record
    DROP FOREIGN KEY medical_record_doctor_fk;
"""


class Migration(migrations.Migration):
    dependencies = [
        ("lung_cancer", "0010_add_patient_foreign_keys"),
        ("auth", "__latest__"),  # auth_user 테이블 의존성
    ]

    operations = [
        migrations.RunSQL(
            sql=ADD_DOCTOR_COLUMN,
            reverse_sql=DROP_DOCTOR_COLUMN,
        ),
        migrations.RunSQL(
            sql=POPULATE_DOCTOR_COLUMN,
            reverse_sql=migrations.RunSQL.noop,
        ),
        migrations.RunSQL(
            sql=FINALIZE_DOCTOR_COLUMN,
            reverse_sql=REVERT_DOCTOR_FINALIZE,
        ),
    ]

