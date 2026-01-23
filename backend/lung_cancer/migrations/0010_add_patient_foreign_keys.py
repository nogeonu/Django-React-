from django.db import migrations

ADD_MEDICAL_COLUMN = """
ALTER TABLE medical_record
    ADD COLUMN patient_fk_id BIGINT NULL;
"""

POPULATE_MEDICAL_COLUMN = """
UPDATE medical_record m
INNER JOIN patients_patient p
    ON m.patient_id COLLATE utf8mb4_unicode_ci = p.patient_id COLLATE utf8mb4_unicode_ci
SET m.patient_fk_id = p.id;
"""

FINALIZE_MEDICAL_COLUMN = """
ALTER TABLE medical_record
    MODIFY COLUMN patient_fk_id BIGINT NOT NULL,
    ADD CONSTRAINT medical_record_patient_fk
        FOREIGN KEY (patient_fk_id) REFERENCES patients_patient(id)
        ON DELETE RESTRICT ON UPDATE CASCADE;
"""

REVERT_MEDICAL_FINALIZE = """
ALTER TABLE medical_record
    DROP FOREIGN KEY medical_record_patient_fk,
    MODIFY COLUMN patient_fk_id BIGINT NULL;
"""

DROP_MEDICAL_COLUMN = """
ALTER TABLE medical_record
    DROP COLUMN patient_fk_id;
"""

ADD_LUNG_COLUMN = """
ALTER TABLE lung_record
    ADD COLUMN patient_fk_id BIGINT NULL;
"""

POPULATE_LUNG_COLUMN = """
UPDATE lung_record lr
INNER JOIN patients_patient p
    ON lr.patient_id COLLATE utf8mb4_unicode_ci = p.patient_id COLLATE utf8mb4_unicode_ci
SET lr.patient_fk_id = p.id;
"""

FINALIZE_LUNG_COLUMN = """
ALTER TABLE lung_record
    MODIFY COLUMN patient_fk_id BIGINT NOT NULL,
    ADD CONSTRAINT lung_record_patient_fk
        FOREIGN KEY (patient_fk_id) REFERENCES patients_patient(id)
        ON DELETE RESTRICT ON UPDATE CASCADE;
"""

REVERT_LUNG_FINALIZE = """
ALTER TABLE lung_record
    DROP FOREIGN KEY lung_record_patient_fk,
    MODIFY COLUMN patient_fk_id BIGINT NULL;
"""

DROP_LUNG_COLUMN = """
ALTER TABLE lung_record
    DROP COLUMN patient_fk_id;
"""


class Migration(migrations.Migration):

    dependencies = [
        ("lung_cancer", "0009_delete_medicalrecordnew"),
        ("patients", "0011_appointment"),
    ]

    operations = [
        migrations.RunSQL(sql=ADD_MEDICAL_COLUMN, reverse_sql=DROP_MEDICAL_COLUMN),
        migrations.RunSQL(sql=POPULATE_MEDICAL_COLUMN, reverse_sql=migrations.RunSQL.noop),
        migrations.RunSQL(sql=FINALIZE_MEDICAL_COLUMN, reverse_sql=REVERT_MEDICAL_FINALIZE),
        migrations.RunSQL(sql=ADD_LUNG_COLUMN, reverse_sql=DROP_LUNG_COLUMN),
        migrations.RunSQL(sql=POPULATE_LUNG_COLUMN, reverse_sql=migrations.RunSQL.noop),
        migrations.RunSQL(sql=FINALIZE_LUNG_COLUMN, reverse_sql=REVERT_LUNG_FINALIZE),
    ]
