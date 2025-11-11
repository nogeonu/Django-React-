from django.db import connection, migrations, IntegrityError
from django.utils import timezone


def add_doctor_id_column(apps, schema_editor):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'auth_user'
              AND COLUMN_NAME = 'doctor_id'
            """
        )
        exists = cursor.fetchone()[0]
        if not exists:
            cursor.execute(
                "ALTER TABLE auth_user ADD COLUMN doctor_id VARCHAR(20) UNIQUE AFTER email"
            )

    populate_missing_doctor_ids()


def populate_missing_doctor_ids():
    current_year = timezone.now().year
    prefix = f"D{current_year}"

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT doctor_id FROM auth_user WHERE doctor_id IS NOT NULL AND doctor_id != ''"
        )
        existing = {row[0] for row in cursor.fetchall()}

        cursor.execute(
            "SELECT id FROM auth_user WHERE doctor_id IS NULL OR doctor_id = '' ORDER BY id"
        )
        pending_ids = [row[0] for row in cursor.fetchall()]

    seq_start = 1
    prefix_len = len(prefix)
    for doctor_id in existing:
        if doctor_id and doctor_id.startswith(prefix):
            suffix = doctor_id[prefix_len:]
            if suffix.isdigit():
                seq_start = max(seq_start, int(suffix) + 1)

    seq = seq_start
    for user_id in pending_ids:
        assigned = False
        while not assigned:
            candidate = f"{prefix}{seq:03d}"
            seq += 1
            if candidate in existing:
                continue
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "UPDATE auth_user SET doctor_id=%s WHERE id=%s",
                        [candidate, user_id],
                    )
                existing.add(candidate)
                assigned = True
            except IntegrityError:
                continue


def drop_doctor_id_column(apps, schema_editor):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'auth_user'
              AND COLUMN_NAME = 'doctor_id'
            """
        )
        exists = cursor.fetchone()[0]
        if exists:
            cursor.execute("ALTER TABLE auth_user DROP COLUMN doctor_id")


class Migration(migrations.Migration):

    dependencies = [
        ("patients", "0008_patient_age"),
    ]

    operations = [
        migrations.RunPython(
            add_doctor_id_column,
            reverse_code=drop_doctor_id_column,
        ),
    ]
