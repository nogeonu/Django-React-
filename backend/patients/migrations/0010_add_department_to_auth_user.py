from django.db import migrations, connection


DEPARTMENT_ADMIN = "admin"


def add_department_column(apps, schema_editor):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'auth_user'
              AND COLUMN_NAME = 'department'
            """
        )
        exists = cursor.fetchone()[0]
        if not exists:
            cursor.execute(
                "ALTER TABLE auth_user ADD COLUMN department VARCHAR(30) NOT NULL DEFAULT %s AFTER doctor_id",
                [DEPARTMENT_ADMIN],
            )


def drop_department_column(apps, schema_editor):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'auth_user'
              AND COLUMN_NAME = 'department'
            """
        )
        exists = cursor.fetchone()[0]
        if exists:
            cursor.execute("ALTER TABLE auth_user DROP COLUMN department")


class Migration(migrations.Migration):

    dependencies = [
        ("patients", "0009_add_doctor_id_to_auth_user"),
    ]

    operations = [
        migrations.RunPython(add_department_column, reverse_code=drop_department_column),
    ]
