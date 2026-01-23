from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("patients", "0007_patient_blood_type"),
    ]

    operations = [
        migrations.AddField(
            model_name="patient",
            name="age",
            field=models.PositiveIntegerField(blank=True, null=True, verbose_name="나이"),
        ),
    ]

