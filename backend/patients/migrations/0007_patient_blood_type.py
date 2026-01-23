from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("patients", "0006_merge_20251111_0405"),
    ]

    operations = [
        migrations.AddField(
            model_name="patient",
            name="blood_type",
            field=models.CharField(blank=True, max_length=3, null=True, verbose_name="혈액형"),
        ),
    ]

