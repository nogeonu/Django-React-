# Generated migration

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='MRIStudy',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('patient_id', models.CharField(max_length=100, verbose_name='환자 ID')),
                ('study_date', models.DateField(blank=True, null=True, verbose_name='검사 날짜')),
                ('scanner_manufacturer', models.CharField(blank=True, max_length=100, verbose_name='스캐너 제조사')),
                ('scanner_model', models.CharField(blank=True, max_length=100, verbose_name='스캐너 모델')),
                ('field_strength', models.FloatField(blank=True, null=True, verbose_name='자기장 세기(T)')),
                ('age', models.IntegerField(blank=True, null=True, verbose_name='나이')),
                ('menopausal_status', models.CharField(blank=True, max_length=50, verbose_name='폐경 상태')),
                ('tumor_subtype', models.CharField(blank=True, max_length=100, verbose_name='종양 유형')),
                ('data_directory', models.CharField(max_length=500, verbose_name='데이터 디렉토리')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='등록일')),
                ('updated_at', models.DateTimeField(auto_now=True, verbose_name='수정일')),
            ],
            options={
                'verbose_name': 'MRI 검사',
                'verbose_name_plural': 'MRI 검사 목록',
                'db_table': 'mri_studies',
                'ordering': ['-created_at'],
            },
        ),
    ]

