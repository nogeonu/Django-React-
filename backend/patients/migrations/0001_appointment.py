# Generated manually for Django migration

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Appointment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('appointment_date', models.DateTimeField(verbose_name='예약 일시')),
                ('appointment_type', models.CharField(choices=[('검진', '검진'), ('회의', '회의'), ('내근', '내근'), ('외근', '외근')], default='검진', max_length=10, verbose_name='예약 종류')),
                ('title', models.CharField(max_length=200, verbose_name='예약 제목')),
                ('description', models.TextField(blank=True, verbose_name='설명')),
                ('status', models.CharField(choices=[('예약됨', '예약됨'), ('완료', '완료'), ('취소', '취소')], default='예약됨', max_length=10, verbose_name='상태')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='등록일')),
                ('updated_at', models.DateTimeField(auto_now=True, verbose_name='수정일')),
                ('created_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL, verbose_name='생성자')),
                ('patient', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='appointments', to='patients.patient', verbose_name='환자')),
            ],
            options={
                'verbose_name': '예약',
                'verbose_name_plural': '예약들',
                'ordering': ['-appointment_date'],
            },
        ),
        migrations.AddIndex(
            model_name='appointment',
            index=models.Index(fields=['patient', 'appointment_date'], name='patients_ap_patient_abc123_idx'),
        ),
        migrations.AddIndex(
            model_name='appointment',
            index=models.Index(fields=['appointment_date'], name='patients_ap_appoint_idx'),
        ),
    ]
