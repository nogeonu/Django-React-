from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('patients', '0010_add_department_to_auth_user'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Appointment',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('title', models.CharField(max_length=200, verbose_name='제목')),
                ('type', models.CharField(choices=[('예약', '일반 예약'), ('검진', '검진'), ('회의', '회의'), ('내근', '내근'), ('외근', '외근')], default='예약', max_length=30, verbose_name='유형')),
                ('start_time', models.DateTimeField(verbose_name='시작 일시')),
                ('end_time', models.DateTimeField(blank=True, null=True, verbose_name='종료 일시')),
                ('patient_identifier', models.CharField(blank=True, max_length=50, verbose_name='환자 ID')),
                ('patient_name', models.CharField(blank=True, max_length=100, verbose_name='환자 이름')),
                ('patient_gender', models.CharField(blank=True, choices=[('M', '남성'), ('F', '여성')], max_length=1, verbose_name='환자 성별')),
                ('patient_age', models.PositiveIntegerField(blank=True, null=True, verbose_name='환자 나이')),
                ('doctor_code', models.CharField(blank=True, default='', max_length=20, verbose_name='의사 코드')),
                ('doctor_username', models.CharField(max_length=150, verbose_name='의사 계정')),
                ('doctor_name', models.CharField(blank=True, max_length=150, verbose_name='의사 이름')),
                ('doctor_department', models.CharField(blank=True, max_length=30, verbose_name='진료과')),
                ('status', models.CharField(choices=[('scheduled', '예약됨'), ('completed', '완료'), ('cancelled', '취소')], default='scheduled', max_length=20, verbose_name='상태')),
                ('memo', models.TextField(blank=True, verbose_name='메모')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='등록일')),
                ('updated_at', models.DateTimeField(auto_now=True, verbose_name='수정일')),
                ('created_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='created_appointments', to=settings.AUTH_USER_MODEL, verbose_name='등록자')),
                ('doctor', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='appointments', to=settings.AUTH_USER_MODEL, verbose_name='담당 의사')),
                ('patient', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='appointments', to='patients.patient', verbose_name='환자')),
            ],
            options={
                'verbose_name': '예약',
                'verbose_name_plural': '예약들',
                'ordering': ['-start_time', '-created_at'],
            },
        ),
    ]
