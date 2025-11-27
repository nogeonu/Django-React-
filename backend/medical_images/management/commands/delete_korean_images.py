"""
한국어 파일명을 가진 의료 이미지 삭제 명령어

사용법:
    python manage.py delete_korean_images --dry-run  # 삭제할 파일만 확인
    python manage.py delete_korean_images            # 실제 삭제
"""
from django.core.management.base import BaseCommand
from medical_images.models import MedicalImage
import os
import re
from django.conf import settings


class Command(BaseCommand):
    help = '한국어 파일명을 가진 의료 이미지를 삭제합니다'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='실제로 삭제하지 않고 삭제할 파일만 표시',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        
        # 한국어 문자를 포함하는 정규표현식
        korean_pattern = re.compile(r'[가-힣]')
        
        # 모든 의료 이미지 가져오기
        images = MedicalImage.objects.all()
        deleted_count = 0
        file_deleted_count = 0
        error_count = 0
        
        self.stdout.write(self.style.WARNING(f'\n{"="*60}'))
        self.stdout.write(self.style.WARNING('한국어 파일명 이미지 검색 중...'))
        self.stdout.write(self.style.WARNING(f'{"="*60}\n'))
        
        for image in images:
            if not image.image_file or not image.image_file.name:
                continue
                
            file_name = image.image_file.name
            # 파일명에 한국어가 포함되어 있는지 확인
            if korean_pattern.search(file_name):
                self.stdout.write(f'발견: ID={image.id}, 파일명={file_name}')
                
                if not dry_run:
                    try:
                        # 파일 시스템에서 삭제
                        file_paths = []
                        
                        # 여러 경로 시도
                        possible_paths = [
                            os.path.join(settings.MEDIA_ROOT, file_name),
                            os.path.join(settings.MEDIA_ROOT, 'medical_images', os.path.basename(file_name)),
                        ]
                        
                        # image_file.path가 있으면 사용
                        try:
                            if hasattr(image.image_file, 'path'):
                                possible_paths.insert(0, image.image_file.path)
                        except:
                            pass
                        
                        file_deleted = False
                        for file_path in possible_paths:
                            if os.path.exists(file_path) and os.path.isfile(file_path):
                                os.remove(file_path)
                                self.stdout.write(self.style.SUCCESS(f'  ✅ 파일 삭제: {file_path}'))
                                file_deleted = True
                                file_deleted_count += 1
                                break
                        
                        if not file_deleted:
                            self.stdout.write(self.style.WARNING(f'  ⚠️ 파일을 찾을 수 없음: {file_name}'))
                        
                        # 데이터베이스에서 삭제
                        image.delete()
                        self.stdout.write(self.style.SUCCESS(f'  ✅ 데이터베이스 레코드 삭제: ID={image.id}'))
                        deleted_count += 1
                        
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f'  ❌ 삭제 실패: {str(e)}'))
                        error_count += 1
        
        self.stdout.write(self.style.WARNING(f'\n{"="*60}'))
        if dry_run:
            self.stdout.write(self.style.WARNING(f'검색 완료: {deleted_count}개의 한국어 파일명 이미지 발견'))
            self.stdout.write(self.style.WARNING('--dry-run 옵션을 제거하면 실제로 삭제됩니다.'))
        else:
            self.stdout.write(self.style.SUCCESS(f'삭제 완료:'))
            self.stdout.write(f'  - 데이터베이스 레코드: {deleted_count}개')
            self.stdout.write(f'  - 파일 시스템: {file_deleted_count}개')
            if error_count > 0:
                self.stdout.write(self.style.ERROR(f'  - 오류: {error_count}개'))
        self.stdout.write(self.style.WARNING(f'{"="*60}\n'))

