import csv
import io
from datetime import datetime
from rest_framework import viewsets, status, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.db.models import Q
from .models import LabTest, RNATest
from .serializers import LabTestSerializer, RNATestSerializer
from patients.models import Patient
import logging

logger = logging.getLogger(__name__)

# Import pCR predictor
try:
    from .pcr_predictor import PCRPredictor
    pcr_predictor = PCRPredictor()
    PCR_AVAILABLE = True
except Exception as e:
    logger.warning(f"pCR predictor not available: {e}")
    PCR_AVAILABLE = False


class LabTestViewSet(viewsets.ModelViewSet):
    queryset = LabTest.objects.all()
    serializer_class = LabTestSerializer
    permission_classes = [permissions.AllowAny]
    
    def get_queryset(self):
        queryset = LabTest.objects.all()
        search = self.request.query_params.get('search', None)
        
        if search:
            queryset = queryset.filter(
                Q(patient__patient_id__icontains=search) |
                Q(patient__name__icontains=search) |
                Q(accession_number__icontains=search)
            )
        
        return queryset
    
    @action(detail=False, methods=['post'], parser_classes=[MultiPartParser, FormParser])
    def upload_csv(self, request):
        """혈액검사 CSV 파일 업로드"""
        if 'file' not in request.FILES:
            return Response(
                {'error': 'CSV 파일이 필요합니다.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        csv_file = request.FILES['file']
        
        if not csv_file.name.endswith('.csv'):
            return Response(
                {'error': 'CSV 파일만 업로드 가능합니다.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            decoded_file = csv_file.read().decode('utf-8')
            io_string = io.StringIO(decoded_file)
            reader = csv.DictReader(io_string)
            
            created_count = 0
            updated_count = 0
            errors = []
            
            for row in reader:
                try:
                    patient_id = row.get('patient_id', '').strip()
                    
                    if not patient_id:
                        errors.append(f"환자 ID가 없습니다: {row}")
                        continue
                    
                    try:
                        patient = Patient.objects.get(patient_id=patient_id)
                    except Patient.DoesNotExist:
                        errors.append(f"환자를 찾을 수 없습니다: {patient_id}")
                        continue
                    
                    test_date = datetime.strptime(row.get('test_date', ''), '%Y-%m-%d').date()
                    accession_number = f"L{test_date.strftime('%Y%m%d')}-{patient_id}"
                    
                    lab_test, created = LabTest.objects.update_or_create(
                        accession_number=accession_number,
                        defaults={
                            'patient': patient,
                            'test_date': test_date,
                            'result_date': test_date,
                            'wbc': float(row.get('wbc', 0)) if row.get('wbc') else None,
                            'wbc_unit': row.get('wbc_unit', 'x10^9/L'),
                            'hemoglobin': float(row.get('hemoglobin', 0)) if row.get('hemoglobin') else None,
                            'hemoglobin_unit': row.get('hemoglobin_unit', 'g/dL'),
                            'neutrophils': float(row.get('neutrophils', 0)) if row.get('neutrophils') else None,
                            'neutrophils_unit': row.get('neutrophils_unit', 'x10^9/L'),
                            'lymphocytes': float(row.get('lymphocytes', 0)) if row.get('lymphocytes') else None,
                            'lymphocytes_unit': row.get('lymphocytes_unit', 'x10^9/L'),
                            'platelets': float(row.get('platelets', 0)) if row.get('platelets') else None,
                            'platelets_unit': row.get('platelets_unit', 'x10^9/L'),
                            'nlr': float(row.get('nlr', 0)) if row.get('nlr') else None,
                            'crp': float(row.get('crp', 0)) if row.get('crp') else None,
                            'crp_unit': row.get('crp_unit', 'mg/L'),
                            'ldh': float(row.get('ldh', 0)) if row.get('ldh') else None,
                            'ldh_unit': row.get('ldh_unit', 'U/L'),
                            'albumin': float(row.get('albumin', 0)) if row.get('albumin') else None,
                            'albumin_unit': row.get('albumin_unit', 'g/dL'),
                        }
                    )
                    
                    if created:
                        created_count += 1
                    else:
                        updated_count += 1
                        
                except Exception as e:
                    errors.append(f"행 처리 오류: {str(e)} - {row}")
                    logger.error(f"Lab test upload error: {e}", exc_info=True)
            
            return Response({
                'success': True,
                'created': created_count,
                'updated': updated_count,
                'errors': errors
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"CSV upload error: {e}", exc_info=True)
            return Response(
                {'error': f'CSV 파일 처리 중 오류가 발생했습니다: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class RNATestViewSet(viewsets.ModelViewSet):
    queryset = RNATest.objects.all()
    serializer_class = RNATestSerializer
    permission_classes = [permissions.AllowAny]
    
    def get_queryset(self):
        queryset = RNATest.objects.all()
        search = self.request.query_params.get('search', None)
        
        if search:
            queryset = queryset.filter(
                Q(patient__patient_id__icontains=search) |
                Q(patient__name__icontains=search) |
                Q(accession_number__icontains=search)
            )
        
        return queryset
    
    @action(detail=False, methods=['post'], parser_classes=[MultiPartParser, FormParser])
    def upload_csv(self, request):
        """RNA 검사 CSV 파일 업로드"""
        if 'file' not in request.FILES:
            return Response(
                {'error': 'CSV 파일이 필요합니다.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        csv_file = request.FILES['file']
        
        if not csv_file.name.endswith('.csv'):
            return Response(
                {'error': 'CSV 파일만 업로드 가능합니다.'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            decoded_file = csv_file.read().decode('utf-8')
            io_string = io.StringIO(decoded_file)
            reader = csv.DictReader(io_string)
            
            created_count = 0
            updated_count = 0
            errors = []
            
            for row in reader:
                try:
                    patient_id = row.get('patient_id', '').strip()
                    
                    if not patient_id:
                        errors.append(f"환자 ID가 없습니다: {row}")
                        continue
                    
                    try:
                        patient = Patient.objects.get(patient_id=patient_id)
                    except Patient.DoesNotExist:
                        errors.append(f"환자를 찾을 수 없습니다: {patient_id}")
                        continue
                    
                    from datetime import date
                    test_date = date.today()
                    accession_number = f"R{test_date.strftime('%Y%m%d')}-{patient_id}"
                    
                    # 27개 유전자 발현값 추출
                    gene_data = {}
                    gene_names = ['CXCL13', 'CD8A', 'CCR7', 'C1QA', 'LY9', 'CXCL10', 'CXCL9', 'STAT1',
                                  'CCND1', 'MKI67', 'TOP2A', 'BRCA1', 'RAD51', 'PRKDC', 'POLD3', 'POLB',
                                  'LIG1', 'ERBB2', 'ESR1', 'PGR', 'ARAF', 'PIK3CA', 'AKT1', 'MTOR',
                                  'TP53', 'PTEN', 'MYC']
                    
                    for gene in gene_names:
                        value = row.get(gene, '')
                        gene_data[gene] = float(value) if value else None
                    
                    rna_test, created = RNATest.objects.update_or_create(
                        accession_number=accession_number,
                        defaults={
                            'patient': patient,
                            'test_date': test_date,
                            'result_date': test_date,
                            **gene_data
                        }
                    )
                    
                    if created:
                        created_count += 1
                    else:
                        updated_count += 1
                        
                except Exception as e:
                    errors.append(f"행 처리 오류: {str(e)} - {row}")
                    logger.error(f"RNA test upload error: {e}", exc_info=True)
            
            return Response({
                'success': True,
                'created': created_count,
                'updated': updated_count,
                'errors': errors
            }, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"CSV upload error: {e}", exc_info=True)
            return Response(
                {'error': f'CSV 파일 처리 중 오류가 발생했습니다: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['post'])
    def predict_pcr(self, request, pk=None):
        """pCR 예측 수행"""
        if not PCR_AVAILABLE:
            return Response(
                {'error': 'pCR 예측 모델을 사용할 수 없습니다.'},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        try:
            rna_test = self.get_object()
            
            # 유전자 발현값 추출
            gene_values = {
                'CXCL13': rna_test.CXCL13, 'CD8A': rna_test.CD8A, 'CCR7': rna_test.CCR7,
                'C1QA': rna_test.C1QA, 'LY9': rna_test.LY9, 'CXCL10': rna_test.CXCL10,
                'CXCL9': rna_test.CXCL9, 'STAT1': rna_test.STAT1, 'CCND1': rna_test.CCND1,
                'MKI67': rna_test.MKI67, 'TOP2A': rna_test.TOP2A, 'BRCA1': rna_test.BRCA1,
                'RAD51': rna_test.RAD51, 'PRKDC': rna_test.PRKDC, 'POLD3': rna_test.POLD3,
                'POLB': rna_test.POLB, 'LIG1': rna_test.LIG1, 'ERBB2': rna_test.ERBB2,
                'ESR1': rna_test.ESR1, 'PGR': rna_test.PGR, 'ARAF': rna_test.ARAF,
                'PIK3CA': rna_test.PIK3CA, 'AKT1': rna_test.AKT1, 'MTOR': rna_test.MTOR,
                'TP53': rna_test.TP53, 'PTEN': rna_test.PTEN, 'MYC': rna_test.MYC
            }
            
            # 환자 정보
            from datetime import date
            age = (date.today() - rna_test.patient.birth_date).days // 365 if rna_test.patient.birth_date else 0
            gender_display = '여성 (Female)' if rna_test.patient.gender == 'F' else '남성 (Male)'
            
            patient_info = {
                'patient_id': rna_test.patient.patient_id,
                'name': rna_test.patient.name,
                'age': age,
                'gender': gender_display,
                'test_date': str(rna_test.test_date)
            }
            
            # pCR 예측 수행
            result = pcr_predictor.generate_report_image(gene_values, patient_info)
            
            return Response({
                'success': True,
                'probability': result['probability'],
                'prediction': result['prediction'],
                'image': result['image'],
                'top_genes': result['top_genes'],
                'pathway_scores': result['pathway_scores']
            })
            
        except Exception as pcr_error:
            logger.error(f"pCR prediction error: {pcr_error}", exc_info=True)
            return Response(
                {'error': f'pCR 예측 중 오류가 발생했습니다: {str(pcr_error)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
