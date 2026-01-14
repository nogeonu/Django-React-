import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, FileText, User, Calendar, CheckCircle, AlertTriangle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { getImagingAnalysisByIdApi, getOrderApi } from '@/lib/api';
import { format } from 'date-fns';
import { ko } from 'date-fns/locale';

export default function ImagingAnalysisDetail() {
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const orderId = searchParams.get('order');
  const navigate = useNavigate();

  // 영상 분석 결과 조회
  const { data: analysis, isLoading: isLoadingAnalysis } = useQuery({
    queryKey: ['imaging-analysis', id],
    queryFn: () => getImagingAnalysisByIdApi(id!),
    enabled: !!id,
  });

  // 주문 정보 조회
  const { data: order, isLoading: isLoadingOrder } = useQuery({
    queryKey: ['order', orderId],
    queryFn: () => getOrderApi(orderId!),
    enabled: !!orderId,
  });

  if (isLoadingAnalysis || isLoadingOrder) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-center py-12">
          <p className="text-muted-foreground">로딩 중...</p>
        </div>
      </div>
    );
  }

  if (!analysis || !order) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-center py-12">
          <p className="text-muted-foreground">분석 결과를 찾을 수 없습니다.</p>
          <Button onClick={() => navigate('/ocs')} className="mt-4">
            OCS로 돌아가기
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" onClick={() => navigate('/ocs')}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            돌아가기
          </Button>
          <div>
            <h1 className="text-3xl font-bold">영상 분석 결과</h1>
            <p className="text-muted-foreground mt-1">
              {order.patient_name}님의 {order.order_data?.imaging_type || '영상'} 분석 결과
            </p>
          </div>
        </div>
      </div>

      {/* 환자 정보 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <User className="h-5 w-5" />
            환자 정보
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-muted-foreground">환자명</p>
              <p className="font-medium">{order.patient_name}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">환자번호</p>
              <p className="font-medium">{order.patient_number}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">의사</p>
              <p className="font-medium">{order.doctor_name}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">촬영 유형</p>
              <p className="font-medium">{order.order_data?.imaging_type || '-'}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 분석 정보 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            분석 정보
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">분석자</p>
                <p className="font-medium">{analysis.analyzed_by_name || '-'}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">분석일시</p>
                <p className="font-medium">
                  {format(new Date(analysis.created_at), 'yyyy-MM-dd HH:mm', { locale: ko })}
                </p>
              </div>
            </div>
            {analysis.confidence_score !== null && (
              <div>
                <p className="text-sm text-muted-foreground mb-2">신뢰도</p>
                <div className="flex items-center gap-2">
                  <div className="flex-1 bg-slate-200 rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full transition-all"
                      style={{ width: `${analysis.confidence_score * 100}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium">
                    {(analysis.confidence_score * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* 소견 */}
      {analysis.findings && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5" />
              소견
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="prose max-w-none">
              <p className="whitespace-pre-wrap">{analysis.findings}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 권고사항 */}
      {analysis.recommendations && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5" />
              권고사항
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="prose max-w-none">
              <p className="whitespace-pre-wrap">{analysis.recommendations}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 분석 결과 상세 (JSON) */}
      {analysis.analysis_result && Object.keys(analysis.analysis_result).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>분석 결과 상세</CardTitle>
            <CardDescription>자세한 분석 데이터</CardDescription>
          </CardHeader>
          <CardContent>
            <pre className="bg-slate-100 dark:bg-slate-800 p-4 rounded-lg overflow-auto text-sm">
              {JSON.stringify(analysis.analysis_result, null, 2)}
            </pre>
          </CardContent>
        </Card>
      )}

      {/* 액션 버튼 */}
      <div className="flex gap-2">
        <Button onClick={() => navigate(`/ocs/orders/${order.id}`)}>
          주문 상세 보기
        </Button>
        <Button variant="outline" onClick={() => navigate('/ocs')}>
          OCS로 돌아가기
        </Button>
      </div>
    </div>
  );
}
