import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Users, TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface Statistics {
  total_patients: number;
  positive_patients: number;
  negative_patients: number;
  positive_rate: number;
  gender_statistics: {
    male: {
      total: number;
      positive: number;
      negative: number;
      positive_rate: number;
    };
    female: {
      total: number;
      positive: number;
      negative: number;
      positive_rate: number;
    };
  };
  average_risk_score: number;
}

export default function LungCancerStats() {
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    fetchStatistics();
  }, []);

  const fetchStatistics = async () => {
    try {
      const response = await fetch('/api/lung_cancer/api/results/statistics/');
      if (!response.ok) {
        throw new Error('통계 데이터를 가져오는데 실패했습니다.');
      }
      const data = await response.json();
      setStatistics(data);
    } catch (error) {
      console.error('Error:', error);
      toast({
        title: "오류 발생",
        description: "통계 데이터를 불러오는데 실패했습니다.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="container mx-auto p-6 max-w-6xl">
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin" />
          <span className="ml-2">통계 데이터를 불러오는 중...</span>
        </div>
      </div>
    );
  }

  if (!statistics) {
    return (
      <div className="container mx-auto p-6 max-w-6xl">
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            통계 데이터를 불러올 수 없습니다. 다시 시도해주세요.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">폐암 예측 통계</h1>
        <p className="text-gray-600 mt-2">
          폐암 예측 시스템의 전체 통계 및 분석 결과입니다.
        </p>
      </div>

      {/* 전체 통계 카드 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">총 환자 수</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{statistics.total_patients}</div>
            <p className="text-xs text-muted-foreground">전체 예측 환자</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">양성 예측</CardTitle>
            <AlertTriangle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">{statistics.positive_patients}</div>
            <p className="text-xs text-muted-foreground">
              {statistics.positive_rate}% 비율
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">음성 예측</CardTitle>
            <CheckCircle className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{statistics.negative_patients}</div>
            <p className="text-xs text-muted-foreground">
              {100 - statistics.positive_rate}% 비율
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">평균 위험도</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{statistics.average_risk_score}%</div>
            <p className="text-xs text-muted-foreground">전체 평균</p>
          </CardContent>
        </Card>
      </div>

      {/* 성별 통계 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <Card>
          <CardHeader>
            <CardTitle>남성 환자 통계</CardTitle>
            <CardDescription>남성 환자의 폐암 예측 결과</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">총 환자 수</span>
                <Badge variant="outline">{statistics.gender_statistics.male.total}명</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">양성 예측</span>
                <Badge variant="destructive">{statistics.gender_statistics.male.positive}명</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">음성 예측</span>
                <Badge variant="secondary">{statistics.gender_statistics.male.negative}명</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">양성 비율</span>
                <Badge variant={statistics.gender_statistics.male.positive_rate > 50 ? "destructive" : "secondary"}>
                  {statistics.gender_statistics.male.positive_rate}%
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>여성 환자 통계</CardTitle>
            <CardDescription>여성 환자의 폐암 예측 결과</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">총 환자 수</span>
                <Badge variant="outline">{statistics.gender_statistics.female.total}명</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">양성 예측</span>
                <Badge variant="destructive">{statistics.gender_statistics.female.positive}명</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">음성 예측</span>
                <Badge variant="secondary">{statistics.gender_statistics.female.negative}명</Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium">양성 비율</span>
                <Badge variant={statistics.gender_statistics.female.positive_rate > 50 ? "destructive" : "secondary"}>
                  {statistics.gender_statistics.female.positive_rate}%
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 시각화 차트 영역 */}
      <Card>
        <CardHeader>
          <CardTitle>데이터 시각화</CardTitle>
          <CardDescription>
            폐암 예측 결과의 시각적 분석을 위한 차트입니다.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <TrendingUp className="h-12 w-12 mx-auto mb-4 text-gray-400" />
            <p>차트 데이터를 불러오는 중...</p>
            <p className="text-sm">실제 구현에서는 Chart.js 또는 Recharts를 사용하여 차트를 렌더링합니다.</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
