import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Users, TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { apiRequest } from '@/lib/api';

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

interface VisualizationData {
  prediction_distribution: { [key: string]: number };
  gender_distribution: { [key: string]: { [key: string]: number } };
  age_distribution: { [key: string]: { [key: string]: number } };
  total_patients: number;
  average_age: number;
  average_risk: number;
}

const COLORS = {
  positive: '#ef4444',  // red-500
  negative: '#10b981',  // green-500
  male: '#3b82f6',      // blue-500
  female: '#f472b6',    // pink-400
};

export default function LungCancerStats() {
  const [statistics, setStatistics] = useState<Statistics | null>(null);
  const [visualizationData, setVisualizationData] = useState<VisualizationData | null>(null);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    fetchStatistics();
    fetchVisualizationData();
  }, []);

  const fetchStatistics = async () => {
    try {
      const data = await apiRequest('GET', '/api/lung_cancer/results/statistics/');
      setStatistics(data);
    } catch (error) {
      console.error('Error:', error);
      toast({
        title: "오류 발생",
        description: "통계 데이터를 불러오는데 실패했습니다.",
        variant: "destructive",
      });
    }
  };

  const fetchVisualizationData = async () => {
    try {
      const data = await apiRequest('GET', '/api/lung_cancer/visualization/');
      setVisualizationData(data);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  useEffect(() => {
    if (statistics) {
      setLoading(false);
    }
  }, [statistics]);

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

  // 데이터 가공
  const predictionData = Object.entries(statistics.gender_statistics).map(([gender, stats]) => ({
    name: gender === 'male' ? '남성' : '여성',
    양성: stats.positive,
    음성: stats.negative,
  }));

  const pieData = [
    { name: '양성', value: statistics.positive_patients, color: COLORS.positive },
    { name: '음성', value: statistics.negative_patients, color: COLORS.negative },
  ];

  const ageChartData = visualizationData?.age_distribution ? 
    Object.entries(visualizationData.age_distribution).map(([age, data]) => ({
      age: `${age}대`,
      양성: data['YES'] || 0,
      음성: data['NO'] || 0,
    })) : [];

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

      {/* 차트 영역 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* 예측 결과 분포 (원형 차트) */}
        <Card>
          <CardHeader>
            <CardTitle>예측 결과 분포</CardTitle>
            <CardDescription>전체 환자의 양성/음성 예측 비율</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* 성별 예측 결과 (막대 차트) */}
        <Card>
          <CardHeader>
            <CardTitle>성별 예측 결과 비교</CardTitle>
            <CardDescription>성별에 따른 양성/음성 예측 비교</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={predictionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="양성" fill={COLORS.positive} />
                <Bar dataKey="음성" fill={COLORS.negative} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* 연령대별 분석 */}
        {ageChartData.length > 0 && (
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle>연령대별 예측 결과</CardTitle>
              <CardDescription>연령대별 양성/음성 예측 분석</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={ageChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="age" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="양성" fill={COLORS.positive} />
                  <Bar dataKey="음성" fill={COLORS.negative} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )}
      </div>

      {/* 상세 통계 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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
    </div>
  );
}
