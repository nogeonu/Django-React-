import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { 
  FlaskConical, 
  Upload, 
  Search, 
  CheckCircle2, 
  Clock, 
  FileText,
  TrendingUp,
  Users,
  Activity
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import {
  getLabTestsApi,
  getRNATestsApi,
  uploadLabTestCsvApi,
  uploadRNATestCsvApi,
} from '@/lib/api';

interface LabTest {
  id: number;
  accession_number: string;
  patient_name: string;
  patient_id: string;
  patient_birth_date: string;
  patient_age: number;
  patient_gender: string;
  test_date: string;
  result_date: string;
  wbc: number | null;
  hemoglobin: number | null;
  neutrophils: number | null;
  lymphocytes: number | null;
  platelets: number | null;
  nlr: number | null;
  crp: number | null;
  ldh: number | null;
  albumin: number | null;
}

interface RNATest {
  id: number;
  accession_number: string;
  patient_name: string;
  patient_id: string;
  patient_birth_date: string;
  patient_age: number;
  patient_gender: string;
  test_date: string;
  result_date: string;
  [key: string]: any;
}

export default function LaboratoryDashboard() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState('pending');
  const [searchTerm, setSearchTerm] = useState('');
  const [labTests, setLabTests] = useState<LabTest[]>([]);
  const [rnaTests, setRNATests] = useState<RNATest[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadLabTests();
    loadRNATests();
  }, []);

  const loadLabTests = async () => {
    try {
      const data = await getLabTestsApi();
      setLabTests(data.results || data);
    } catch (error) {
      console.error('Failed to load lab tests:', error);
    }
  };

  const loadRNATests = async () => {
    try {
      const data = await getRNATestsApi();
      setRNATests(data.results || data);
    } catch (error) {
      console.error('Failed to load RNA tests:', error);
    }
  };

  const handleLabTestUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    try {
      const result = await uploadLabTestCsvApi(file);
      toast({
        title: '업로드 성공',
        description: `${result.created}개 생성, ${result.updated}개 업데이트`,
      });
      loadLabTests();
    } catch (error: any) {
      toast({
        title: '업로드 실패',
        description: error?.response?.data?.error || '파일 업로드 중 오류가 발생했습니다.',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
      event.target.value = '';
    }
  };

  const handleRNATestUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setLoading(true);
    try {
      const result = await uploadRNATestCsvApi(file);
      toast({
        title: 'RNA 업로드 성공',
        description: `${result.created}개 생성, ${result.updated}개 업데이트`,
      });
      loadRNATests();
    } catch (error: any) {
      toast({
        title: '업로드 실패',
        description: error?.response?.data?.error || '파일 업로드 중 오류가 발생했습니다.',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
      event.target.value = '';
    }
  };

  const handleSearch = async () => {
    if (!searchTerm.trim()) {
      loadLabTests();
      loadRNATests();
      return;
    }

    setLoading(true);
    try {
      const [labData, rnaData] = await Promise.all([
        getLabTestsApi({ search: searchTerm }),
        getRNATestsApi({ search: searchTerm }),
      ]);
      setLabTests(labData.results || labData);
      setRNATests(rnaData.results || rnaData);
      
      const totalResults = (labData.results?.length || labData.length || 0) + 
                          (rnaData.results?.length || rnaData.length || 0);
      toast({
        title: '검색 완료',
        description: `${totalResults}개의 결과를 찾았습니다.`,
      });
    } catch (error: any) {
      toast({
        title: '검색 실패',
        description: error?.response?.data?.error || '검색 중 오류가 발생했습니다.',
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  // 통계 계산
  const stats = {
    total: labTests.length + rnaTests.length,
    lab: labTests.length,
    rna: rnaTests.length,
    today: labTests.filter(test => {
      const today = new Date().toISOString().split('T')[0];
      return test.test_date === today;
    }).length,
  };

  const statsCards = [
    {
      title: '전체 검사',
      value: stats.total,
      icon: FileText,
      bgColor: 'bg-blue-100',
      color: 'text-blue-600',
    },
    {
      title: '혈액검사',
      value: stats.lab,
      icon: Activity,
      bgColor: 'bg-green-100',
      color: 'text-green-600',
    },
    {
      title: 'RNA 검사',
      value: stats.rna,
      icon: TrendingUp,
      bgColor: 'bg-purple-100',
      color: 'text-purple-600',
    },
    {
      title: '오늘 검사',
      value: stats.today,
      icon: Users,
      bgColor: 'bg-orange-100',
      color: 'text-orange-600',
    },
  ];

  return (
    <div className="space-y-8 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="bg-gradient-to-br from-primary to-accent p-3 rounded-2xl">
            <FlaskConical className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">검사실 관리 시스템</h1>
            <p className="text-sm text-gray-500">Laboratory Information System</p>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statsCards.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <Card key={index} className="border-none shadow-sm hover:shadow-md transition-all duration-300 group overflow-hidden bg-white">
              <CardContent className="p-6 relative">
                <div className="flex justify-between items-start mb-4">
                  <div className={`${stat.bgColor} p-3 rounded-2xl transition-transform group-hover:scale-110 duration-300`}>
                    <Icon className={`w-5 h-5 ${stat.color}`} />
                  </div>
                </div>
                <div>
                  <p className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-1">{stat.title}</p>
                  <p className="text-3xl font-black text-gray-900 tracking-tight">
                    {stat.value}
                  </p>
                </div>
                <Icon className={`absolute -right-4 -bottom-4 w-24 h-24 opacity-[0.03] ${stat.color} rotate-12`} />
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Search and Upload Section */}
      <Card className="border-none shadow-sm bg-white">
        <CardHeader className="border-b border-gray-50 pb-4">
          <CardTitle className="text-lg font-bold text-gray-900">검사 관리</CardTitle>
        </CardHeader>
        <CardContent className="pt-6">
          <div className="flex flex-col md:flex-row gap-4">
            {/* Search */}
            <div className="flex flex-1 gap-2">
              <Input
                placeholder="환자 이름, ID, 검사번호로 검색..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                className="flex-1"
              />
              <Button onClick={handleSearch} disabled={loading} className="bg-primary hover:bg-primary/90">
                <Search className="mr-2 h-4 w-4" />
                검색
              </Button>
            </div>

            {/* Upload Buttons */}
            <div className="flex gap-2">
              <label htmlFor="lab-upload">
                <Button variant="outline" disabled={loading} asChild className="hover:bg-gray-50">
                  <span>
                    <Upload className="mr-2 h-4 w-4" />
                    혈액검사 업로드
                  </span>
                </Button>
              </label>
              <input
                id="lab-upload"
                type="file"
                accept=".csv"
                onChange={handleLabTestUpload}
                className="hidden"
              />
              
              <label htmlFor="rna-upload">
                <Button variant="outline" disabled={loading} asChild className="hover:bg-gray-50">
                  <span>
                    <Upload className="mr-2 h-4 w-4" />
                    RNA 업로드
                  </span>
                </Button>
              </label>
              <input
                id="rna-upload"
                type="file"
                accept=".csv"
                onChange={handleRNATestUpload}
                className="hidden"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Tabs for Test Lists */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-4 bg-white">
          <TabsTrigger value="pending">
            <Clock className="mr-2 h-4 w-4" />
            대기 중 ({labTests.length})
          </TabsTrigger>
          <TabsTrigger value="completed">
            <CheckCircle2 className="mr-2 h-4 w-4" />
            완료 ({rnaTests.length})
          </TabsTrigger>
          <TabsTrigger value="all">
            <FileText className="mr-2 h-4 w-4" />
            전체 ({stats.total})
          </TabsTrigger>
        </TabsList>

        {/* Pending Tests */}
        <TabsContent value="pending">
          <Card className="border-none shadow-sm bg-white">
            <CardHeader className="border-b border-gray-50 pb-4">
              <CardTitle className="text-lg font-bold text-gray-900 flex items-center gap-2">
                <Clock className="h-5 w-5 text-orange-600" />
                대기 중인 혈액검사
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="space-y-3">
                {labTests.map((test) => (
                  <div
                    key={test.id}
                    className="group flex items-center gap-4 p-5 rounded-xl border border-gray-100 hover:bg-gray-50 transition-colors cursor-pointer"
                  >
                    <div className="w-12 h-12 flex-shrink-0 bg-white shadow-sm border border-gray-100 rounded-2xl flex items-center justify-center text-orange-600 font-black text-lg group-hover:bg-orange-600 group-hover:text-white transition-all">
                      <Clock className="w-5 h-5" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-bold text-gray-900">{test.patient_name}</span>
                        <Badge variant="secondary" className="text-xs">
                          {test.patient_id}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          {test.patient_gender}, {test.patient_age}세
                        </Badge>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-gray-500">
                        <span>검사번호: <span className="font-medium text-gray-700">{test.accession_number}</span></span>
                        <span>접수일: {test.test_date}</span>
                        {test.result_date && <span>결과일: {test.result_date}</span>}
                      </div>
                    </div>
                    <Button size="sm" className="bg-primary hover:bg-primary/90">
                      결과 입력
                    </Button>
                  </div>
                ))}
                {labTests.length === 0 && (
                  <div className="py-12 text-center text-gray-500">
                    <Clock className="mx-auto h-12 w-12 mb-3 text-gray-300" />
                    <p className="text-lg font-medium">대기 중인 검사가 없습니다</p>
                    <p className="text-sm">CSV 파일을 업로드하여 검사를 등록하세요</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Completed Tests */}
        <TabsContent value="completed">
          <Card className="border-none shadow-sm bg-white">
            <CardHeader className="border-b border-gray-50 pb-4">
              <CardTitle className="text-lg font-bold text-gray-900 flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-green-600" />
                완료된 RNA 검사
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-6">
              <div className="space-y-3">
                {rnaTests.map((test) => (
                  <div
                    key={test.id}
                    className="group flex items-center gap-4 p-5 rounded-xl border border-gray-100 hover:bg-gray-50 transition-colors cursor-pointer"
                  >
                    <div className="w-12 h-12 flex-shrink-0 bg-white shadow-sm border border-gray-100 rounded-2xl flex items-center justify-center text-green-600 font-black text-lg group-hover:bg-green-600 group-hover:text-white transition-all">
                      <CheckCircle2 className="w-5 h-5" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-bold text-gray-900">{test.patient_name}</span>
                        <Badge variant="secondary" className="text-xs">
                          {test.patient_id}
                        </Badge>
                        <Badge variant="outline" className="text-xs">
                          {test.patient_gender}, {test.patient_age}세
                        </Badge>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-gray-500">
                        <span>검사번호: <span className="font-medium text-gray-700">{test.accession_number}</span></span>
                        <span>접수일: {test.test_date}</span>
                        {test.result_date && <span>결과일: {test.result_date}</span>}
                      </div>
                    </div>
                    <Button size="sm" variant="outline" className="hover:bg-gray-50">
                      결과 조회
                    </Button>
                  </div>
                ))}
                {rnaTests.length === 0 && (
                  <div className="py-12 text-center text-gray-500">
                    <CheckCircle2 className="mx-auto h-12 w-12 mb-3 text-gray-300" />
                    <p className="text-lg font-medium">완료된 검사가 없습니다</p>
                    <p className="text-sm">검사 결과를 입력하세요</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* All Tests */}
        <TabsContent value="all">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Lab Tests */}
            <Card className="border-none shadow-sm bg-white">
              <CardHeader className="border-b border-gray-50 pb-4">
                <CardTitle className="text-lg font-bold text-gray-900 flex items-center gap-2">
                  <Activity className="h-5 w-5 text-blue-600" />
                  혈액검사 ({labTests.length})
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                <div className="space-y-2 max-h-[600px] overflow-y-auto">
                  {labTests.map((test) => (
                    <div
                      key={test.id}
                      className="cursor-pointer rounded-lg border border-gray-100 p-3 hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-semibold text-sm">{test.patient_name}</p>
                          <p className="text-xs text-gray-500">{test.accession_number}</p>
                        </div>
                        <Badge variant="outline" className="text-xs">
                          대기
                        </Badge>
                      </div>
                    </div>
                  ))}
                  {labTests.length === 0 && (
                    <p className="py-8 text-center text-gray-500">혈액검사 결과가 없습니다</p>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* RNA Tests */}
            <Card className="border-none shadow-sm bg-white">
              <CardHeader className="border-b border-gray-50 pb-4">
                <CardTitle className="text-lg font-bold text-gray-900 flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-purple-600" />
                  RNA 검사 ({rnaTests.length})
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                <div className="space-y-2 max-h-[600px] overflow-y-auto">
                  {rnaTests.map((test) => (
                    <div
                      key={test.id}
                      className="cursor-pointer rounded-lg border border-gray-100 p-3 hover:bg-gray-50 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-semibold text-sm">{test.patient_name}</p>
                          <p className="text-xs text-gray-500">{test.accession_number}</p>
                        </div>
                        <Badge variant="outline" className="text-xs">
                          완료
                        </Badge>
                      </div>
                    </div>
                  ))}
                  {rnaTests.length === 0 && (
                    <p className="py-8 text-center text-gray-500">RNA 검사 결과가 없습니다</p>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
