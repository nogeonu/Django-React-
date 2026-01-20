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

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-4 mb-2">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-600 shadow-lg">
              <FlaskConical className="h-7 w-7 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">검사실 관리 시스템</h1>
              <p className="text-sm text-gray-600">Laboratory Information System - 검사실 전용</p>
            </div>
          </div>
        </div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <Card className="border-l-4 border-blue-500 shadow-md hover:shadow-lg transition-shadow">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">전체 검사</p>
                  <p className="text-3xl font-bold text-gray-900">{stats.total}</p>
                </div>
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-100">
                  <FileText className="h-6 w-6 text-blue-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-l-4 border-green-500 shadow-md hover:shadow-lg transition-shadow">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">혈액검사</p>
                  <p className="text-3xl font-bold text-gray-900">{stats.lab}</p>
                </div>
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-100">
                  <Activity className="h-6 w-6 text-green-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-l-4 border-purple-500 shadow-md hover:shadow-lg transition-shadow">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">RNA 검사</p>
                  <p className="text-3xl font-bold text-gray-900">{stats.rna}</p>
                </div>
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-purple-100">
                  <TrendingUp className="h-6 w-6 text-purple-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-l-4 border-orange-500 shadow-md hover:shadow-lg transition-shadow">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">오늘 검사</p>
                  <p className="text-3xl font-bold text-gray-900">{stats.today}</p>
                </div>
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-orange-100">
                  <Users className="h-6 w-6 text-orange-600" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Search and Upload Section */}
        <Card className="mb-6 shadow-md">
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
                <Button onClick={handleSearch} disabled={loading} className="bg-blue-600 hover:bg-blue-700">
                  <Search className="mr-2 h-4 w-4" />
                  검색
                </Button>
              </div>

              {/* Upload Buttons */}
              <div className="flex gap-2">
                <label htmlFor="lab-upload">
                  <Button variant="outline" disabled={loading} asChild className="border-green-500 text-green-700 hover:bg-green-50">
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
                  <Button variant="outline" disabled={loading} asChild className="border-purple-500 text-purple-700 hover:bg-purple-50">
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
          <TabsList className="mb-4 bg-white shadow-sm">
            <TabsTrigger value="pending" className="data-[state=active]:bg-orange-100 data-[state=active]:text-orange-900">
              <Clock className="mr-2 h-4 w-4" />
              대기 중 ({labTests.length})
            </TabsTrigger>
            <TabsTrigger value="completed" className="data-[state=active]:bg-green-100 data-[state=active]:text-green-900">
              <CheckCircle2 className="mr-2 h-4 w-4" />
              완료 ({rnaTests.length})
            </TabsTrigger>
            <TabsTrigger value="all" className="data-[state=active]:bg-blue-100 data-[state=active]:text-blue-900">
              <FileText className="mr-2 h-4 w-4" />
              전체 ({stats.total})
            </TabsTrigger>
          </TabsList>

          {/* Pending Tests */}
          <TabsContent value="pending">
            <Card className="shadow-md">
              <CardHeader className="bg-gradient-to-r from-orange-50 to-yellow-50">
                <CardTitle className="flex items-center gap-2">
                  <Clock className="h-5 w-5 text-orange-600" />
                  대기 중인 혈액검사
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                <div className="space-y-3">
                  {labTests.map((test) => (
                    <div
                      key={test.id}
                      className="cursor-pointer rounded-lg border border-gray-200 p-4 hover:bg-orange-50 hover:border-orange-300 transition-all"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <p className="font-semibold text-lg">{test.patient_name}</p>
                            <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-300">
                              {test.patient_id}
                            </Badge>
                            <Badge variant="outline" className="bg-gray-50">
                              {test.patient_gender}, {test.patient_age}세
                            </Badge>
                          </div>
                          <div className="flex items-center gap-4 text-sm text-gray-600">
                            <span>검사번호: <span className="font-medium text-gray-900">{test.accession_number}</span></span>
                            <span>접수일: {test.test_date}</span>
                            {test.result_date && <span>결과일: {test.result_date}</span>}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge className="bg-orange-100 text-orange-800 hover:bg-orange-200">
                            <Clock className="mr-1 h-3 w-3" />
                            대기 중
                          </Badge>
                          <Button size="sm" className="bg-blue-600 hover:bg-blue-700">
                            결과 입력
                          </Button>
                        </div>
                      </div>
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
            <Card className="shadow-md">
              <CardHeader className="bg-gradient-to-r from-green-50 to-emerald-50">
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5 text-green-600" />
                  완료된 RNA 검사
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                <div className="space-y-3">
                  {rnaTests.map((test) => (
                    <div
                      key={test.id}
                      className="cursor-pointer rounded-lg border border-gray-200 p-4 hover:bg-green-50 hover:border-green-300 transition-all"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <p className="font-semibold text-lg">{test.patient_name}</p>
                            <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-300">
                              {test.patient_id}
                            </Badge>
                            <Badge variant="outline" className="bg-gray-50">
                              {test.patient_gender}, {test.patient_age}세
                            </Badge>
                          </div>
                          <div className="flex items-center gap-4 text-sm text-gray-600">
                            <span>검사번호: <span className="font-medium text-gray-900">{test.accession_number}</span></span>
                            <span>접수일: {test.test_date}</span>
                            {test.result_date && <span>결과일: {test.result_date}</span>}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge className="bg-green-100 text-green-800 hover:bg-green-200">
                            <CheckCircle2 className="mr-1 h-3 w-3" />
                            완료
                          </Badge>
                          <Button size="sm" variant="outline" className="border-purple-300 text-purple-700 hover:bg-purple-50">
                            결과 조회
                          </Button>
                        </div>
                      </div>
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
              <Card className="shadow-md">
                <CardHeader className="bg-gradient-to-r from-blue-50 to-cyan-50">
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5 text-blue-600" />
                    혈액검사 ({labTests.length})
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="space-y-2 max-h-[600px] overflow-y-auto">
                    {labTests.map((test) => (
                      <div
                        key={test.id}
                        className="cursor-pointer rounded-lg border border-gray-200 p-3 hover:bg-blue-50 transition-colors"
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="font-semibold">{test.patient_name}</p>
                            <p className="text-xs text-gray-500">{test.accession_number}</p>
                          </div>
                          <Badge variant="outline" className="bg-orange-50 text-orange-700">
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
              <Card className="shadow-md">
                <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50">
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5 text-purple-600" />
                    RNA 검사 ({rnaTests.length})
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="space-y-2 max-h-[600px] overflow-y-auto">
                    {rnaTests.map((test) => (
                      <div
                        key={test.id}
                        className="cursor-pointer rounded-lg border border-gray-200 p-3 hover:bg-purple-50 transition-colors"
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="font-semibold">{test.patient_name}</p>
                            <p className="text-xs text-gray-500">{test.accession_number}</p>
                          </div>
                          <Badge variant="outline" className="bg-green-50 text-green-700">
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
    </div>
  );
}
