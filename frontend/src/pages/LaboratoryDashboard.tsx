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
  Activity,
  Dna,
  Brain,
  Printer,
  AlertCircle
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import {
  getLabTestsApi,
  getRNATestsApi,
  uploadLabTestCsvApi,
  uploadRNATestCsvApi,
  predictPCRApi,
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
  wbc_unit?: string;
  hemoglobin: number | null;
  hemoglobin_unit?: string;
  neutrophils: number | null;
  neutrophils_unit?: string;
  lymphocytes: number | null;
  lymphocytes_unit?: string;
  platelets: number | null;
  platelets_unit?: string;
  nlr: number | null;
  crp: number | null;
  crp_unit?: string;
  ldh: number | null;
  ldh_unit?: string;
  albumin: number | null;
  albumin_unit?: string;
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

const GENE_NAMES = [
  'CXCL13', 'CD8A', 'CCR7', 'C1QA', 'LY9', 'CXCL10', 'CXCL9', 'STAT1',
  'CCND1', 'MKI67', 'TOP2A', 'BRCA1', 'RAD51', 'PRKDC', 'POLD3', 'POLB',
  'LIG1', 'ERBB2', 'ESR1', 'PGR', 'ARAF', 'PIK3CA', 'AKT1', 'MTOR',
  'TP53', 'PTEN', 'MYC'
];

const GENE_PATHWAYS: Record<string, string> = {
  'CXCL13': '면역 (Immune)',
  'CD8A': '면역 (Immune)',
  'CCR7': '면역 (Immune)',
  'C1QA': '면역 (Immune)',
  'LY9': '면역 (Immune)',
  'CXCL10': '면역 (Immune)',
  'CXCL9': '면역 (Immune)',
  'STAT1': '면역 (Immune)',
  'CCND1': '세포증식 (Proliferation)',
  'MKI67': '세포증식 (Proliferation)',
  'TOP2A': '세포증식 (Proliferation)',
  'BRCA1': 'DNA 복구 (DNA Repair)',
  'RAD51': 'DNA 복구 (DNA Repair)',
  'PRKDC': 'DNA 복구 (DNA Repair)',
  'POLD3': 'DNA 복구 (DNA Repair)',
  'POLB': 'DNA 복구 (DNA Repair)',
  'LIG1': 'DNA 복구 (DNA Repair)',
  'ERBB2': 'HER2 수용체',
  'ESR1': '호르몬 수용체 (ER/PR)',
  'PGR': '호르몬 수용체 (ER/PR)',
  'ARAF': '신호전달 (AKT/mTOR)',
  'PIK3CA': '신호전달 (AKT/mTOR)',
  'AKT1': '신호전달 (AKT/mTOR)',
  'MTOR': '신호전달 (AKT/mTOR)',
  'TP53': '신호전달 (AKT/mTOR)',
  'PTEN': '신호전달 (AKT/mTOR)',
  'MYC': '신호전달 (AKT/mTOR)',
};

export default function LaboratoryDashboard() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState('lab-requests');
  const [searchTerm, setSearchTerm] = useState('');
  const [labTests, setLabTests] = useState<LabTest[]>([]);
  const [rnaTests, setRNATests] = useState<RNATest[]>([]);
  const [selectedLabTest, setSelectedLabTest] = useState<LabTest | null>(null);
  const [selectedRNATest, setSelectedRNATest] = useState<RNATest | null>(null);
  const [loading, setLoading] = useState(false);
  const [pcrPrediction, setPcrPrediction] = useState<any>(null);
  const [predictingPCR, setPredictingPCR] = useState(false);
  const [showReportModal, setShowReportModal] = useState(false);

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
      const updatedData = await getRNATestsApi();
      const updatedTests = updatedData.results || updatedData;
      setRNATests(updatedTests);
      
      if (updatedTests.length > 0) {
        setActiveTab('rna-results');
        setSelectedRNATest(updatedTests[0]);
      }
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

  const handlePCRPredict = async () => {
    const testToPredict = selectedRNATest || rnaTests[0];
    if (!testToPredict) {
      toast({
        title: 'RNA 검사 선택 필요',
        description: 'pCR 예측을 위해 RNA 검사를 선택해주세요.',
        variant: 'destructive',
      });
      return;
    }

    setPredictingPCR(true);
    try {
      const result = await predictPCRApi(testToPredict.id);
      setPcrPrediction(result);
      toast({
        title: 'pCR 예측 완료',
        description: `예측 확률: ${(result.probability * 100).toFixed(1)}%`,
      });
    } catch (error: any) {
      toast({
        title: 'pCR 예측 실패',
        description: error?.response?.data?.error || 'pCR 예측 중 오류가 발생했습니다.',
        variant: 'destructive',
      });
    } finally {
      setPredictingPCR(false);
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

  const getLabTestFlag = (value: number | null, refMin: number, refMax: number) => {
    if (value === null || value === undefined) return { flag: 'N/A', color: 'text-gray-500', bgColor: 'bg-gray-100' };
    if (value < refMin) return { flag: 'Low', color: 'text-yellow-700', bgColor: 'bg-yellow-100' };
    if (value > refMax) return { flag: 'High', color: 'text-red-700', bgColor: 'bg-red-100' };
    return { flag: 'Normal', color: 'text-green-700', bgColor: 'bg-green-100' };
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
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 p-6">
      <div className="mx-auto max-w-7xl space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="bg-gradient-to-br from-blue-600 to-indigo-600 p-4 rounded-2xl shadow-lg">
              <FlaskConical className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">검사실 정보 시스템</h1>
              <p className="text-sm text-gray-600 mt-1">Laboratory Information System (LIS) - CDSS 통합</p>
            </div>
          </div>
          <Button variant="outline" size="icon" className="hidden md:flex">
            <Printer className="h-5 w-5" />
          </Button>
        </div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="border-l-4 border-l-blue-500 shadow-md hover:shadow-lg transition-shadow">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">전체 검사</p>
                  <p className="text-3xl font-bold text-gray-900">{stats.total}</p>
                </div>
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-100">
                  <FileText className="h-6 w-6 text-blue-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-l-4 border-l-green-500 shadow-md hover:shadow-lg transition-shadow">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">혈액검사</p>
                  <p className="text-3xl font-bold text-gray-900">{stats.lab}</p>
                </div>
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-100">
                  <Activity className="h-6 w-6 text-green-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-l-4 border-l-purple-500 shadow-md hover:shadow-lg transition-shadow">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">RNA 검사</p>
                  <p className="text-3xl font-bold text-gray-900">{stats.rna}</p>
                </div>
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-purple-100">
                  <TrendingUp className="h-6 w-6 text-purple-600" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-l-4 border-l-orange-500 shadow-md hover:shadow-lg transition-shadow">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">오늘 검사</p>
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
        <Card className="shadow-md">
          <CardHeader className="border-b bg-gradient-to-r from-gray-50 to-blue-50">
            <CardTitle className="text-lg font-bold text-gray-900">검사 관리</CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="flex flex-col md:flex-row gap-4">
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

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-4 bg-white shadow-sm">
            <TabsTrigger value="lab-requests" className="data-[state=active]:bg-orange-100 data-[state=active]:text-orange-900">
              <Clock className="mr-2 h-4 w-4" />
              검사 요청 ({labTests.length})
            </TabsTrigger>
            <TabsTrigger value="lab-results" className="data-[state=active]:bg-blue-100 data-[state=active]:text-blue-900">
              <FileText className="mr-2 h-4 w-4" />
              혈액검사 결과
            </TabsTrigger>
            <TabsTrigger value="rna-results" className="data-[state=active]:bg-purple-100 data-[state=active]:text-purple-900">
              <Dna className="mr-2 h-4 w-4" />
              RNA 검사 결과 ({rnaTests.length})
            </TabsTrigger>
          </TabsList>

          {/* Lab Requests Tab */}
          <TabsContent value="lab-requests">
            <Card className="shadow-md">
              <CardHeader className="bg-gradient-to-r from-orange-50 to-yellow-50 border-b">
                <CardTitle className="flex items-center gap-2">
                  <Clock className="h-5 w-5 text-orange-600" />
                  대기 중인 검사 요청
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                <div className="space-y-3">
                  {labTests.map((test) => (
                    <div
                      key={test.id}
                      className="cursor-pointer rounded-lg border border-gray-200 p-4 hover:bg-orange-50 hover:border-orange-300 transition-all"
                      onClick={() => {
                        setSelectedLabTest(test);
                        setActiveTab('lab-results');
                      }}
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
                          <Badge className="bg-orange-100 text-orange-800">
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

          {/* Lab Results Tab */}
          <TabsContent value="lab-results">
            {selectedLabTest ? (
              <Card className="shadow-md">
                <CardHeader className="border-b bg-gradient-to-r from-blue-50 to-cyan-50">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">
                        Accession # <span className="font-semibold text-gray-900">{selectedLabTest.accession_number}</span>
                        {' '}{selectedLabTest.patient_gender}, {selectedLabTest.patient_age}세
                      </p>
                      <p className="text-sm text-gray-600 mt-1">
                        Order Date: {selectedLabTest.test_date} {selectedLabTest.result_date && `| Result Date: ${selectedLabTest.result_date}`}
                      </p>
                    </div>
                    <Button variant="ghost" size="icon">
                      <Printer className="h-5 w-5" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="pt-6">
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                      <thead className="bg-gray-100">
                        <tr>
                          <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700 border">검사명</th>
                          <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700 border">결과</th>
                          <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700 border">단위</th>
                          <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700 border">참고치</th>
                          <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700 border">판정</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        <tr className="hover:bg-gray-50">
                          <td className="px-4 py-3 border font-medium">WBC</td>
                          <td className="px-4 py-3 border font-semibold">
                            {selectedLabTest.wbc?.toFixed(1) || 'N/A'} <span className="text-xs text-gray-500">×10³/L</span>
                          </td>
                          <td className="px-4 py-3 border">{selectedLabTest.wbc_unit || '×10³/L'}</td>
                          <td className="px-4 py-3 border">4.0-10.0</td>
                          <td className="px-4 py-3 border">
                            {(() => {
                              const flag = getLabTestFlag(selectedLabTest.wbc, 4.0, 10.0);
                              return (
                                <span className={`inline-flex items-center px-2 py-1 rounded text-xs ${flag.bgColor} ${flag.color}`}>
                                  {flag.flag === 'Normal' ? '✓' : flag.flag === 'High' ? '↑' : '↓'} {flag.flag}
                                </span>
                              );
                            })()}
                          </td>
                        </tr>
                        <tr className="hover:bg-gray-50">
                          <td className="px-4 py-3 border font-medium">Hemoglobin</td>
                          <td className="px-4 py-3 border font-semibold">
                            {selectedLabTest.hemoglobin?.toFixed(1) || 'N/A'} <span className="text-xs text-gray-500">g/dL</span>
                          </td>
                          <td className="px-4 py-3 border">{selectedLabTest.hemoglobin_unit || 'g/dL'}</td>
                          <td className="px-4 py-3 border">12.0-16.0</td>
                          <td className="px-4 py-3 border">
                            {(() => {
                              const flag = getLabTestFlag(selectedLabTest.hemoglobin, 12.0, 16.0);
                              return (
                                <span className={`inline-flex items-center px-2 py-1 rounded text-xs ${flag.bgColor} ${flag.color}`}>
                                  {flag.flag === 'Normal' ? '✓' : flag.flag === 'High' ? '↑' : '↓'} {flag.flag}
                                </span>
                              );
                            })()}
                          </td>
                        </tr>
                        <tr className="hover:bg-gray-50">
                          <td className="px-4 py-3 border font-medium">Neutrophils</td>
                          <td className="px-4 py-3 border font-semibold">
                            {selectedLabTest.neutrophils?.toFixed(1) || 'N/A'} <span className="text-xs text-gray-500">×10³/L</span>
                          </td>
                          <td className="px-4 py-3 border">{selectedLabTest.neutrophils_unit || '×10³/L'}</td>
                          <td className="px-4 py-3 border">1.5-7.0</td>
                          <td className="px-4 py-3 border">
                            {(() => {
                              const flag = getLabTestFlag(selectedLabTest.neutrophils, 1.5, 7.0);
                              return (
                                <span className={`inline-flex items-center px-2 py-1 rounded text-xs ${flag.bgColor} ${flag.color}`}>
                                  {flag.flag === 'Normal' ? '✓' : flag.flag === 'High' ? '↑' : '↓'} {flag.flag}
                                </span>
                              );
                            })()}
                          </td>
                        </tr>
                        <tr className="hover:bg-gray-50">
                          <td className="px-4 py-3 border font-medium">Lymphocytes</td>
                          <td className="px-4 py-3 border font-semibold">
                            {selectedLabTest.lymphocytes?.toFixed(1) || 'N/A'} <span className="text-xs text-gray-500">×10³/L</span>
                          </td>
                          <td className="px-4 py-3 border">{selectedLabTest.lymphocytes_unit || '×10³/L'}</td>
                          <td className="px-4 py-3 border">1.0-3.0</td>
                          <td className="px-4 py-3 border">
                            {(() => {
                              const flag = getLabTestFlag(selectedLabTest.lymphocytes, 1.0, 3.0);
                              return (
                                <span className={`inline-flex items-center px-2 py-1 rounded text-xs ${flag.bgColor} ${flag.color}`}>
                                  {flag.flag === 'Normal' ? '✓' : flag.flag === 'High' ? '↑' : '↓'} {flag.flag}
                                </span>
                              );
                            })()}
                          </td>
                        </tr>
                        <tr className="hover:bg-gray-50">
                          <td className="px-4 py-3 border font-medium">Platelets</td>
                          <td className="px-4 py-3 border font-semibold">
                            {selectedLabTest.platelets || 'N/A'} <span className="text-xs text-gray-500">×10³/μL</span>
                          </td>
                          <td className="px-4 py-3 border">{selectedLabTest.platelets_unit || '×10³/μL'}</td>
                          <td className="px-4 py-3 border">150-400</td>
                          <td className="px-4 py-3 border">
                            {(() => {
                              const flag = getLabTestFlag(selectedLabTest.platelets, 150, 400);
                              return (
                                <span className={`inline-flex items-center px-2 py-1 rounded text-xs ${flag.bgColor} ${flag.color}`}>
                                  {flag.flag === 'Normal' ? '✓' : flag.flag === 'High' ? '↑' : '↓'} {flag.flag}
                                </span>
                              );
                            })()}
                          </td>
                        </tr>
                        <tr className="bg-gray-50 hover:bg-gray-100">
                          <td className="px-4 py-3 border font-medium">
                            <span>NLR</span>
                            <span className="ml-2 text-xs text-gray-500">(Neutrophils / Lymphocytes ratio)</span>
                          </td>
                          <td className="px-4 py-3 border font-semibold">{selectedLabTest.nlr?.toFixed(2) || 'N/A'}</td>
                          <td className="px-4 py-3 border"></td>
                          <td className="px-4 py-3 border font-medium text-orange-600">&lt; 3.0</td>
                          <td className="px-4 py-3 border">
                            {(() => {
                              const flag = selectedLabTest.nlr ? getLabTestFlag(selectedLabTest.nlr, 0, 3.0) : { flag: 'N/A', color: 'text-gray-500', bgColor: 'bg-gray-100' };
                              return (
                                <span className={`inline-flex items-center px-2 py-1 rounded text-xs ${flag.bgColor} ${flag.color}`}>
                                  {flag.flag === 'Normal' ? '✓' : flag.flag === 'High' ? '↑' : ''} {flag.flag}
                                </span>
                              );
                            })()}
                          </td>
                        </tr>
                        <tr className="bg-gray-50 hover:bg-gray-100">
                          <td className="px-4 py-3 border font-medium">CRP</td>
                          <td className="px-4 py-3 border font-semibold">{selectedLabTest.crp?.toFixed(1) || 'N/A'}</td>
                          <td className="px-4 py-3 border">{selectedLabTest.crp_unit || 'mg/L'}</td>
                          <td className="px-4 py-3 border font-medium text-orange-600">&lt; 5.0</td>
                          <td className="px-4 py-3 border">
                            {(() => {
                              const flag = selectedLabTest.crp ? getLabTestFlag(selectedLabTest.crp, 0, 5.0) : { flag: 'N/A', color: 'text-gray-500', bgColor: 'bg-gray-100' };
                              return (
                                <span className={`inline-flex items-center px-2 py-1 rounded text-xs ${flag.bgColor} ${flag.color}`}>
                                  {flag.flag === 'Normal' ? '✓' : flag.flag === 'High' ? '↑' : ''} {flag.flag}
                                </span>
                              );
                            })()}
                          </td>
                        </tr>
                        <tr className="hover:bg-gray-50">
                          <td className="px-4 py-3 border font-medium">LDH</td>
                          <td className="px-4 py-3 border font-semibold">{selectedLabTest.ldh || 'N/A'}</td>
                          <td className="px-4 py-3 border">{selectedLabTest.ldh_unit || 'U/L'}</td>
                          <td className="px-4 py-3 border">120-250</td>
                          <td className="px-4 py-3 border">
                            {(() => {
                              const flag = getLabTestFlag(selectedLabTest.ldh, 120, 250);
                              return (
                                <span className={`inline-flex items-center px-2 py-1 rounded text-xs ${flag.bgColor} ${flag.color}`}>
                                  {flag.flag === 'Normal' ? '✓' : flag.flag === 'High' ? '↑' : '↓'} {flag.flag}
                                </span>
                              );
                            })()}
                          </td>
                        </tr>
                        <tr className="hover:bg-gray-50">
                          <td className="px-4 py-3 border font-medium">Albumin</td>
                          <td className="px-4 py-3 border font-semibold">{selectedLabTest.albumin?.toFixed(1) || 'N/A'}</td>
                          <td className="px-4 py-3 border">{selectedLabTest.albumin_unit || 'g/dL'}</td>
                          <td className="px-4 py-3 border">3.5-5.5</td>
                          <td className="px-4 py-3 border">
                            {(() => {
                              const flag = getLabTestFlag(selectedLabTest.albumin, 3.5, 5.5);
                              return (
                                <span className={`inline-flex items-center px-2 py-1 rounded text-xs ${flag.bgColor} ${flag.color}`}>
                                  {flag.flag === 'Normal' ? '✓' : flag.flag === 'High' ? '↑' : '↓'} {flag.flag}
                                </span>
                              );
                            })()}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                    <p className="text-sm text-gray-700">
                      <strong>Tip:</strong> 비정상 결과는 색상 플래그로 표시되어 쉽게 식별할 수 있습니다.
                    </p>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card className="shadow-md">
                <CardContent className="py-12 text-center text-gray-500">
                  <FileText className="mx-auto h-12 w-12 mb-4 text-gray-400" />
                  <p className="text-lg font-semibold mb-2">검사 결과를 선택해주세요</p>
                  <p className="text-sm">검사 요청 탭에서 검사를 선택하면 상세 결과를 확인할 수 있습니다</p>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* RNA Results Tab */}
          <TabsContent value="rna-results">
            {rnaTests.length > 0 ? (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left: RNA Test List */}
                <Card className="shadow-md">
                  <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50 border-b">
                    <CardTitle className="flex items-center gap-2">
                      <Dna className="h-5 w-5 text-purple-600" />
                      RNA 검사 목록
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-6">
                    <div className="space-y-2 max-h-[600px] overflow-y-auto">
                      {rnaTests.map((test) => (
                        <div
                          key={test.id}
                          className={`cursor-pointer rounded-lg border p-3 transition-all ${
                            selectedRNATest?.id === test.id
                              ? 'bg-purple-50 border-purple-300'
                              : 'border-gray-200 hover:bg-gray-50'
                          }`}
                          onClick={() => setSelectedRNATest(test)}
                        >
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="font-semibold">{test.patient_name}</p>
                              <p className="text-xs text-gray-500">{test.accession_number}</p>
                            </div>
                            {selectedRNATest?.id === test.id && (
                              <Badge className="bg-purple-600 text-white">선택됨</Badge>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* Middle: Gene Expression Table */}
                <Card className="shadow-md lg:col-span-1">
                  <CardHeader className="border-b bg-gradient-to-r from-purple-50 to-indigo-50">
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="flex items-center gap-2 mb-2">
                          <Dna className="h-5 w-5 text-purple-600" />
                          <CardTitle className="text-lg font-bold text-gray-900">유전자 발현값</CardTitle>
                        </div>
                        {(selectedRNATest || rnaTests[0]) && (
                          <p className="text-sm text-gray-500">
                            Patient: {(selectedRNATest || rnaTests[0]).patient_name} ({(selectedRNATest || rnaTests[0]).patient_id})
                          </p>
                        )}
                      </div>
                      <Button 
                        onClick={handlePCRPredict} 
                        disabled={predictingPCR || !selectedRNATest}
                        className="bg-purple-600 hover:bg-purple-700"
                        size="sm"
                      >
                        <Brain className="mr-2 h-4 w-4" />
                        {predictingPCR ? '예측 중...' : 'pCR 예측'}
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent className="pt-6">
                    <div className="max-h-[600px] overflow-y-auto">
                      <table className="w-full">
                        <thead className="bg-gray-50 sticky top-0">
                          <tr>
                            <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700">유전자명</th>
                            <th className="px-3 py-2 text-right text-xs font-semibold text-gray-700">발현값</th>
                            <th className="px-3 py-2 text-left text-xs font-semibold text-gray-700">Pathway</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                          {GENE_NAMES.map((gene) => {
                            const test = selectedRNATest || rnaTests[0];
                            const value = test?.[gene];
                            const pathway = GENE_PATHWAYS[gene] || '기타';
                            return (
                              <tr key={gene} className="hover:bg-gray-50">
                                <td className="px-3 py-2 font-mono text-xs font-medium text-purple-700">{gene}</td>
                                <td className="px-3 py-2 text-right font-semibold text-gray-900 text-xs">
                                  {value !== null && value !== undefined ? value.toFixed(3) : 'N/A'}
                                </td>
                                <td className="px-3 py-2 text-xs text-gray-600">{pathway}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>

                {/* Right: pCR Prediction Results */}
                <div className="space-y-6 lg:col-span-1">
                  {pcrPrediction ? (
                    <>
                      <Card className="border-2 border-green-500 shadow-md">
                        <CardHeader className="bg-green-50 border-b">
                          <CardTitle className="text-lg font-bold text-green-800">pCR 예측 결과</CardTitle>
                        </CardHeader>
                        <CardContent className="pt-6">
                          <div className="text-center">
                            <p className="text-sm text-gray-600 mb-2">예측 확률</p>
                            <p className="text-5xl font-bold text-green-600 mb-4">
                              {(pcrPrediction.probability * 100).toFixed(1)}%
                            </p>
                            <p className="text-xl font-semibold">
                              {pcrPrediction.prediction === 'Positive' ? (
                                <span className="text-green-600">✓ 양성 (Positive)</span>
                              ) : (
                                <span className="text-red-600">✗ 음성 (Negative)</span>
                              )}
                            </p>
                          </div>
                        </CardContent>
                      </Card>

                      <Card className="shadow-md">
                        <CardHeader className="bg-indigo-50 border-b">
                          <CardTitle className="text-lg font-bold text-indigo-800">AI 맞춤 치료 제안</CardTitle>
                        </CardHeader>
                        <CardContent className="pt-4">
                          {pcrPrediction.probability >= 0.342 ? (
                            <div className="space-y-3 text-sm">
                              <div className="flex items-start gap-2">
                                <span className="text-lg">📋</span>
                                <div>
                                  <p className="font-semibold">HER2 양성 특성</p>
                                  <p className="text-gray-600">• Trastuzumab/Pertuzumab 표적치료 권장</p>
                                </div>
                              </div>
                              <div className="flex items-start gap-2">
                                <span className="text-lg">📋</span>
                                <div>
                                  <p className="font-semibold">높은 면역 활성</p>
                                  <p className="text-gray-600">• 면역관문억제제 병용 고려 가능</p>
                                </div>
                              </div>
                              <div className="flex items-start gap-2">
                                <span className="text-lg">📋</span>
                                <div>
                                  <p className="font-semibold">빠른 세포 증식</p>
                                  <p className="text-gray-600">• 세포독성 항암제 반응성 우수 예상</p>
                                </div>
                              </div>
                            </div>
                          ) : (
                            <div className="space-y-3 text-sm">
                              <div className="flex items-start gap-2">
                                <span className="text-lg">📋</span>
                                <div>
                                  <p className="font-semibold">관찰 요망</p>
                                  <p className="text-gray-600">• 표준 프로토콜 준수<br/>• 정밀 추적 검사 권장</p>
                                </div>
                              </div>
                            </div>
                          )}
                        </CardContent>
                      </Card>

                      {pcrPrediction.image && (
                        <Card className="shadow-md">
                          <CardHeader className="bg-purple-50 border-b">
                            <CardTitle className="text-lg font-bold text-purple-800">AI 임상 리포트</CardTitle>
                          </CardHeader>
                          <CardContent className="pt-6">
                            <div 
                              className="cursor-pointer hover:opacity-90 transition-opacity"
                              onClick={() => setShowReportModal(true)}
                            >
                              <img 
                                src={`data:image/png;base64,${pcrPrediction.image}`}
                                alt="pCR Clinical Report"
                                className="w-full rounded-lg shadow-lg"
                              />
                              <p className="text-xs text-center text-gray-500 mt-2">클릭하여 확대</p>
                            </div>
                          </CardContent>
                        </Card>
                      )}
                    </>
                  ) : (
                    <Card className="shadow-md">
                      <CardContent className="py-12 text-center text-gray-500">
                        <Brain className="mx-auto h-12 w-12 mb-4 text-gray-400" />
                        <p className="text-lg font-semibold mb-2">예측 결과 없음</p>
                        <p className="text-sm">RNA 검사를 선택하고 "pCR 예측" 버튼을 클릭하세요</p>
                      </CardContent>
                    </Card>
                  )}
                </div>
              </div>
            ) : (
              <Card className="shadow-md">
                <CardContent className="py-12 text-center text-gray-500">
                  <Dna className="mx-auto h-12 w-12 mb-4 text-gray-400" />
                  <p className="text-lg font-semibold mb-2">RNA 검사 결과가 없습니다</p>
                  <p className="text-sm">CSV 파일을 업로드해주세요</p>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>

        {/* Report Image Modal */}
        {showReportModal && pcrPrediction && (
          <div 
            className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75 p-4"
            onClick={() => setShowReportModal(false)}
          >
            <div className="relative max-w-7xl max-h-[95vh] overflow-auto">
              <button
                onClick={() => setShowReportModal(false)}
                className="absolute top-4 right-4 z-10 rounded-full bg-white p-2 shadow-lg hover:bg-gray-100"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
              <img 
                src={`data:image/png;base64,${pcrPrediction.image}`}
                alt="pCR Clinical Report - Full Size"
                className="w-full h-auto rounded-lg shadow-2xl"
                onClick={(e) => e.stopPropagation()}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
