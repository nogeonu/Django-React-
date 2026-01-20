import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { 
  Brain, 
  Upload, 
  Dna,
  Activity,
  FileText,
  Loader2
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import {
  uploadLabTestCsvApi,
  uploadRNATestCsvApi,
  predictPCRApi,
  getRNATestsApi,
} from '@/lib/api';

interface RNATest {
  id: number;
  accession_number: string;
  patient_name: string;
  patient_id: string;
  patient_age: number;
  patient_gender: string;
  test_date: string;
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

export default function LaboratoryAIAnalysis() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState('upload');
  const [rnaTests, setRNATests] = useState<RNATest[]>([]);
  const [selectedRNATest, setSelectedRNATest] = useState<RNATest | null>(null);
  const [uploading, setUploading] = useState(false);
  const [pcrPrediction, setPcrPrediction] = useState<any>(null);
  const [predictingPCR, setPredictingPCR] = useState(false);
  const [showReportModal, setShowReportModal] = useState(false);

  useEffect(() => {
    loadRNATests();
  }, []);

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

    setUploading(true);
    try {
      const result = await uploadLabTestCsvApi(file);
      toast({
        title: '업로드 성공',
        description: `${result.created}개 생성, ${result.updated}개 업데이트`,
      });
    } catch (error: any) {
      toast({
        title: '업로드 실패',
        description: error?.response?.data?.error || '파일 업로드 중 오류가 발생했습니다.',
        variant: 'destructive',
      });
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  };

  const handleRNATestUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    try {
      const result = await uploadRNATestCsvApi(file);
      toast({
        title: 'RNA 업로드 성공',
        description: `${result.created}개 생성, ${result.updated}개 업데이트`,
      });
      await loadRNATests();
      
      if (rnaTests.length > 0 || (result.created > 0 || result.updated > 0)) {
        setActiveTab('analysis');
      }
    } catch (error: any) {
      toast({
        title: '업로드 실패',
        description: error?.response?.data?.error || '파일 업로드 중 오류가 발생했습니다.',
        variant: 'destructive',
      });
    } finally {
      setUploading(false);
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

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">AI 분석 시스템</h1>
          <p className="text-muted-foreground mt-1">
            검사 데이터 업로드 및 AI 모델 추론 결과 확인
          </p>
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="mb-4">
          <TabsTrigger value="upload">
            <Upload className="mr-2 h-4 w-4" />
            데이터 업로드
          </TabsTrigger>
          <TabsTrigger value="analysis">
            <Brain className="mr-2 h-4 w-4" />
            AI 분석 ({rnaTests.length})
          </TabsTrigger>
        </TabsList>

        {/* Upload Tab */}
        <TabsContent value="upload">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Lab Test Upload */}
            <Card>
              <CardHeader className="bg-gradient-to-r from-blue-50 to-cyan-50">
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5 text-blue-600" />
                  혈액검사 데이터 업로드
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                <div className="space-y-4">
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                    <FileText className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                    <p className="text-sm text-muted-foreground mb-4">
                      CSV 파일을 업로드하여 혈액검사 데이터를 등록하세요
                    </p>
                    <label htmlFor="lab-upload">
                      <Button 
                        variant="outline" 
                        disabled={uploading} 
                        asChild
                        className="cursor-pointer"
                      >
                        <span>
                          {uploading ? (
                            <>
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                              업로드 중...
                            </>
                          ) : (
                            <>
                              <Upload className="mr-2 h-4 w-4" />
                              CSV 파일 선택
                            </>
                          )}
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
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* RNA Test Upload */}
            <Card>
              <CardHeader className="bg-gradient-to-r from-purple-50 to-pink-50">
                <CardTitle className="flex items-center gap-2">
                  <Dna className="h-5 w-5 text-purple-600" />
                  RNA 검사 데이터 업로드
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-6">
                <div className="space-y-4">
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                    <Dna className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                    <p className="text-sm text-muted-foreground mb-4">
                      CSV 파일을 업로드하여 RNA 검사 데이터를 등록하세요
                    </p>
                    <label htmlFor="rna-upload">
                      <Button 
                        variant="outline" 
                        disabled={uploading} 
                        asChild
                        className="cursor-pointer"
                      >
                        <span>
                          {uploading ? (
                            <>
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                              업로드 중...
                            </>
                          ) : (
                            <>
                              <Upload className="mr-2 h-4 w-4" />
                              CSV 파일 선택
                            </>
                          )}
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
          </div>
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis">
          {rnaTests.length > 0 ? (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left: RNA Test List */}
              <Card>
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
                            <p className="text-xs text-muted-foreground">{test.accession_number}</p>
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
              <Card className="lg:col-span-1">
                <CardHeader className="border-b bg-gradient-to-r from-purple-50 to-indigo-50">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <Dna className="h-5 w-5 text-purple-600" />
                        <CardTitle className="text-lg font-bold">유전자 발현값</CardTitle>
                      </div>
                      {(selectedRNATest || rnaTests[0]) && (
                        <p className="text-sm text-muted-foreground">
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
                      {predictingPCR ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          예측 중...
                        </>
                      ) : (
                        <>
                          <Brain className="mr-2 h-4 w-4" />
                          pCR 예측
                        </>
                      )}
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
                    <Card className="border-2 border-green-500">
                      <CardHeader className="bg-green-50 border-b">
                        <CardTitle className="text-lg font-bold text-green-800">pCR 예측 결과</CardTitle>
                      </CardHeader>
                      <CardContent className="pt-6">
                        <div className="text-center">
                          <p className="text-sm text-muted-foreground mb-2">예측 확률</p>
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

                    <Card>
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
                                <p className="text-muted-foreground">• Trastuzumab/Pertuzumab 표적치료 권장</p>
                              </div>
                            </div>
                            <div className="flex items-start gap-2">
                              <span className="text-lg">📋</span>
                              <div>
                                <p className="font-semibold">높은 면역 활성</p>
                                <p className="text-muted-foreground">• 면역관문억제제 병용 고려 가능</p>
                              </div>
                            </div>
                            <div className="flex items-start gap-2">
                              <span className="text-lg">📋</span>
                              <div>
                                <p className="font-semibold">빠른 세포 증식</p>
                                <p className="text-muted-foreground">• 세포독성 항암제 반응성 우수 예상</p>
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="space-y-3 text-sm">
                            <div className="flex items-start gap-2">
                              <span className="text-lg">📋</span>
                              <div>
                                <p className="font-semibold">관찰 요망</p>
                                <p className="text-muted-foreground">• 표준 프로토콜 준수<br/>• 정밀 추적 검사 권장</p>
                              </div>
                            </div>
                          </div>
                        )}
                      </CardContent>
                    </Card>

                    {pcrPrediction.image && (
                      <Card>
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
                            <p className="text-xs text-center text-muted-foreground mt-2">클릭하여 확대</p>
                          </div>
                        </CardContent>
                      </Card>
                    )}
                  </>
                ) : (
                  <Card>
                    <CardContent className="py-12 text-center text-muted-foreground">
                      <Brain className="mx-auto h-12 w-12 mb-4 text-gray-400" />
                      <p className="text-lg font-semibold mb-2">예측 결과 없음</p>
                      <p className="text-sm">RNA 검사를 선택하고 "pCR 예측" 버튼을 클릭하세요</p>
                    </CardContent>
                  </Card>
                )}
              </div>
            </div>
          ) : (
            <Card>
              <CardContent className="py-12 text-center text-muted-foreground">
                <Dna className="mx-auto h-12 w-12 mb-4 text-gray-400" />
                <p className="text-lg font-semibold mb-2">RNA 검사 데이터가 없습니다</p>
                <p className="text-sm mb-4">데이터 업로드 탭에서 CSV 파일을 업로드해주세요</p>
                <Button 
                  onClick={() => setActiveTab('upload')}
                  variant="outline"
                >
                  데이터 업로드로 이동
                </Button>
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
  );
}
