import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Loader2, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { apiRequest } from '@/lib/api';

interface PredictionResult {
  patient_id: number;
  prediction: 'YES' | 'NO';
  probability: number;
  risk_level: '낮음' | '중간' | '높음';
  risk_message: string;
  external_db_saved: boolean;
  symptoms: Record<string, any>;
}

export default function LungCancerPrediction() {
  const [formData, setFormData] = useState({
    name: '',
    gender: '',
    age: '',
    smoking: '',
    yellow_fingers: '',
    anxiety: '',
    peer_pressure: '',
    chronic_disease: '',
    fatigue: '',
    allergy: '',
    wheezing: '',
    alcohol_consuming: '',
    coughing: '',
    shortness_of_breath: '',
    swallowing_difficulty: '',
    chest_pain: '',
  });

  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [patientSearchResults, setPatientSearchResults] = useState<any[]>([]);
  const [showPatientSearch, setShowPatientSearch] = useState(false);
  const [searchingPatients, setSearchingPatients] = useState(false);
  const { toast } = useToast();

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    
    // 이름 입력 시 환자 검색
    if (field === 'name' && value.length > 1) {
      searchPatients(value);
    } else if (field === 'name' && value.length === 0) {
      setPatientSearchResults([]);
      setShowPatientSearch(false);
    }
  };

  const searchPatients = async (query: string) => {
    setSearchingPatients(true);
    try {
      const response = await apiRequest('GET', `/api/lung_cancer/medical-records/search_patients/?query=${encodeURIComponent(query)}`);
      setPatientSearchResults(response.patients || []);
      setShowPatientSearch(true);
    } catch (error) {
      console.error('환자 검색 오류:', error);
      setPatientSearchResults([]);
    } finally {
      setSearchingPatients(false);
    }
  };

  const selectPatient = (patient: any) => {
    setFormData(prev => ({
      ...prev,
      name: patient.name,
      gender: patient.gender === '남성' || patient.gender === 'M' ? '1' : '0',
      age: patient.age?.toString() || ''
    }));
    setShowPatientSearch(false);
    setPatientSearchResults([]);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      // 나이로부터 생년월일 계산 (대략적인 birth_date 생성)
      const age = parseInt(formData.age);
      const birthYear = new Date().getFullYear() - age;
      const birth_date = `${birthYear}-01-01`; // 대략적인 생년월일
      
      const response = await fetch('/api/lung_cancer/patients/predict/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: formData.name,
          birth_date: birth_date,
          gender: formData.gender === '1' ? 'M' : 'F',
          age: age,
          smoking: formData.smoking === '2',
          yellow_fingers: formData.yellow_fingers === '2',
          anxiety: formData.anxiety === '2',
          peer_pressure: formData.peer_pressure === '2',
          chronic_disease: formData.chronic_disease === '2',
          fatigue: formData.fatigue === '2',
          allergy: formData.allergy === '2',
          wheezing: formData.wheezing === '2',
          alcohol_consuming: formData.alcohol_consuming === '2',
          coughing: formData.coughing === '2',
          shortness_of_breath: formData.shortness_of_breath === '2',
          swallowing_difficulty: formData.swallowing_difficulty === '2',
          chest_pain: formData.chest_pain === '2',
        }),
      });

      if (!response.ok) {
        throw new Error('예측 요청에 실패했습니다.');
      }

      const data = await response.json();
      setResult(data);
      
      toast({
        title: "예측 완료",
        description: "폐암 예측이 성공적으로 완료되었습니다.",
      });
    } catch (error) {
      console.error('Error:', error);
      toast({
        title: "오류 발생",
        description: "예측 중 오류가 발생했습니다. 다시 시도해주세요.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case '높음':
        return 'bg-red-100 text-red-800 border-red-200';
      case '중간':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case '낮음':
        return 'bg-green-100 text-green-800 border-green-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case '높음':
        return <AlertTriangle className="h-4 w-4" />;
      case '중간':
        return <Info className="h-4 w-4" />;
      case '낮음':
        return <CheckCircle className="h-4 w-4" />;
      default:
        return <Info className="h-4 w-4" />;
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">폐암 예측 시스템</h1>
        <p className="text-gray-600 mt-2">
          환자의 증상과 생활 습관을 입력하여 폐암 위험도를 예측합니다.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 입력 폼 */}
        <Card>
          <CardHeader>
            <CardTitle>환자 정보 입력</CardTitle>
            <CardDescription>
              환자의 기본 정보와 증상을 입력해주세요.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="relative">
                  <Label htmlFor="name">환자명 (선택사항)</Label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e) => handleInputChange('name', e.target.value)}
                    placeholder="환자명을 입력하세요"
                  />
                  {showPatientSearch && patientSearchResults.length > 0 && (
                    <div className="absolute z-10 mt-1 w-full bg-white border border-gray-300 rounded-md shadow-lg max-h-48 overflow-y-auto">
                      {patientSearchResults.map((patient) => (
                        <div
                          key={patient.id}
                          className="px-3 py-2 hover:bg-blue-50 cursor-pointer border-b last:border-b-0"
                          onClick={() => selectPatient(patient)}
                        >
                          <div className="font-medium">{patient.name}</div>
                          <div className="text-sm text-gray-500">
                            {patient.id} | {patient.gender} | {patient.age}세
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                <div>
                  <Label htmlFor="gender">성별 *</Label>
                  <Select value={formData.gender} onValueChange={(value) => handleInputChange('gender', value)}>
                    <SelectTrigger>
                      <SelectValue placeholder="성별을 선택하세요" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">남성</SelectItem>
                      <SelectItem value="0">여성</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <Label htmlFor="age">나이 *</Label>
                <Input
                  id="age"
                  type="number"
                  value={formData.age}
                  onChange={(e) => handleInputChange('age', e.target.value)}
                  placeholder="나이를 입력하세요"
                  min="0"
                  max="120"
                  required
                />
              </div>

              <div className="space-y-3">
                <h3 className="text-lg font-semibold">증상 및 생활 습관</h3>
                
                {[
                  { key: 'smoking', label: '흡연' },
                  { key: 'yellow_fingers', label: '손가락 변색' },
                  { key: 'anxiety', label: '불안' },
                  { key: 'peer_pressure', label: '또래 압박' },
                  { key: 'chronic_disease', label: '만성 질환' },
                  { key: 'fatigue', label: '피로' },
                  { key: 'allergy', label: '알레르기' },
                  { key: 'wheezing', label: '쌕쌕거림' },
                  { key: 'alcohol_consuming', label: '음주' },
                  { key: 'coughing', label: '기침' },
                  { key: 'shortness_of_breath', label: '호흡 곤란' },
                  { key: 'swallowing_difficulty', label: '삼킴 곤란' },
                  { key: 'chest_pain', label: '가슴 통증' },
                ].map(({ key, label }) => (
                  <div key={key} className="flex items-center justify-between">
                    <Label htmlFor={key} className="flex-1">{label}</Label>
                    <Select 
                      value={formData[key as keyof typeof formData]} 
                      onValueChange={(value) => handleInputChange(key, value)}
                    >
                      <SelectTrigger className="w-32">
                        <SelectValue placeholder="선택" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="1">아니오</SelectItem>
                        <SelectItem value="2">예</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                ))}
              </div>

              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    예측 중...
                  </>
                ) : (
                  '폐암 위험도 예측하기'
                )}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* 결과 표시 */}
        <Card>
          <CardHeader>
            <CardTitle>예측 결과</CardTitle>
            <CardDescription>
              입력된 정보를 바탕으로 한 폐암 위험도 예측 결과입니다.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {result ? (
              <div className="space-y-4">
                <div className="text-center">
                  <div className="text-4xl font-bold mb-2">
                    {result.probability}%
                  </div>
                  <Badge className={`${getRiskColor(result.risk_level)} flex items-center gap-2 w-fit mx-auto`}>
                    {getRiskIcon(result.risk_level)}
                    {result.risk_level} 위험도
                  </Badge>
                </div>

                <Alert className={result.prediction === 'YES' ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}>
                  <AlertDescription>
                    <strong>예측 결과:</strong> {result.prediction === 'YES' ? '폐암 양성' : '폐암 음성'}
                  </AlertDescription>
                </Alert>

                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertDescription>
                    {result.risk_message}
                  </AlertDescription>
                </Alert>

                <div className="text-sm text-gray-600">
                  <p><strong>환자 ID:</strong> {result.patient_id}</p>
                  <p><strong>외부 DB 저장:</strong> {result.external_db_saved ? '성공' : '실패'}</p>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500 py-8">
                <Info className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                <p>환자 정보를 입력하고 예측 버튼을 클릭하세요.</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
