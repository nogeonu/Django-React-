import { useState, useEffect } from "react";
import { useSearchParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Loader2, RefreshCw, Download, ArrowLeft, Search } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/api";

interface Patient {
  id: string;
  name: string;
  age: number;
  gender: string;
}

export default function Visualization3D() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [patientId, setPatientId] = useState<string | null>(searchParams.get("patient_id") || null);
  const [visualizationType, setVisualizationType] = useState<string>("slices"); // 기본값을 slices로 변경
  const [htmlContent, setHtmlContent] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [patientSearchTerm, setPatientSearchTerm] = useState<string>("");
  const [showPatientList, setShowPatientList] = useState<boolean>(false);

  // 환자 목록 불러오기
  const { data: patients = [] } = useQuery({
    queryKey: ["patients"],
    queryFn: async () => {
      const response = await apiRequest("GET", "/api/lung_cancer/patients/");
      return response.results || response || [];
    },
  });

  // 환자 검색 필터링
  const filteredPatients = (patients as Patient[]).filter((patient: Patient) =>
    patient.name.toLowerCase().includes(patientSearchTerm.toLowerCase()) ||
    patient.id.toString().includes(patientSearchTerm)
  );

  // 세션 스토리지에서 데이터 가져오기
  useEffect(() => {
    const storedData = sessionStorage.getItem("3d_visualization_data");
    if (storedData) {
      try {
        const data = JSON.parse(storedData);
        if (data.patientId && !patientId) {
          setPatientId(data.patientId);
        }
      } catch (e) {
        console.error("세션 스토리지 데이터 파싱 실패:", e);
      }
    }
  }, [patientId]);

  // 3D 시각화 생성
  const generateVisualization = async () => {
    if (!patientId) {
      toast({
        title: "환자 ID 필요",
        description: "환자 ID를 입력해주세요.",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    setError(null);
    setHtmlContent("");

    try {
      const response = await fetch(
        `${import.meta.env.VITE_API_URL || "http://34.42.223.43"}/api/medical-images/generate_3d_visualization/?patient_id=${patientId}&visualization_type=${visualizationType}`,
        {
          method: "GET",
          headers: {
            "Content-Type": "text/html",
          },
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: "알 수 없는 오류가 발생했습니다." }));
        throw new Error(errorData.error || `HTTP ${response.status}`);
      }

      const html = await response.text();
      setHtmlContent(html);
      toast({
        title: "3D 시각화 생성 완료",
        description: "3D 시각화가 성공적으로 생성되었습니다.",
      });
    } catch (err: any) {
      const errorMessage = err.message || "3D 시각화 생성 중 오류가 발생했습니다.";
      setError(errorMessage);
      toast({
        title: "3D 시각화 생성 실패",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  // 페이지 외부 클릭 시 환자 목록 닫기
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (!target.closest('.relative')) {
        setShowPatientList(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // 페이지 로드 시 자동 생성
  useEffect(() => {
    if (patientId) {
      generateVisualization();
    }
  }, [patientId, visualizationType]);

  // HTML 다운로드
  const handleDownload = () => {
    if (!htmlContent) {
      toast({
        title: "다운로드 실패",
        description: "다운로드할 내용이 없습니다.",
        variant: "destructive",
      });
      return;
    }

    const blob = new Blob([htmlContent], { type: "text/html" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `3d_visualization_patient_${patientId}_${visualizationType}.html`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);

    toast({
      title: "다운로드 완료",
      description: "3D 시각화 HTML 파일이 다운로드되었습니다.",
    });
  };

  return (
    <div className="container mx-auto p-6">
      <div className="mb-6">
        <Button
          variant="outline"
          onClick={() => navigate(-1)}
          className="mb-4"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          뒤로 가기
        </Button>
        <h1 className="text-3xl font-bold text-gray-800">3D 시각화</h1>
        <p className="text-gray-600 mt-2">
          환자의 의료 이미지를 3D로 시각화합니다.
        </p>
      </div>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle>시각화 설정</CardTitle>
          <CardDescription>
            환자 ID와 시각화 타입을 선택하세요.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="relative">
              <label className="text-sm font-medium mb-2 block">환자 선택</label>
              <div className="relative">
                <Input
                  type="text"
                  value={patientSearchTerm}
                  onChange={(e) => {
                    setPatientSearchTerm(e.target.value);
                    setShowPatientList(true);
                  }}
                  onFocus={() => setShowPatientList(true)}
                  placeholder="환자 이름 또는 ID 검색"
                  className="w-full"
                />
                <Search className="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              </div>
              {showPatientList && filteredPatients.length > 0 && (
                <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-y-auto">
                  {filteredPatients.map((patient: Patient) => (
                    <div
                      key={patient.id}
                      onClick={() => {
                        setPatientId(patient.id.toString());
                        setPatientSearchTerm(patient.name);
                        setShowPatientList(false);
                      }}
                      className="px-4 py-2 hover:bg-gray-100 cursor-pointer border-b border-gray-200 last:border-b-0"
                    >
                      <div className="font-medium">{patient.name}</div>
                      <div className="text-sm text-gray-500">
                        ID: {patient.id} | {patient.age}세 | {patient.gender === 'M' ? '남' : '여'}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              {showPatientList && filteredPatients.length === 0 && patientSearchTerm && (
                <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg p-4 text-center text-gray-500">
                  검색 결과가 없습니다.
                </div>
              )}
              {patientId && (
                <div className="mt-2 text-sm text-gray-600">
                  선택된 환자 ID: <span className="font-semibold">{patientId}</span>
                </div>
              )}
            </div>
            <div className="flex flex-col">
              <label className="text-sm font-medium mb-2 block">시각화 타입</label>
              <Select value={visualizationType} onValueChange={setVisualizationType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="slices">슬라이스 탐색기</SelectItem>
                  <SelectItem value="voxel">복셀 기반 (점 마커)</SelectItem>
                  <SelectItem value="mesh">메쉬 기반</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col">
              <label className="text-sm font-medium mb-2 block opacity-0">시각화 생성</label>
              <Button
                onClick={generateVisualization}
                disabled={loading || !patientId}
                className="w-full"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    생성 중...
                  </>
                ) : (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2" />
                    시각화 생성
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="mb-6 border-red-500">
          <CardContent className="pt-6">
            <div className="text-red-600">
              <strong>오류:</strong> {error}
            </div>
          </CardContent>
        </Card>
      )}

      {htmlContent && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>3D 시각화 결과</CardTitle>
                <CardDescription>
                  환자 ID: {patientId} | 타입: {visualizationType}
                </CardDescription>
              </div>
              <Button onClick={handleDownload} variant="outline">
                <Download className="w-4 h-4 mr-2" />
                HTML 다운로드
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="border rounded-lg overflow-hidden" style={{ minHeight: "800px" }}>
              <iframe
                srcDoc={htmlContent}
                className="w-full h-full"
                style={{ minHeight: "800px", border: "none" }}
                title="3D Visualization"
              />
            </div>
          </CardContent>
        </Card>
      )}

      {loading && !htmlContent && (
        <Card>
          <CardContent className="pt-12 pb-12">
            <div className="flex flex-col items-center justify-center">
              <Loader2 className="w-12 h-12 animate-spin text-blue-600 mb-4" />
              <p className="text-gray-600">3D 시각화를 생성하는 중입니다...</p>
              <p className="text-sm text-gray-500 mt-2">
                이미지가 많을 경우 시간이 걸릴 수 있습니다.
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

