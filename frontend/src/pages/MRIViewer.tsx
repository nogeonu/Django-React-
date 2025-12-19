import { useState, useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, User, Activity, Scan, Eye, EyeOff } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface PatientInfo {
  patient_id: string;
  age?: number;
  tumor_subtype?: string;
  scanner_manufacturer?: string;
}

interface SeriesInfo {
  filename: string;
  index: number;
}

interface PatientDetailInfo {
  patient_id: string;
  patient_info: {
    clinical_data: {
      age: number;
      menopausal_status: string;
      breast_density: string;
    };
    primary_lesion: {
      pcr: number;
      tumor_subtype: string;
    };
    imaging_data: {
      scanner_manufacturer: string;
      scanner_model: string;
      field_strength: number;
    };
  };
  series: SeriesInfo[];
  has_segmentation: boolean;
  volume_shape: number[];
  num_slices: number;
}

const API_BASE_URL = "http://localhost:5000/api/mri";

export default function MRIViewer() {
  const [patients, setPatients] = useState<PatientInfo[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<string | null>(null);
  const [patientDetail, setPatientDetail] = useState<PatientDetailInfo | null>(null);
  const [currentSlice, setCurrentSlice] = useState(0);
  const [currentSeries, setCurrentSeries] = useState(0);
  const [sliceImage, setSliceImage] = useState<string | null>(null);
  const [showSegmentation, setShowSegmentation] = useState(false);
  const [loading, setLoading] = useState(false);
  const [imageLoading, setImageLoading] = useState(false);
  const [axis, setAxis] = useState<"axial" | "sagittal" | "coronal">("axial");
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { toast } = useToast();

  // 환자 목록 로드
  useEffect(() => {
    fetchPatients();
  }, []);

  // 환자 선택 시 상세 정보 로드
  useEffect(() => {
    if (selectedPatient) {
      fetchPatientDetail(selectedPatient);
    }
  }, [selectedPatient]);

  // 슬라이스 이미지 로드
  useEffect(() => {
    if (selectedPatient && patientDetail) {
      fetchSliceImage();
    }
  }, [selectedPatient, currentSlice, currentSeries, axis, showSegmentation]);

  const fetchPatients = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/patients/`);
      const data = await response.json();
      if (data.success) {
        setPatients(data.patients);
        if (data.patients.length > 0) {
          setSelectedPatient(data.patients[0].patient_id);
        }
      }
    } catch (error) {
      toast({
        title: "오류",
        description: "환자 목록을 불러오는데 실패했습니다.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const fetchPatientDetail = async (patientId: string) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/patients/${patientId}/`);
      const data = await response.json();
      if (data.success) {
        setPatientDetail(data);
        setCurrentSlice(Math.floor(data.num_slices / 2)); // 중간 슬라이스로 시작
        setCurrentSeries(0);
      }
    } catch (error) {
      toast({
        title: "오류",
        description: "환자 정보를 불러오는데 실패했습니다.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const fetchSliceImage = async () => {
    if (!selectedPatient) return;

    setImageLoading(true);
    try {
      const url = `${API_BASE_URL}/patients/${selectedPatient}/slice/?series=${currentSeries}&slice=${currentSlice}&axis=${axis}&segmentation=${showSegmentation}`;
      const response = await fetch(url);
      const data = await response.json();
      if (data.success) {
        setSliceImage(data.image);
      }
    } catch (error) {
      console.error("Failed to fetch slice image:", error);
    } finally {
      setImageLoading(false);
    }
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    if (!patientDetail) return;

    const delta = e.deltaY > 0 ? 1 : -1;
    const newSlice = Math.max(0, Math.min(patientDetail.num_slices - 1, currentSlice + delta));
    setCurrentSlice(newSlice);
  };

  const handleSliderChange = (value: number[]) => {
    setCurrentSlice(value[0]);
  };

  if (loading && !patientDetail) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">유방 MRI 뷰어</h1>
          <p className="text-gray-500 mt-1">3D 의료 영상 분석 및 세그멘테이션</p>
        </div>
        <Badge variant="outline" className="text-lg px-4 py-2">
          <Activity className="h-4 w-4 mr-2" />
          실시간 분석
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* 좌측: 환자 정보 및 컨트롤 */}
        <div className="lg:col-span-1 space-y-4">
          {/* 환자 선택 */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <User className="h-5 w-5" />
                환자 선택
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Select value={selectedPatient || ""} onValueChange={setSelectedPatient}>
                <SelectTrigger>
                  <SelectValue placeholder="환자를 선택하세요" />
                </SelectTrigger>
                <SelectContent>
                  {patients.map((patient) => (
                    <SelectItem key={patient.patient_id} value={patient.patient_id}>
                      {patient.patient_id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </CardContent>
          </Card>

          {/* 환자 정보 */}
          {patientDetail && (
            <Card>
              <CardHeader>
                <CardTitle>환자 정보</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <Label className="text-xs text-gray-500">환자 ID</Label>
                  <p className="font-medium">{patientDetail.patient_id}</p>
                </div>
                <div>
                  <Label className="text-xs text-gray-500">나이</Label>
                  <p className="font-medium">{patientDetail.patient_info.clinical_data.age}세</p>
                </div>
                <div>
                  <Label className="text-xs text-gray-500">종양 유형</Label>
                  <p className="font-medium">
                    {patientDetail.patient_info.primary_lesion.tumor_subtype}
                  </p>
                </div>
                <div>
                  <Label className="text-xs text-gray-500">폐경 상태</Label>
                  <p className="font-medium">
                    {patientDetail.patient_info.clinical_data.menopausal_status}
                  </p>
                </div>
                <div>
                  <Label className="text-xs text-gray-500">스캐너</Label>
                  <p className="font-medium text-sm">
                    {patientDetail.patient_info.imaging_data.scanner_manufacturer}{" "}
                    {patientDetail.patient_info.imaging_data.scanner_model}
                  </p>
                </div>
                <div>
                  <Label className="text-xs text-gray-500">자기장 세기</Label>
                  <p className="font-medium">
                    {patientDetail.patient_info.imaging_data.field_strength}T
                  </p>
                </div>
              </CardContent>
            </Card>
          )}

          {/* 뷰어 컨트롤 */}
          {patientDetail && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Scan className="h-5 w-5" />
                  뷰어 설정
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* 시리즈 선택 */}
                <div>
                  <Label>시퀀스 선택</Label>
                  <Select
                    value={currentSeries.toString()}
                    onValueChange={(val) => setCurrentSeries(parseInt(val))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {patientDetail.series.map((series) => (
                        <SelectItem key={series.index} value={series.index.toString()}>
                          {series.filename}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* 축 선택 */}
                <div>
                  <Label>단면 방향</Label>
                  <Tabs value={axis} onValueChange={(val) => setAxis(val as any)}>
                    <TabsList className="grid w-full grid-cols-3">
                      <TabsTrigger value="axial">Axial</TabsTrigger>
                      <TabsTrigger value="sagittal">Sagittal</TabsTrigger>
                      <TabsTrigger value="coronal">Coronal</TabsTrigger>
                    </TabsList>
                  </Tabs>
                </div>

                {/* 세그멘테이션 토글 */}
                {patientDetail.has_segmentation && (
                  <div className="flex items-center justify-between">
                    <Label htmlFor="segmentation" className="flex items-center gap-2">
                      {showSegmentation ? (
                        <Eye className="h-4 w-4" />
                      ) : (
                        <EyeOff className="h-4 w-4" />
                      )}
                      세그멘테이션 표시
                    </Label>
                    <Switch
                      id="segmentation"
                      checked={showSegmentation}
                      onCheckedChange={setShowSegmentation}
                    />
                  </div>
                )}

                {/* 볼륨 정보 */}
                <div className="pt-4 border-t">
                  <Label className="text-xs text-gray-500">볼륨 크기</Label>
                  <p className="font-mono text-sm">
                    {patientDetail.volume_shape.join(" × ")}
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* 우측: 이미지 뷰어 */}
        <div className="lg:col-span-3">
          <Card className="h-full">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>MRI 이미지</CardTitle>
                {patientDetail && (
                  <Badge variant="secondary">
                    슬라이스 {currentSlice + 1} / {patientDetail.num_slices}
                  </Badge>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {patientDetail ? (
                <div className="space-y-4">
                  {/* 이미지 뷰어 */}
                  <div
                    className="relative bg-black rounded-lg overflow-hidden"
                    style={{ height: "600px" }}
                    onWheel={handleWheel}
                  >
                    {imageLoading && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
                        <Loader2 className="h-8 w-8 animate-spin text-white" />
                      </div>
                    )}
                    {sliceImage ? (
                      <img
                        src={sliceImage}
                        alt={`Slice ${currentSlice}`}
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <div className="flex items-center justify-center h-full text-white">
                        이미지를 불러오는 중...
                      </div>
                    )}
                    <canvas ref={canvasRef} className="hidden" />
                  </div>

                  {/* 슬라이스 슬라이더 */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label>슬라이스 위치</Label>
                      <span className="text-sm text-gray-500">
                        마우스 휠로 이동 가능
                      </span>
                    </div>
                    <Slider
                      value={[currentSlice]}
                      onValueChange={handleSliderChange}
                      max={patientDetail.num_slices - 1}
                      step={1}
                      className="w-full"
                    />
                  </div>

                  {/* 네비게이션 버튼 */}
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      onClick={() => setCurrentSlice(Math.max(0, currentSlice - 10))}
                      disabled={currentSlice === 0}
                    >
                      -10
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => setCurrentSlice(Math.max(0, currentSlice - 1))}
                      disabled={currentSlice === 0}
                    >
                      이전
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() =>
                        setCurrentSlice(
                          Math.min(patientDetail.num_slices - 1, currentSlice + 1)
                        )
                      }
                      disabled={currentSlice === patientDetail.num_slices - 1}
                    >
                      다음
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() =>
                        setCurrentSlice(
                          Math.min(patientDetail.num_slices - 1, currentSlice + 10)
                        )
                      }
                      disabled={currentSlice === patientDetail.num_slices - 1}
                    >
                      +10
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center h-96 text-gray-500">
                  환자를 선택하세요
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

