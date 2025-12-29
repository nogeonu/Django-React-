import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
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
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Loader2,
  User,
  Scan,
  Upload,
  Database,
  Image as ImageIcon,
  ChevronRight,
  ChevronLeft,
  Maximize2,
  Info,
  Settings2,
  Cpu,
  Plus
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/context/AuthContext";
import { getPatientsApi } from "@/lib/api";
import CornerstoneViewer from "@/components/CornerstoneViewer";


interface SystemPatient {
  id: number;
  patient_id: string;
  name: string;
  age?: number;
  gender?: string;
  phone?: string;
}

interface OrthancImage {
  instance_id: string;
  series_id: string;
  study_id: string;
  series_description: string;
  instance_number: string;
  preview_url: string;
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

const API_BASE_URL = "/api/mri";

export default function MRIViewer() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const isRadiologyTech = user?.department === '방사선과'; // 방사선과 = 촬영 담당

  const [systemPatients, setSystemPatients] = useState<SystemPatient[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<string | null>(null);
  const [patientDetail, setPatientDetail] = useState<PatientDetailInfo | null>(null);
  const [currentSlice, setCurrentSlice] = useState(0);
  const [currentSeries, setCurrentSeries] = useState(0);
  const [sliceImage, setSliceImage] = useState<string | null>(null);
  const [showSegmentation, setShowSegmentation] = useState(false);
  const [loading, setLoading] = useState(false);
  const [imageLoading, setImageLoading] = useState(false);
  const [axis, setAxis] = useState<"axial" | "sagittal" | "coronal">("axial");
  const [orthancImages, setOrthancImages] = useState<OrthancImage[]>([]);
  const [showOrthancImages, setShowOrthancImages] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedImage, setSelectedImage] = useState<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  useEffect(() => {
    fetchPatients();
  }, []);

  useEffect(() => {
    if (selectedPatient) {
      fetchPatientDetail(selectedPatient);
      fetchOrthancImages(selectedPatient);
    }
  }, [selectedPatient]);

  useEffect(() => {
    if (selectedPatient && patientDetail) {
      fetchSliceImage();
    }
  }, [selectedPatient, currentSlice, currentSeries, axis, showSegmentation, patientDetail]);

  const fetchPatients = async () => {
    setLoading(true);
    try {
      const systemData = await getPatientsApi({ page_size: 1000 });
      if (systemData.results && systemData.results.length > 0) {
        setSystemPatients(systemData.results);
        try {
          const mriResponse = await fetch(`${API_BASE_URL}/patients/`);
          const mriData = await mriResponse.json();
          if (mriData.success && mriData.patients) {
            // MRI patient data fetched but not used in this view currently
          }
        } catch (mriError) { }
        if (systemData.results.length > 0) {
          setSelectedPatient(systemData.results[0].patient_id);
        }
      } else {
        const response = await fetch(`${API_BASE_URL}/patients/`);
        const data = await response.json();
        if (data.success && data.patients) {
          if (data.patients.length > 0) {
            setSelectedPatient(data.patients[0].patient_id);
          }
        }
      }
    } catch (error) {
      toast({ title: "오류", description: "환자 목록 로드 실패", variant: "destructive" });
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
        setCurrentSlice(Math.floor(data.num_slices / 2));
        setCurrentSeries(0);
      }
    } catch (error) {
      toast({ title: "오류", description: "환자 상세 정보 로드 실패", variant: "destructive" });
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
    if (!patientDetail) return;
    // Don't prevent default if we want to allow page scroll, but here we want to scroll slices
    // e.preventDefault(); 
    const delta = e.deltaY > 0 ? 1 : -1;
    const newSlice = Math.max(0, Math.min(patientDetail.num_slices - 1, currentSlice + delta));
    setCurrentSlice(newSlice);
  };

  const handleDetailView = () => {
    if (orthancImages.length > 0 && orthancImages[selectedImage]) {
      const instanceId = orthancImages[selectedImage].instance_id;
      // 환자 정보를 세션 스토리지에 저장
      if (selectedPatient) {
        sessionStorage.setItem('currentPatientId', selectedPatient);
        const patient = systemPatients.find(p => p.patient_id === selectedPatient);
        if (patient) {
          sessionStorage.setItem('currentPatientName', patient.name);
        }
      }
      // 상세 뷰어로 이동
      navigate(`/dicom-viewer/${instanceId}`);
    }
  };

  const handleOrthancWheel = (e: React.WheelEvent) => {
    if (orthancImages.length === 0) return;
    const delta = e.deltaY > 0 ? 1 : -1;
    const newImage = Math.max(0, Math.min(orthancImages.length - 1, selectedImage + delta));
    setSelectedImage(newImage);
  };

  const fetchOrthancImages = async (patientId: string) => {
    setImageLoading(true);
    try {
      const response = await fetch(`/api/mri/orthanc/patients/${patientId}/`);
      const data = await response.json();
      if (data.success && data.images && data.images.length > 0) {
        setOrthancImages(data.images);
        setShowOrthancImages(true);
        setSelectedImage(0);
      } else {
        setOrthancImages([]);
        setShowOrthancImages(false);
      }
    } catch (error) {
      setOrthancImages([]);
      setShowOrthancImages(false);
    } finally {
      setImageLoading(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    setUploading(true);
    let successCount = 0;
    try {
      for (let i = 0; i < files.length; i++) {
        const formData = new FormData();
        formData.append('file', files[i]);
        if (selectedPatient) formData.append('patient_id', selectedPatient);
        const response = await fetch('/api/mri/orthanc/upload/', { method: 'POST', body: formData });
        if (response.ok) successCount++;
      }
      if (successCount > 0) {
        toast({ title: "업로드 완료", description: `${successCount}개 파일이 저장되었습니다.` });
        if (selectedPatient) fetchOrthancImages(selectedPatient);
      }
    } catch (error) {
      toast({ title: "오류", description: "업로드 중 문제가 발생했습니다.", variant: "destructive" });
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  if (loading && !patientDetail) {
    return (
      <div className="flex flex-col items-center justify-center h-[70vh] gap-4">
        <Loader2 className="h-12 w-12 animate-spin text-blue-600" />
        <p className="text-gray-400 font-bold animate-pulse uppercase tracking-widest text-xs">원격 판독 워크스테이션 로드 중...</p>
      </div>
    );
  }

  return (
    <div className="space-y-8 pb-12">
      {/* Workstation Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="bg-blue-600 p-2 rounded-xl shadow-lg shadow-blue-200">
              <Scan className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-3xl font-black text-gray-900 tracking-tight">영상 판독 워크스테이션</h1>
          </div>
          <p className="text-sm font-medium text-gray-400">유방 MRI 3D 분석 및 Orthanc PACS 연동 시스템</p>
        </div>
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => selectedPatient && fetchOrthancImages(selectedPatient)}
            disabled={!selectedPatient}
            className="rounded-xl border-gray-200 font-bold text-xs h-10 px-4 hover:bg-gray-50"
          >
            <Database className="h-4 w-4 mr-2 text-blue-600" />
            PACS 연동
          </Button>
          <Badge className="bg-emerald-50 text-emerald-600 border-none px-4 py-2 rounded-xl flex items-center gap-2 h-10">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
            <span className="font-bold text-xs uppercase tracking-widest">분석 활성화됨</span>
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Side: Controls & Info (4 cols) */}
        <div className="lg:col-span-4 space-y-6">
          {/* Patient Selector */}
          <Card className="border-none shadow-sm rounded-3xl overflow-hidden bg-white">
            <CardHeader className="bg-gray-50/50 border-b border-gray-100">
              <CardTitle className="text-sm font-black text-gray-900 flex items-center gap-2 tracking-tight uppercase">
                <User className="h-4 w-4 text-blue-600" />
                환자 정보
              </CardTitle>
            </CardHeader>
            <CardContent className="p-6 space-y-6">
              <div className="space-y-2">
                <Label className="text-[10px] font-black text-gray-400 uppercase tracking-widest">대상 환자 선택</Label>
                <Select value={selectedPatient || ""} onValueChange={setSelectedPatient}>
                  <SelectTrigger className="h-11 rounded-xl bg-gray-50 border-none font-bold text-sm focus:ring-2 focus:ring-blue-600/20">
                    <SelectValue placeholder="환자를 선택하세요" />
                  </SelectTrigger>
                  <SelectContent className="rounded-xl border-none shadow-xl">
                    {systemPatients.map((p) => (
                      <SelectItem key={p.id} value={p.patient_id} className="rounded-lg">
                        {p.name} <span className="text-gray-400 ml-1">({p.patient_id})</span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {patientDetail && (
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 p-4 rounded-2xl">
                    <p className="text-[10px] font-black text-gray-400 uppercase mb-1">나이</p>
                    <p className="font-black text-gray-900">{patientDetail.patient_info.clinical_data.age}세</p>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-2xl">
                    <p className="text-[10px] font-black text-gray-400 uppercase mb-1">밀도</p>
                    <p className="font-black text-gray-900">{patientDetail.patient_info.clinical_data.breast_density}</p>
                  </div>
                  <div className="col-span-2 bg-blue-50/50 p-4 rounded-2xl border border-blue-50">
                    <p className="text-[10px] font-black text-blue-600 uppercase mb-1">Tumor Subtype</p>
                    <p className="font-black text-blue-900">{patientDetail.patient_info.primary_lesion.tumor_subtype}</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Imaging Controls */}
          {patientDetail && (
            <Card className="border-none shadow-sm rounded-3xl overflow-hidden bg-white">
              <CardHeader className="bg-gray-50/50 border-b border-gray-100">
                <CardTitle className="text-sm font-black text-gray-900 flex items-center gap-2 tracking-tight uppercase">
                  <Settings2 className="h-4 w-4 text-purple-600" />
                  워크스테이션 제어
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6 space-y-6">
                <div className="space-y-2">
                  <Label className="text-[10px] font-black text-gray-400 uppercase tracking-widest">MRI 시퀀스</Label>
                  <Select value={currentSeries.toString()} onValueChange={(v) => setCurrentSeries(parseInt(v))}>
                    <SelectTrigger className="h-11 rounded-xl bg-gray-50 border-none font-bold text-xs truncate">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="rounded-xl border-none shadow-xl">
                      {patientDetail.series.map((s) => (
                        <SelectItem key={s.index} value={s.index.toString()} className="rounded-lg">
                          {s.filename}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-[10px] font-black text-gray-400 uppercase tracking-widest">단면 방향 (Axis)</Label>
                  <Tabs value={axis} onValueChange={(v) => setAxis(v as any)} className="w-full">
                    <TabsList className="grid w-full grid-cols-3 bg-gray-50 rounded-xl p-1 h-11">
                      <TabsTrigger value="axial" className="rounded-lg font-bold text-[10px] uppercase">Axial</TabsTrigger>
                      <TabsTrigger value="sagittal" className="rounded-lg font-bold text-[10px] uppercase">Sagittal</TabsTrigger>
                      <TabsTrigger value="coronal" className="rounded-lg font-bold text-[10px] uppercase">Coronal</TabsTrigger>
                    </TabsList>
                  </Tabs>
                </div>

                <div className="flex items-center justify-between p-4 bg-emerald-50/30 rounded-2xl border border-emerald-50">
                  <div className="flex flex-col">
                    <Label className="text-[10px] font-black text-emerald-600 uppercase tracking-widest mb-1">AI 병변 분할 (Segmentation)</Label>
                    <p className="text-[9px] font-medium text-emerald-600/70">자동 병변 탐지 활성화</p>
                  </div>
                  <Switch checked={showSegmentation} onCheckedChange={setShowSegmentation} className="data-[state=checked]:bg-emerald-500" />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Upload Card - Only for Radiology department */}
          {user?.department === "방사선과" && (
            <Card className="border-none shadow-sm rounded-3xl overflow-hidden bg-gray-900 text-white relative group">
              <div className="absolute inset-0 bg-blue-600 opacity-0 group-hover:opacity-10 transition-opacity duration-500"></div>
              <CardHeader>
                <CardTitle className="text-sm font-black flex items-center gap-2 tracking-tight uppercase">
                  <Upload className="h-4 w-4 text-blue-400" />
                  데이터 업로드
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-[10px] font-medium text-gray-400 leading-relaxed">
                  DICOM 폴더 또는 NIfTI 파일을 서버로 전송합니다. 전송 후 실시간 3D 변환이 시작됩니다.
                </p>
                <div className="relative">
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    onChange={handleFileUpload}
                    disabled={uploading}
                    className="hidden"
                    id="file-upload-input"
                  />
                  <Button
                    className="w-full h-11 rounded-xl bg-white text-gray-900 hover:bg-gray-100 font-black text-xs gap-2"
                    onClick={() => document.getElementById('file-upload-input')?.click()}
                    disabled={uploading}
                  >
                    {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
                    파일 선택 및 업로드
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Side: Main Viewer (8 cols) */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          <Card className="flex-1 border-none shadow-sm rounded-[2rem] overflow-hidden bg-white flex flex-col">
            <CardHeader className="border-b border-gray-50 flex flex-row items-center justify-between py-4 px-8">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded-2xl bg-gray-900 flex items-center justify-center">
                  <ImageIcon className="w-5 h-5 text-white" />
                </div>
                <div>
                  <CardTitle className="text-lg font-black text-gray-900 tracking-tight">
                    {showOrthancImages ? "PACS 뷰어" : "분석용 뷰어"}
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-[9px] font-bold border-gray-100 text-gray-400 uppercase py-0 px-2 h-4">
                      3D 재구성 완료
                    </Badge>
                    <span className="text-[9px] font-bold text-gray-300">|</span>
                    <span className="text-[9px] font-black text-blue-600 uppercase tracking-widest">
                      {showOrthancImages ? "원본 DICOM" : "처리된 NIfTI"}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="icon"
                  className="rounded-xl h-10 w-10 hover:bg-gray-50"
                  onClick={() => setShowOrthancImages(!showOrthancImages)}
                  title="DICOM/NIfTI 전환"
                >
                  <Cpu className={`w-4 h-4 ${showOrthancImages ? 'text-blue-600' : 'text-gray-400'}`} />
                </Button>
                {showOrthancImages && orthancImages.length > 0 && (
                  <Button 
                    variant="default"
                    size="sm"
                    className="rounded-xl h-10 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium"
                    onClick={handleDetailView}
                    title="자세히 보기"
                  >
                    <Maximize2 className="w-4 h-4 mr-2" />
                    자세히 보기
                  </Button>
                )}
              </div>
            </CardHeader>

            <CardContent className="p-8 flex-1 flex flex-col gap-8">
              {/* Cornerstone3D 뷰어 */}
              {showOrthancImages && orthancImages.length > 0 ? (
                <div className="flex-1 min-h-[500px] bg-gray-950 rounded-[2.5rem] overflow-hidden shadow-inner">
                  <CornerstoneViewer
                    key={`cornerstone-${selectedPatient}-${orthancImages.length}`}
                    instanceIds={orthancImages.map(img => img.instance_id)}
                    currentIndex={selectedImage}
                    onIndexChange={setSelectedImage}
                    showMeasurementTools={!isRadiologyTech}
                  />
                </div>
              ) : (
                <>
                  {/* Main Image Viewport */}
                  <div
                    className="relative flex-1 min-h-[500px] bg-gray-950 rounded-[2.5rem] overflow-hidden shadow-inner group"
                    onWheel={showOrthancImages ? handleOrthancWheel : handleWheel}
                  >
                    <AnimatePresence mode="wait">
                      {imageLoading ? (
                        <motion.div
                          key="loading"
                          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                          className="absolute inset-0 flex items-center justify-center z-20 bg-black/40 backdrop-blur-sm"
                        >
                          <div className="flex flex-col items-center gap-4">
                            <Loader2 className="w-10 h-10 animate-spin text-blue-500" />
                            <span className="text-[10px] font-black text-white/50 uppercase tracking-[0.2em]">데이터 동기화 중</span>
                          </div>
                        </motion.div>
                      ) : null}
                    </AnimatePresence>

                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none p-8">
                      {sliceImage || (showOrthancImages && orthancImages.length > 0) ? (
                        <motion.img
                          key={showOrthancImages ? `orthanc-${selectedImage}` : `slice-${currentSlice}`}
                          initial={{ opacity: 0, scale: 0.98 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.2 }}
                          src={showOrthancImages ? orthancImages[selectedImage].preview_url : (sliceImage || "")}
                          className="max-w-full max-h-full object-contain pointer-events-auto"
                        />
                      ) : (
                        <div className="text-white/20 flex flex-col items-center gap-4">
                          <Info className="w-12 h-12" />
                          <p className="text-xs font-bold uppercase tracking-widest">데이터 스트림 없음</p>
                        </div>
                      )}
                    </div>

                    {/* Overlays */}
                    <div className="absolute top-6 left-6 flex flex-col gap-2 pointer-events-none">
                      <Badge className="bg-black/60 backdrop-blur-md text-white border-none py-1.5 px-4 rounded-xl text-[10px] font-black uppercase tracking-widest w-fit">
                        {showOrthancImages ? `S: ${orthancImages[selectedImage]?.series_description || 'Raw'}` : `Axis: ${axis}`}
                      </Badge>
                      <Badge className="bg-blue-600/80 backdrop-blur-md text-white border-none py-1.5 px-4 rounded-xl text-[10px] font-black uppercase tracking-widest w-fit">
                        {showOrthancImages ? `Inst: ${orthancImages[selectedImage]?.instance_number}` : `Slice: ${currentSlice + 1}`}
                      </Badge>
                    </div>

                    {showOrthancImages && (
                      <div className="absolute top-6 right-6">
                        <Button
                          size="sm"
                          className="rounded-xl bg-white/90 hover:bg-white text-gray-900 font-black text-[10px] uppercase shadow-xl pointer-events-auto"
                          onClick={() => {
                            if (selectedPatient && orthancImages[selectedImage]) {
                              sessionStorage.setItem('currentPatientId', selectedPatient);
                              const p = systemPatients.find(x => x.patient_id === selectedPatient);
                              if (p) sessionStorage.setItem('currentPatientName', p.name);
                              navigate(`/dicom-viewer/${orthancImages[selectedImage].instance_id}`);
                            }
                          }}
                        >
                          자세히 보기 <ChevronRight className="w-3 h-3 ml-1" />
                        </Button>
                      </div>
                    )}

                    {/* Mouse Wheel Hint */}
                    <div className="absolute bottom-6 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
                      <div className="bg-white/10 backdrop-blur-md px-4 py-2 rounded-full flex items-center gap-2 border border-white/5">
                        <div className="w-1 h-3 rounded-full bg-white animate-bounce"></div>
                        <span className="text-[10px] font-black text-white uppercase tracking-widest">휠을 사용하여 탐색</span>
                      </div>
                    </div>
                  </div>

                  {/* Navigation Slider */}
                  <div className="px-4 space-y-4">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest">워크스페이스 내비게이션</p>
                      <div className="flex items-center gap-2">
                        <Button
                          variant="ghost" size="icon" className="h-8 w-8 rounded-lg hover:bg-gray-50"
                          onClick={() => showOrthancImages ? setSelectedImage(Math.max(0, selectedImage - 1)) : setCurrentSlice(Math.max(0, currentSlice - 1))}
                        >
                          <ChevronLeft className="w-4 h-4" />
                        </Button>
                        <span className="text-xs font-black text-gray-900 mx-2">
                          {showOrthancImages ? `${selectedImage + 1} / ${orthancImages.length}` : `${currentSlice + 1} / ${patientDetail?.num_slices || 0}`}
                        </span>
                        <Button
                          variant="ghost" size="icon" className="h-8 w-8 rounded-lg hover:bg-gray-50"
                          onClick={() => showOrthancImages ? setSelectedImage(Math.min(orthancImages.length - 1, selectedImage + 1)) : setCurrentSlice(Math.min((patientDetail?.num_slices || 1) - 1, currentSlice + 1))}
                        >
                          <ChevronRight className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                    <Slider
                      value={[showOrthancImages ? selectedImage : currentSlice]}
                      onValueChange={(v) => showOrthancImages ? setSelectedImage(v[0]) : setCurrentSlice(v[0])}
                      max={showOrthancImages ? (orthancImages.length - 1 || 0) : ((patientDetail?.num_slices || 1) - 1)}
                      step={1}
                      className="w-full"
                    />
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
