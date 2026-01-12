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
  Info,
  Settings2,
  Cpu,
  Plus,
  Maximize2
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
  modality?: string;  // MG, MR, 등
  view_position?: string;  // CC, MLO
  image_laterality?: string;  // L, R
  mammography_view?: string;  // LCC, RCC, LMLO, RMLO
}

interface SeriesInfo {
  filename: string;
  index: number;
}

interface PatientDetailInfo {
  patient_id: string;
  name?: string;
  age?: number;
  gender?: string;
  phone?: string;
  tumor_subtype?: string;
  patient_info?: {
    clinical_data?: {
      age?: number;
      menopausal_status?: string;
      breast_density?: string;
    };
    primary_lesion?: {
      pcr?: number;
      tumor_subtype?: string;
    };
    imaging_data?: {
      scanner_manufacturer?: string;
      scanner_model?: string;
      field_strength?: number;
    };
  };
  series?: SeriesInfo[];
  has_segmentation?: boolean;
  volume_shape?: number[];
  num_slices?: number;
}

const API_BASE_URL = "/api/mri";

export default function MRIViewer() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const isRadiologyTech = user?.department === '방사선과'; // 방사선과 = 촬영 담당

  // 페이지 제목 결정: 방사선과는 "영상 업로드", 영상의학과/외과는 "영상 판독"
  const pageTitle = isRadiologyTech ? '영상 업로드' : '영상 판독';

  const [imageType, setImageType] = useState<'유방촬영술 영상' | '병리 영상' | 'MRI 영상'>('유방촬영술 영상');
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
  const [allOrthancImages, setAllOrthancImages] = useState<OrthancImage[]>([]); // 필터링 전 모든 이미지
  const [showOrthancImages, setShowOrthancImages] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [selectedImage, setSelectedImage] = useState<number>(0);
  const [isDragging, setIsDragging] = useState(false);
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

  // imageType 변경 시 이미지 필터링
  useEffect(() => {
    filterImagesByType();
  }, [imageType, allOrthancImages]);

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
        setCurrentSlice(Math.floor((data.num_slices || 1) / 2));
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
    const newSlice = Math.max(0, Math.min((patientDetail.num_slices || 1) - 1, currentSlice + delta));
    setCurrentSlice(newSlice);
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
      console.log(`[fetchOrthancImages] 환자 ID로 이미지 요청: ${patientId}`);
      // 병렬 요청으로 성능 개선
      const response = await fetch(`/api/mri/orthanc/patients/${patientId}/`, {
        cache: 'no-cache', // 캐시 비활성화 (최신 데이터 확인)
      });
      
      console.log(`[fetchOrthancImages] 응답 상태: ${response.status} ${response.statusText}`);
      const data = await response.json();
      console.log(`[fetchOrthancImages] 응답 데이터:`, data);
      
      if (!response.ok) {
        console.error(`[fetchOrthancImages] API 오류:`, data);
        throw new Error(data.error || `서버 오류 (${response.status})`);
      }
      
      if (data.success && data.images && Array.isArray(data.images)) {
        console.log(`[fetchOrthancImages] 이미지 개수: ${data.images.length}`);
        console.log(`[fetchOrthancImages] 첫 번째 이미지 샘플:`, data.images[0]);
        
        if (data.images.length > 0) {
        setAllOrthancImages(data.images); // 모든 이미지 저장
        // 필터링은 useEffect에서 자동으로 처리됨
        
        // 이미지 프리로딩 (첫 3개 이미지만 먼저 로드)
          const previewUrlsToPreload = data.images.slice(0, Math.min(3, data.images.length))
            .map((img: OrthancImage) => img.preview_url);
          
          // 백그라운드에서 프리로드 (사용자 경험 개선)
          previewUrlsToPreload.forEach((url: string) => {
            const img = new Image();
            img.src = url;
          });
        } else {
          console.warn(`[fetchOrthancImages] 이미지 배열이 비어있음`);
          setAllOrthancImages([]);
          setOrthancImages([]);
          setShowOrthancImages(true); // 이미지 없음 메시지 표시를 위해 true로 설정
        }
      } else {
        console.warn(`[fetchOrthancImages] 응답 형식 오류:`, {
          success: data.success,
          hasImages: !!data.images,
          imagesType: Array.isArray(data.images),
          imagesLength: data.images?.length
        });
        setAllOrthancImages([]);
        setOrthancImages([]);
        setShowOrthancImages(true); // 이미지 없음 메시지 표시를 위해 true로 설정
      }
    } catch (error) {
      console.error('[fetchOrthancImages] Orthanc 이미지 로드 실패:', error);
      setAllOrthancImages([]);
      setOrthancImages([]);
      setShowOrthancImages(true); // 에러 발생 시에도 뷰어 표시
      toast({
        title: "오류",
        description: error instanceof Error ? error.message : "Orthanc 이미지를 불러오는데 실패했습니다.",
        variant: "destructive",
      });
    } finally {
      setImageLoading(false);
    }
  };

  // imageType에 따라 이미지 필터링
  const filterImagesByType = () => {
    console.log(`[filterImagesByType] 전체 이미지 개수: ${allOrthancImages.length}, 선택된 영상 유형: ${imageType || '(없음)'}`);
    
    if (allOrthancImages.length === 0) {
      console.log(`[filterImagesByType] 이미지가 없음 - 빈 배열 설정`);
      setOrthancImages([]);
      // 이미지가 없어도 뷰어는 표시하되 "이미지 없음" 메시지를 보여줌
      setShowOrthancImages(true);
      return;
    }

    // 이미지 모달리티 확인
    const modalities = allOrthancImages.map(img => img.modality).filter(Boolean);
    console.log(`[filterImagesByType] 사용 가능한 모달리티:`, [...new Set(modalities)]);
    console.log(`[filterImagesByType] 첫 번째 이미지 샘플:`, allOrthancImages[0]);

    let filtered: OrthancImage[] = [];

    // imageType이 없거나 빈 문자열이면 모든 이미지 표시
    if (!imageType || imageType.trim() === '') {
      filtered = allOrthancImages;
      console.log(`[filterImagesByType] 영상 유형 미선택 - 전체 이미지 표시: ${filtered.length}개`);
    } else {
    switch (imageType) {
      case '유방촬영술 영상':
        // MG (Mammography) 모달리티만
        filtered = allOrthancImages.filter(img => img.modality === 'MG');
          console.log(`[filterImagesByType] 유방촬영술 필터링 결과: ${filtered.length}개 (전체 ${allOrthancImages.length}개 중)`);
        break;
      case 'MRI 영상':
        // MR (Magnetic Resonance) 모달리티만
        filtered = allOrthancImages.filter(img => img.modality === 'MR');
          console.log(`[filterImagesByType] MRI 필터링 결과: ${filtered.length}개 (전체 ${allOrthancImages.length}개 중)`);
        break;
      case '병리 영상':
        // 병리 영상: SM (Slide Microscopy) 모달리티만
        filtered = allOrthancImages.filter(img => img.modality === 'SM');
        console.log(`[filterImagesByType] 병리 영상 필터링 결과: ${filtered.length}개 (전체 ${allOrthancImages.length}개 중)`);
        console.log(`[filterImagesByType] 병리 영상 모달리티:`, filtered.map(img => img.modality));
        break;
      default:
        filtered = allOrthancImages;
          console.log(`[filterImagesByType] 알 수 없는 영상 유형 "${imageType}" - 전체 이미지 표시: ${filtered.length}개`);
      }
    }

    setOrthancImages(filtered);
    // 필터링 결과와 관계없이 뷰어는 항상 표시 (이미지가 없으면 "이미지 없음" 메시지 표시)
      setShowOrthancImages(true);
    if (filtered.length > 0) {
      setSelectedImage(0);
      console.log(`[filterImagesByType] 이미지 표시 설정 완료: ${filtered.length}개`);
    } else {
      setSelectedImage(0);
      console.log(`[filterImagesByType] 필터링 후 이미지 없음 - "이미지 없음" 메시지 표시`);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!uploading) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (!selectedPatient) {
      toast({ 
        title: "오류", 
        description: "먼저 환자를 선택해주세요.", 
        variant: "destructive" 
      });
      return;
    }

    const items = Array.from(e.dataTransfer.items);
    const files: File[] = [];

    // 드롭된 파일 수집
    for (const item of items) {
      if (item.kind === 'file') {
        const file = item.getAsFile();
        if (file) {
          // DICOM 또는 NIfTI 파일만 허용
          if (file.name.endsWith('.dicom') || 
              file.name.endsWith('.dcm') || 
              file.name.endsWith('.nii') || 
              file.name.endsWith('.nii.gz')) {
            files.push(file);
          }
        }
      }
    }

    if (files.length === 0) {
      toast({ 
        title: "오류", 
        description: "DICOM 또는 NIfTI 파일을 드롭해주세요.", 
        variant: "destructive" 
      });
      return;
    }

    await uploadFiles(files);
  };

  const processFiles = async (files: FileList | File[]) => {
    if (!files || files.length === 0) return;
    if (!selectedPatient) {
      toast({ 
        title: "오류", 
        description: "먼저 환자를 선택해주세요.", 
        variant: "destructive" 
      });
      return;
    }
    
    const fileArray = Array.from(files);
    await uploadFiles(fileArray);
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    await processFiles(files);
  };

  const uploadFiles = async (files: File[]) => {
    if (!selectedPatient) return;
    if (!imageType) {
      toast({ 
        title: "오류", 
        description: "먼저 영상 유형을 선택해주세요.", 
        variant: "destructive" 
      });
      return;
    }
    setUploading(true);
    let successCount = 0;
    let errorMessages: string[] = [];
    
    try {
      for (let i = 0; i < files.length; i++) {
        try {
          const formData = new FormData();
          formData.append('file', files[i]);
          formData.append('patient_id', selectedPatient);
          formData.append('image_type', imageType); // 영상 유형 전달
          
          // 병리 이미지는 별도 엔드포인트 사용
          const uploadUrl = imageType === '병리 영상' 
            ? '/api/mri/pathology/upload/' 
            : '/api/mri/orthanc/upload/';
          
          const response = await fetch(uploadUrl, { 
            method: 'POST', 
            body: formData 
          });
          
          let data;
          try {
            data = await response.json();
          } catch (jsonError) {
            const text = await response.text();
            errorMessages.push(`${files[i].name}: 서버 응답 파싱 실패 (${response.status})`);
            console.error(`❌ 파일 ${i + 1} 응답 파싱 실패:`, text);
            continue;
          }
          
          if (response.ok && data.success) {
            successCount++;
            console.log(`✅ 파일 ${i + 1} 업로드 성공:`, files[i].name);
          } else {
            const errorMsg = data.error || data.message || data.error_type || `파일 ${i + 1} 업로드 실패`;
            const fullErrorMsg = data.traceback 
              ? `${errorMsg}\n\n상세 오류:\n${data.traceback.split('\n').slice(0, 5).join('\n')}`
              : errorMsg;
            errorMessages.push(`${files[i].name}: ${fullErrorMsg}`);
            console.error(`❌ 파일 ${i + 1} 업로드 실패:`, {
              error: errorMsg,
              error_type: data.error_type,
              traceback: data.traceback,
              full_data: data
            });
          }
        } catch (fileError) {
          const errorMsg = fileError instanceof Error ? fileError.message : `파일 ${i + 1} 업로드 중 오류`;
          errorMessages.push(`${files[i].name}: ${errorMsg}`);
          console.error(`❌ 파일 ${i + 1} 업로드 예외:`, fileError);
        }
      }
      
      if (successCount > 0) {
        toast({ 
          title: "업로드 완료", 
          description: `${successCount}개 파일이 저장되었습니다.${errorMessages.length > 0 ? ` (${errorMessages.length}개 실패)` : ''}` 
        });
        if (selectedPatient) fetchOrthancImages(selectedPatient);
      } else {
        toast({ 
          title: "업로드 실패", 
          description: errorMessages.length > 0 
            ? errorMessages.slice(0, 3).join(', ') + (errorMessages.length > 3 ? '...' : '')
            : "모든 파일 업로드에 실패했습니다.",
          variant: "destructive" 
        });
      }
    } catch (error) {
      console.error('업로드 중 예외 발생:', error);
      toast({ 
        title: "오류", 
        description: error instanceof Error ? error.message : "업로드 중 문제가 발생했습니다.", 
        variant: "destructive" 
      });
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
            <h1 className="text-3xl font-black text-gray-900 tracking-tight">{pageTitle}</h1>
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
                <Label className="text-[10px] font-black text-gray-400 uppercase tracking-widest">영상 유형 선택</Label>
                <Select value={imageType} onValueChange={(value) => setImageType(value as '유방촬영술 영상' | '병리 영상' | 'MRI 영상')}>
                  <SelectTrigger className="h-11 rounded-xl bg-gray-50 border-none font-bold text-sm focus:ring-2 focus:ring-blue-600/20">
                    <SelectValue placeholder="영상 유형을 선택하세요" />
                  </SelectTrigger>
                  <SelectContent className="rounded-xl border-none shadow-xl">
                    <SelectItem value="유방촬영술 영상" className="rounded-lg">유방촬영술 영상</SelectItem>
                    <SelectItem value="병리 영상" className="rounded-lg">병리 영상</SelectItem>
                    <SelectItem value="MRI 영상" className="rounded-lg">MRI 영상</SelectItem>
                  </SelectContent>
                </Select>
              </div>
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
                    <p className="font-black text-gray-900">
                      {patientDetail.patient_info?.clinical_data?.age ?? patientDetail.age ?? '-'}세
                    </p>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-2xl">
                    <p className="text-[10px] font-black text-gray-400 uppercase mb-1">밀도</p>
                    <p className="font-black text-gray-900">
                      {patientDetail.patient_info?.clinical_data?.breast_density ?? '-'}
                    </p>
                  </div>
                  <div className="col-span-2 bg-blue-50/50 p-4 rounded-2xl border border-blue-50">
                    <p className="text-[10px] font-black text-blue-600 uppercase mb-1">Tumor Subtype</p>
                    <p className="font-black text-blue-900">
                      {patientDetail.patient_info?.primary_lesion?.tumor_subtype ?? patientDetail.tumor_subtype ?? '-'}
                    </p>
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
                      {(patientDetail.series || []).map((s) => (
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
                
                {/* 드래그 앤 드롭 영역 */}
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  className={`relative border-2 border-dashed rounded-xl p-6 transition-all duration-200 ${
                    isDragging 
                      ? 'border-blue-400 bg-blue-500/10 scale-[1.02]' 
                      : 'border-gray-700 hover:border-gray-600'
                  }`}
                >
                  <div className="text-center space-y-3">
                    <div className={`mx-auto w-12 h-12 rounded-full flex items-center justify-center transition-colors ${
                      isDragging ? 'bg-blue-500' : 'bg-gray-800'
                    }`}>
                      <Upload className={`w-6 h-6 ${isDragging ? 'text-white' : 'text-gray-400'}`} />
                    </div>
                    <div>
                      <p className="text-sm font-bold text-white">
                        {isDragging ? '여기에 놓으세요!' : '폴더를 드래그하세요'}
                      </p>
                      <p className="text-[10px] text-gray-500 mt-1">
                        또는 아래 버튼으로 파일 선택 (Cmd+A로 전체 선택)
                      </p>
                    </div>
                  </div>
                </div>

                <div className="relative">
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept={imageType === '병리 영상' ? '.svs' : '.dicom,.dcm,.nii,.nii.gz'}
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
                {showOrthancImages && orthancImages.length > 0 && (
                  <Button
                    variant="default"
                    size="sm"
                    className="rounded-xl bg-blue-600 hover:bg-blue-700 text-white font-bold h-10 px-4"
                    onClick={() => {
                      if (selectedPatient) {
                        navigate(`/mri-viewer/${selectedPatient}?imageType=${encodeURIComponent(imageType)}&index=${selectedImage}`);
                      }
                    }}
                  >
                    <Maximize2 className="w-4 h-4 mr-2" />
                    자세히 보기
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="icon"
                  className="rounded-xl h-10 w-10 hover:bg-gray-50"
                  onClick={() => setShowOrthancImages(!showOrthancImages)}
                  title="DICOM/NIfTI 전환"
                >
                  <Cpu className={`w-4 h-4 ${showOrthancImages ? 'text-blue-600' : 'text-gray-400'}`} />
                </Button>
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
                      {showOrthancImages && orthancImages.length > 0 ? (
                        <motion.img
                          key={`orthanc-${selectedImage}`}
                          initial={{ opacity: 0, scale: 0.98 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.2 }}
                          src={orthancImages[selectedImage]?.preview_url}
                          className="max-w-full max-h-full object-contain pointer-events-auto"
                          loading="eager"
                          decoding="async"
                          onLoadStart={() => {
                            setImageLoading(true);
                          }}
                          onLoad={() => {
                            setImageLoading(false);
                          }}
                          onError={() => {
                            setImageLoading(false);
                          }}
                        />
                      ) : showOrthancImages && orthancImages.length === 0 && allOrthancImages.length > 0 ? (
                        // 필터링 결과 이미지가 없는 경우
                        <div className="text-white/70 flex flex-col items-center gap-6 bg-black/30 backdrop-blur-sm rounded-3xl p-12 border border-white/10 max-w-md">
                          <ImageIcon className="w-16 h-16 text-white/40" />
                          <div className="text-center space-y-2">
                            <p className="text-lg font-black text-white uppercase tracking-widest">이미지 없음</p>
                            <p className="text-sm font-medium text-white/60">
                              선택한 환자에게 <span className="font-bold text-white">{imageType}</span> 이미지가 없습니다.
                            </p>
                            <p className="text-xs font-medium text-white/40 mt-4">
                              다른 영상 유형을 선택하거나 이미지를 업로드해주세요.
                            </p>
                          </div>
                        </div>
                      ) : showOrthancImages && allOrthancImages.length === 0 ? (
                        // Orthanc에 이미지가 전혀 없는 경우
                        <div className="text-white/70 flex flex-col items-center gap-6 bg-black/30 backdrop-blur-sm rounded-3xl p-12 border border-white/10 max-w-md">
                          <Database className="w-16 h-16 text-white/40" />
                          <div className="text-center space-y-2">
                            <p className="text-lg font-black text-white uppercase tracking-widest">이미지 없음</p>
                            <p className="text-sm font-medium text-white/60">
                              선택한 환자(<span className="font-bold text-white">{selectedPatient}</span>)의 Orthanc 이미지가 없습니다.
                            </p>
                            <p className="text-xs font-medium text-white/40 mt-4">
                              이미지를 업로드하면 여기에 표시됩니다.
                            </p>
                          </div>
                        </div>
                      ) : sliceImage ? (
                        <motion.img
                          key={`slice-${currentSlice}`}
                          initial={{ opacity: 0, scale: 0.98 }}
                          animate={{ opacity: 1, scale: 1 }}
                          transition={{ duration: 0.2 }}
                          src={sliceImage}
                          className="max-w-full max-h-full object-contain pointer-events-auto"
                          loading="eager"
                          decoding="async"
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
