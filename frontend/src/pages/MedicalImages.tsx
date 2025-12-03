import { useState, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { 
  Upload, 
  FileImage, 
  Search, 
  Filter,
  Play,
  Download,
  AlertTriangle,
  CheckCircle,
  Clock,
  Box,
  X
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/api";

interface Patient {
  id: string;
  name: string;
  birth_date: string;
  gender: string;
  phone?: string;
  address?: string;
  emergency_contact?: string;
  blood_type?: string;
  age: number;
  created_at: string;
  updated_at: string;
}

interface MedicalImage {
  id: string;
  patient: string;
  patient_name?: string;
  image_type: string;
  image_file?: string;
  image_url?: string;
  description?: string;
  taken_date: string;
  doctor_notes?: string;
  created_at: string;
  analysis_results?: AIAnalysisResult[];
}

interface AIAnalysisResult {
  id: string;
  image: string;
  analysis_type: string;
  results: any;
  confidence?: number;
  findings?: string;
  recommendations?: string;
  model_version?: string;
  analysis_date: string;
}

export default function MedicalImages() {
  const [selectedPatient, setSelectedPatient] = useState<string>("");
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedImage, setSelectedImage] = useState<MedicalImage | null>(null);
  const [selectedImagesForAnalysis, setSelectedImagesForAnalysis] = useState<Set<string>>(new Set());
  const [showAnalysisModal, setShowAnalysisModal] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: patients = [] } = useQuery({
    queryKey: ["patients"],
    queryFn: async () => {
      const response = await apiRequest("GET", "/api/lung_cancer/patients/");
      return response.results || [];
    },
  });

  const { data: images = [], isLoading } = useQuery({
    queryKey: ["medical-images", selectedPatient],
    queryFn: async () => {
      const response = await apiRequest("GET", `/api/medical-images/?patient_id=${selectedPatient}`);
      // DRF pagination 응답일 경우 results 사용
      return response.results || response || [];
    },
    enabled: !!selectedPatient,
  });

  const aiAnalysisMutation = useMutation({
    mutationFn: async (imageId: string) => {
      const response = await apiRequest("POST", `/api/medical-images/${imageId}/analyze/`);
      return response;
    },
    onSuccess: () => {
      toast({
        title: "AI 분석 완료",
        description: "이미지 분석이 완료되었습니다.",
      });
      queryClient.invalidateQueries({ queryKey: ["medical-images", selectedPatient] });
    },
    onError: (error: any) => {
      const errorMessage = error?.response?.data?.error || error?.response?.data?.detail || error?.message || "AI 분석 중 오류가 발생했습니다.";
      const solution = error?.response?.data?.solution;
      
      toast({
        title: "분석 실패",
        description: solution ? `${errorMessage}\n\n해결 방법: ${solution}` : errorMessage,
        variant: "destructive",
      });
    }
  });

  const handleFileUpload = () => {
    if (!selectedPatient) {
      toast({
        title: "환자 선택 필요",
        description: "먼저 환자를 선택해주세요.",
        variant: "destructive",
      });
      return;
    }
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const fileArray = Array.from(files);
    
    toast({
      title: "업로드 시작",
      description: `${fileArray.length}개의 이미지를 업로드합니다...`,
    });

    // 여러 파일을 순차적으로 업로드
    for (let i = 0; i < fileArray.length; i++) {
      const file = fileArray[i];
      try {
        const formData = new FormData();
        formData.append('patient_id', selectedPatient);
        formData.append('image_type', 'MRI'); // 기본값, 실제로는 사용자가 선택
        formData.append('image_file', file);
        formData.append('description', '');
        formData.append('taken_date', new Date().toISOString());
        formData.append('doctor_notes', '');

        await apiRequest("POST", "/api/medical-images/", formData);
      } catch (error) {
        console.error(`이미지 ${i + 1} 업로드 실패:`, error);
        toast({
          title: `파일 ${i + 1} 업로드 실패`,
          description: `${file.name} 업로드 중 오류가 발생했습니다.`,
          variant: "destructive",
        });
      }
    }

    // 모든 업로드 완료 후 목록 갱신
    queryClient.invalidateQueries({ queryKey: ["medical-images", selectedPatient] });
    
    toast({
      title: "업로드 완료",
      description: `${fileArray.length}개의 이미지가 성공적으로 업로드되었습니다.`,
    });

    // 파일 입력 초기화
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleAiAnalysis = async (image: MedicalImage) => {
    try {
      aiAnalysisMutation.mutate(image.id);
    } catch (error) {
      toast({
        title: "분석 시작 실패",
        description: "AI 분석을 시작할 수 없습니다.",
        variant: "destructive",
      });
    }
  };

  const handleBatchAnalysis = () => {
    if (selectedImagesForAnalysis.size === 0) {
      toast({
        title: "이미지 선택 필요",
        description: "분석할 이미지를 선택해주세요.",
        variant: "destructive",
      });
      return;
    }
    setShowAnalysisModal(true);
  };

  const handleConfirmBatchAnalysis = async () => {
    const selectedIds = Array.from(selectedImagesForAnalysis);
    
    toast({
      title: "일괄 분석 시작",
      description: `${selectedIds.length}개의 이미지를 분석합니다...`,
    });

    setShowAnalysisModal(false);

    // 선택된 이미지들을 순차적으로 분석
    for (const imageId of selectedIds) {
      try {
        await apiRequest("POST", `/api/medical-images/${imageId}/analyze/`);
      } catch (error) {
        console.error(`이미지 ${imageId} 분석 실패:`, error);
      }
    }

    // 분석 완료 후 목록 갱신
    queryClient.invalidateQueries({ queryKey: ["medical-images", selectedPatient] });
    
    toast({
      title: "일괄 분석 완료",
      description: `${selectedIds.length}개의 이미지 분석이 완료되었습니다.`,
    });

    // 선택 초기화
    setSelectedImagesForAnalysis(new Set());
  };

  const toggleImageSelection = (imageId: string) => {
    const newSelection = new Set(selectedImagesForAnalysis);
    if (newSelection.has(imageId)) {
      newSelection.delete(imageId);
    } else {
      newSelection.add(imageId);
    }
    setSelectedImagesForAnalysis(newSelection);
  };

  const selectAllImages = () => {
    const allImageIds = filteredImages.map((img: MedicalImage) => img.id);
    setSelectedImagesForAnalysis(new Set(allImageIds));
  };

  const deselectAllImages = () => {
    setSelectedImagesForAnalysis(new Set());
  };

  const handleDownload = (image: MedicalImage) => {
    if (!image.image_url) {
      toast({
        title: "다운로드 실패",
        description: "이미지 URL이 없습니다.",
        variant: "destructive",
      });
      return;
    }

    try {
      // 이미지 URL에서 파일명 추출
      const url = new URL(image.image_url);
      const pathParts = url.pathname.split('/');
      const filename = pathParts[pathParts.length - 1] || `medical_image_${image.id}.jpg`;
      
      // 이미지 다운로드
      fetch(image.image_url)
        .then(response => response.blob())
        .then(blob => {
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = decodeURIComponent(filename);
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);
          
          toast({
            title: "다운로드 완료",
            description: "이미지가 다운로드되었습니다.",
          });
        })
        .catch(error => {
          console.error('다운로드 오류:', error);
          toast({
            title: "다운로드 실패",
            description: "이미지를 다운로드할 수 없습니다.",
            variant: "destructive",
          });
        });
    } catch (error) {
      console.error('다운로드 오류:', error);
      toast({
        title: "다운로드 실패",
        description: "이미지를 다운로드할 수 없습니다.",
        variant: "destructive",
      });
    }
  };

  const handle3DVisualization = (image: MedicalImage) => {
    if (!image.analysis_results || image.analysis_results.length === 0) {
      toast({
        title: "3D 시각화 불가",
        description: "먼저 이미지 분석을 완료해주세요.",
        variant: "destructive",
      });
      return;
    }

    // 3D 시각화 페이지로 이동 (분석 결과와 이미지 정보 전달)
    const visualizationData = {
      imageId: image.id,
      imageUrl: image.image_url,
      analysisResult: image.analysis_results[0],
      imageType: image.image_type,
      patientId: image.patient,
    };
    
    // 세션 스토리지에 데이터 저장
    sessionStorage.setItem('3d_visualization_data', JSON.stringify(visualizationData));
    
    // 새 탭에서 3D 시각화 페이지 열기
    window.open('/3d-visualization', '_blank');
    
    toast({
      title: "3D 시각화 열기",
      description: "새 탭에서 3D 시각화를 확인할 수 있습니다.",
    });
  };

  const getAnalysisStatusBadge = (image: MedicalImage) => {
    if (image.analysis_results && image.analysis_results.length > 0) {
      return (
        <Badge variant="default" className="bg-green-100 text-green-800">
          <CheckCircle className="w-3 h-3 mr-1" />
          분석 완료
        </Badge>
      );
    }
    return (
      <Badge variant="outline" className="text-gray-600">
        <Clock className="w-3 h-3 mr-1" />
        분석 대기
      </Badge>
    );
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 80) return "text-green-600";
    if (confidence >= 60) return "text-yellow-600";
    return "text-red-600";
  };

  const filteredImages = (images as MedicalImage[]).filter((image: MedicalImage) =>
    image.image_type.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (image.description && image.description.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div>
              <h1 className="text-xl font-bold text-gray-900">의료 이미지 관리</h1>
              <p className="text-sm text-gray-500">MRI, CT 등 의료 이미지를 업로드하고 AI 분석을 수행합니다</p>
            </div>
            <Button onClick={handleFileUpload} data-testid="button-upload-image">
              <Upload className="w-4 h-4 mr-2" />
              이미지 업로드
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Controls */}
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  환자 선택
                </label>
                <Select value={selectedPatient} onValueChange={setSelectedPatient} data-testid="select-patient">
                  <SelectTrigger>
                    <SelectValue placeholder="환자를 선택하세요" />
                  </SelectTrigger>
                  <SelectContent>
                    {(patients as Patient[]).map((patient: Patient) => (
                      <SelectItem key={patient.id} value={patient.id}>
                        {patient.name} ({patient.id})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  검색
                </label>
                <Input
                  placeholder="이미지 유형, 부위로 검색..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  data-testid="input-search-images"
                />
              </div>
              
              <div className="flex items-end">
                <Button variant="outline" className="mr-2" data-testid="button-filter-images">
                  <Filter className="w-4 h-4 mr-2" />
                  필터
                </Button>
                <Button variant="outline" data-testid="button-search-images">
                  <Search className="w-4 h-4 mr-2" />
                  검색
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Hidden File Input - multiple 속성 추가 */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={handleFileChange}
          data-testid="input-file-upload"
        />

        {/* 일괄 분석 버튼 (이미지가 선택된 경우에만 표시) */}
        {selectedPatient && filteredImages.length > 0 && (
          <Card className="mb-6 bg-blue-50 border-blue-200">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className="text-sm font-medium text-gray-700">
                    선택된 이미지: <span className="text-blue-600 font-bold">{selectedImagesForAnalysis.size}개</span>
                  </div>
                  <div className="flex space-x-2">
                    <Button 
                      size="sm" 
                      variant="outline" 
                      onClick={selectAllImages}
                      data-testid="button-select-all"
                    >
                      전체 선택
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline" 
                      onClick={deselectAllImages}
                      data-testid="button-deselect-all"
                    >
                      선택 해제
                    </Button>
                  </div>
                </div>
                <Button 
                  onClick={handleBatchAnalysis}
                  disabled={selectedImagesForAnalysis.size === 0}
                  data-testid="button-batch-analysis"
                >
                  <Play className="w-4 h-4 mr-2" />
                  선택한 이미지 분석 ({selectedImagesForAnalysis.size})
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Image Grid */}
        {selectedPatient ? (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                의료 이미지 목록
                <Badge variant="secondary" data-testid="text-image-count">
                  총 {filteredImages.length}개
                </Badge>
              </CardTitle>
              <CardDescription>
                선택된 환자의 의료 이미지를 확인하고 AI 분석을 수행할 수 있습니다
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {Array.from({ length: 6 }).map((_, i) => (
                    <div key={i} className="animate-pulse">
                      <div className="aspect-video bg-gray-200 rounded-lg mb-4"></div>
                      <div className="space-y-2">
                        <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                        <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : filteredImages.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {filteredImages.map((image: MedicalImage) => (
                    <div
                      key={image.id}
                      className="bg-white border rounded-lg overflow-hidden hover:shadow-md transition-shadow cursor-pointer"
                      onClick={() => setSelectedImage(image)}
                      data-testid={`image-card-${image.id}`}
                    >
                      <div className="aspect-video bg-gray-100 relative">
                        {/* 체크박스 - 왼쪽 상단 */}
                        <div 
                          className="absolute top-3 left-3 z-10"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <input
                            type="checkbox"
                            checked={selectedImagesForAnalysis.has(image.id)}
                            onChange={() => toggleImageSelection(image.id)}
                            className="w-5 h-5 cursor-pointer accent-blue-600"
                            data-testid={`checkbox-image-${image.id}`}
                          />
                        </div>

                        {image.image_url ? (
                          <img
                            src={image.image_url}
                            alt={image.image_type}
                            className="w-full h-full object-cover"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement;
                              target.style.display = 'none';
                              const placeholder = target.nextElementSibling as HTMLElement;
                              if (placeholder) placeholder.style.display = 'flex';
                            }}
                          />
                        ) : null}
                        <div 
                          className="w-full h-full flex items-center justify-center bg-gray-100"
                          style={{ display: image.image_url ? 'none' : 'flex' }}
                        >
                          <div className="text-center">
                            <FileImage className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                            <p className="text-xs text-gray-500">이미지 없음</p>
                          </div>
                        </div>
                        
                        {/* 분석 상태 배지 - 오른쪽 상단 */}
                        <div className="absolute top-2 right-2">
                          {getAnalysisStatusBadge(image)}
                        </div>
                        
                        {/* 이미지 타입 배지 - 하단 왼쪽 */}
                        <div className="absolute bottom-2 left-2">
                          <Badge 
                            variant={image.image_type === 'MRI' ? 'default' : 'secondary'}
                            className={image.image_type === 'MRI' ? 'bg-purple-600 hover:bg-purple-700' : ''}
                          >
                            {image.image_type}
                          </Badge>
                        </div>
                      </div>
                      
                      <div className="p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="font-medium text-gray-900" data-testid={`text-image-type-${image.id}`}>
                            {image.image_type}
                          </h3>
                          {image.analysis_results && image.analysis_results.length > 0 && (
                            <Badge variant="default" className="bg-green-500 hover:bg-green-600">
                              <CheckCircle className="w-3 h-3 mr-1" />
                              분석 완료
                            </Badge>
                          )}
                        </div>
                        <p className="text-sm text-gray-500 mb-3">
                          {new Date(image.taken_date || '').toLocaleDateString('ko-KR')}
                        </p>
                        
                        {image.analysis_results && image.analysis_results.length > 0 && (
                          <div className="mb-3 bg-gray-50 rounded-lg p-3">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-xs font-medium text-gray-700">신뢰도</span>
                              <span className={`text-lg font-bold ${getConfidenceColor(image.analysis_results[0].confidence || 0)}`}>
                                {image.analysis_results[0].confidence}%
                              </span>
                            </div>
                            <p className="text-xs text-gray-700 line-clamp-2">
                              {image.analysis_results[0].findings}
                            </p>
                          </div>
                        )}
                        
                        <div className="flex space-x-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleAiAnalysis(image);
                            }}
                            disabled={aiAnalysisMutation.isPending}
                            data-testid={`button-analyze-${image.id}`}
                          >
                            <Play className="w-3 h-3 mr-1" />
                            {image.analysis_results && image.analysis_results.length > 0 ? "재분석" : "분석"}
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDownload(image);
                            }}
                            data-testid={`button-download-${image.id}`}
                          >
                            <Download className="w-3 h-3" />
                          </Button>
                          {image.analysis_results && image.analysis_results.length > 0 && (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={(e) => {
                                e.stopPropagation();
                                handle3DVisualization(image);
                              }}
                              data-testid={`button-3d-visualization-${image.id}`}
                            >
                              <Box className="w-3 h-3 mr-1" />
                              3D 시각화
                            </Button>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <FileImage className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <div className="text-gray-500 mb-4">
                    {searchTerm ? "검색 결과가 없습니다" : "업로드된 이미지가 없습니다"}
                  </div>
                  {!searchTerm && (
                    <Button onClick={handleFileUpload} data-testid="button-upload-first-image">
                      <Upload className="w-4 h-4 mr-2" />
                      첫 번째 이미지 업로드
                    </Button>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardContent className="pt-6">
              <div className="text-center py-12">
                <AlertTriangle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <div className="text-gray-500 mb-4">
                  환자를 선택해주세요
                </div>
                <p className="text-sm text-gray-400">
                  의료 이미지를 보려면 먼저 환자를 선택해야 합니다
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </main>

      {/* Image Detail Modal */}
      {selectedImage && (
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedImage(null)}
          data-testid="modal-image-detail"
        >
          <div
            className="bg-white rounded-lg w-full max-w-4xl max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-lg font-semibold">
                  {selectedImage.image_type}
                </h3>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSelectedImage(null)}
                  data-testid="button-close-image-modal"
                >
                  닫기
                </Button>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  {selectedImage.image_url ? (
                    <img
                      src={selectedImage.image_url}
                      alt={selectedImage.image_type}
                      className="w-full rounded-lg"
                      onError={(e) => {
                        const target = e.target as HTMLImageElement;
                        target.style.display = 'none';
                        const placeholder = target.nextElementSibling as HTMLElement;
                        if (placeholder) placeholder.style.display = 'flex';
                      }}
                    />
                  ) : null}
                  <div 
                    className="w-full aspect-video flex items-center justify-center bg-gray-100 rounded-lg"
                    style={{ display: selectedImage.image_url ? 'none' : 'flex' }}
                  >
                    <div className="text-center">
                      <FileImage className="w-16 h-16 text-gray-400 mx-auto mb-2" />
                      <p className="text-sm text-gray-500">이미지를 불러올 수 없습니다</p>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-gray-900 mb-2">이미지 정보</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">유형:</span>
                        <span>{selectedImage.image_type}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">촬영일:</span>
                        <span>{new Date(selectedImage.taken_date || '').toLocaleDateString('ko-KR')}</span>
                      </div>
                      {selectedImage.patient_name && (
                        <div className="flex justify-between">
                          <span className="text-gray-600">환자:</span>
                          <span>{selectedImage.patient_name}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {selectedImage.analysis_results && selectedImage.analysis_results.length > 0 && (
                    <div>
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="font-medium text-gray-900">AI 분석 결과</h4>
                        <Badge variant="default" className="bg-green-500 hover:bg-green-600">
                          <CheckCircle className="w-3 h-3 mr-1" />
                          분석 완료
                        </Badge>
                      </div>
                      <div className="space-y-3">
                        <div className="bg-gray-50 rounded-lg p-4">
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-sm font-medium text-gray-700">신뢰도</span>
                            <span className={`text-2xl font-bold ${getConfidenceColor(selectedImage.analysis_results[0].confidence || 0)}`}>
                              {selectedImage.analysis_results[0].confidence}%
                            </span>
                          </div>
                        </div>
                        
                        <div>
                          <span className="text-sm text-gray-600 block mb-1">발견사항:</span>
                          <p className="text-sm text-gray-900 bg-blue-50 p-3 rounded-lg">{selectedImage.analysis_results[0].findings}</p>
                        </div>
                        
                        <div>
                          <span className="text-sm text-gray-600 block mb-1">권장사항:</span>
                          <p className="text-sm text-gray-900 bg-amber-50 p-3 rounded-lg">{selectedImage.analysis_results[0].recommendations}</p>
                        </div>
                        
                        <div className="flex justify-between text-xs text-gray-500 pt-2 border-t">
                          <span>분석일: {new Date(selectedImage.analysis_results[0].analysis_date || '').toLocaleDateString('ko-KR')}</span>
                          <span>모델: {selectedImage.analysis_results[0].model_version}</span>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  <div className="flex space-x-3 pt-4">
                    <Button
                      onClick={() => handleAiAnalysis(selectedImage)}
                      disabled={aiAnalysisMutation.isPending}
                      data-testid="button-analyze-modal"
                    >
                      <Play className="w-4 h-4 mr-2" />
                      {selectedImage.analysis_results && selectedImage.analysis_results.length > 0 ? "재분석" : "AI 분석"}
                    </Button>
                    <Button 
                      variant="outline" 
                      onClick={() => handleDownload(selectedImage)}
                      data-testid="button-download-modal"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      다운로드
                    </Button>
                    {selectedImage.analysis_results && selectedImage.analysis_results.length > 0 && (
                      <Button 
                        variant="outline" 
                        onClick={() => handle3DVisualization(selectedImage)}
                        data-testid="button-3d-visualization-modal"
                      >
                        <Box className="w-4 h-4 mr-2" />
                        3D 시각화
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 일괄 분석 확인 모달 */}
      {showAnalysisModal && (
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setShowAnalysisModal(false)}
          data-testid="modal-batch-analysis"
        >
          <div
            className="bg-white rounded-lg w-full max-w-2xl max-h-[80vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-lg font-semibold">일괄 분석 확인</h3>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowAnalysisModal(false)}
                  data-testid="button-close-batch-modal"
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
              
              <div className="mb-6">
                <p className="text-sm text-gray-600 mb-4">
                  다음 <span className="font-bold text-blue-600">{selectedImagesForAnalysis.size}개</span>의 이미지를 분석하시겠습니까?
                </p>
                
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 max-h-96 overflow-y-auto">
                  {filteredImages
                    .filter((img: MedicalImage) => selectedImagesForAnalysis.has(img.id))
                    .map((image: MedicalImage) => (
                      <div
                        key={image.id}
                        className="border rounded-lg overflow-hidden"
                      >
                        <div className="aspect-video bg-gray-100 relative">
                          {image.image_url ? (
                            <img
                              src={image.image_url}
                              alt={image.image_type}
                              className="w-full h-full object-cover"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center">
                              <FileImage className="w-8 h-8 text-gray-400" />
                            </div>
                          )}
                          <div className="absolute top-1 right-1">
                            <Badge 
                              variant={image.image_type === 'MRI' ? 'default' : 'secondary'}
                              className="text-xs"
                            >
                              {image.image_type}
                            </Badge>
                          </div>
                        </div>
                        <div className="p-2">
                          <p className="text-xs text-gray-600 truncate">
                            {new Date(image.taken_date || '').toLocaleDateString('ko-KR')}
                          </p>
                        </div>
                      </div>
                    ))}
                </div>
              </div>
              
              <div className="flex justify-end space-x-3">
                <Button
                  variant="outline"
                  onClick={() => setShowAnalysisModal(false)}
                  data-testid="button-cancel-batch-analysis"
                >
                  취소
                </Button>
                <Button
                  onClick={handleConfirmBatchAnalysis}
                  data-testid="button-confirm-batch-analysis"
                >
                  <Play className="w-4 h-4 mr-2" />
                  분석 시작
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
