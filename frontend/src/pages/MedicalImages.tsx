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
  Clock
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

  const uploadImageMutation = useMutation({
    mutationFn: async (imageData: any) => {
      const response = await apiRequest("POST", "/api/medical-images/", imageData);
      return response;
    },
    onSuccess: () => {
      toast({
        title: "업로드 완료",
        description: "의료 이미지가 성공적으로 업로드되었습니다.",
      });
      queryClient.invalidateQueries({ queryKey: ["medical-images", selectedPatient] });
    },
    onError: (error: any) => {
      console.error('이미지 업로드 오류:', error);
      const errorMessage = error?.response?.data?.error || error?.response?.data?.detail || error?.message || "이미지 업로드 중 오류가 발생했습니다.";
      toast({
        title: "업로드 실패",
        description: errorMessage,
        variant: "destructive",
      });
    }
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
    onError: () => {
      toast({
        title: "분석 실패",
        description: "AI 분석 중 오류가 발생했습니다.",
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
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const formData = new FormData();
      formData.append('patient_id', selectedPatient);
      formData.append('image_type', 'MRI'); // 기본값, 실제로는 사용자가 선택
      formData.append('image_file', file);
      formData.append('description', '');
      formData.append('taken_date', new Date().toISOString());
      formData.append('doctor_notes', '');

      uploadImageMutation.mutate(formData);
    } catch (error) {
      toast({
        title: "파일 처리 실패",
        description: "파일을 처리하는 중 오류가 발생했습니다.",
        variant: "destructive",
      });
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

        {/* Hidden File Input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleFileChange}
          data-testid="input-file-upload"
        />

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
                        <div className="absolute top-2 right-2">
                          {getAnalysisStatusBadge(image)}
                        </div>
                        <div className="absolute top-2 left-2">
                          <Badge variant="secondary">
                            {image.image_type}
                          </Badge>
                        </div>
                      </div>
                      
                      <div className="p-4">
                        <h3 className="font-medium text-gray-900 mb-1" data-testid={`text-image-type-${image.id}`}>
                          {image.image_type}
                        </h3>
                        <p className="text-sm text-gray-500 mb-3">
                          {new Date(image.taken_date || '').toLocaleDateString('ko-KR')}
                        </p>
                        
                        {image.analysis_results && image.analysis_results.length > 0 && (
                          <div className="mb-3">
                            <div className="flex items-center justify-between mb-1">
                              <span className="text-xs text-gray-600">신뢰도</span>
                              <span className={`text-xs font-medium ${getConfidenceColor(image.analysis_results[0].confidence || 0)}`}>
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
                      <h4 className="font-medium text-gray-900 mb-2">AI 분석 결과</h4>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-600">신뢰도:</span>
                          <span className={`font-medium ${getConfidenceColor(selectedImage.analysis_results[0].confidence || 0)}`}>
                            {selectedImage.analysis_results[0].confidence}%
                          </span>
                        </div>
                        
                        <div>
                          <span className="text-sm text-gray-600 block mb-1">발견사항:</span>
                          <p className="text-sm text-gray-900">{selectedImage.analysis_results[0].findings}</p>
                        </div>
                        
                        <div>
                          <span className="text-sm text-gray-600 block mb-1">권장사항:</span>
                          <p className="text-sm text-gray-900">{selectedImage.analysis_results[0].recommendations}</p>
                        </div>
                        
                        <div className="flex justify-between text-xs text-gray-500">
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
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
