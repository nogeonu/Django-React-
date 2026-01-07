import { useState, useEffect } from "react";
import { useParams, useNavigate, useSearchParams } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import {
  Loader2,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  RotateCw,
  Maximize2,
  X,
  ArrowLeft,
  Scan,
  Image as ImageIcon,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import CornerstoneViewer from "@/components/CornerstoneViewer";

interface OrthancImage {
  instance_id: string;
  series_id: string;
  study_id: string;
  series_description: string;
  instance_number: string;
  preview_url: string;
  modality?: string;
  view_position?: string;
  image_laterality?: string;
  mammography_view?: string;
}

export default function MRIImageDetail() {
  const { patientId } = useParams<{ patientId: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { toast } = useToast();

  const imageType = searchParams.get("imageType") || "유방촬영술 영상";
  const initialIndex = parseInt(searchParams.get("index") || "0");

  const [orthancImages, setOrthancImages] = useState<OrthancImage[]>([]);
  const [allOrthancImages, setAllOrthancImages] = useState<OrthancImage[]>([]);
  const [selectedImage, setSelectedImage] = useState(initialIndex);
  const [loading, setLoading] = useState(false);
  const [zoom, setZoom] = useState(100);
  const [rotation, setRotation] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    if (patientId) {
      fetchOrthancImages(patientId);
    }
  }, [patientId]);

  useEffect(() => {
    filterImagesByType();
  }, [imageType, allOrthancImages]);

  const fetchOrthancImages = async (patientId: string) => {
    setLoading(true);
    try {
      const response = await fetch(`/api/mri/orthanc/patients/${patientId}/`, {
        cache: 'no-cache',
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || `서버 오류 (${response.status})`);
      }
      
      if (data.success && data.images && Array.isArray(data.images)) {
        setAllOrthancImages(data.images);
      } else {
        setAllOrthancImages([]);
        setOrthancImages([]);
      }
    } catch (error) {
      console.error('Orthanc 이미지 로드 실패:', error);
      toast({
        title: "오류",
        description: error instanceof Error ? error.message : "이미지를 불러오는데 실패했습니다.",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const filterImagesByType = () => {
    if (allOrthancImages.length === 0) {
      setOrthancImages([]);
      return;
    }

    let filtered: OrthancImage[] = [];

    switch (imageType) {
      case '유방촬영술 영상':
        filtered = allOrthancImages.filter(img => img.modality === 'MG');
        break;
      case 'MRI 영상':
        filtered = allOrthancImages.filter(img => img.modality === 'MR');
        break;
      case '병리 영상':
        filtered = allOrthancImages.filter(img => 
          img.modality === 'SM' || img.modality === 'OT' || 
          (img.modality && img.modality !== 'MG' && img.modality !== 'MR')
        );
        break;
      default:
        filtered = allOrthancImages;
    }

    setOrthancImages(filtered);
    
    // 초기 인덱스가 범위를 벗어난 경우 조정
    if (initialIndex >= filtered.length) {
      setSelectedImage(0);
    }
  };


  const handleZoomIn = () => {
    setZoom(prev => Math.min(300, prev + 25));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(50, prev - 25));
  };

  const handleRotate = () => {
    setRotation(prev => (prev + 90) % 360);
  };

  const handleReset = () => {
    setZoom(100);
    setRotation(0);
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (orthancImages.length === 0) return;

    switch (e.key) {
      case 'ArrowLeft':
        setSelectedImage(prev => Math.max(0, prev - 1));
        break;
      case 'ArrowRight':
        setSelectedImage(prev => Math.min(orthancImages.length - 1, prev + 1));
        break;
      case '+':
      case '=':
        handleZoomIn();
        break;
      case '-':
      case '_':
        handleZoomOut();
        break;
      case 'r':
      case 'R':
        handleRotate();
        break;
      case 'Escape':
        if (isFullscreen) {
          setIsFullscreen(false);
        }
        break;
    }
  };

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [orthancImages.length, selectedImage, isFullscreen]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen gap-4 bg-gray-950">
        <Loader2 className="h-12 w-12 animate-spin text-blue-600" />
        <p className="text-gray-400 font-bold animate-pulse uppercase tracking-widest text-xs">이미지 로드 중...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white w-full h-screen overflow-hidden">
      {/* Header */}
      <div className="sticky top-0 z-50 bg-gray-900/95 backdrop-blur-sm border-b border-gray-800">
        <div className="w-full px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => navigate('/mri-viewer')}
                className="text-white hover:bg-gray-800 rounded-xl"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                목록으로
              </Button>
              <div className="h-8 w-px bg-gray-700"></div>
              <div className="flex items-center gap-3">
                <div className="bg-blue-600 p-2 rounded-xl">
                  <Scan className="w-4 h-4 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-black tracking-tight">영상 상세 보기</h1>
                  <p className="text-xs text-gray-400">환자 ID: {patientId}</p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <Badge className="bg-blue-600/20 text-blue-400 border border-blue-600/30 px-4 py-2 rounded-xl">
                {imageType}
              </Badge>
              <Badge className="bg-gray-800 text-gray-300 border border-gray-700 px-4 py-2 rounded-xl">
                {selectedImage + 1} / {orthancImages.length}
              </Badge>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="w-full h-[calc(100vh-73px)] px-6 py-4 overflow-hidden">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-full">
          {/* Left: Controls */}
          <div className="lg:col-span-3 space-y-4">
            <Card className="bg-gray-900 border-gray-800 rounded-2xl overflow-hidden">
              <CardContent className="p-6 space-y-4">
                <h3 className="text-sm font-black text-white uppercase tracking-widest mb-4">
                  뷰어 제어
                </h3>

                {/* Zoom Controls */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-bold text-gray-400">확대/축소</label>
                    <span className="text-sm font-black text-white">{zoom}%</span>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleZoomOut}
                      className="flex-1 bg-gray-800 border-gray-700 hover:bg-gray-700 text-white rounded-xl"
                    >
                      <ZoomOut className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleZoomIn}
                      className="flex-1 bg-gray-800 border-gray-700 hover:bg-gray-700 text-white rounded-xl"
                    >
                      <ZoomIn className="w-4 h-4" />
                    </Button>
                  </div>
                  <Slider
                    value={[zoom]}
                    onValueChange={(v) => setZoom(v[0])}
                    min={50}
                    max={300}
                    step={10}
                    className="w-full"
                  />
                </div>

                {/* Rotation */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-bold text-gray-400">회전</label>
                    <span className="text-sm font-black text-white">{rotation}°</span>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleRotate}
                    className="w-full bg-gray-800 border-gray-700 hover:bg-gray-700 text-white rounded-xl"
                  >
                    <RotateCw className="w-4 h-4 mr-2" />
                    90° 회전
                  </Button>
                </div>

                {/* Reset */}
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleReset}
                  className="w-full bg-gray-800 border-gray-700 hover:bg-gray-700 text-white rounded-xl"
                >
                  초기화
                </Button>

                {/* Fullscreen */}
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setIsFullscreen(!isFullscreen)}
                  className="w-full bg-blue-600/20 border-blue-600/30 hover:bg-blue-600/30 text-blue-400 rounded-xl"
                >
                  <Maximize2 className="w-4 h-4 mr-2" />
                  {isFullscreen ? "전체화면 종료" : "전체화면"}
                </Button>

                {/* Keyboard Shortcuts */}
                <div className="pt-4 border-t border-gray-800">
                  <h4 className="text-xs font-black text-gray-400 uppercase tracking-widest mb-3">
                    단축키
                  </h4>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">이전/다음</span>
                      <span className="text-white font-mono">← →</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">확대/축소</span>
                      <span className="text-white font-mono">+ -</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">회전</span>
                      <span className="text-white font-mono">R</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Ctrl + 휠</span>
                      <span className="text-white">줌</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Image Info */}
            {orthancImages[selectedImage] && (
              <Card className="bg-gray-900 border-gray-800 rounded-2xl overflow-hidden">
                <CardContent className="p-6 space-y-3">
                  <h3 className="text-sm font-black text-white uppercase tracking-widest mb-4">
                    영상 정보
                  </h3>
                  <div className="space-y-2 text-xs">
                    <div>
                      <span className="text-gray-400">Series:</span>
                      <p className="text-white font-mono mt-1">
                        {orthancImages[selectedImage].series_description || 'N/A'}
                      </p>
                    </div>
                    <div>
                      <span className="text-gray-400">Instance:</span>
                      <p className="text-white font-mono mt-1">
                        {orthancImages[selectedImage].instance_number}
                      </p>
                    </div>
                    {orthancImages[selectedImage].modality && (
                      <div>
                        <span className="text-gray-400">Modality:</span>
                        <p className="text-white font-mono mt-1">
                          {orthancImages[selectedImage].modality}
                        </p>
                      </div>
                    )}
                    {orthancImages[selectedImage].view_position && (
                      <div>
                        <span className="text-gray-400">View:</span>
                        <p className="text-white font-mono mt-1">
                          {orthancImages[selectedImage].view_position}
                        </p>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Right: Image Viewer */}
          <div className="lg:col-span-9 h-full">
            <Card className={`bg-gray-900 border-gray-800 rounded-2xl overflow-hidden h-full ${isFullscreen ? 'fixed inset-0 z-50 rounded-none' : ''}`}>
              <CardContent className={`p-0 h-full ${isFullscreen ? 'h-screen' : ''}`}>
                {orthancImages.length > 0 ? (
                  <div className="relative h-full">
                    {/* Cornerstone Viewer */}
                    <div className="h-full">
                      <CornerstoneViewer
                        key={`cornerstone-detail-${patientId}-${orthancImages.length}`}
                        instanceIds={orthancImages.map(img => img.instance_id)}
                        currentIndex={selectedImage}
                        onIndexChange={setSelectedImage}
                        showMeasurementTools={true}
                      />
                    </div>

                    {/* Close Fullscreen Button */}
                    {isFullscreen && (
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => setIsFullscreen(false)}
                        className="absolute top-4 right-4 z-50 bg-black/50 hover:bg-black/70 text-white rounded-xl"
                      >
                        <X className="w-6 h-6" />
                      </Button>
                    )}

                    {/* Navigation Controls */}
                    {!isFullscreen && (
                      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-6">
                        <div className="flex items-center gap-4">
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => setSelectedImage(Math.max(0, selectedImage - 1))}
                            disabled={selectedImage === 0}
                            className="bg-white/10 hover:bg-white/20 text-white rounded-xl"
                          >
                            <ChevronLeft className="w-6 h-6" />
                          </Button>
                          
                          <div className="flex-1">
                            <Slider
                              value={[selectedImage]}
                              onValueChange={(v) => setSelectedImage(v[0])}
                              max={orthancImages.length - 1}
                              step={1}
                              className="w-full"
                            />
                          </div>

                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => setSelectedImage(Math.min(orthancImages.length - 1, selectedImage + 1))}
                            disabled={selectedImage === orthancImages.length - 1}
                            className="bg-white/10 hover:bg-white/20 text-white rounded-xl"
                          >
                            <ChevronRight className="w-6 h-6" />
                          </Button>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center">
                    <div className="text-center space-y-4">
                      <ImageIcon className="w-16 h-16 text-gray-600 mx-auto" />
                      <p className="text-gray-400">이미지가 없습니다</p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

