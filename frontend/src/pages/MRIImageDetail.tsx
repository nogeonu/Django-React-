import { useState, useEffect } from "react";
import { useParams, useNavigate, useSearchParams } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import {
  Loader2,
  ZoomIn,
  ZoomOut,
  RotateCw,
  Maximize2,
  X,
  ArrowLeft,
  Scan,
  Image as ImageIcon,
  Brain,
  CheckCircle,
  Layers,
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

interface SeriesGroup {
  series_id: string;
  series_description: string;
  images: OrthancImage[];
  modality: string;
}

export default function MRIImageDetail() {
  const { patientId } = useParams<{ patientId: string }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { toast } = useToast();

  const imageType = searchParams.get("imageType") || "MRI 영상";
  const initialIndex = parseInt(searchParams.get("index") || "0");

  // 데이터 상태
  const [allOrthancImages, setAllOrthancImages] = useState<OrthancImage[]>([]);
  const [seriesGroups, setSeriesGroups] = useState<SeriesGroup[]>([]);
  const [selectedSeriesIndex, setSelectedSeriesIndex] = useState(0);
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);
  const [loading, setLoading] = useState(false);

  // 뷰어 제어 상태
  const [zoom, setZoom] = useState(100);
  const [rotation, setRotation] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // AI 분석 상태
  const [aiAnalyzing, setAiAnalyzing] = useState(false);
  const [aiResult, setAiResult] = useState<any>(null);
  const [showAiResult, setShowAiResult] = useState(false);
  
  // 4-channel DCE-MRI를 위한 Series 선택
  const [selectedSeriesFor4Channel, setSelectedSeriesFor4Channel] = useState<number[]>([]);
  
  // 시리즈 전체 세그멘테이션 상태
  const [seriesSegmentationResults, setSeriesSegmentationResults] = useState<{[seriesId: string]: any}>({});
  const [showSegmentationOverlay, setShowSegmentationOverlay] = useState(false);

  // 현재 선택된 Series의 이미지들
  const currentImages = seriesGroups[selectedSeriesIndex]?.images || [];
  const currentImage = currentImages[selectedImageIndex];

  useEffect(() => {
    if (patientId) {
      fetchOrthancImages(patientId);
    }
  }, [patientId]);

  useEffect(() => {
    if (allOrthancImages.length > 0) {
      groupImagesBySeries();
    }
  }, [allOrthancImages, imageType]);

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

  const groupImagesBySeries = () => {
    // imageType에 따라 필터링
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

    // Series ID별로 그룹화
    const seriesMap: { [key: string]: OrthancImage[] } = {};
    filtered.forEach(img => {
      if (!seriesMap[img.series_id]) {
        seriesMap[img.series_id] = [];
      }
      seriesMap[img.series_id].push(img);
    });

    // SeriesGroup 배열로 변환
    const groups: SeriesGroup[] = Object.keys(seriesMap).map(seriesId => {
      const images = seriesMap[seriesId].sort((a, b) =>
        parseInt(a.instance_number) - parseInt(b.instance_number)
      );
      return {
        series_id: seriesId,
        series_description: images[0]?.series_description || 'Unknown Series',
        images,
        modality: images[0]?.modality || 'Unknown',
      };
    });

    setSeriesGroups(groups);

    // 초기 선택
    if (groups.length > 0) {
      setSelectedSeriesIndex(0);
      setSelectedImageIndex(Math.min(initialIndex, groups[0].images.length - 1));
    }
  };

  const handleSeriesChange = (index: number) => {
    setSelectedSeriesIndex(index);
    setSelectedImageIndex(0);
    setAiResult(null);
    setShowAiResult(false);
  };

  const handleAiAnalysis = async () => {
    // AI 분석 = 시리즈 전체 세그멘테이션
    if (seriesGroups.length === 0 || selectedSeriesIndex === -1) {
      toast({
        title: "오류",
        description: "시리즈를 선택해주세요.",
        variant: "destructive",
      });
      return;
    }

    const currentSeries = seriesGroups[selectedSeriesIndex];
    const seriesId = currentSeries.series_id;
    const isMRI = imageType === 'MRI 영상';

    // 유방촬영술은 단일 이미지 분석
    if (!isMRI) {
      if (!currentImage) {
        toast({
          title: "오류",
          description: "이미지를 선택해주세요.",
          variant: "destructive",
        });
        return;
      }

      setAiAnalyzing(true);
      setAiResult(null);

      try {
        const instanceId = currentImage.instance_id;
        const response = await fetch(`/api/mri/yolo/instances/${instanceId}/detect/`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || `서버 오류 (${response.status})`);
        }

        setAiResult({ ...data, isMRI: false });
        setShowAiResult(true);

        toast({
          title: "AI 디텍션 완료",
          description: `${data.detection_count}개의 병변이 감지되었습니다.`,
        });
      } catch (error) {
        console.error('AI 분석 오류:', error);
        toast({
          title: "AI 분석 실패",
          description: error instanceof Error ? error.message : "AI 분석 중 오류가 발생했습니다.",
          variant: "destructive",
        });
      } finally {
        setAiAnalyzing(false);
      }
      return;
    }

    // MRI: 시리즈 전체 세그멘테이션
    setAiAnalyzing(true);

    try {
      // 4-channel 모드 확인
      let payload: any = {};
      if (selectedSeriesFor4Channel.length === 4) {
        const sequenceSeriesIds = selectedSeriesFor4Channel.map(idx => seriesGroups[idx].series_id);
        payload.sequence_series_ids = sequenceSeriesIds;
        
        toast({
          title: "시리즈 전체 세그멘테이션 시작",
          description: `4-channel 모드로 ${currentSeries.images.length}개 슬라이스 분석 중...`,
        });
      } else {
        toast({
          title: "시리즈 전체 세그멘테이션 시작",
          description: `${currentSeries.images.length}개 슬라이스 분석 중...`,
        });
      }

      const response = await fetch(`/api/mri/segmentation/series/${seriesId}/segment/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `서버 오류 (${response.status})`);
      }

      // 결과 저장
      setSeriesSegmentationResults({
        ...seriesSegmentationResults,
        [seriesId]: data
      });

      toast({
        title: "시리즈 세그멘테이션 완료",
        description: `${data.successful_slices}/${data.total_slices} 슬라이스 분석 완료. 병변 탐지 버튼으로 오버레이를 확인하세요.`,
      });
    } catch (error) {
      console.error('시리즈 세그멘테이션 오류:', error);
      toast({
        title: "세그멘테이션 실패",
        description: error instanceof Error ? error.message : "세그멘테이션 중 오류가 발생했습니다.",
        variant: "destructive",
      });
    } finally {
      setAiAnalyzing(false);
    }
  };

  const handleZoomIn = () => setZoom(prev => Math.min(300, prev + 25));
  const handleZoomOut = () => setZoom(prev => Math.max(50, prev - 25));
  const handleRotate = () => setRotation(prev => (prev + 90) % 360);
  const handleReset = () => {
    setZoom(100);
    setRotation(0);
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-screen gap-4 bg-gray-950">
        <Loader2 className="h-12 w-12 animate-spin text-blue-600" />
        <p className="text-gray-400 font-bold animate-pulse uppercase tracking-widest text-xs">
          이미지 로드 중...
        </p>
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
              {currentImages.length > 0 && (
                <Badge className="bg-gray-800 text-gray-300 border border-gray-700 px-4 py-2 rounded-xl">
                  {selectedImageIndex + 1} / {currentImages.length}
                </Badge>
              )}

              <Button
                onClick={handleAiAnalysis}
                disabled={aiAnalyzing || !currentImage}
                className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-bold px-6 py-2 rounded-xl flex items-center gap-2 shadow-lg"
              >
                {aiAnalyzing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    AI 분석 중...
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4" />
                    AI 분석
                    {imageType === 'MRI 영상' && selectedSeriesFor4Channel.length === 4 && (
                      <Badge className="ml-2 bg-purple-800 text-white text-xs">4CH</Badge>
                    )}
                  </>
                )}
              </Button>

              {/* 병변 탐지 토글 버튼 (세그멘테이션 결과가 있을 때만 표시) */}
              {imageType === 'MRI 영상' && 
               seriesGroups.length > 0 && 
               selectedSeriesIndex !== -1 &&
               seriesSegmentationResults[seriesGroups[selectedSeriesIndex]?.series_id] && (
                <Button
                  onClick={() => setShowSegmentationOverlay(!showSegmentationOverlay)}
                  className={`font-bold px-6 py-2 rounded-xl flex items-center gap-2 shadow-lg ${
                    showSegmentationOverlay
                      ? 'bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700'
                      : 'bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700'
                  } text-white`}
                >
                  {showSegmentationOverlay ? (
                    <>
                      <Scan className="w-4 h-4" />
                      병변 탐지 OFF
                    </>
                  ) : (
                    <>
                      <Scan className="w-4 h-4" />
                      병변 탐지 ON
                    </>
                  )}
                </Button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content - 3 Column Layout */}
      <div className="w-full h-[calc(100vh-73px)] px-6 py-4 overflow-hidden">
        <div className="grid grid-cols-12 gap-4 h-full">
          {/* Left Sidebar: Series Selection & Image Info */}
          <div className="col-span-2 space-y-4 overflow-y-auto">
            {/* Series Selection */}
            <Card className="bg-gray-900 border-gray-800 rounded-2xl overflow-hidden">
              <CardContent className="p-4 space-y-3">
                <div className="flex items-center gap-2 mb-3">
                  <Layers className="w-4 h-4 text-blue-400" />
                  <h3 className="text-xs font-black text-white uppercase tracking-widest">
                    Series 선택
                  </h3>
                </div>
                
                {/* 4-channel 모드 안내 */}
                {imageType === 'MRI 영상' && seriesGroups.length >= 4 && (
                  <div className="bg-purple-900/20 border border-purple-700/30 rounded-lg p-2 mb-2">
                    <p className="text-xs text-purple-300 font-bold">
                      4-channel DCE-MRI 분석
                    </p>
                    <p className="text-xs text-purple-400 mt-1">
                      4개 Series 선택: {selectedSeriesFor4Channel.length}/4
                    </p>
                  </div>
                )}
                
                <div className="space-y-2">
                  {seriesGroups.length > 0 ? (
                    seriesGroups.map((series, idx) => {
                      const is4ChannelSelected = selectedSeriesFor4Channel.includes(idx);
                      const isCurrentSeries = selectedSeriesIndex === idx;
                      
                      return (
                        <div key={series.series_id} className="relative">
                          <button
                            onClick={() => handleSeriesChange(idx)}
                            className={`w-full p-3 rounded-xl text-left transition-all ${
                              isCurrentSeries
                                ? 'bg-blue-600 text-white shadow-lg'
                                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                            }`}
                          >
                            <div className="text-xs font-bold">Series {idx + 1}</div>
                            <div className="text-xs opacity-75 mt-1 truncate">
                              {series.series_description}
                            </div>
                            <div className="text-xs opacity-60 mt-1">
                              {series.images.length} images
                            </div>
                          </button>
                          
                          {/* 4-channel 선택 체크박스 (MRI만) */}
                          {imageType === 'MRI 영상' && (
                            <input
                              type="checkbox"
                              checked={is4ChannelSelected}
                              onChange={(e) => {
                                e.stopPropagation();
                                if (e.target.checked) {
                                  if (selectedSeriesFor4Channel.length < 4) {
                                    setSelectedSeriesFor4Channel([...selectedSeriesFor4Channel, idx]);
                                  }
                                } else {
                                  setSelectedSeriesFor4Channel(selectedSeriesFor4Channel.filter(i => i !== idx));
                                }
                              }}
                              className="absolute top-2 right-2 w-4 h-4 rounded border-2 border-purple-500 bg-gray-800 checked:bg-purple-600"
                            />
                          )}
                        </div>
                      );
                    })
                  ) : (
                    <div className="text-center text-gray-500 text-xs py-4">
                      Series가 없습니다
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Image Info */}
            {currentImage && (
              <Card className="bg-gray-900 border-gray-800 rounded-2xl overflow-hidden">
                <CardContent className="p-4 space-y-2">
                  <h3 className="text-xs font-black text-white uppercase tracking-widest mb-3">
                    영상 정보
                  </h3>
                  <div className="space-y-2 text-xs">
                    <div>
                      <span className="text-gray-400">Instance:</span>
                      <p className="text-white font-mono mt-1">
                        {currentImage.instance_number}
                      </p>
                    </div>
                    <div>
                      <span className="text-gray-400">Modality:</span>
                      <p className="text-white font-mono mt-1">
                        {currentImage.modality}
                      </p>
                    </div>
                    <div>
                      <span className="text-gray-400">Series ID:</span>
                      <p className="text-white font-mono mt-1 text-[10px] break-all">
                        {currentImage.series_id.substring(0, 16)}...
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Center: Image Viewer */}
          <div className="col-span-7 h-full">
            <Card className={`bg-gray-900 border-gray-800 rounded-2xl overflow-hidden h-full ${isFullscreen ? 'fixed inset-0 z-50 rounded-none' : ''
              }`}>
              <CardContent className={`p-0 h-full ${isFullscreen ? 'h-screen' : ''}`}>
                {currentImages.length > 0 ? (
                  <div className="relative h-full">
                    <CornerstoneViewer
                      key={`viewer-${seriesGroups[selectedSeriesIndex]?.series_id}-${currentImages.length}`}
                      instanceIds={currentImages.map(img => img.instance_id)}
                      currentIndex={selectedImageIndex}
                      onIndexChange={setSelectedImageIndex}
                      showMeasurementTools={true}
                    />
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
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-400">
                    <div className="text-center">
                      <ImageIcon className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p>Series를 선택하세요</p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right Sidebar: Viewer Controls */}
          <div className="col-span-3 space-y-4 overflow-y-auto">
            <Card className="bg-gray-900 border-gray-800 rounded-2xl overflow-hidden">
              <CardContent className="p-4 space-y-4">
                <h3 className="text-xs font-black text-white uppercase tracking-widest mb-4">
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

                {/* AI Analysis Results */}
                {aiResult && showAiResult && (
                  <div className="space-y-2 pt-4 border-t border-gray-800">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs font-bold text-gray-400 flex items-center gap-2">
                        <Brain className="w-4 h-4 text-purple-400" />
                        AI 분석 결과
                      </h4>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setShowAiResult(false)}
                        className="h-6 w-6 p-0 hover:bg-gray-800 rounded-lg"
                      >
                        <X className="w-3 h-3" />
                      </Button>
                    </div>

                    <div className="bg-gradient-to-r from-purple-900/30 to-pink-900/30 border border-purple-700/30 rounded-xl p-4 space-y-3">
                      {aiResult.isMRI ? (
                        <>
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-bold text-white">종양 영역</span>
                            <Badge className="bg-purple-600 text-white font-bold">
                              {(aiResult.tumor_ratio_percent || (aiResult.tumor_ratio * 100) || 0).toFixed(1)}%
                            </Badge>
                          </div>

                          <div className="space-y-2">
                            <div className="flex justify-between text-xs">
                              <span className="text-gray-400">종양 픽셀</span>
                              <span className="text-white font-bold">
                                {(aiResult.tumor_pixel_count || aiResult.tumor_pixels || 0).toLocaleString()}
                              </span>
                            </div>
                            <div className="flex justify-between text-xs">
                              <span className="text-gray-400">전체 픽셀</span>
                              <span className="text-white font-bold">
                                {(aiResult.total_pixel_count || aiResult.total_pixels || 0).toLocaleString()}
                              </span>
                            </div>
                          </div>

                          {(aiResult.segmentation_mask_base64 || aiResult.mask_base64) && (
                            <div className="pt-3 border-t border-gray-800">
                              <p className="text-xs text-gray-400 mb-2">세그멘테이션 마스크</p>
                              <img
                                src={`data:image/png;base64,${aiResult.segmentation_mask_base64 || aiResult.mask_base64}`}
                                alt="Segmentation Mask"
                                className="w-full rounded-lg border border-gray-700"
                              />
                            </div>
                          )}
                        </>
                      ) : (
                        <>
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-bold text-white">검출된 병변</span>
                            <Badge className="bg-purple-600 text-white font-bold">
                              {aiResult.detection_count}개
                            </Badge>
                          </div>

                          {aiResult.detections && aiResult.detections.length > 0 ? (
                            <div className="space-y-2 max-h-60 overflow-y-auto">
                              {aiResult.detections.map((det: any, idx: number) => (
                                <div key={idx} className="bg-gray-900/50 rounded-lg p-3 space-y-1">
                                  <div className="flex items-center justify-between">
                                    <span className="text-xs font-bold text-purple-300">
                                      {det.class_name || `객체 ${idx + 1}`}
                                    </span>
                                    <Badge className="bg-green-600/20 text-green-400 text-xs">
                                      {(det.confidence * 100).toFixed(1)}%
                                    </Badge>
                                  </div>
                                  <div className="text-xs text-gray-400 font-mono">
                                    위치: [{det.bbox[0].toFixed(0)}, {det.bbox[1].toFixed(0)}] -
                                    [{det.bbox[2].toFixed(0)}, {det.bbox[3].toFixed(0)}]
                                  </div>
                                </div>
                              ))}
                            </div>
                          ) : (
                            <div className="flex items-center gap-2 text-sm text-gray-400">
                              <CheckCircle className="w-4 h-4 text-green-400" />
                              <span>병변이 검출되지 않았습니다</span>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                )}

                {/* Reset & Fullscreen */}
                <div className="space-y-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleReset}
                    className="w-full bg-gray-800 border-gray-700 hover:bg-gray-700 text-white rounded-xl"
                  >
                    초기화
                  </Button>

                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setIsFullscreen(!isFullscreen)}
                    className="w-full bg-blue-600/20 border-blue-600/30 hover:bg-blue-600/30 text-blue-400 rounded-xl"
                  >
                    <Maximize2 className="w-4 h-4 mr-2" />
                    {isFullscreen ? "전체화면 종료" : "전체화면"}
                  </Button>
                </div>

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
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
