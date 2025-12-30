import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ArrowLeft, ZoomIn, ZoomOut, ChevronLeft, ChevronRight, Maximize2, Brain, Activity, Columns2 } from 'lucide-react';
import { apiRequest } from '@/lib/api';
import { useAuth } from '@/context/AuthContext';
import CornerstoneViewer from '@/components/CornerstoneViewer';


interface OrthancImage {
    instance_id: string;
    preview_url: string;
    series_description?: string;
    view_position?: string;  // CC, MLO
    image_laterality?: string;  // L, R
    mammography_view?: string;  // LCC, RCC, LMLO, RMLO
}

export default function DicomDetailViewer() {
    const { instanceId } = useParams<{ instanceId: string }>();
    const navigate = useNavigate();
    const { user } = useAuth();
    const isRadiologyTech = user?.department === '방사선과'; // 방사선과 = 촬영 담당

    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisComplete, setAnalysisComplete] = useState(false);
    const [isSplitView, setIsSplitView] = useState(false); // 분할 뷰 토글
    const [activeViewport, setActiveViewport] = useState<1 | 2>(1); // 활성 뷰포트
    const [viewport1Index, setViewport1Index] = useState(0);
    const [viewport2Index, setViewport2Index] = useState(0);
    const [zoom, setZoom] = useState(100);
    const [allImages, setAllImages] = useState<OrthancImage[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [patientInfo, setPatientInfo] = useState<any>(null);
    const [instanceIds, setInstanceIds] = useState<string[]>([]); // Cornerstone용 instance ID 배열


    useEffect(() => {
        if (instanceId) {
            loadImage();
        }
    }, [instanceId]);

    const loadImage = async () => {
        try {
            // Try to get patient context from session storage
            const patientId = sessionStorage.getItem('currentPatientId');
            // imageType is no longer used in split view

            console.log('Loading patient data for:', patientId);

            if (patientId) {
                const response = await apiRequest('GET', `/api/mri/orthanc/patients/${patientId}/`);
                console.log('Patient API response:', response);

                if (response.success && response.images) {
                    setAllImages(response.images);
                    // Cornerstone3D용 instance ID 배열 설정
                    const ids = response.images.map((img: OrthancImage) => img.instance_id);
                    setInstanceIds(ids);

                    const index = response.images.findIndex((img: OrthancImage) => img.instance_id === instanceId);
                    if (index !== -1) {
                        setCurrentIndex(index);
                    }
                }

                // Set patient info with fallback to sessionStorage
                setPatientInfo({
                    patient_id: response.patient_id || patientId,
                    patient_name: response.patient_name || response.name || sessionStorage.getItem('currentPatientName') || 'Unknown',
                    ...response
                });
            }
        } catch (error) {
            console.error('Failed to load image:', error);
        }
    };

    const handleZoomIn = () => {
        setZoom(prev => Math.min(prev + 25, 300));
    };

    const handleZoomOut = () => {
        setZoom(prev => Math.max(prev - 25, 50));
    };

    const handlePrevImage = () => {
        if (currentIndex > 0 && allImages.length > 0) {
            const prevImage = allImages[currentIndex - 1];
            navigate(`/dicom-viewer/${prevImage.instance_id}`);
        }
    };

    const handleNextImage = () => {
        if (currentIndex < allImages.length - 1 && allImages.length > 0) {
            const nextImage = allImages[currentIndex + 1];
            navigate(`/dicom-viewer/${nextImage.instance_id}`);
        }
    };

    const handleFullscreen = () => {
        const elem = document.getElementById('dicom-viewer-container');
        if (elem) {
            if (elem.requestFullscreen) {
                elem.requestFullscreen();
            } else if ((elem as any).webkitRequestFullscreen) {
                // Safari 지원
                (elem as any).webkitRequestFullscreen();
            } else if ((elem as any).msRequestFullscreen) {
                // IE11 지원
                (elem as any).msRequestFullscreen();
            }
        }
    };


    return (
        <div className="min-h-screen bg-gray-900 text-white">
            {/* Header */}
            <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => navigate('/mri-viewer')}
                            className="text-gray-300 hover:text-white"
                        >
                            <ArrowLeft className="h-4 w-4 mr-2" />
                            MRI 뷰어로 돌아가기
                        </Button>
                        <div className="h-6 w-px bg-gray-600" />
                        <h1 className="text-xl font-bold">DICOM 상세 뷰어</h1>
                    </div>
                    <div className="flex items-center gap-4">
                        {patientInfo && (
                            <div className="flex items-center gap-4 text-sm">
                                <div>
                                    <span className="text-gray-400">환자 번호: </span>
                                    <span className="text-white font-medium">{patientInfo.patient_id || '미상'}</span>
                                </div>
                                <div className="h-4 w-px bg-gray-600" />
                                <div>
                                    <span className="text-gray-400">이름: </span>
                                    <span className="text-white font-medium">{patientInfo.patient_name || '미상'}</span>
                                </div>
                            </div>
                        )}
                        {allImages.length > 0 && (
                            <Badge variant="secondary" className="text-sm">
                                이미지 {currentIndex + 1} / {allImages.length}
                            </Badge>
                        )}
                        {/* 분할 뷰 토글 버튼 */}
                        <Button
                            variant={isSplitView ? "secondary" : "outline"}
                            className={`ml-4 ${isSplitView ? 'bg-purple-600 hover:bg-purple-700' : 'bg-gray-700 hover:bg-gray-600'} text-white font-bold border-none`}
                            onClick={() => {
                                setIsSplitView(!isSplitView);
                                if (!isSplitView) {
                                    // 분할 뷰 진입 시 두 뷰포트 모두 현재 이미지로 초기화
                                    setViewport1Index(currentIndex);
                                    setViewport2Index(currentIndex);
                                }
                            }}
                        >
                            <Columns2 className="w-4 h-4 mr-2" />
                            {isSplitView ? '단일 뷰' : '분할 뷰'}
                        </Button>
                        {!isRadiologyTech && (
                            <Button
                                variant={analysisComplete ? "secondary" : "default"}
                                className={`ml-4 ${analysisComplete ? 'bg-green-600 hover:bg-green-700' : 'bg-blue-600 hover:bg-blue-700'} text-white font-bold border-none`}
                                onClick={() => {
                                    setIsAnalyzing(true);
                                    // Simulate AI analysis delay
                                    setTimeout(() => {
                                        setIsAnalyzing(false);
                                        setAnalysisComplete(true);
                                    }, 2000);
                                }}
                                disabled={isAnalyzing || analysisComplete}
                            >
                                {isAnalyzing ? (
                                    <>
                                        <Activity className="w-4 h-4 mr-2 animate-spin" />
                                        AI 분석 중...
                                    </>
                                ) : analysisComplete ? (
                                    <>
                                        <Brain className="w-4 h-4 mr-2" />
                                        분석 완료
                                    </>
                                ) : (
                                    <>
                                        <Brain className="w-4 h-4 mr-2" />
                                        AI 정밀 분석 실행
                                    </>
                                )}
                            </Button>
                        )}
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="flex h-[calc(100vh-73px)] overflow-hidden">
                {/* Left Sidebar - Thumbnails */}
                {instanceIds.length > 0 && (
                    <div className="w-32 bg-gray-800 border-r border-gray-700 overflow-y-auto">
                        <div className="p-2 space-y-2">
                            {instanceIds.map((id, index) => {
                                const isViewport1Active = isSplitView && viewport1Index === index;
                                const isViewport2Active = isSplitView && viewport2Index === index;
                                const isSingleViewActive = !isSplitView && currentIndex === index;

                                return (
                                    <div
                                        key={id}
                                        className={`relative cursor-pointer rounded overflow-hidden transition-all ${isViewport1Active ? 'ring-2 ring-blue-500' :
                                            isViewport2Active ? 'ring-2 ring-green-500' :
                                                isSingleViewActive ? 'ring-2 ring-purple-500' :
                                                    'ring-1 ring-gray-600 hover:ring-gray-500'
                                            }`}
                                        onClick={() => {
                                            if (isSplitView) {
                                                // 분할 뷰: 활성 뷰포트에 이미지 설정
                                                if (activeViewport === 1) {
                                                    setViewport1Index(index);
                                                } else {
                                                    setViewport2Index(index);
                                                }
                                            } else {
                                                // 단일 뷰: 현재 이미지 변경
                                                setCurrentIndex(index);
                                                navigate(`/dicom-viewer/${allImages[index].instance_id}`);
                                            }
                                        }}
                                    >
                                        <img
                                            src={allImages[index]?.preview_url || ''}
                                            alt={`Slice ${index + 1}`}
                                            className="w-full h-24 object-cover"
                                        />
                                        <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-70 text-white text-xs text-center py-1">
                                            {index + 1}
                                        </div>
                                        {/* 뷰포트 인디케이터 */}
                                        {isViewport1Active && (
                                            <div className="absolute top-1 right-1 bg-blue-500 text-white text-xs px-1 rounded">
                                                1
                                            </div>
                                        )}
                                        {isViewport2Active && (
                                            <div className="absolute top-1 right-1 bg-green-500 text-white text-xs px-1 rounded">
                                                2
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}

                {/* Center - Image Display */}
                <div className="flex-1 flex flex-col min-h-0" id="dicom-viewer-container">
                    {instanceIds.length > 0 && (
                        <div className="flex-1 bg-gray-900 min-h-0">
                            {isSplitView ? (
                                // 분할 뷰: 2개 뷰포트
                                <div className="flex h-full gap-1">
                                    {/* Viewport 1 */}
                                    <div
                                        className={`flex-1 relative ${activeViewport === 1 ? 'ring-2 ring-blue-500 ring-inset' : ''
                                            }`}
                                        onClick={() => setActiveViewport(1)}
                                    >
                                        <div className="absolute top-2 left-2 z-50 bg-blue-600 text-white text-xs font-bold px-2 py-1 rounded shadow-lg border border-blue-400">
                                            뷰포트 1
                                        </div>
                                        <CornerstoneViewer
                                            key={`viewport1-${viewport1Index}`}
                                            instanceIds={instanceIds}
                                            currentIndex={viewport1Index}
                                            onIndexChange={(index) => setViewport1Index(index)}
                                            showMeasurementTools={!isRadiologyTech}
                                            viewportId="split-viewport-1"
                                        />
                                    </div>

                                    {/* Viewport 2 */}
                                    <div
                                        className={`flex-1 relative ${activeViewport === 2 ? 'ring-2 ring-green-500 ring-inset' : ''
                                            }`}
                                        onClick={() => setActiveViewport(2)}
                                    >
                                        <div className="absolute top-2 left-2 z-50 bg-green-600 text-white text-xs font-bold px-2 py-1 rounded shadow-lg border border-green-400">
                                            뷰포트 2
                                        </div>
                                        <CornerstoneViewer
                                            key={`viewport2-${viewport2Index}`}
                                            instanceIds={instanceIds}
                                            currentIndex={viewport2Index}
                                            onIndexChange={(index) => setViewport2Index(index)}
                                            showMeasurementTools={!isRadiologyTech}
                                            viewportId="split-viewport-2"
                                        />
                                    </div>
                                </div>
                            ) : (
                                // 단일 뷰
                                <CornerstoneViewer
                                    key={`cornerstone-${instanceId}-${instanceIds.length}`}
                                    instanceIds={instanceIds}
                                    currentIndex={currentIndex}
                                    onIndexChange={(index) => {
                                        setCurrentIndex(index);
                                        if (allImages[index]) {
                                            navigate(`/dicom-viewer/${allImages[index].instance_id}`);
                                        }
                                    }}
                                    showMeasurementTools={!isRadiologyTech}
                                />
                            )}
                        </div>
                    )}

                    {/* Bottom Controls */}
                    <div className="flex-shrink-0 bg-gray-800 border-t border-gray-700 px-6 py-3">
                        <div className="flex items-center justify-between">
                            {/* Navigation Controls */}
                            <div className="flex items-center gap-2">
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={handlePrevImage}
                                    disabled={currentIndex === 0 || allImages.length === 0}
                                    className="bg-gray-700 border-gray-600 text-white hover:bg-gray-600"
                                >
                                    <ChevronLeft className="h-4 w-4 mr-1" />
                                    이전
                                </Button>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={handleNextImage}
                                    disabled={currentIndex === allImages.length - 1 || allImages.length === 0}
                                    className="bg-gray-700 border-gray-600 text-white hover:bg-gray-600"
                                >
                                    다음
                                    <ChevronRight className="h-4 w-4 ml-1" />
                                </Button>
                            </div>

                            {/* Zoom Controls */}
                            <div className="flex items-center gap-4">
                                <div className="flex items-center gap-2">
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={handleZoomOut}
                                        disabled={zoom <= 50}
                                        className="bg-gray-700 border-gray-600 text-white hover:bg-gray-600"
                                    >
                                        <ZoomOut className="h-4 w-4" />
                                    </Button>
                                    <span className="text-sm text-gray-300 w-16 text-center">{zoom}%</span>
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={handleZoomIn}
                                        disabled={zoom >= 300}
                                        className="bg-gray-700 border-gray-600 text-white hover:bg-gray-600"
                                    >
                                        <ZoomIn className="h-4 w-4" />
                                    </Button>
                                </div>
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={handleFullscreen}
                                    className="bg-gray-700 border-gray-600 text-white hover:bg-gray-600"
                                >
                                    <Maximize2 className="h-4 w-4 mr-2" />
                                    전체화면
                                </Button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
