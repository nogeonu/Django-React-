import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ArrowLeft, ZoomIn, ZoomOut, ChevronLeft, ChevronRight, Maximize2, Brain, Activity, Grid3x3 } from 'lucide-react';
import { apiRequest } from '@/lib/api';
import { useAuth } from '@/context/AuthContext';
import CornerstoneViewer from '@/components/CornerstoneViewer';
import SurgicalQuadView from '@/components/SurgicalQuadView';

interface OrthancImage {
    instance_id: string;
    preview_url: string;
    series_description?: string;
}

export default function DicomDetailViewer() {
    const { instanceId } = useParams<{ instanceId: string }>();
    const navigate = useNavigate();
    const { user } = useAuth();
    const isRadiologyTech = user?.department === '방사선과'; // 방사선과 = 촬영 담당
    const isSurgeon = user?.department === '외과'; // 외과 의사
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisComplete, setAnalysisComplete] = useState(false);
    const [showQuadView, setShowQuadView] = useState(false); // 4분할 뷰 토글
    const [zoom, setZoom] = useState(100);
    const [allImages, setAllImages] = useState<OrthancImage[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [patientInfo, setPatientInfo] = useState<any>(null);
    const [instanceIds, setInstanceIds] = useState<string[]>([]); // Cornerstone용 instance ID 배열
    const [imageType, setImageType] = useState<'유방촬영술 영상' | '병리 영상' | 'MRI 영상'>('MRI 영상');

    useEffect(() => {
        if (instanceId) {
            loadImage();
        }
    }, [instanceId]);

    const loadImage = async () => {
        try {
            // Try to get patient context from session storage
            const patientId = sessionStorage.getItem('currentPatientId');
            const storedImageType = sessionStorage.getItem('currentImageType') as '유방촬영술 영상' | '병리 영상' | 'MRI 영상' | null;
            if (storedImageType) {
                setImageType(storedImageType);
            }
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
                        {/* 외과 의사 전용 4분할 뷰 버튼 */}
                        {isSurgeon && (
                            <Button
                                variant={showQuadView ? "secondary" : "outline"}
                                className={`ml-4 ${showQuadView ? 'bg-purple-600 hover:bg-purple-700' : 'bg-gray-700 hover:bg-gray-600'} text-white font-bold border-none`}
                                onClick={() => setShowQuadView(!showQuadView)}
                            >
                                <Grid3x3 className="w-4 h-4 mr-2" />
                                {showQuadView ? '단일 뷰' : '4분할 뷰'}
                            </Button>
                        )}
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
            <div className="flex flex-col h-[calc(100vh-73px)] overflow-hidden">
                {/* Center - Image Display (Full Width) */}
                <div className="flex-1 flex flex-col w-full min-h-0" id="dicom-viewer-container">
                    {/* 외과 의사 4분할 뷰 또는 일반 뷰어 */}
                    {instanceIds.length > 0 && (
                        <div className="flex-1 bg-gray-900 min-h-0">
                            {showQuadView && isSurgeon ? (
                                <SurgicalQuadView
                                    instanceIds={instanceIds}
                                    currentIndex={currentIndex}
                                    patientId={patientInfo?.patient_id || ''}
                                    imageType={imageType}
                                    onIndexChange={(index) => {
                                        setCurrentIndex(index);
                                        if (allImages[index]) {
                                            navigate(`/dicom-viewer/${allImages[index].instance_id}`);
                                        }
                                    }}
                                />
                            ) : (
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
