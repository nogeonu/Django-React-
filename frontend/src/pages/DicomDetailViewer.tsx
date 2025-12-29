import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ArrowLeft, ZoomIn, ZoomOut, ChevronLeft, ChevronRight, Maximize2, Grid3x3 } from 'lucide-react';
import { apiRequest } from '@/lib/api';
import { useAuth } from '@/context/AuthContext';
import { Brain, Layers, Box, ScanLine, Activity } from 'lucide-react';
import { motion } from 'framer-motion';
import CornerstoneViewer from '@/components/CornerstoneViewer';

interface OrthancImage {
    instance_id: string;
    preview_url: string;
    series_description?: string;
}

export default function DicomDetailViewer() {
    const { instanceId } = useParams<{ instanceId: string }>();
    const navigate = useNavigate();
    const { user } = useAuth();
    const isRadiology = user?.department === '영상의학과';
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisComplete, setAnalysisComplete] = useState(false);
    const [imageUrl, setImageUrl] = useState<string>('');
    const [zoom, setZoom] = useState(100);
    const [loading, setLoading] = useState(true);
    const [allImages, setAllImages] = useState<OrthancImage[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [patientInfo, setPatientInfo] = useState<any>(null);
    const [useCornerstoneViewer, setUseCornerstoneViewer] = useState(true); // Cornerstone3D 사용 여부
    const [instanceIds, setInstanceIds] = useState<string[]>([]); // Cornerstone용 instance ID 배열

    useEffect(() => {
        if (instanceId) {
            loadImage();
        }
    }, [instanceId]);

    const loadImage = async () => {
        try {
            setLoading(true);
            // Get the image preview URL
            const url = `/api/mri/orthanc/instances/${instanceId}/preview/`;
            setImageUrl(url);

            // Try to get patient context from session storage
            const patientId = sessionStorage.getItem('currentPatientId');
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
        } finally {
            setLoading(false);
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
        const elem = document.getElementById('dicom-image-container');
        if (elem) {
            if (elem.requestFullscreen) {
                elem.requestFullscreen();
            }
        }
    };

    const handleWheel = (e: React.WheelEvent) => {
        e.preventDefault();
        if (allImages.length === 0) return;

        const delta = e.deltaY > 0 ? 1 : -1;
        const newIndex = Math.max(0, Math.min(allImages.length - 1, currentIndex + delta));

        if (newIndex !== currentIndex) {
            const newImage = allImages[newIndex];
            navigate(`/dicom-viewer/${newImage.instance_id}`);
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
                        <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setUseCornerstoneViewer(!useCornerstoneViewer)}
                            className="text-gray-300 hover:text-white"
                            title={useCornerstoneViewer ? "기본 뷰어로 전환" : "Cornerstone3D 뷰어로 전환"}
                        >
                            <Grid3x3 className={`h-4 w-4 mr-2 ${useCornerstoneViewer ? 'text-blue-500' : ''}`} />
                            {useCornerstoneViewer ? 'Cornerstone3D' : '기본 뷰어'}
                        </Button>
                        {isRadiology && (
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
            <div className="flex h-[calc(100vh-73px)]">
                {/* Left Sidebar - Patient Info (Hidden for better image viewing) */}
                {/* 사이드바는 숨기고 이미지만 전체 화면으로 표시 */}
                {/* 환자 정보는 헤더에 표시되거나 필요시 주석 해제 가능 */}

                {/* Center - Image Display (Full Width) */}
                <div className="flex-1 flex flex-col w-full">
                    {/* Cornerstone3D 뷰어 또는 기본 뷰어 */}
                    {useCornerstoneViewer && instanceIds.length > 0 ? (
                        <div className="flex-1 bg-gray-900">
                            <CornerstoneViewer
                                instanceIds={instanceIds}
                                currentIndex={currentIndex}
                                onIndexChange={(index) => {
                                    setCurrentIndex(index);
                                    if (allImages[index]) {
                                        navigate(`/dicom-viewer/${allImages[index].instance_id}`);
                                    }
                                }}
                            />
                        </div>
                    ) : isRadiology ? (
                        <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-1 p-1 bg-black h-[calc(100vh-140px)] w-full">
                            {/* Top-Left: Original DICOM */}
                            <div className="relative bg-black border border-gray-800 rounded-lg overflow-hidden flex items-center justify-center group">
                                <div className="absolute top-2 left-2 z-10 bg-black/50 px-2 py-1 rounded text-xs text-blue-400 font-bold border border-blue-500/30 flex items-center gap-1">
                                    <ScanLine className="w-3 h-3" /> Original
                                </div>
                                {loading ? (
                                    <div className="text-gray-500 text-xs">Loading...</div>
                                ) : (
                                    <img
                                        src={imageUrl}
                                        alt="Original"
                                        className="max-w-full max-h-full object-contain"
                                        style={{ transform: `scale(${zoom / 100})` }}
                                    />
                                )}
                            </div>

                            {/* Top-Right: Tumor Segmentation Only */}
                            <div className="relative bg-black border border-gray-800 rounded-lg overflow-hidden flex items-center justify-center group">
                                <div className="absolute top-2 left-2 z-10 bg-black/50 px-2 py-1 rounded text-xs text-red-400 font-bold border border-red-500/30 flex items-center gap-1">
                                    <Brain className="w-3 h-3" /> Tumor Seg.
                                </div>
                                {analysisComplete ? (
                                    // Simulated Segmentation View (using filter for demo effect)
                                    <img
                                        src={imageUrl}
                                        alt="Segmentation"
                                        className="max-w-full max-h-full object-contain opacity-80"
                                        style={{ filter: 'brightness(0.7) contrast(200%) hue-rotate(90deg)', transform: `scale(${zoom / 100})` }}
                                    />
                                ) : (
                                    <div className="flex flex-col items-center justify-center text-gray-700 gap-2">
                                        <Brain className="w-8 h-8 opacity-20" />
                                        <span className="text-xs">분석 대기 중</span>
                                    </div>
                                )}
                            </div>

                            {/* Bottom-Left: Overlay */}
                            <div className="relative bg-black border border-gray-800 rounded-lg overflow-hidden flex items-center justify-center group">
                                <div className="absolute top-2 left-2 z-10 bg-black/50 px-2 py-1 rounded text-xs text-purple-400 font-bold border border-purple-500/30 flex items-center gap-1">
                                    <Layers className="w-3 h-3" /> Overlay
                                </div>
                                {analysisComplete ? (
                                    <div className="relative w-full h-full flex items-center justify-center">
                                        <img
                                            src={imageUrl}
                                            alt="Base"
                                            className="absolute inset-0 max-w-full max-h-full object-contain mx-auto"
                                            style={{ transform: `scale(${zoom / 100})` }}
                                        />
                                        {/* Simulated Overlay Layer */}
                                        <div className="absolute inset-0 bg-red-500/20 mix-blend-overlay max-w-full max-h-full mx-auto" style={{ maskImage: `url(${imageUrl})`, transform: `scale(${zoom / 100})` }}></div>
                                    </div>
                                ) : (
                                    <div className="flex flex-col items-center justify-center text-gray-700 gap-2">
                                        <Layers className="w-8 h-8 opacity-20" />
                                        <span className="text-xs">분석 대기 중</span>
                                    </div>
                                )}
                            </div>

                            {/* Bottom-Right: 3D Visualization */}
                            <div className="relative bg-black border border-gray-800 rounded-lg overflow-hidden flex items-center justify-center group">
                                <div className="absolute top-2 left-2 z-10 bg-black/50 px-2 py-1 rounded text-xs text-green-400 font-bold border border-green-500/30 flex items-center gap-1">
                                    <Box className="w-3 h-3" /> 3D Volume
                                </div>
                                {analysisComplete ? (
                                    <motion.div
                                        initial={{ rotateY: 0 }}
                                        animate={{ rotateY: 360 }}
                                        transition={{ repeat: Infinity, duration: 10, ease: "linear" }}
                                        className="w-32 h-32 border-4 border-green-500/30 rounded-full flex items-center justify-center bg-green-500/10"
                                    >
                                        <Box className="w-16 h-16 text-green-500" />
                                    </motion.div>
                                ) : (
                                    <div className="flex flex-col items-center justify-center text-gray-700 gap-2">
                                        <Box className="w-8 h-8 opacity-20" />
                                        <span className="text-xs">분석 대기 중</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    ) : (
                        <div
                            id="dicom-image-container"
                            className="flex-1 bg-black flex items-center justify-center overflow-hidden relative"
                            onWheel={handleWheel}
                        >
                            {loading ? (
                                <div className="text-gray-400">이미지 로딩 중...</div>
                            ) : (
                                <img
                                    src={imageUrl}
                                    alt="DICOM Image"
                                    className="max-w-full max-h-full object-contain transition-transform duration-200"
                                    style={{ transform: `scale(${zoom / 100})` }}
                                />
                            )}
                        </div>
                    )}

                    {/* Bottom Controls */}
                    <div className="bg-gray-800 border-t border-gray-700 px-6 py-4">
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
