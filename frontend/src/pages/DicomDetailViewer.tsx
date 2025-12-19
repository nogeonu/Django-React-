import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ArrowLeft, ZoomIn, ZoomOut, ChevronLeft, ChevronRight, Maximize2 } from 'lucide-react';
import { apiRequest } from '@/lib/api';

interface OrthancImage {
    instance_id: string;
    preview_url: string;
    series_description?: string;
}

export default function DicomDetailViewer() {
    const { instanceId } = useParams<{ instanceId: string }>();
    const navigate = useNavigate();
    const [imageUrl, setImageUrl] = useState<string>('');
    const [zoom, setZoom] = useState(100);
    const [loading, setLoading] = useState(true);
    const [allImages, setAllImages] = useState<OrthancImage[]>([]);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [patientInfo, setPatientInfo] = useState<any>(null);

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
            if (patientId) {
                const response = await apiRequest('GET', `/api/mri/orthanc/patients/${patientId}/`);
                if (response.success && response.images) {
                    setAllImages(response.images);
                    const index = response.images.findIndex((img: OrthancImage) => img.instance_id === instanceId);
                    if (index !== -1) {
                        setCurrentIndex(index);
                    }
                }
                setPatientInfo(response);
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
                    {allImages.length > 0 && (
                        <Badge variant="secondary" className="text-sm">
                            이미지 {currentIndex + 1} / {allImages.length}
                        </Badge>
                    )}
                </div>
            </div>

            {/* Main Content */}
            <div className="flex h-[calc(100vh-73px)]">
                {/* Left Sidebar - Patient Info */}
                {patientInfo && (
                    <div className="w-80 bg-gray-800 border-r border-gray-700 p-6 overflow-y-auto">
                        <Card className="bg-gray-900 border-gray-700">
                            <CardHeader>
                                <CardTitle className="text-white">환자 정보</CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-3 text-sm">
                                <div>
                                    <p className="text-gray-400">Patient ID</p>
                                    <p className="text-white font-medium">{patientInfo.patient_id || 'N/A'}</p>
                                </div>
                                <div>
                                    <p className="text-gray-400">이름</p>
                                    <p className="text-white font-medium">{patientInfo.patient_name || 'N/A'}</p>
                                </div>
                                {allImages[currentIndex]?.series_description && (
                                    <div>
                                        <p className="text-gray-400">Series</p>
                                        <p className="text-white font-medium">{allImages[currentIndex].series_description}</p>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    </div>
                )}

                {/* Center - Image Display */}
                <div className="flex-1 flex flex-col">
                    <div
                        id="dicom-image-container"
                        className="flex-1 bg-black flex items-center justify-center overflow-hidden relative"
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
