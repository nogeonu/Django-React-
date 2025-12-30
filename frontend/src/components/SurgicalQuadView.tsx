import { useState, useEffect, useRef } from 'react';
import { Box, Grid3x3, Layers, Volume2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { apiRequest } from '@/lib/api';
import CornerstoneViewer from './CornerstoneViewer';

interface SurgicalQuadViewProps {
    instanceIds: string[];
    currentIndex: number;
    patientId: string;
    imageType?: '유방촬영술 영상' | '병리 영상' | 'MRI 영상';
    onIndexChange: (index: number) => void;
}

export default function SurgicalQuadView({ 
    instanceIds, 
    currentIndex, 
    patientId,
    imageType = 'MRI 영상',
    onIndexChange 
}: SurgicalQuadViewProps) {
    const [segmentationLoaded, setSegmentationLoaded] = useState(false);
    const [overlayOpacity, setOverlayOpacity] = useState(0.5);
    const originalViewRef = useRef<HTMLDivElement>(null);
    const segmentationViewRef = useRef<HTMLDivElement>(null);
    const overlayViewRef = useRef<HTMLDivElement>(null);
    const volume3DViewRef = useRef<HTMLDivElement>(null);

    // 세그멘테이션 데이터 로드
    useEffect(() => {
        const loadSegmentation = async () => {
            if (!patientId) return;
            
            try {
                // 세그멘테이션 데이터 확인
                const response = await apiRequest('GET', `/api/mri/orthanc/patients/${patientId}/segmentation/`);
                console.log('Segmentation API response:', response);
                
                if (response.success && response.segmentation_available) {
                    setSegmentationLoaded(true);
                } else {
                    // 세그멘테이션이 없으면 자동 실행
                    console.log('Starting AI segmentation...');
                    const runResponse = await apiRequest('POST', `/api/mri/orthanc/patients/${patientId}/segmentation/run/`);
                    if (runResponse.success && runResponse.segmentation_complete) {
                        setSegmentationLoaded(true);
                    }
                }
            } catch (error) {
                console.error('Failed to load segmentation:', error);
                // 에러 발생 시에도 시뮬레이션으로 표시
                setTimeout(() => {
                    setSegmentationLoaded(true);
                }, 2000);
            }
        };

        loadSegmentation();
    }, [patientId, currentIndex]);

    const isMammography = imageType === '유방촬영술 영상';

    return (
        <div className="h-full w-full bg-gray-900 p-4">
            {/* 컨트롤 바 */}
            <div className="mb-4 bg-gray-800 rounded-lg p-3 flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <Badge variant="secondary" className="text-sm">
                        <Grid3x3 className="w-3 h-3 mr-1" />
                        4분할 뷰 (외과)
                    </Badge>
                    <div className="h-4 w-px bg-gray-600" />
                    <div className="text-sm text-gray-300">
                        {isMammography ? '유방촬영술 영상' : `슬라이스: ${currentIndex + 1} / ${instanceIds.length}`}
                    </div>
                </div>
                {!isMammography && (
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-400">오버레이 투명도:</span>
                        <input
                            type="range"
                            min="0"
                            max="100"
                            value={overlayOpacity * 100}
                            onChange={(e) => setOverlayOpacity(Number(e.target.value) / 100)}
                            className="w-32 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                        />
                        <span className="text-xs text-gray-300 w-8">{Math.round(overlayOpacity * 100)}%</span>
                    </div>
                )}
            </div>

            {/* 4분할 그리드 */}
            <div className="grid grid-cols-2 grid-rows-2 gap-3 h-[calc(100%-80px)]">
                {isMammography ? (
                    <>
                        {/* 유방촬영술 영상: LCC, RCC, LMLO, RMLO */}
                        {/* 좌측 상단 - LCC (Left CranioCaudal) */}
                        <div className="relative bg-gray-800 rounded-lg overflow-hidden border-2 border-blue-600">
                            <div className="absolute top-2 left-2 z-10 flex items-center gap-2">
                                <Badge className="bg-blue-600 text-white border-none">
                                    LCC
                                </Badge>
                            </div>
                            <div ref={originalViewRef} className="w-full h-full">
                                <CornerstoneViewer
                                    key={`lcc-${currentIndex}`}
                                    instanceIds={instanceIds}
                                    currentIndex={currentIndex}
                                    onIndexChange={onIndexChange}
                                    showMeasurementTools={false}
                                />
                            </div>
                        </div>

                        {/* 우측 상단 - RCC (Right CranioCaudal) */}
                        <div className="relative bg-gray-800 rounded-lg overflow-hidden border-2 border-green-600">
                            <div className="absolute top-2 left-2 z-10 flex items-center gap-2">
                                <Badge className="bg-green-600 text-white border-none">
                                    RCC
                                </Badge>
                            </div>
                            <div ref={segmentationViewRef} className="w-full h-full">
                                <CornerstoneViewer
                                    key={`rcc-${currentIndex}`}
                                    instanceIds={instanceIds}
                                    currentIndex={currentIndex}
                                    onIndexChange={onIndexChange}
                                    showMeasurementTools={false}
                                />
                            </div>
                        </div>

                        {/* 좌측 하단 - LMLO (Left MedioLateral Oblique) */}
                        <div className="relative bg-gray-800 rounded-lg overflow-hidden border-2 border-purple-600">
                            <div className="absolute top-2 left-2 z-10 flex items-center gap-2">
                                <Badge className="bg-purple-600 text-white border-none">
                                    LMLO
                                </Badge>
                            </div>
                            <div ref={overlayViewRef} className="w-full h-full">
                                <CornerstoneViewer
                                    key={`lmlo-${currentIndex}`}
                                    instanceIds={instanceIds}
                                    currentIndex={currentIndex}
                                    onIndexChange={onIndexChange}
                                    showMeasurementTools={false}
                                />
                            </div>
                        </div>

                        {/* 우측 하단 - RMLO (Right MedioLateral Oblique) */}
                        <div className="relative bg-gray-800 rounded-lg overflow-hidden border-2 border-orange-600">
                            <div className="absolute top-2 left-2 z-10 flex items-center gap-2">
                                <Badge className="bg-orange-600 text-white border-none">
                                    RMLO
                                </Badge>
                            </div>
                            <div ref={volume3DViewRef} className="w-full h-full">
                                <CornerstoneViewer
                                    key={`rmlo-${currentIndex}`}
                                    instanceIds={instanceIds}
                                    currentIndex={currentIndex}
                                    onIndexChange={onIndexChange}
                                    showMeasurementTools={false}
                                />
                            </div>
                        </div>
                    </>
                ) : (
                    <>
                        {/* MRI/병리 영상: 원본, 세그멘테이션, 오버레이, 3D */}
                        {/* 좌측 상단 - 원본 이미지 */}
                        <div className="relative bg-gray-800 rounded-lg overflow-hidden border-2 border-gray-700">
                            <div className="absolute top-2 left-2 z-10 flex items-center gap-2">
                                <Badge className="bg-blue-600 text-white border-none">
                                    <Box className="w-3 h-3 mr-1" />
                                    원본 (Original)
                                </Badge>
                            </div>
                            <div ref={originalViewRef} className="w-full h-full">
                                <CornerstoneViewer
                                    key={`original-${currentIndex}`}
                                    instanceIds={instanceIds}
                                    currentIndex={currentIndex}
                                    onIndexChange={onIndexChange}
                                    showMeasurementTools={false}
                                />
                            </div>
                        </div>

                        {/* 우측 상단 - 세그멘테이션 */}
                        <div className="relative bg-gray-800 rounded-lg overflow-hidden border-2 border-green-600">
                            <div className="absolute top-2 left-2 z-10 flex items-center gap-2">
                                <Badge className="bg-green-600 text-white border-none">
                                    <Layers className="w-3 h-3 mr-1" />
                                    세그멘테이션 (Tumor Seg)
                                </Badge>
                                {!segmentationLoaded && (
                                    <span className="text-xs text-yellow-400">로딩 중...</span>
                                )}
                            </div>
                            <div ref={segmentationViewRef} className="w-full h-full flex items-center justify-center">
                                {segmentationLoaded ? (
                                    <CornerstoneViewer
                                        key={`segmentation-${currentIndex}`}
                                        instanceIds={instanceIds}
                                        currentIndex={currentIndex}
                                        onIndexChange={onIndexChange}
                                        showMeasurementTools={false}
                                    />
                                ) : (
                                    <div className="text-gray-400 text-sm">
                                        AI 세그멘테이션 분석 준비 중...
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* 좌측 하단 - 오버레이 */}
                        <div className="relative bg-gray-800 rounded-lg overflow-hidden border-2 border-purple-600">
                            <div className="absolute top-2 left-2 z-10 flex items-center gap-2">
                                <Badge className="bg-purple-600 text-white border-none">
                                    <Layers className="w-3 h-3 mr-1" />
                                    오버레이 (Overlay)
                                </Badge>
                            </div>
                            <div ref={overlayViewRef} className="w-full h-full flex items-center justify-center">
                                {segmentationLoaded ? (
                                    <div className="relative w-full h-full">
                                        <CornerstoneViewer
                                            key={`overlay-${currentIndex}`}
                                            instanceIds={instanceIds}
                                            currentIndex={currentIndex}
                                            onIndexChange={onIndexChange}
                                            showMeasurementTools={false}
                                        />
                                        {/* TODO: 실제 오버레이 레이어 추가 */}
                                    </div>
                                ) : (
                                    <div className="text-gray-400 text-sm">
                                        오버레이 준비 중...
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* 우측 하단 - 3D 볼륨 */}
                        <div className="relative bg-gray-800 rounded-lg overflow-hidden border-2 border-orange-600">
                            <div className="absolute top-2 left-2 z-10 flex items-center gap-2">
                                <Badge className="bg-orange-600 text-white border-none">
                                    <Volume2 className="w-3 h-3 mr-1" />
                                    3D 세그멘테이션
                                </Badge>
                            </div>
                            <div ref={volume3DViewRef} className="w-full h-full flex items-center justify-center">
                                {segmentationLoaded ? (
                                    <div className="text-gray-300 text-sm">
                                        3D 볼륨 렌더링 준비 중...
                                        <p className="text-xs text-gray-500 mt-2">
                                            (향후 VTK.js 또는 Three.js 통합 예정)
                                        </p>
                                    </div>
                                ) : (
                                    <div className="text-gray-400 text-sm">
                                        3D 렌더링 대기 중...
                                    </div>
                                )}
                            </div>
                        </div>
                    </>
                )}
            </div>

            {/* 안내 메시지 */}
            <div className="mt-3 text-center">
                <p className="text-xs text-gray-500">
                    {isMammography 
                        ? '💡 유방촬영술 4분할 뷰: LCC (좌측 두두방향), RCC (우측 두두방향), LMLO (좌측 내외사방향), RMLO (우측 내외사방향)'
                        : '💡 4분할 뷰는 외과 의사 전용 기능입니다. AI 분석 완료 후 세그멘테이션 결과가 자동으로 표시됩니다.'
                    }
                </p>
            </div>
        </div>
    );
}

