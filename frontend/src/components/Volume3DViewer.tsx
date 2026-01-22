/**
 * Cornerstone3D 기반 3D 볼륨 렌더링 뷰어
 * DICOM 이미지와 세그멘테이션을 3D로 시각화하여 여러 각도에서 종양 모양을 관찰할 수 있음
 */
import { useEffect, useRef, useState } from 'react';
import {
  RenderingEngine,
  Enums,
  type Types,
  volumeLoader,
  cache,
} from '@cornerstonejs/core';
import {
  addTool,
  ToolGroupManager,
  ZoomTool,
  PanTool,
  WindowLevelTool,
} from '@cornerstonejs/tools';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { RotateCw, ZoomIn, ZoomOut, Layers, Eye, EyeOff } from 'lucide-react';
import { initCornerstone, createImageId } from '@/lib/cornerstone';

interface Volume3DViewerProps {
  instanceIds: string[]; // DICOM 인스턴스 ID 배열
  segmentationInstanceId?: string; // 세그멘테이션 인스턴스 ID (선택)
  patientId?: string;
}

export default function Volume3DViewer({
  instanceIds,
  segmentationInstanceId,
  patientId,
}: Volume3DViewerProps) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [showSegmentation, setShowSegmentation] = useState(true);
  const [volumeOpacity, setVolumeOpacity] = useState(0.7);
  const [segmentationOpacity, setSegmentationOpacity] = useState(0.8);
  const renderingEngineRef = useRef<RenderingEngine | null>(null);
  const volumeIdRef = useRef<string | null>(null);
  const segmentationVolumeIdRef = useRef<string | null>(null);
  const uniqueId = useRef<string>(`volume3d_${Date.now()}_${Math.random()}`);
  const viewportIdRef = useRef<string>(`viewport_${uniqueId.current}`);
  const toolGroupIdRef = useRef<string>(`toolGroup_${uniqueId.current}`);

  // Cornerstone 초기화
  useEffect(() => {
    const initialize = async () => {
      try {
        await initCornerstone();

        // 3D 도구 등록
        addTool(ZoomTool);
        addTool(PanTool);
        addTool(WindowLevelTool);

        setIsInitialized(true);
      } catch (error) {
        console.error('Failed to initialize Cornerstone3D:', error);
      }
    };

    initialize();
  }, []);

  // 볼륨 로드 및 렌더링
  useEffect(() => {
    if (!isInitialized || !viewportRef.current || instanceIds.length === 0) {
      return;
    }

    const loadVolume = async () => {
      try {
        setIsLoading(true);

        const renderingEngineId = `volume3d_engine_${uniqueId.current}`;
        const renderingEngine = new RenderingEngine(renderingEngineId);
        renderingEngineRef.current = renderingEngine;

        // Viewport 생성
        const viewportInput = {
          viewportId: viewportIdRef.current,
          type: Enums.ViewportType.VOLUME_3D,
          element: viewportRef.current,
        };

        renderingEngine.setViewports([viewportInput]);

        const viewport = renderingEngine.getViewport(viewportIdRef.current) as Types.IVolumeViewport;

        // 이미지 ID 생성
        const imageIds = instanceIds.map(id => createImageId(id));

        // 볼륨 로드
        const volume = await volumeLoader.createAndCacheVolume('cornerstoneStreamingImageVolume', {
          imageIds,
        });

        volumeIdRef.current = volume.volumeId;
        await volume.load();

        // 볼륨을 뷰포트에 설정
        viewport.setVolumes([
          {
            volumeId: volume.volumeId,
            callback: ({ volumeActor }) => {
              // 볼륨 렌더링 설정
              const volumeProperty = volumeActor.getProperty();
              volumeProperty.setScalarOpacity(0, volumeOpacity);
              volumeProperty.setRGBTransferFunction(0, [
                { value: 0, opacity: 0.0, rgb: [0, 0, 0] },
                { value: 500, opacity: 0.2, rgb: [0.5, 0.5, 0.5] },
                { value: 1000, opacity: 0.4, rgb: [1, 1, 1] },
                { value: 2000, opacity: 0.6, rgb: [1, 0.9, 0.8] },
              ]);
              volumeProperty.setInterpolationTypeToLinear();
              volumeProperty.setShade(true);
              volumeProperty.setAmbient(0.2);
              volumeProperty.setDiffuse(0.7);
              volumeProperty.setSpecular(0.3);
              volumeProperty.setSpecularPower(10);
            },
          },
        ]);

        // 세그멘테이션 볼륨 로드 (있는 경우)
        if (segmentationInstanceId && showSegmentation) {
          try {
            const segImageId = createImageId(segmentationInstanceId);
            const segVolume = await volumeLoader.createAndCacheVolume('cornerstoneStreamingImageVolume', {
              imageIds: [segImageId],
            });

            segmentationVolumeIdRef.current = segVolume.volumeId;
            await segVolume.load();

            // 세그멘테이션을 뷰포트에 추가
            viewport.addVolumes([
              {
                volumeId: segVolume.volumeId,
                callback: ({ volumeActor }) => {
                  const volumeProperty = volumeActor.getProperty();
                  // 세그멘테이션은 빨간색/핑크색으로 표시
                  volumeProperty.setScalarOpacity(0, segmentationOpacity);
                  volumeProperty.setRGBTransferFunction(0, [
                    { value: 0, opacity: 0.0, rgb: [0, 0, 0] },
                    { value: 1, opacity: segmentationOpacity, rgb: [1, 0, 0.5] }, // 핑크색
                  ]);
                  volumeProperty.setInterpolationTypeToNearest();
                },
              },
            ]);
          } catch (segError) {
            console.warn('Failed to load segmentation volume:', segError);
          }
        }

        // 렌더링
        viewport.render();

        // 도구 그룹 설정
        const toolGroup = ToolGroupManager.createToolGroup(toolGroupIdRef.current);
        if (toolGroup) {
          toolGroup.addViewport(viewportIdRef.current, renderingEngineId);
          // 3D 볼륨 뷰포트는 기본적으로 마우스로 회전 가능
          toolGroup.setToolActive(ZoomTool.toolName, {
            bindings: [{ mouseButton: Enums.MouseBindings.Secondary }],
          });
          toolGroup.setToolActive(PanTool.toolName, {
            bindings: [{ mouseButton: Enums.MouseBindings.Auxiliary }],
          });
        }

        setIsLoading(false);
      } catch (error) {
        console.error('Failed to load volume:', error);
        setIsLoading(false);
      }
    };

    loadVolume();

    // Cleanup
    return () => {
      if (renderingEngineRef.current) {
        try {
          const toolGroup = ToolGroupManager.getToolGroup(toolGroupIdRef.current);
          if (toolGroup) {
            toolGroup.removeViewports(renderingEngineRef.current.id, viewportIdRef.current);
            ToolGroupManager.destroyToolGroup(toolGroupIdRef.current);
          }

          renderingEngineRef.current.destroy();
          renderingEngineRef.current = null;
        } catch (error) {
          console.warn('Error cleaning up rendering engine:', error);
        }
      }

      // 볼륨 캐시 정리
      if (volumeIdRef.current) {
        cache.removeVolumeLoadObject(volumeIdRef.current);
      }
      if (segmentationVolumeIdRef.current) {
        cache.removeVolumeLoadObject(segmentationVolumeIdRef.current);
      }
    };
  }, [isInitialized, instanceIds, segmentationInstanceId, showSegmentation, volumeOpacity, segmentationOpacity]);

  // 볼륨 투명도 업데이트
  useEffect(() => {
    if (!renderingEngineRef.current || !volumeIdRef.current) return;

    try {
      const viewport = renderingEngineRef.current.getViewport(viewportIdRef.current) as Types.IVolumeViewport;
      if (!viewport) return;

      const volumeActor = viewport.getActor(volumeIdRef.current);
      if (volumeActor) {
        const volumeProperty = volumeActor.getProperty();
        volumeProperty.setScalarOpacity(0, volumeOpacity);
        viewport.render();
      }
    } catch (error) {
      console.warn('Failed to update volume opacity:', error);
    }
  }, [volumeOpacity]);

  // 세그멘테이션 투명도 업데이트
  useEffect(() => {
    if (!renderingEngineRef.current || !segmentationVolumeIdRef.current || !showSegmentation) return;

    try {
      const viewport = renderingEngineRef.current.getViewport(viewportIdRef.current) as Types.IVolumeViewport;
      if (!viewport) return;

      const volumeActor = viewport.getActor(segmentationVolumeIdRef.current);
      if (volumeActor) {
        const volumeProperty = volumeActor.getProperty();
        volumeProperty.setScalarOpacity(0, segmentationOpacity);
        volumeProperty.setRGBTransferFunction(0, [
          { value: 0, opacity: 0.0, rgb: [0, 0, 0] },
          { value: 1, opacity: segmentationOpacity, rgb: [1, 0, 0.5] },
        ]);
        viewport.render();
      }
    } catch (error) {
      console.warn('Failed to update segmentation opacity:', error);
    }
  }, [segmentationOpacity, showSegmentation]);

  const handleResetView = () => {
    if (!renderingEngineRef.current) return;

    try {
      const viewport = renderingEngineRef.current.getViewport(viewportIdRef.current) as Types.IVolumeViewport;
      if (viewport) {
        viewport.resetCamera();
        viewport.render();
      }
    } catch (error) {
      console.warn('Failed to reset view:', error);
    }
  };

  return (
    <div className="w-full h-full flex flex-col bg-gray-950 rounded-lg overflow-hidden">
      {/* 컨트롤 패널 */}
      <div className="flex items-center justify-between p-4 bg-gray-900 border-b border-gray-800">
        <div className="flex items-center gap-4">
          <Badge variant="outline" className="text-xs">
            {instanceIds.length} slices
          </Badge>
          {segmentationInstanceId && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowSegmentation(!showSegmentation)}
              className="text-xs"
            >
              {showSegmentation ? <Eye className="w-4 h-4 mr-1" /> : <EyeOff className="w-4 h-4 mr-1" />}
              세그멘테이션
            </Button>
          )}
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <Layers className="w-4 h-4" />
            <span>볼륨 투명도:</span>
            <Slider
              value={[volumeOpacity]}
              onValueChange={([value]) => setVolumeOpacity(value)}
              min={0}
              max={1}
              step={0.1}
              className="w-24"
            />
            <span className="w-8 text-right">{Math.round(volumeOpacity * 100)}%</span>
          </div>

          {showSegmentation && segmentationInstanceId && (
            <div className="flex items-center gap-2 text-xs text-gray-400">
              <span>세그멘테이션 투명도:</span>
              <Slider
                value={[segmentationOpacity]}
                onValueChange={([value]) => setSegmentationOpacity(value)}
                min={0}
                max={1}
                step={0.1}
                className="w-24"
              />
              <span className="w-8 text-right">{Math.round(segmentationOpacity * 100)}%</span>
            </div>
          )}

          <Button variant="outline" size="sm" onClick={handleResetView}>
            <RotateCw className="w-4 h-4 mr-1" />
            뷰 리셋
          </Button>
        </div>
      </div>

      {/* 3D 뷰포트 */}
      <div className="flex-1 relative">
        <div
          ref={viewportRef}
          className="w-full h-full"
          style={{ minHeight: '600px' }}
        />
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-950/80">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
              <p className="text-white text-sm">3D 볼륨 로딩 중...</p>
            </div>
          </div>
        )}

        {/* 사용 안내 */}
        {!isLoading && (
          <div className="absolute bottom-4 left-4 bg-black/70 text-white text-xs p-2 rounded">
            <p>🖱️ 왼쪽 클릭 + 드래그: 회전</p>
            <p>🖱️ 오른쪽 클릭 + 드래그: 줌</p>
            <p>🖱️ 휠: 줌 인/아웃</p>
          </div>
        )}
      </div>
    </div>
  );
}
