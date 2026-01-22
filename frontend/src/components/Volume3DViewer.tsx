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
import { RotateCw, Layers, Eye, EyeOff } from 'lucide-react';
import { initCornerstone, createImageId } from '@/lib/cornerstone';
import { Enums as ToolEnums } from '@cornerstonejs/tools';

interface Volume3DViewerProps {
  instanceIds: string[]; // DICOM 인스턴스 ID 배열
  segmentationInstanceId?: string; // 세그멘테이션 인스턴스 ID (선택)
  segmentationFrames?: Array<{ index: number; mask_base64: string }>; // 세그멘테이션 프레임 (base64 마스크)
  patientId?: string;
}

export default function Volume3DViewer({
  instanceIds,
  segmentationInstanceId,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  segmentationFrames: _segmentationFrames = [], // 향후 세그멘테이션 마스킹 정보 직접 사용 예정
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
        if (!viewportRef.current) {
          setIsLoading(false);
          return;
        }

        const viewportInput = {
          viewportId: viewportIdRef.current,
          type: Enums.ViewportType.VOLUME_3D,
          element: viewportRef.current,
        };

        renderingEngine.setViewports([viewportInput]);

        const viewport = renderingEngine.getViewport(viewportIdRef.current) as Types.IVolumeViewport;

        // 이미지 ID 생성 (CornerstoneViewer와 동일한 형식 사용)
        const imageIds = instanceIds.map(id => createImageId(`/api/mri/orthanc/instances/${id}/file`));
        
        console.log('[Volume3DViewer] 이미지 ID 생성:', imageIds.length, '개');

        // 볼륨 로드
        console.log('[Volume3DViewer] 볼륨 로드 시작...');
        const volume = await volumeLoader.createAndCacheVolume('cornerstoneStreamingImageVolume', {
          imageIds,
        });

        volumeIdRef.current = volume.volumeId;
        console.log('[Volume3DViewer] 볼륨 로딩 중...');
        await volume.load();
        console.log('[Volume3DViewer] 볼륨 로드 완료:', volume.volumeId);

        // 볼륨을 뷰포트에 설정
        viewport.setVolumes([
          {
            volumeId: volume.volumeId,
            callback: ({ volumeActor }) => {
              // 볼륨 렌더링 설정
              // @ts-ignore - Cornerstone3D volume property API
              const volumeProperty = volumeActor.getProperty();
              if (volumeProperty) {
                // @ts-ignore - VTK API types
                const scalarOpacity = volumeProperty.getScalarOpacity();
                if (scalarOpacity) {
                  scalarOpacity.removeAllPoints();
                  scalarOpacity.addPoint(0, 0.0);
                  scalarOpacity.addPoint(500, 0.2 * volumeOpacity);
                  scalarOpacity.addPoint(1000, 0.4 * volumeOpacity);
                  scalarOpacity.addPoint(2000, 0.6 * volumeOpacity);
                }
                // @ts-ignore - VTK API types
                const rgbTransferFunction = volumeProperty.getRGBTransferFunction();
                if (rgbTransferFunction) {
                  rgbTransferFunction.removeAllPoints();
                  rgbTransferFunction.addRGBPoint(0, 0, 0, 0);
                  rgbTransferFunction.addRGBPoint(500, 0.5, 0.5, 0.5);
                  rgbTransferFunction.addRGBPoint(1000, 1, 1, 1);
                  rgbTransferFunction.addRGBPoint(2000, 1, 0.9, 0.8);
                }
                // @ts-ignore - VTK API types
                volumeProperty.setInterpolationTypeToLinear();
                // @ts-ignore - VTK API types
                volumeProperty.setShade(true);
                // @ts-ignore - VTK API types
                volumeProperty.setAmbient(0.2);
                // @ts-ignore - VTK API types
                volumeProperty.setDiffuse(0.7);
                // @ts-ignore - VTK API types
                volumeProperty.setSpecular(0.3);
                // @ts-ignore - VTK API types
                volumeProperty.setSpecularPower(10);
              }
            },
          },
        ]);
        
        // 렌더링 (볼륨 설정 후 즉시)
        viewport.render();

        // 세그멘테이션 볼륨 로드 (있는 경우) - Orthanc에 저장된 DICOM SEG 파일 사용
        if (showSegmentation && (segmentationInstanceId || _segmentationFrames.length > 0)) {
          try {
            console.log('[Volume3DViewer] 🎯 세그멘테이션 볼륨 로드 시작...', {
              segmentationInstanceId,
              hasInstanceId: !!segmentationInstanceId,
              hasFrames: _segmentationFrames.length > 0,
              showSegmentation,
            });

            let segImageIds: string[] = [];
            
            if (segmentationInstanceId) {
              // DICOM SEG 파일에서 로드
              const segImageId = createImageId(`/api/mri/orthanc/instances/${segmentationInstanceId}/file`);
              segImageIds = [segImageId];
            } else if (_segmentationFrames.length > 0) {
              // 세그멘테이션 프레임이 있으면 나중에 직접 볼륨 생성 (현재는 DICOM SEG 우선)
              console.log('[Volume3DViewer] 세그멘테이션 프레임이 있지만 DICOM SEG 인스턴스 ID가 없습니다.');
            }

            if (segImageIds.length > 0) {
              const segVolume = await volumeLoader.createAndCacheVolume('cornerstoneStreamingImageVolume', {
                imageIds: segImageIds,
              });

              segmentationVolumeIdRef.current = segVolume.volumeId;
              await segVolume.load();
              console.log('[Volume3DViewer] 세그멘테이션 볼륨 로드 완료:', segVolume.volumeId);

              // 세그멘테이션을 뷰포트에 추가 (빨간색으로 표시)
              viewport.addVolumes([
                {
                  volumeId: segVolume.volumeId,
                  callback: ({ volumeActor }) => {
                    // @ts-ignore - Cornerstone3D volume property API
                    const volumeProperty = volumeActor.getProperty();
                    if (volumeProperty) {
                      // @ts-ignore - VTK API types
                      const scalarOpacity = volumeProperty.getScalarOpacity();
                      if (scalarOpacity) {
                        scalarOpacity.removeAllPoints();
                        scalarOpacity.addPoint(0, 0.0);
                        scalarOpacity.addPoint(1, segmentationOpacity); // 종양 영역만 표시
                      }
                      // @ts-ignore - VTK API types
                      const rgbTransferFunction = volumeProperty.getRGBTransferFunction();
                      if (rgbTransferFunction) {
                        rgbTransferFunction.removeAllPoints();
                        rgbTransferFunction.addRGBPoint(0, 0, 0, 0); // 배경: 검은색
                        rgbTransferFunction.addRGBPoint(1, 1, 0, 0); // 종양: 빨간색 (두 번째 이미지처럼)
                      }
                      // @ts-ignore - VTK API types
                      volumeProperty.setInterpolationTypeToNearest();
                    }
                  },
                },
              ]);
              
              // 세그멘테이션 추가 후 렌더링
              viewport.render();
              console.log('[Volume3DViewer] 세그멘테이션 볼륨 추가 및 렌더링 완료');
            }
          } catch (segError) {
            console.error('[Volume3DViewer] 세그멘테이션 볼륨 로드 실패:', segError);
            console.error('[Volume3DViewer] 에러 상세:', {
              segmentationInstanceId,
              hasFrames: _segmentationFrames.length > 0,
              errorMessage: segError instanceof Error ? segError.message : String(segError),
            });
          }
        }

        // 최종 렌더링
        console.log('[Volume3DViewer] 최종 렌더링 시작...');
        viewport.render();
        console.log('[Volume3DViewer] 렌더링 완료');

        // 도구 그룹 설정
        const toolGroup = ToolGroupManager.createToolGroup(toolGroupIdRef.current);
        if (toolGroup) {
          toolGroup.addViewport(viewportIdRef.current, renderingEngineId);
          // 3D 볼륨 뷰포트는 기본적으로 마우스로 회전 가능
          toolGroup.setToolActive(ZoomTool.toolName, {
            bindings: [{ mouseButton: ToolEnums.MouseBindings.Secondary }],
          });
          toolGroup.setToolActive(PanTool.toolName, {
            bindings: [{ mouseButton: ToolEnums.MouseBindings.Auxiliary }],
          });
        }

        setIsLoading(false);
      } catch (error) {
        console.error('[Volume3DViewer] 볼륨 로드 실패:', error);
        console.error('[Volume3DViewer] 에러 상세:', {
          instanceIds,
          segmentationInstanceId,
          errorMessage: error instanceof Error ? error.message : String(error),
          errorStack: error instanceof Error ? error.stack : undefined,
        });
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
        // @ts-ignore - Cornerstone3D volume property API
        const volumeProperty = volumeActor.getProperty();
        if (volumeProperty) {
          // @ts-ignore - VTK API types
          const scalarOpacity = volumeProperty.getScalarOpacity();
          if (scalarOpacity) {
            scalarOpacity.removeAllPoints();
            scalarOpacity.addPoint(0, 0.0);
            scalarOpacity.addPoint(500, 0.2 * volumeOpacity);
            scalarOpacity.addPoint(1000, 0.4 * volumeOpacity);
            scalarOpacity.addPoint(2000, 0.6 * volumeOpacity);
          }
        }
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
        // @ts-ignore - Cornerstone3D volume property API
        const volumeProperty = volumeActor.getProperty();
        if (volumeProperty) {
          // @ts-ignore - VTK API types
          const scalarOpacity = volumeProperty.getScalarOpacity();
          if (scalarOpacity) {
            scalarOpacity.removeAllPoints();
            scalarOpacity.addPoint(0, 0.0);
            scalarOpacity.addPoint(1, segmentationOpacity);
          }
          // @ts-ignore - VTK API types
          const rgbTransferFunction = volumeProperty.getRGBTransferFunction();
          if (rgbTransferFunction) {
            rgbTransferFunction.removeAllPoints();
            rgbTransferFunction.addRGBPoint(0, 0, 0, 0); // 배경: 검은색
            rgbTransferFunction.addRGBPoint(1, 1, 0, 0); // 종양: 빨간색
          }
        }
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
