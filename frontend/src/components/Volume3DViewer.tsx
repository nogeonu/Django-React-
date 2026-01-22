/**
 * Cornerstone3D 기반 3D 볼륨 렌더링 뷰어
 * 세그멘테이션(종양)을 3D로 시각화하여 여러 각도에서 종양 모양을 관찰할 수 있음
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
  TrackballRotateTool,
} from '@cornerstonejs/tools';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Badge } from '@/components/ui/badge';
import { RotateCw, Loader2, AlertCircle, Box } from 'lucide-react';
import { initCornerstone, createImageId } from '@/lib/cornerstone';
import { Enums as ToolEnums } from '@cornerstonejs/tools';

interface Volume3DViewerProps {
  instanceIds: string[]; // DICOM 인스턴스 ID 배열 (원본 MRI)
  segmentationInstanceId?: string; // 세그멘테이션 인스턴스 ID (DICOM SEG)
  segmentationFrames?: Array<{ index: number; mask_base64: string }>; // 세그멘테이션 프레임
  patientId?: string;
}

export default function Volume3DViewer({
  instanceIds,
  segmentationInstanceId,
  segmentationFrames = [],
}: Volume3DViewerProps) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingMessage, setLoadingMessage] = useState('초기화 중...');
  const [error, setError] = useState<string | null>(null);
  const [tumorOpacity, setTumorOpacity] = useState(1.0);
  const renderingEngineRef = useRef<RenderingEngine | null>(null);
  const volumeIdRef = useRef<string | null>(null);
  const uniqueId = useRef<string>(`volume3d_${Date.now()}_${Math.random()}`);
  const viewportIdRef = useRef<string>(`viewport_${uniqueId.current}`);
  const toolGroupIdRef = useRef<string>(`toolGroup_${uniqueId.current}`);
  const isCleanedUpRef = useRef(false);

  // 디버깅: props 확인
  useEffect(() => {
    console.log('[Volume3DViewer] 🔍 Props 확인:', {
      instanceIds: instanceIds?.length || 0,
      segmentationInstanceId: segmentationInstanceId || 'undefined',
      segmentationFrames: segmentationFrames?.length || 0,
    });
  }, [instanceIds, segmentationInstanceId, segmentationFrames]);

  // Cornerstone 초기화
  useEffect(() => {
    const initialize = async () => {
      try {
        await initCornerstone();

        // 3D 도구 등록
        try {
          addTool(ZoomTool);
          addTool(PanTool);
          addTool(TrackballRotateTool);
        } catch (e) {
          // 이미 등록된 도구는 무시
        }

        setIsInitialized(true);
      } catch (error) {
        console.error('Failed to initialize Cornerstone3D:', error);
        setError('Cornerstone3D 초기화 실패');
      }
    };

    initialize();

    return () => {
      isCleanedUpRef.current = true;
    };
  }, []);

  // 종양 3D 볼륨 로드
  useEffect(() => {
    if (!isInitialized || !viewportRef.current) {
      return;
    }

    if (!segmentationInstanceId) {
      setError('세그멘테이션 인스턴스 ID가 없습니다. AI 분석을 먼저 실행해주세요.');
      setIsLoading(false);
      return;
    }

    const loadTumorVolume = async () => {
      try {
        setIsLoading(true);
        setError(null);
        setLoadingMessage('종양 3D 볼륨 데이터 로드 중...');

        console.log('[Volume3DViewer] 🎯 종양 3D 볼륨 로드 시작:', segmentationInstanceId);

        // 1. DICOM SEG → 개별 인스턴스 변환 API 호출
        setLoadingMessage('세그멘테이션 데이터 변환 중...');
        const volumeInstancesResponse = await fetch(
          `/api/mri/segmentation/instances/${segmentationInstanceId}/volume-instances/`
        );
        const volumeInstancesData = await volumeInstancesResponse.json();

        if (!volumeInstancesData.success || !volumeInstancesData.instance_ids || volumeInstancesData.instance_ids.length === 0) {
          throw new Error('세그멘테이션 볼륨 인스턴스 변환 실패: ' + (volumeInstancesData.error || '인스턴스가 없습니다'));
        }

        console.log('[Volume3DViewer] ✅ 세그멘테이션 인스턴스 변환 완료:', {
          count: volumeInstancesData.instance_ids.length,
        });

        // 렌더링 엔진 생성
        setLoadingMessage('3D 렌더링 엔진 초기화 중...');
        const renderingEngineId = `volume3d_engine_${uniqueId.current}`;
        const renderingEngine = new RenderingEngine(renderingEngineId);
        renderingEngineRef.current = renderingEngine;

        if (!viewportRef.current || isCleanedUpRef.current) {
          console.log('[Volume3DViewer] 컴포넌트가 언마운트됨, 중단');
          return;
        }

        // 3D 뷰포트 생성
        const viewportInput = {
          viewportId: viewportIdRef.current,
          type: Enums.ViewportType.VOLUME_3D,
          element: viewportRef.current,
          defaultOptions: {
            background: [0.1, 0.1, 0.1] as Types.Point3,
          },
        };

        renderingEngine.setViewports([viewportInput]);
        const viewport = renderingEngine.getViewport(viewportIdRef.current) as Types.IVolumeViewport;

        if (!viewport) {
          throw new Error('뷰포트 생성 실패');
        }

        // 2. 이미지 ID 생성 (세그멘테이션 인스턴스들)
        setLoadingMessage('종양 볼륨 데이터 로드 중...');
        const segImageIds = volumeInstancesData.instance_ids.map((id: string) =>
          createImageId(`/api/mri/orthanc/instances/${id}/file`)
        );

        console.log('[Volume3DViewer] 이미지 ID 생성 완료:', segImageIds.length, '개');

        // 3. 볼륨 로드
        const volumeId = `cornerstoneStreamingImageVolume:tumor_${uniqueId.current}`;

        const volume = await volumeLoader.createAndCacheVolume(volumeId, {
          imageIds: segImageIds,
        });

        volumeIdRef.current = volumeId;

        setLoadingMessage('종양 3D 모델 생성 중...');
        await volume.load();

        if (isCleanedUpRef.current) {
          console.log('[Volume3DViewer] 로드 완료 후 컴포넌트 언마운트됨');
          return;
        }

        console.log('[Volume3DViewer] ✅ 종양 볼륨 로드 완료');

        // 4. 볼륨을 뷰포트에 설정 (빨간색 종양)
        setLoadingMessage('3D 렌더링 중...');
        await viewport.setVolumes([
          {
            volumeId: volumeId,
            callback: ({ volumeActor }) => {
              console.log('[Volume3DViewer] 종양 볼륨 액터 설정 중...');

              // @ts-ignore
              const volumeProperty = volumeActor.getProperty();
              if (volumeProperty) {
                // 투명도 설정 (배경은 투명, 종양은 불투명)
                // @ts-ignore
                const scalarOpacity = volumeProperty.getScalarOpacity();
                if (scalarOpacity) {
                  scalarOpacity.removeAllPoints();
                  // 배경(0)은 완전히 투명
                  scalarOpacity.addPoint(0, 0.0);
                  scalarOpacity.addPoint(1, 0.0);
                  // 종양(128-255)은 불투명
                  scalarOpacity.addPoint(127, 0.0);
                  scalarOpacity.addPoint(128, tumorOpacity);
                  scalarOpacity.addPoint(255, tumorOpacity);
                }

                // 색상 설정 (빨간색)
                // @ts-ignore
                const rgbTransferFunction = volumeProperty.getRGBTransferFunction();
                if (rgbTransferFunction) {
                  rgbTransferFunction.removeAllPoints();
                  // 배경은 검은색
                  rgbTransferFunction.addRGBPoint(0, 0, 0, 0);
                  rgbTransferFunction.addRGBPoint(127, 0, 0, 0);
                  // 종양은 빨간색
                  rgbTransferFunction.addRGBPoint(128, 1, 0, 0);
                  rgbTransferFunction.addRGBPoint(255, 1, 0.2, 0.2);
                }

                // 렌더링 품질 설정
                // @ts-ignore
                volumeProperty.setInterpolationTypeToNearest();
                // @ts-ignore
                volumeProperty.setShade(true);
                // @ts-ignore
                volumeProperty.setAmbient(0.3);
                // @ts-ignore
                volumeProperty.setDiffuse(0.7);
                // @ts-ignore
                volumeProperty.setSpecular(0.3);
                // @ts-ignore
                volumeProperty.setSpecularPower(10);
              }
            },
          },
        ]);

        // 카메라 리셋
        viewport.resetCamera();
        viewport.render();

        // 5. 도구 그룹 설정
        let toolGroup = ToolGroupManager.getToolGroup(toolGroupIdRef.current);
        if (!toolGroup) {
          toolGroup = ToolGroupManager.createToolGroup(toolGroupIdRef.current);
        }

        if (toolGroup) {
          try {
            toolGroup.addTool(TrackballRotateTool.toolName);
            toolGroup.addTool(ZoomTool.toolName);
            toolGroup.addTool(PanTool.toolName);
          } catch (e) {
            // 이미 추가된 도구 무시
          }

          toolGroup.addViewport(viewportIdRef.current, renderingEngineId);

          // 마우스 왼쪽 버튼: 회전
          toolGroup.setToolActive(TrackballRotateTool.toolName, {
            bindings: [{ mouseButton: ToolEnums.MouseBindings.Primary }],
          });
          // 마우스 오른쪽 버튼: 줌
          toolGroup.setToolActive(ZoomTool.toolName, {
            bindings: [{ mouseButton: ToolEnums.MouseBindings.Secondary }],
          });
          // 마우스 휠 버튼: 이동
          toolGroup.setToolActive(PanTool.toolName, {
            bindings: [{ mouseButton: ToolEnums.MouseBindings.Auxiliary }],
          });
        }

        console.log('[Volume3DViewer] ✅ 종양 3D 렌더링 완료!');
        setIsLoading(false);

      } catch (error) {
        console.error('[Volume3DViewer] ❌ 종양 볼륨 로드 실패:', error);
        setError(error instanceof Error ? error.message : '종양 3D 볼륨 로드 실패');
        setIsLoading(false);
      }
    };

    loadTumorVolume();

    // Cleanup
    return () => {
      console.log('[Volume3DViewer] 클린업 시작');
      isCleanedUpRef.current = true;

      try {
        const toolGroup = ToolGroupManager.getToolGroup(toolGroupIdRef.current);
        if (toolGroup && renderingEngineRef.current) {
          toolGroup.removeViewports(renderingEngineRef.current.id, viewportIdRef.current);
          ToolGroupManager.destroyToolGroup(toolGroupIdRef.current);
        }
      } catch (e) {
        console.warn('Tool group cleanup error:', e);
      }

      try {
        if (renderingEngineRef.current) {
          renderingEngineRef.current.destroy();
          renderingEngineRef.current = null;
        }
      } catch (e) {
        console.warn('Rendering engine cleanup error:', e);
      }

      try {
        if (volumeIdRef.current) {
          cache.removeVolumeLoadObject(volumeIdRef.current);
        }
      } catch (e) {
        console.warn('Volume cache cleanup error:', e);
      }
    };
  }, [isInitialized, segmentationInstanceId]);

  // 투명도 변경
  useEffect(() => {
    if (!renderingEngineRef.current || !volumeIdRef.current) return;

    try {
      const viewport = renderingEngineRef.current.getViewport(viewportIdRef.current) as Types.IVolumeViewport;
      if (!viewport) return;

      const actors = viewport.getActors();
      if (actors && actors.length > 0) {
        const volumeActor = actors[0].actor;
        // @ts-ignore
        const volumeProperty = volumeActor?.getProperty?.();
        if (volumeProperty) {
          // @ts-ignore
          const scalarOpacity = volumeProperty.getScalarOpacity();
          if (scalarOpacity) {
            scalarOpacity.removeAllPoints();
            scalarOpacity.addPoint(0, 0.0);
            scalarOpacity.addPoint(127, 0.0);
            scalarOpacity.addPoint(128, tumorOpacity);
            scalarOpacity.addPoint(255, tumorOpacity);
          }
        }
        viewport.render();
      }
    } catch (error) {
      console.warn('Failed to update tumor opacity:', error);
    }
  }, [tumorOpacity]);

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
          <Badge variant="outline" className="text-red-400 border-red-600 flex items-center gap-2">
            <Box className="w-4 h-4" />
            종양 3D 뷰
          </Badge>
          {segmentationInstanceId && (
            <span className="text-xs text-gray-500">
              SEG ID: {segmentationInstanceId.substring(0, 8)}...
            </span>
          )}
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-xs text-gray-400">
            <span>종양 투명도:</span>
            <Slider
              value={[tumorOpacity]}
              onValueChange={([value]) => setTumorOpacity(value)}
              min={0.1}
              max={1}
              step={0.1}
              className="w-24"
            />
            <span className="w-8 text-right">{Math.round(tumorOpacity * 100)}%</span>
          </div>

          <Button variant="outline" size="sm" onClick={handleResetView} disabled={isLoading}>
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
          style={{ minHeight: '500px' }}
        />

        {/* 로딩 상태 */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-950/90">
            <div className="text-center">
              <Loader2 className="w-12 h-12 text-red-500 animate-spin mx-auto mb-4" />
              <p className="text-white text-sm font-medium">{loadingMessage}</p>
              <p className="text-gray-400 text-xs mt-2">종양 데이터를 3D로 변환하고 있습니다...</p>
            </div>
          </div>
        )}

        {/* 에러 상태 */}
        {error && !isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-950/90">
            <div className="text-center max-w-md p-6">
              <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
              <p className="text-white text-sm font-medium mb-2">3D 뷰 로드 실패</p>
              <p className="text-gray-400 text-xs">{error}</p>
              <p className="text-gray-500 text-xs mt-4">
                AI 분석을 먼저 실행하여 세그멘테이션 데이터를 생성해주세요.
              </p>
            </div>
          </div>
        )}

        {/* 사용 안내 */}
        {!isLoading && !error && (
          <div className="absolute bottom-4 left-4 bg-black/70 text-white text-xs p-3 rounded-lg">
            <p className="font-bold text-red-400 mb-2">🔴 종양 3D 뷰</p>
            <p>🖱️ 왼쪽 클릭 + 드래그: 회전</p>
            <p>🖱️ 오른쪽 클릭 + 드래그: 줌</p>
            <p>🖱️ 휠 클릭 + 드래그: 이동</p>
          </div>
        )}
      </div>
    </div>
  );
}
