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
  instanceIds = [], // 기본값으로 빈 배열 설정
  segmentationInstanceId,
  segmentationFrames = [], // 세그멘테이션 프레임 (base64 마스크 이미지들)
}: Volume3DViewerProps) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showSegmentation, setShowSegmentation] = useState(true);
  const [volumeOpacity, setVolumeOpacity] = useState(0.7);
  const [segmentationOpacity, setSegmentationOpacity] = useState(1.0); // 종양을 더 선명하게 보이도록 1.0으로 설정
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
    // 안전성 검사: instanceIds가 없거나 빈 배열이면 조기 반환
    if (!isInitialized || !viewportRef.current || !instanceIds || instanceIds.length === 0) {
      if (!instanceIds || instanceIds.length === 0) {
        setError('DICOM 이미지가 없습니다. 3D 볼륨을 표시할 수 없습니다.');
        setIsLoading(false);
      }
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
        // 안전성 검사: instanceIds가 유효한지 확인
        if (!instanceIds || instanceIds.length === 0) {
          throw new Error('DICOM 인스턴스 ID가 없습니다.');
        }
        
        const imageIds = instanceIds
          .filter(id => id != null && id !== '') // null, undefined, 빈 문자열 필터링
          .map(id => createImageId(`/api/mri/orthanc/instances/${id}/file`));
        
        if (imageIds.length === 0) {
          throw new Error('유효한 DICOM 인스턴스 ID가 없습니다.');
        }
        
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

        // 세그멘테이션 볼륨 로드 (있는 경우)
        // DICOM SEG 파일의 각 프레임을 개별 인스턴스로 변환하여 볼륨 생성
        if (showSegmentation && segmentationInstanceId) {
          try {
            console.log('[Volume3DViewer] 🎯 세그멘테이션 볼륨 로드 시작...', {
              segmentationInstanceId,
            });
            
            // 1. DICOM SEG 파일의 각 프레임을 개별 DICOM 인스턴스로 변환
            const volumeInstancesResponse = await fetch(
              `/api/mri/segmentation/instances/${segmentationInstanceId}/volume-instances/`
            );
            const volumeInstancesData = await volumeInstancesResponse.json();
            
            if (!volumeInstancesData.success || !volumeInstancesData.instance_ids || volumeInstancesData.instance_ids.length === 0) {
              throw new Error('세그멘테이션 볼륨 인스턴스 변환 실패');
            }
            
            console.log('[Volume3DViewer] 세그멘테이션 인스턴스 변환 완료:', {
              count: volumeInstancesData.instance_ids.length,
              instance_ids: volumeInstancesData.instance_ids.slice(0, 5), // 처음 5개만 로그
            });
            
            // 2. 각 인스턴스를 imageId로 변환
            const segImageIds = volumeInstancesData.instance_ids.map((id: string) =>
              createImageId(`/api/mri/orthanc/instances/${id}/file`)
            );
            
            // 3. 볼륨 로드
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
                  console.log('[Volume3DViewer] 세그멘테이션 볼륨 액터 콜백 실행');
                  // @ts-ignore - Cornerstone3D volume property API
                  const volumeProperty = volumeActor.getProperty();
                  if (volumeProperty) {
                    console.log('[Volume3DViewer] 볼륨 속성 설정 시작...');
                    
                    // @ts-ignore - VTK API types
                    const scalarOpacity = volumeProperty.getScalarOpacity();
                    if (scalarOpacity) {
                      scalarOpacity.removeAllPoints();
                      // 배경(0)은 투명, 종양(255)은 불투명
                      scalarOpacity.addPoint(0, 0.0);
                      scalarOpacity.addPoint(1, segmentationOpacity);
                      scalarOpacity.addPoint(255, segmentationOpacity);
                      console.log('[Volume3DViewer] 투명도 설정 완료:', segmentationOpacity);
                    } else {
                      console.warn('[Volume3DViewer] scalarOpacity를 가져올 수 없습니다');
                    }
                    
                    // @ts-ignore - VTK API types
                    const rgbTransferFunction = volumeProperty.getRGBTransferFunction();
                    if (rgbTransferFunction) {
                      rgbTransferFunction.removeAllPoints();
                      // 배경(0)은 검은색, 종양(1 이상)은 빨간색
                      rgbTransferFunction.addRGBPoint(0, 0, 0, 0); // 배경: 검은색
                      rgbTransferFunction.addRGBPoint(1, 1, 0, 0); // 종양: 빨간색
                      rgbTransferFunction.addRGBPoint(255, 1, 0, 0); // 종양: 빨간색
                      console.log('[Volume3DViewer] 색상 설정 완료: 빨간색');
                    } else {
                      console.warn('[Volume3DViewer] rgbTransferFunction을 가져올 수 없습니다');
                    }
                    
                    // @ts-ignore - VTK API types
                    volumeProperty.setInterpolationTypeToNearest();
                    console.log('[Volume3DViewer] 볼륨 속성 설정 완료');
                  } else {
                    console.error('[Volume3DViewer] volumeProperty를 가져올 수 없습니다');
                  }
                },
              },
            ]);
            
            // 세그멘테이션 추가 후 렌더링
            viewport.render();
            console.log('[Volume3DViewer] ✅ 세그멘테이션 볼륨 추가 및 렌더링 완료 (빨간색)');
            
            // 디버깅: 볼륨 정보 확인
            try {
              const volumeInfo = cache.getVolume(segVolume.volumeId);
              console.log('[Volume3DViewer] 세그멘테이션 볼륨 정보:', {
                volumeId: segVolume.volumeId,
                dimensions: volumeInfo?.dimensions,
                spacing: volumeInfo?.spacing,
              });
            } catch (e) {
              console.warn('[Volume3DViewer] 볼륨 정보 확인 실패:', e);
            }
          } catch (segError) {
            console.error('[Volume3DViewer] ❌ 세그멘테이션 볼륨 로드 실패:', segError);
            console.error('[Volume3DViewer] 에러 상세:', {
              segmentationInstanceId,
              errorMessage: segError instanceof Error ? segError.message : String(segError),
              errorStack: segError instanceof Error ? segError.stack : undefined,
            });
            
            // 사용자에게 에러 표시
            alert(`세그멘테이션 볼륨 로드 실패: ${segError instanceof Error ? segError.message : String(segError)}\n\n브라우저 콘솔을 확인하세요.`);
          }
        } else if (showSegmentation && segmentationFrames.length > 0) {
          // 방법 2: segmentationFrames 사용 (향후 구현)
          try {
            console.log('[Volume3DViewer] 🎯 세그멘테이션 볼륨 생성 시작...', {
              framesCount: segmentationFrames.length,
              showSegmentation,
            });

            // base64 마스크 이미지들을 로드하여 3D 배열 생성
            const maskSlices: number[][] = [];
            let maskWidth = 0;
            let maskHeight = 0;

            for (const frame of segmentationFrames.sort((a, b) => a.index - b.index)) {
              try {
                const img = new Image();
                await new Promise((resolve, reject) => {
                  img.onload = resolve;
                  img.onerror = reject;
                  img.src = `data:image/png;base64,${frame.mask_base64}`;
                });

                // Canvas를 사용하여 이미지 데이터 추출
                const canvas = document.createElement('canvas');
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                if (!ctx) continue;

                ctx.drawImage(img, 0, 0);
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                
                if (maskWidth === 0) {
                  maskWidth = canvas.width;
                  maskHeight = canvas.height;
                }

                // 그레이스케일로 변환 (빨간색 채널만 사용)
                const slice: number[] = [];
                for (let i = 0; i < imageData.data.length; i += 4) {
                  const r = imageData.data[i];
                  const g = imageData.data[i + 1];
                  const b = imageData.data[i + 2];
                  const a = imageData.data[i + 3];
                  // 마스크가 있는 픽셀은 1, 없으면 0
                  const maskValue = (r > 0 || g > 0 || b > 0) && a > 0 ? 1 : 0;
                  slice.push(maskValue);
                }
                maskSlices.push(slice);
              } catch (frameError) {
                console.warn(`[Volume3DViewer] 프레임 ${frame.index} 로드 실패:`, frameError);
              }
            }

            if (maskSlices.length > 0 && maskWidth > 0 && maskHeight > 0) {
              console.log('[Volume3DViewer] 세그멘테이션 마스크 배열 생성 완료:', {
                depth: maskSlices.length,
                height: maskHeight,
                width: maskWidth,
              });

              // 백엔드 API를 통해 세그멘테이션 프레임들을 개별 DICOM 인스턴스로 변환
              // 또는 DICOM SEG 파일의 각 프레임을 개별 이미지로 제공하는 API 사용
              console.log('[Volume3DViewer] ⚠️ segmentationFrames를 직접 볼륨으로 변환하는 기능은 아직 구현되지 않았습니다.');
              console.log('[Volume3DViewer] 💡 DICOM SEG 파일을 사용하거나, 백엔드에서 프레임을 개별 인스턴스로 제공하는 API가 필요합니다.');
              
              // 임시: segmentationInstanceId가 있으면 그것을 사용
              if (segmentationInstanceId) {
                console.log('[Volume3DViewer] segmentationInstanceId를 사용하여 DICOM SEG 로드 시도...');
                try {
                  const segImageId = createImageId(`/api/mri/orthanc/instances/${segmentationInstanceId}/file`);
                  const segVolume = await volumeLoader.createAndCacheVolume('cornerstoneStreamingImageVolume', {
                    imageIds: [segImageId],
                  });

                  segmentationVolumeIdRef.current = segVolume.volumeId;
                  await segVolume.load();
                  console.log('[Volume3DViewer] DICOM SEG 볼륨 로드 완료:', segVolume.volumeId);

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
                            scalarOpacity.addPoint(0.5, 0.0);
                            scalarOpacity.addPoint(1, segmentationOpacity); // 종양 영역만 표시
                          }
                          // @ts-ignore - VTK API types
                          const rgbTransferFunction = volumeProperty.getRGBTransferFunction();
                          if (rgbTransferFunction) {
                            rgbTransferFunction.removeAllPoints();
                            rgbTransferFunction.addRGBPoint(0, 0, 0, 0); // 배경: 검은색
                            rgbTransferFunction.addRGBPoint(0.5, 0, 0, 0); // 중간값: 검은색
                            rgbTransferFunction.addRGBPoint(1, 1, 0, 0); // 종양: 빨간색
                          }
                          // @ts-ignore - VTK API types
                          volumeProperty.setInterpolationTypeToNearest();
                        }
                      },
                    },
                  ]);
                  
                  // 세그멘테이션 추가 후 렌더링
                  viewport.render();
                  console.log('[Volume3DViewer] ✅ 세그멘테이션 볼륨 추가 및 렌더링 완료 (빨간색)');
                } catch (volError) {
                  console.error('[Volume3DViewer] DICOM SEG 볼륨 로드 실패:', volError);
                }
              } else {
                console.warn('[Volume3DViewer] ⚠️ segmentationInstanceId도 없어서 세그멘테이션을 표시할 수 없습니다.');
              }
            } else {
              console.warn('[Volume3DViewer] 세그멘테이션 마스크 데이터가 없습니다.');
            }
          } catch (segError) {
            console.error('[Volume3DViewer] 세그멘테이션 볼륨 생성 실패:', segError);
            console.error('[Volume3DViewer] 에러 상세:', {
              framesCount: segmentationFrames.length,
              errorMessage: segError instanceof Error ? segError.message : String(segError),
              errorStack: segError instanceof Error ? segError.stack : undefined,
            });
          }
          console.log('[Volume3DViewer] ⚠️ segmentationFrames를 직접 볼륨으로 변환하는 기능은 아직 구현되지 않았습니다.');
          console.log('[Volume3DViewer] 💡 DICOM SEG 파일(segmentationInstanceId)을 사용하세요.');
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
        setError(null); // 성공 시 에러 초기화
      } catch (error) {
        console.error('[Volume3DViewer] 볼륨 로드 실패:', error);
        console.error('[Volume3DViewer] 에러 상세:', {
          instanceIds: instanceIds?.length || 0,
          hasInstanceIds: !!instanceIds,
          segmentationInstanceId,
          errorMessage: error instanceof Error ? error.message : String(error),
          errorStack: error instanceof Error ? error.stack : undefined,
        });
        setIsLoading(false);
        setError(error instanceof Error ? error.message : '3D 볼륨 로드 실패');
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
  }, [isInitialized, instanceIds, segmentationInstanceId, segmentationFrames, showSegmentation, volumeOpacity, segmentationOpacity]);

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
