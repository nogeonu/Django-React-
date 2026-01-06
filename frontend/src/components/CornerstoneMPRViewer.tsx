/**
 * Cornerstone3D Volume-based MPR 뷰어
 * DICOM 메타데이터를 활용한 자동 3D 볼륨 재구성 및 MPR
 * VolView 스타일의 멀티플래너 렌더링
 */
import { useEffect, useRef, useState } from 'react';
import {
  RenderingEngine,
  Enums,
  type Types,
  volumeLoader,
  cache,
  setVolumesForViewports,
} from '@cornerstonejs/core';
import {
  addTool,
  ToolGroupManager,
  Enums as ToolEnums,
  WindowLevelTool,
  PanTool,
  ZoomTool,
} from '@cornerstonejs/tools';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Grid3x3, Loader2 } from 'lucide-react';
import { initCornerstone, createImageId, WINDOW_LEVEL_PRESETS } from '@/lib/cornerstone';

interface CornerstoneMPRViewerProps {
  instanceIds: string[];
  onClose?: () => void;
}

export default function CornerstoneMPRViewer({
  instanceIds,
  onClose,
}: CornerstoneMPRViewerProps) {
  const axialRef = useRef<HTMLDivElement>(null);
  const sagittalRef = useRef<HTMLDivElement>(null);
  const coronalRef = useRef<HTMLDivElement>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [windowLevel] = useState(WINDOW_LEVEL_PRESETS.MRI_BRAIN);
  const renderingEngineRef = useRef<RenderingEngine | null>(null);
  const volumeIdRef = useRef<string>('');
  const toolGroupIdRef = useRef<string>('MPR_TOOL_GROUP');

  // Cornerstone 초기화
  useEffect(() => {
    const initialize = async () => {
      try {
        await initCornerstone();
        
        // 도구 등록
        addTool(WindowLevelTool);
        addTool(PanTool);
        addTool(ZoomTool);

        setIsInitialized(true);
      } catch (error) {
        console.error('[MPR] Failed to initialize Cornerstone:', error);
      }
    };

    initialize();
  }, []);

  // Volume-based MPR 뷰포트 설정
  useEffect(() => {
    if (
      !isInitialized ||
      !axialRef.current ||
      !sagittalRef.current ||
      !coronalRef.current ||
      instanceIds.length === 0
    ) {
      return;
    }

    const setupVolumeMPR = async () => {
      setIsLoading(true);
      setLoadingProgress(0);
      
      try {
        console.log('[MPR Volume] 🚀 Setting up Volume MPR with', instanceIds.length, 'instances');
        const renderingEngineId = 'volumeMprRenderingEngine';
        const volumeId = `mpr_volume_${Date.now()}`;
        volumeIdRef.current = volumeId;

        // 기존 렌더링 엔진 정리
        if (renderingEngineRef.current) {
          console.log('[MPR Volume] 🧹 Cleaning up existing rendering engine');
          renderingEngineRef.current.destroy();
        }

        // 기존 볼륨 캐시 정리
        try {
          const existingVolume = cache.getVolume(volumeId);
          if (existingVolume) {
            cache.removeVolumeLoadObject(volumeId);
          }
        } catch (e) {
          // 볼륨이 없으면 무시
        }

        setLoadingProgress(10);

        // 렌더링 엔진 생성
        console.log('[MPR Volume] 🎨 Creating rendering engine');
        const renderingEngine = new RenderingEngine(renderingEngineId);
        renderingEngineRef.current = renderingEngine;

        // 이미지 ID 생성
        const imageIds = instanceIds.map((id) =>
          createImageId(`/api/mri/orthanc/instances/${id}/file`)
        );
        
        console.log('[MPR Volume] 📸 Created', imageIds.length, 'image IDs');
        setLoadingProgress(20);

        // Volume 로드 (DICOM 메타데이터 자동 분석 및 3D 볼륨 재구성)
        console.log('[MPR Volume] 🧊 Creating volume from DICOM series...');
        console.log('[MPR Volume] 📋 DICOM will be automatically sorted by ImagePositionPatient (z-axis)');
        
        const volume = await volumeLoader.createAndCacheVolume(volumeId, {
          imageIds,
        });

        setLoadingProgress(50);
        console.log('[MPR Volume] ✅ Volume created, loading...');

        // Volume 로드 (픽셀 데이터 로드)
        volume.load((progress: { loaded: number; total: number }) => {
          const percent = 50 + (progress.loaded / progress.total) * 30;
          setLoadingProgress(Math.round(percent));
          console.log(`[MPR Volume] 📊 Loading progress: ${Math.round(percent)}%`);
        });

        setLoadingProgress(80);
        console.log('[MPR Volume] 🎯 Volume loaded successfully');

        // 3개의 Volume 뷰포트 생성 (Axial, Sagittal, Coronal)
        console.log('[MPR Volume] 🖼️ Setting up MPR viewports...');
        const viewportInputArray: Types.PublicViewportInput[] = [
          {
            viewportId: 'MPR_AXIAL',
            type: Enums.ViewportType.ORTHOGRAPHIC,
            element: axialRef.current,
            defaultOptions: {
              orientation: Enums.OrientationAxis.AXIAL,
              background: [0, 0, 0] as Types.Point3,
            },
          },
          {
            viewportId: 'MPR_SAGITTAL',
            type: Enums.ViewportType.ORTHOGRAPHIC,
            element: sagittalRef.current,
            defaultOptions: {
              orientation: Enums.OrientationAxis.SAGITTAL,
              background: [0, 0, 0] as Types.Point3,
            },
          },
          {
            viewportId: 'MPR_CORONAL',
            type: Enums.ViewportType.ORTHOGRAPHIC,
            element: coronalRef.current,
            defaultOptions: {
              orientation: Enums.OrientationAxis.CORONAL,
              background: [0, 0, 0] as Types.Point3,
            },
          },
        ];

        // 뷰포트 활성화
        renderingEngine.setViewports(viewportInputArray);
        setLoadingProgress(85);

        // 모든 뷰포트에 동일한 볼륨 설정
        console.log('[MPR Volume] 🔗 Linking volume to all viewports...');
        await setVolumesForViewports(
          renderingEngine,
          [{ volumeId }],
          ['MPR_AXIAL', 'MPR_SAGITTAL', 'MPR_CORONAL']
        );

        setLoadingProgress(90);

        // Window/Level 설정
        console.log('[MPR Volume] 🎚️ Applying window/level settings...');
        ['MPR_AXIAL', 'MPR_SAGITTAL', 'MPR_CORONAL'].forEach((viewportId) => {
          const viewport = renderingEngine.getViewport(viewportId);
          if (viewport) {
            // @ts-ignore - setProperties exists but types are incomplete
            viewport.setProperties({
              voiRange: {
                lower: windowLevel.windowCenter - windowLevel.windowWidth / 2,
                upper: windowLevel.windowCenter + windowLevel.windowWidth / 2,
              },
            });
          }
        });

        // 렌더링
        renderingEngine.render();
        setLoadingProgress(95);

        // 도구 그룹 설정
        setupTools(['MPR_AXIAL', 'MPR_SAGITTAL', 'MPR_CORONAL']);
        
        setLoadingProgress(100);
        setIsLoading(false);
        
        console.log('[MPR Volume] 🎉 Volume MPR setup complete!');
        console.log('[MPR Volume] 📐 All views are automatically synchronized');
      } catch (error) {
        console.error('[MPR Volume] ❌ Failed to setup Volume MPR:', error);
        setIsLoading(false);
      }
    };

    setupVolumeMPR();

    return () => {
      console.log('[MPR Volume] 🧹 Cleaning up...');
      if (renderingEngineRef.current) {
        try {
          renderingEngineRef.current.destroy();
        } catch (e) {
          console.warn('[MPR Volume] Error destroying rendering engine:', e);
        }
        renderingEngineRef.current = null;
      }
      
      // 볼륨 캐시 정리
      if (volumeIdRef.current) {
        try {
          cache.removeVolumeLoadObject(volumeIdRef.current);
        } catch (e) {
          console.warn('[MPR Volume] Error removing volume cache:', e);
        }
      }
    };
  }, [isInitialized, instanceIds]);

  // 도구 설정
  const setupTools = (viewportIds: string[]) => {
    try {
      console.log('[MPR Volume] 🛠️ Setting up tools...');
      
      // 기존 도구 그룹 제거
      try {
        const existingToolGroup = ToolGroupManager.getToolGroup(toolGroupIdRef.current);
        if (existingToolGroup) {
          ToolGroupManager.destroyToolGroup(toolGroupIdRef.current);
        }
      } catch (e) {
        // 도구 그룹이 없으면 무시
      }

      // 새 도구 그룹 생성
      const toolGroup = ToolGroupManager.createToolGroup(toolGroupIdRef.current);

      if (toolGroup) {
        // 도구 추가
        toolGroup.addTool(WindowLevelTool.toolName);
        toolGroup.addTool(PanTool.toolName);
        toolGroup.addTool(ZoomTool.toolName);

        // 기본 도구 활성화
        toolGroup.setToolActive(WindowLevelTool.toolName, {
          bindings: [{ mouseButton: ToolEnums.MouseBindings.Primary }],
        });
        toolGroup.setToolActive(PanTool.toolName, {
          bindings: [{ mouseButton: ToolEnums.MouseBindings.Auxiliary }],
        });
        toolGroup.setToolActive(ZoomTool.toolName, {
          bindings: [{ mouseButton: ToolEnums.MouseBindings.Secondary }],
        });

        // 모든 뷰포트에 도구 그룹 연결
        viewportIds.forEach((viewportId) => {
          toolGroup.addViewport(viewportId, renderingEngineRef.current!.id);
        });
        
        console.log('[MPR Volume] ✅ Tools setup complete');
      }
    } catch (error) {
      console.error('[MPR Volume] Failed to setup tools:', error);
    }
  };

  if (!isInitialized) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900 text-white">
        <div className="flex flex-col items-center gap-4">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-white text-sm font-medium">Cornerstone3D 초기화 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-gray-900 relative">
      {/* 도구 바 */}
      <div className="bg-gray-800 border-b border-gray-700 px-4 py-2 flex items-center gap-2 flex-wrap">
        <Badge variant="outline" className="text-white border-gray-600 font-bold">
          🧊 Volume MPR
        </Badge>
        <Badge className="bg-green-600/80 backdrop-blur-md text-white border-none text-xs">
          DICOM 자동 재구성
        </Badge>
        <Badge className="bg-blue-600/80 backdrop-blur-md text-white border-none text-xs">
          {instanceIds.length}장
        </Badge>
        <div className="ml-auto flex items-center gap-2">
          <Badge className="bg-blue-600/80 backdrop-blur-md text-white border-none">
            W: {windowLevel.windowWidth} / L: {windowLevel.windowCenter}
          </Badge>
          {onClose && (
            <Button
              size="sm"
              variant="outline"
              onClick={onClose}
              className="h-8 bg-gray-700 hover:bg-gray-600 text-white border-gray-600"
            >
              <Grid3x3 className="w-4 h-4 mr-1" />
              단일 뷰
            </Button>
          )}
        </div>
      </div>

      {/* 로딩 오버레이 */}
      {isLoading && (
        <div className="absolute inset-0 bg-gray-900/95 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="flex flex-col items-center gap-6 max-w-md">
            <div className="relative">
              <div className="w-20 h-20 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-white font-bold text-sm">{loadingProgress}%</span>
              </div>
            </div>
            <div className="text-center space-y-2">
              <p className="text-white text-lg font-bold">🧊 3D 볼륨 재구성 중...</p>
              <p className="text-white/80 text-sm">DICOM 메타데이터 분석 및 공간 정렬</p>
              <div className="w-64 h-2 bg-gray-700 rounded-full overflow-hidden mt-4">
                <div 
                  className="h-full bg-gradient-to-r from-blue-500 to-green-500 transition-all duration-300"
                  style={{ width: `${loadingProgress}%` }}
                />
              </div>
            </div>
            <div className="space-y-1 text-xs text-gray-400 text-center">
              <p>✓ Image Position (Patient) 분석</p>
              <p>✓ Image Orientation 확인</p>
              <p>✓ Pixel Spacing 적용</p>
              <p>✓ 3D 좌표계 재구성</p>
            </div>
          </div>
        </div>
      )}

      {/* 4분할 뷰포트 그리드 - VolView 스타일 */}
      <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-1 p-1">
        {/* ① Sagittal (시상면) - 좌상단 */}
        <div className="relative bg-black border border-gray-800 rounded-lg overflow-hidden">
          <div
            ref={sagittalRef}
            className="w-full h-full"
            style={{ minHeight: '300px' }}
          />
          <div className="absolute top-2 left-2 pointer-events-none z-10">
            <Badge className="bg-green-600/90 backdrop-blur-md text-white border-none font-bold shadow-lg">
              ① Sagittal (시상면)
            </Badge>
          </div>
          <div className="absolute top-2 right-2 pointer-events-none z-10 flex flex-col gap-1">
            <Badge className="bg-black/70 backdrop-blur-md text-white border-none text-xs">
              S
            </Badge>
          </div>
          <div className="absolute bottom-2 left-2 pointer-events-none z-10">
            <Badge className="bg-black/70 backdrop-blur-md text-green-400 border-none text-xs">
              좌 ← → 우
            </Badge>
          </div>
        </div>

        {/* ② 3D Volume Info - 우상단 */}
        <div className="relative bg-gradient-to-br from-gray-900 via-blue-900/20 to-purple-900/20 border border-blue-700/30 rounded-lg overflow-hidden flex items-center justify-center">
          <div className="text-center space-y-3 p-6">
            <div className="text-5xl mb-2">🧊</div>
            <div className="absolute top-2 left-2 pointer-events-none z-10">
              <Badge className="bg-blue-600/90 backdrop-blur-md text-white border-none font-bold shadow-lg">
                ② Volume MPR
              </Badge>
            </div>
            <h3 className="text-base font-bold text-white">자동 3D 재구성</h3>
            <div className="space-y-1.5 text-xs text-gray-300">
              <p className="flex items-center justify-center gap-2">
                <span className="text-green-400">✓</span>
                <span>DICOM 메타데이터 자동 분석</span>
              </p>
              <p className="flex items-center justify-center gap-2">
                <span className="text-blue-400">✓</span>
                <span>공간 좌표 기반 정렬</span>
              </p>
              <p className="flex items-center justify-center gap-2">
                <span className="text-purple-400">✓</span>
                <span>실시간 MPR 재구성</span>
              </p>
            </div>
            <div className="mt-3 pt-3 border-t border-gray-700/50">
              <p className="text-xs text-gray-400">
                마우스 휠: 슬라이스 스크롤<br />
                좌클릭: 윈도우/레벨 조정<br />
                우클릭: 줌
              </p>
            </div>
          </div>
        </div>

        {/* ③ Axial (횡단면) - 좌하단 */}
        <div className="relative bg-black border border-gray-800 rounded-lg overflow-hidden">
          <div
            ref={axialRef}
            className="w-full h-full"
            style={{ minHeight: '300px' }}
          />
          <div className="absolute top-2 left-2 pointer-events-none z-10">
            <Badge className="bg-blue-600/90 backdrop-blur-md text-white border-none font-bold shadow-lg">
              ③ Axial (횡단면)
            </Badge>
          </div>
          <div className="absolute top-2 right-2 pointer-events-none z-10">
            <Badge className="bg-black/70 backdrop-blur-md text-white border-none text-xs">
              A
            </Badge>
          </div>
          <div className="absolute bottom-2 left-2 pointer-events-none z-10">
            <Badge className="bg-black/70 backdrop-blur-md text-blue-400 border-none text-xs">
              위 ↑ ↓ 아래
            </Badge>
          </div>
        </div>

        {/* ④ Coronal (관상면) - 우하단 */}
        <div className="relative bg-black border border-gray-800 rounded-lg overflow-hidden">
          <div
            ref={coronalRef}
            className="w-full h-full"
            style={{ minHeight: '300px' }}
          />
          <div className="absolute top-2 left-2 pointer-events-none z-10">
            <Badge className="bg-purple-600/90 backdrop-blur-md text-white border-none font-bold shadow-lg">
              ④ Coronal (관상면)
            </Badge>
          </div>
          <div className="absolute top-2 right-2 pointer-events-none z-10">
            <Badge className="bg-black/70 backdrop-blur-md text-white border-none text-xs">
              C
            </Badge>
          </div>
          <div className="absolute bottom-2 left-2 pointer-events-none z-10">
            <Badge className="bg-black/70 backdrop-blur-md text-purple-400 border-none text-xs">
              앞 ← → 뒤
            </Badge>
          </div>
        </div>
      </div>
    </div>
  );
}
