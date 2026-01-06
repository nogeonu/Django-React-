/**
 * Cornerstone3D 4분할 MPR 뷰어 컴포넌트
 * Axial, Sagittal, Coronal 뷰를 동시에 표시
 * VolView 스타일의 MPR 렌더링
 */
import { useEffect, useRef, useState } from 'react';
import {
  RenderingEngine,
  Enums,
  type Types,
  imageLoader,
} from '@cornerstonejs/core';
import {
  addTool,
  ToolGroupManager,
  Enums as ToolEnums,
  WindowLevelTool,
  PanTool,
  ZoomTool,
  StackScrollMouseWheelTool,
} from '@cornerstonejs/tools';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Grid3x3 } from 'lucide-react';
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
  const [windowLevel] = useState(WINDOW_LEVEL_PRESETS.MRI_BRAIN);
  const renderingEngineRef = useRef<RenderingEngine | null>(null);
  const toolGroupIdRef = useRef<string>('MPR_TOOL_GROUP');
  const currentSliceRef = useRef({ axial: 0, sagittal: 0, coronal: 0 });

  // Cornerstone 초기화
  useEffect(() => {
    const initialize = async () => {
      try {
        await initCornerstone();
        
        // 도구 등록
        addTool(WindowLevelTool);
        addTool(PanTool);
        addTool(ZoomTool);
        addTool(StackScrollMouseWheelTool);

        setIsInitialized(true);
      } catch (error) {
        console.error('Failed to initialize Cornerstone:', error);
      }
    };

    initialize();
  }, []);

  // MPR 뷰포트 설정
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

    const setupMPRViewports = async () => {
      setIsLoading(true);
      try {
        console.log('[MPR] Setting up MPR viewports with', instanceIds.length, 'instances');
        const renderingEngineId = 'mprRenderingEngine';

        // 기존 렌더링 엔진 정리
        if (renderingEngineRef.current) {
          renderingEngineRef.current.destroy();
        }

        // 렌더링 엔진 생성
        const renderingEngine = new RenderingEngine(renderingEngineId);
        renderingEngineRef.current = renderingEngine;

        // 이미지 ID 생성 및 프리로드
        const imageIds = instanceIds.map((id) =>
          createImageId(`/api/mri/orthanc/instances/${id}/file`)
        );
        
        console.log('[MPR] Created', imageIds.length, 'image IDs');
        console.log('[MPR] First image ID:', imageIds[0]);

        // 첫 이미지 프리로드 (메타데이터 확인용)
        try {
          console.log('[MPR] Preloading first image for metadata...');
          await imageLoader.loadAndCacheImage(imageIds[0]);
          console.log('[MPR] First image loaded successfully');
        } catch (error) {
          console.error('[MPR] Failed to preload first image:', error);
        }

        // 3개의 뷰포트 생성 (STACK 타입 사용 - 각 방향별로 독립적인 스택)
        const viewportInputs = [
          {
            viewportId: 'MPR_AXIAL',
            type: Enums.ViewportType.STACK,
            element: axialRef.current,
            defaultOptions: {
              background: [0, 0, 0] as Types.Point3,
            },
          },
          {
            viewportId: 'MPR_SAGITTAL',
            type: Enums.ViewportType.STACK,
            element: sagittalRef.current,
            defaultOptions: {
              background: [0, 0, 0] as Types.Point3,
            },
          },
          {
            viewportId: 'MPR_CORONAL',
            type: Enums.ViewportType.STACK,
            element: coronalRef.current,
            defaultOptions: {
              background: [0, 0, 0] as Types.Point3,
            },
          },
        ];

        console.log('[MPR] Enabling viewports...');
        // 뷰포트 활성화
        viewportInputs.forEach((input) => {
          if (input.element) {
            renderingEngine.enableElement(input as any);
          }
        });

        // 각 뷰포트에 이미지 스택 설정
        const viewportIds = ['MPR_AXIAL', 'MPR_SAGITTAL', 'MPR_CORONAL'];
        const middleIndex = Math.floor(imageIds.length / 2);
        
        console.log('[MPR] Setting up stacks for each viewport...');
        for (const viewportId of viewportIds) {
          try {
            const viewport = renderingEngine.getViewport(viewportId);
            if (viewport) {
              console.log(`[MPR] Setting stack for ${viewportId}...`);
              
              // @ts-ignore - setStack exists in StackViewport
              await viewport.setStack(imageIds, middleIndex);
              
              // 윈도우 레벨 설정
              // @ts-ignore - setProperties exists but types are incomplete
              viewport.setProperties({
                voiRange: {
                  lower: windowLevel.windowCenter - windowLevel.windowWidth / 2,
                  upper: windowLevel.windowCenter + windowLevel.windowWidth / 2,
                },
              });
              
              viewport.render();
              console.log(`[MPR] ${viewportId} setup complete`);
              
              // 현재 슬라이스 인덱스 저장
              if (viewportId === 'MPR_AXIAL') currentSliceRef.current.axial = middleIndex;
              if (viewportId === 'MPR_SAGITTAL') currentSliceRef.current.sagittal = middleIndex;
              if (viewportId === 'MPR_CORONAL') currentSliceRef.current.coronal = middleIndex;
            }
          } catch (error) {
            console.error(`[MPR] Failed to setup ${viewportId}:`, error);
          }
        }

        // 도구 그룹 설정
        setupTools(viewportIds);
        
        setIsLoading(false);
        console.log('[MPR] All viewports setup complete');
      } catch (error) {
        console.error('[MPR] Failed to setup MPR viewports:', error);
        setIsLoading(false);
      }
    };

    setupMPRViewports();

    return () => {
      if (renderingEngineRef.current) {
        try {
          renderingEngineRef.current.destroy();
        } catch (e) {
          console.warn('[MPR] Error destroying rendering engine:', e);
        }
        renderingEngineRef.current = null;
      }
    };
  }, [isInitialized, instanceIds]);

  // 도구 설정
  const setupTools = (viewportIds: string[]) => {
    try {
      console.log('[MPR] Setting up tools...');
      // 기존 도구 그룹 제거
      const existingToolGroup = ToolGroupManager.getToolGroup(toolGroupIdRef.current);
      if (existingToolGroup) {
        // @ts-ignore - destroy exists but types are incomplete
        existingToolGroup.destroy();
      }

      // 새 도구 그룹 생성
      const toolGroup = ToolGroupManager.createToolGroup(toolGroupIdRef.current);

      if (toolGroup) {
        // 도구 추가
        toolGroup.addTool(WindowLevelTool.toolName);
        toolGroup.addTool(PanTool.toolName);
        toolGroup.addTool(ZoomTool.toolName);
        toolGroup.addTool(StackScrollMouseWheelTool.toolName);

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
        toolGroup.setToolActive(StackScrollMouseWheelTool.toolName);

        // 모든 뷰포트에 도구 그룹 연결
        viewportIds.forEach((viewportId) => {
          toolGroup.addViewport(viewportId, renderingEngineRef.current!.id);
        });
        
        console.log('[MPR] Tools setup complete');
      }
    } catch (error) {
      console.error('[MPR] Failed to setup tools:', error);
    }
  };

  if (!isInitialized) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900 text-white">
        <div className="flex flex-col items-center gap-4">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-white text-sm font-medium">MPR 뷰어 초기화 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* 도구 바 */}
      <div className="bg-gray-800 border-b border-gray-700 px-4 py-2 flex items-center gap-2 flex-wrap">
        <Badge variant="outline" className="text-white border-gray-600">
          MPR 멀티플래너 뷰
        </Badge>
        <Badge className="bg-blue-600/80 backdrop-blur-md text-white border-none text-xs">
          이미지: {instanceIds.length}장
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
              단일 뷰로 전환
            </Button>
          )}
        </div>
      </div>

      {/* 로딩 오버레이 */}
      {isLoading && (
        <div className="absolute inset-0 bg-gray-900/80 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-white text-sm font-medium">MPR 이미지 로딩 중...</p>
            <p className="text-white/60 text-xs">{instanceIds.length}장의 이미지를 처리하고 있습니다</p>
          </div>
        </div>
      )}

      {/* 3분할 뷰포트 그리드 */}
      <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-1 p-1">
        {/* Axial (횡단면) - 좌상단 */}
        <div className="relative bg-black border border-gray-800 rounded-lg overflow-hidden">
          <div
            ref={axialRef}
            className="w-full h-full"
            style={{ minHeight: '300px' }}
          />
          <div className="absolute top-2 left-2 pointer-events-none z-10">
            <Badge className="bg-black/60 backdrop-blur-md text-white border-none">
              Axial (횡단면)
            </Badge>
          </div>
          <div className="absolute bottom-2 right-2 pointer-events-none z-10">
            <Badge className="bg-black/60 backdrop-blur-md text-white border-none text-xs">
              마우스 휠로 슬라이스 이동
            </Badge>
          </div>
        </div>

        {/* Sagittal (시상면) - 우상단 */}
        <div className="relative bg-black border border-gray-800 rounded-lg overflow-hidden">
          <div
            ref={sagittalRef}
            className="w-full h-full"
            style={{ minHeight: '300px' }}
          />
          <div className="absolute top-2 left-2 pointer-events-none z-10">
            <Badge className="bg-black/60 backdrop-blur-md text-white border-none">
              Sagittal (시상면)
            </Badge>
          </div>
          <div className="absolute bottom-2 right-2 pointer-events-none z-10">
            <Badge className="bg-black/60 backdrop-blur-md text-white border-none text-xs">
              마우스 휠로 슬라이스 이동
            </Badge>
          </div>
        </div>

        {/* Coronal (관상면) - 좌하단 */}
        <div className="relative bg-black border border-gray-800 rounded-lg overflow-hidden">
          <div
            ref={coronalRef}
            className="w-full h-full"
            style={{ minHeight: '300px' }}
          />
          <div className="absolute top-2 left-2 pointer-events-none z-10">
            <Badge className="bg-black/60 backdrop-blur-md text-white border-none">
              Coronal (관상면)
            </Badge>
          </div>
          <div className="absolute bottom-2 right-2 pointer-events-none z-10">
            <Badge className="bg-black/60 backdrop-blur-md text-white border-none text-xs">
              마우스 휠로 슬라이스 이동
            </Badge>
          </div>
        </div>

        {/* 정보 패널 - 우하단 */}
        <div className="relative bg-gradient-to-br from-gray-900 to-gray-800 border border-gray-700 rounded-lg overflow-hidden flex items-center justify-center">
          <div className="text-center space-y-4 p-8">
            <div className="text-6xl mb-4">🏥</div>
            <h3 className="text-xl font-bold text-white mb-2">MPR 멀티플래너 뷰어</h3>
            <div className="space-y-2 text-sm text-gray-300">
              <p className="flex items-center justify-center gap-2">
                <span className="text-blue-400">●</span>
                <span>Axial: 위→아래 단면</span>
              </p>
              <p className="flex items-center justify-center gap-2">
                <span className="text-green-400">●</span>
                <span>Sagittal: 좌→우 단면</span>
              </p>
              <p className="flex items-center justify-center gap-2">
                <span className="text-purple-400">●</span>
                <span>Coronal: 앞→뒤 단면</span>
              </p>
            </div>
            <div className="mt-6 pt-4 border-t border-gray-700">
              <p className="text-xs text-gray-400">
                마우스 휠: 슬라이스 이동<br />
                좌클릭 드래그: 윈도우/레벨<br />
                중간 버튼: 패닝<br />
                우클릭 드래그: 줌
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
