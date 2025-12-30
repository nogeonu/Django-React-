/**
 * Cornerstone3D 기반 DICOM 뷰어 컴포넌트
 */
import { useEffect, useRef, useState } from 'react';
import {
  RenderingEngine,
  Enums,
  type Types,
} from '@cornerstonejs/core';
import {
  addTool,
  ToolGroupManager,
  Enums as ToolEnums,
  LengthTool,
  ProbeTool,
  RectangleROITool,
  EllipticalROITool,
  BidirectionalTool,
  AngleTool,
  ZoomTool,
  PanTool,
  WindowLevelTool,
} from '@cornerstonejs/tools';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Slider } from '@/components/ui/slider';
import {
  Ruler,
  Square,
  Circle,
  MousePointer2,
  Sun,
} from 'lucide-react';
import { initCornerstone, createImageId, WINDOW_LEVEL_PRESETS } from '@/lib/cornerstone';

// 전역 렌더링 엔진 캐시 (WebGL 컨텍스트 재사용)
const renderingEngineCache = new Map<string, RenderingEngine>();

function getOrCreateRenderingEngine(engineId: string): RenderingEngine {
  let engine = renderingEngineCache.get(engineId);
  if (!engine) {
    engine = new RenderingEngine(engineId);
    renderingEngineCache.set(engineId, engine);
  }
  return engine;
}

interface CornerstoneViewerProps {
  instanceIds: string[];
  currentIndex: number;
  onIndexChange: (index: number) => void;
  showMeasurementTools?: boolean; // 측정 도구 표시 여부
  viewportId?: string; // 고유 viewport ID (4분할 뷰 등에서 사용)
}

export default function CornerstoneViewer({
  instanceIds,
  currentIndex,
  onIndexChange,
  showMeasurementTools = true, // 기본값 true
  viewportId, // 외부에서 전달받은 고유 ID
}: CornerstoneViewerProps) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [activeTool, setActiveTool] = useState<string>('WindowLevel');
  const [windowLevel, setWindowLevel] = useState(WINDOW_LEVEL_PRESETS.DEFAULT);
  const renderingEngineRef = useRef<RenderingEngine | null>(null);
  // 고유한 ID 생성 (컴포넌트마다 다른 ID 사용)
  // viewportId가 제공되면 사용, 아니면 랜덤 생성
  const uniqueId = viewportId || `${Date.now()}_${Math.random()}`;
  const renderingEngineIdRef = useRef<string>(`renderingEngine_${uniqueId}`);
  const viewportIdRef = useRef<string>(`viewport_${uniqueId}`);
  const toolGroupIdRef = useRef<string>(`toolGroup_${uniqueId}`);

  // Cornerstone 초기화
  useEffect(() => {
    const initialize = async () => {
      try {
        await initCornerstone();

        // 측정 도구 등록
        addTool(LengthTool);
        addTool(ProbeTool);
        addTool(RectangleROITool);
        addTool(EllipticalROITool);
        addTool(BidirectionalTool);
        addTool(AngleTool);
        addTool(ZoomTool);
        addTool(PanTool);
        addTool(WindowLevelTool);

        setIsInitialized(true);
      } catch (error) {
        console.error('Failed to initialize Cornerstone:', error);
      }
    };

    initialize();
  }, []);

  // 뷰포트 설정
  useEffect(() => {
    if (!isInitialized || !viewportRef.current || instanceIds.length === 0) {
      return;
    }

    const setupViewport = async () => {
      try {
        const element = viewportRef.current!;
        const renderingEngineId = renderingEngineIdRef.current;
        const viewportId = viewportIdRef.current;

        // 전역 캐시에서 렌더링 엔진 가져오기 또는 생성
        const renderingEngine = getOrCreateRenderingEngine(renderingEngineId);
        renderingEngineRef.current = renderingEngine;

        // 기존 뷰포트가 있다면 비활성화
        try {
          const existingViewport = renderingEngine.getViewport(viewportId);
          if (existingViewport) {
            renderingEngine.disableElement(viewportId);
          }
        } catch (e) {
          // 뷰포트가 없으면 무시
        }

        // 뷰포트 생성
        const viewportInput = {
          viewportId,
          type: Enums.ViewportType.STACK,
          element,
          defaultOptions: {
            background: [0, 0, 0] as Types.Point3,
          },
        };

        renderingEngine.enableElement(viewportInput);

        // 이미지 ID 생성
        const imageIds = instanceIds.map((id) =>
          createImageId(`/api/mri/orthanc/instances/${id}/file`)
        );

        // 스택 뷰포트 가져오기
        const viewport = renderingEngine.getViewport(viewportId);

        if (viewport) {
          // @ts-ignore - Stack viewport specific method
          await viewport.setStack(imageIds, currentIndex);

          // 첫 렌더링 (DICOM의 기본 Window/Level 사용)
          viewport.render();

          // DICOM 메타데이터에서 Window/Level 가져오기
          try {
            // @ts-ignore
            const image = viewport.getImageData();
            if (image) {
              // DICOM의 Window Width/Center 사용 (없으면 자동 계산)
              const dicomWindowWidth = image.windowWidth?.[0];
              const dicomWindowCenter = image.windowCenter?.[0];

              if (dicomWindowWidth && dicomWindowCenter) {
                console.log(`Using DICOM Window/Level: W=${dicomWindowWidth}, C=${dicomWindowCenter}`);
                setWindowLevel({
                  windowWidth: dicomWindowWidth,
                  windowCenter: dicomWindowCenter,
                });

                // @ts-ignore
                viewport.setProperties({
                  voiRange: {
                    lower: dicomWindowCenter - dicomWindowWidth / 2,
                    upper: dicomWindowCenter + dicomWindowWidth / 2,
                  },
                });
                viewport.render();
              }
            }
          } catch (e) {
            console.warn('Could not read DICOM Window/Level, using defaults', e);
          }
        }

        // 도구 그룹 설정
        setupTools(viewportId);
      } catch (error) {
        console.error('Failed to setup viewport:', error);
      }
    };

    setupViewport();

    // 클린업 함수: 컴포넌트 언마운트 시 리소스 정리
    return () => {
      // 뷰포트 비활성화
      if (renderingEngineRef.current) {
        try {
          renderingEngineRef.current.disableElement(viewportIdRef.current);
        } catch (e) {
          console.warn('Error disabling viewport on cleanup:', e);
        }
      }

      // 도구 그룹에서 뷰포트 제거
      const toolGroup = ToolGroupManager.getToolGroup(toolGroupIdRef.current);
      if (toolGroup && renderingEngineRef.current) {
        try {
          toolGroup.removeViewports(renderingEngineRef.current.id, viewportIdRef.current);
        } catch (e) {
          console.warn('Error removing viewport from tool group:', e);
        }
      }
    };
  }, [isInitialized, instanceIds]); // 원래대로 복구

  // 슬라이스 변경
  useEffect(() => {
    if (!renderingEngineRef.current) return;

    try {
      const viewport = renderingEngineRef.current.getViewport(viewportIdRef.current);
      if (viewport) {
        // @ts-ignore
        viewport.setImageIdIndex(currentIndex);
        viewport.render();
      }
    } catch (error) {
      console.error('Failed to change slice:', error);
    }
  }, [currentIndex]);

  // 윈도우 레벨 변경
  useEffect(() => {
    if (!renderingEngineRef.current) return;

    try {
      const viewport = renderingEngineRef.current.getViewport(viewportIdRef.current);
      if (viewport) {
        // @ts-ignore - setProperties exists but types are incomplete
        viewport.setProperties({
          voiRange: {
            lower: windowLevel.windowCenter - windowLevel.windowWidth / 2,
            upper: windowLevel.windowCenter + windowLevel.windowWidth / 2,
          },
        });
        viewport.render();
      }
    } catch (error) {
      console.error('Failed to change window level:', error);
    }
  }, [windowLevel]);

  // 도구 설정
  const setupTools = (viewportId: string) => {
    try {
      // 기존 도구 그룹 확인 또는 새로 생성
      let toolGroup = ToolGroupManager.getToolGroup(toolGroupIdRef.current);

      if (!toolGroup) {
        // 도구 그룹이 없으면 새로 생성
        toolGroup = ToolGroupManager.createToolGroup(toolGroupIdRef.current);
      }

      if (toolGroup) {
        // 기존 뷰포트 연결 제거 (있다면)
        try {
          toolGroup.removeViewports(renderingEngineRef.current!.id, viewportId);
        } catch (e) {
          // 뷰포트가 연결되어 있지 않으면 무시
        }

        // 도구 추가 (이미 추가된 경우 무시됨)
        try {
          toolGroup.addTool(WindowLevelTool.toolName);
          toolGroup.addTool(PanTool.toolName);
          toolGroup.addTool(ZoomTool.toolName);
          toolGroup.addTool(LengthTool.toolName);
          toolGroup.addTool(ProbeTool.toolName);
          toolGroup.addTool(RectangleROITool.toolName);
          toolGroup.addTool(EllipticalROITool.toolName);
          toolGroup.addTool(BidirectionalTool.toolName);
          toolGroup.addTool(AngleTool.toolName);
        } catch (e) {
          // 도구가 이미 추가되어 있으면 무시
        }

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

        // 뷰포트에 도구 그룹 연결
        toolGroup.addViewport(viewportId, renderingEngineRef.current!.id);
      }
    } catch (error) {
      console.error('Failed to setup tools:', error);
    }
  };

  // 도구 변경
  const handleToolChange = (toolName: string) => {
    const toolGroup = ToolGroupManager.getToolGroup(toolGroupIdRef.current);
    if (!toolGroup) return;

    // 모든 도구 비활성화
    toolGroup.setToolPassive(LengthTool.toolName);
    toolGroup.setToolPassive(ProbeTool.toolName);
    toolGroup.setToolPassive(RectangleROITool.toolName);
    toolGroup.setToolPassive(EllipticalROITool.toolName);
    toolGroup.setToolPassive(BidirectionalTool.toolName);
    toolGroup.setToolPassive(AngleTool.toolName);
    toolGroup.setToolPassive(WindowLevelTool.toolName);

    // 선택한 도구 활성화
    toolGroup.setToolActive(toolName, {
      bindings: [{ mouseButton: ToolEnums.MouseBindings.Primary }],
    });

    setActiveTool(toolName);
  };

  if (!isInitialized) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900 text-white">
        Cornerstone3D 초기화 중...
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* 도구 바 */}
      {showMeasurementTools && (
        <div className="bg-gray-800 border-b border-gray-700 px-4 py-3 flex items-center gap-3 flex-wrap">
          <Badge variant="outline" className="text-white border-gray-600 font-bold">
            측정 도구
          </Badge>
          <Button
            size="sm"
            variant={activeTool === WindowLevelTool.toolName ? 'default' : 'outline'}
            onClick={() => handleToolChange(WindowLevelTool.toolName)}
            className={`h-9 transition-all ${activeTool === WindowLevelTool.toolName
              ? 'bg-blue-600 hover:bg-blue-700 text-white'
              : 'bg-gray-700 hover:bg-gray-600 text-gray-200 border-gray-600'
              }`}
          >
            <Sun className="w-4 h-4 mr-1" />
            윈도우/레벨
          </Button>
          <div className="w-px h-6 bg-gray-600" />
          <Button
            size="sm"
            variant={activeTool === LengthTool.toolName ? 'default' : 'outline'}
            onClick={() => handleToolChange(LengthTool.toolName)}
            className={`h-9 transition-all ${activeTool === LengthTool.toolName
              ? 'bg-green-600 hover:bg-green-700 text-white'
              : 'bg-gray-700 hover:bg-gray-600 text-gray-200 border-gray-600'
              }`}
          >
            <Ruler className="w-4 h-4 mr-1" />
            거리 측정
          </Button>
          <Button
            size="sm"
            variant={activeTool === RectangleROITool.toolName ? 'default' : 'outline'}
            onClick={() => handleToolChange(RectangleROITool.toolName)}
            className={`h-9 transition-all ${activeTool === RectangleROITool.toolName
              ? 'bg-green-600 hover:bg-green-700 text-white'
              : 'bg-gray-700 hover:bg-gray-600 text-gray-200 border-gray-600'
              }`}
          >
            <Square className="w-4 h-4 mr-1" />
            사각형 ROI
          </Button>
          <Button
            size="sm"
            variant={activeTool === EllipticalROITool.toolName ? 'default' : 'outline'}
            onClick={() => handleToolChange(EllipticalROITool.toolName)}
            className={`h-9 transition-all ${activeTool === EllipticalROITool.toolName
              ? 'bg-green-600 hover:bg-green-700 text-white'
              : 'bg-gray-700 hover:bg-gray-600 text-gray-200 border-gray-600'
              }`}
          >
            <Circle className="w-4 h-4 mr-1" />
            타원 ROI
          </Button>
          <Button
            size="sm"
            variant={activeTool === ProbeTool.toolName ? 'default' : 'outline'}
            onClick={() => handleToolChange(ProbeTool.toolName)}
            className={`h-9 transition-all ${activeTool === ProbeTool.toolName
              ? 'bg-green-600 hover:bg-green-700 text-white'
              : 'bg-gray-700 hover:bg-gray-600 text-gray-200 border-gray-600'
              }`}
          >
            <MousePointer2 className="w-4 h-4 mr-1" />
            픽셀 값
          </Button>
        </div>
      )}

      {/* 뷰포트 */}
      <div
        className="flex-1 relative"
        onWheel={(e) => {
          if (instanceIds.length === 0) return;
          e.preventDefault();
          const delta = e.deltaY > 0 ? 1 : -1;
          const newIndex = Math.max(0, Math.min(instanceIds.length - 1, currentIndex + delta));
          if (newIndex !== currentIndex) {
            onIndexChange(newIndex);
          }
        }}
      >
        <div
          ref={viewportRef}
          className="w-full h-full"
          style={{ minHeight: '400px' }}
        />

        {/* 오버레이 정보 */}
        <div className="absolute top-12 left-4 flex flex-col gap-1.5 pointer-events-none z-20">
          <Badge className="bg-black/60 backdrop-blur-md text-white border-none text-xs px-2 py-0.5">
            슬라이스: {currentIndex + 1} / {instanceIds.length}
          </Badge>
          <Badge className="bg-blue-600/80 backdrop-blur-md text-white border-none text-xs px-2 py-0.5">
            W: {windowLevel.windowWidth} / L: {windowLevel.windowCenter}
          </Badge>
          {activeTool !== WindowLevelTool.toolName && (
            <Badge className="bg-green-600/80 backdrop-blur-md text-white border-none animate-pulse">
              {activeTool === LengthTool.toolName && '📏 클릭하여 거리 측정'}
              {activeTool === RectangleROITool.toolName && '⬜ 드래그하여 사각형 그리기'}
              {activeTool === EllipticalROITool.toolName && '⭕ 드래그하여 타원 그리기'}
              {activeTool === ProbeTool.toolName && '🔍 클릭하여 픽셀 값 확인'}
            </Badge>
          )}
        </div>
      </div>

      {/* 슬라이더 */}
      <div className="bg-gray-800 border-t border-gray-700 px-6 py-4">
        <div className="flex items-center gap-4">
          <span className="text-white text-sm min-w-[120px]">
            {currentIndex + 1} / {instanceIds.length}
          </span>
          <Slider
            value={[currentIndex]}
            onValueChange={(value) => onIndexChange(value[0])}
            max={instanceIds.length - 1}
            step={1}
            className="flex-1"
          />
        </div>
      </div>
    </div>
  );
}

