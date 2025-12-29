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

interface CornerstoneViewerProps {
  instanceIds: string[];
  currentIndex: number;
  onIndexChange: (index: number) => void;
}

export default function CornerstoneViewer({
  instanceIds,
  currentIndex,
  onIndexChange,
}: CornerstoneViewerProps) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [activeTool, setActiveTool] = useState<string>('WindowLevel');
  const [windowLevel, setWindowLevel] = useState(WINDOW_LEVEL_PRESETS.DEFAULT);
  const renderingEngineRef = useRef<RenderingEngine | null>(null);
  const toolGroupIdRef = useRef<string>('DICOM_TOOL_GROUP');

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
        const renderingEngineId = 'myRenderingEngine';
        const viewportId = 'CT_STACK';

        // 기존 렌더링 엔진 정리
        if (renderingEngineRef.current) {
          renderingEngineRef.current.destroy();
        }

        // 렌더링 엔진 생성
        const renderingEngine = new RenderingEngine(renderingEngineId);
        renderingEngineRef.current = renderingEngine;

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
          
          // 윈도우 레벨 설정 (Photometric Interpretation은 자동 처리됨)
          // @ts-ignore - setProperties exists but types are incomplete
          viewport.setProperties({
            voiRange: {
              lower: windowLevel.windowCenter - windowLevel.windowWidth / 2,
              upper: windowLevel.windowCenter + windowLevel.windowWidth / 2,
            },
          });

          viewport.render();
        }

        // 도구 그룹 설정
        setupTools(viewportId);
      } catch (error) {
        console.error('Failed to setup viewport:', error);
      }
    };

    setupViewport();

    return () => {
      if (renderingEngineRef.current) {
        renderingEngineRef.current.destroy();
        renderingEngineRef.current = null;
      }
    };
  }, [isInitialized, instanceIds]);

  // 슬라이스 변경
  useEffect(() => {
    if (!renderingEngineRef.current) return;

    try {
      const viewport = renderingEngineRef.current.getViewport('CT_STACK');
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
      const viewport = renderingEngineRef.current.getViewport('CT_STACK');
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
        toolGroup.addTool(LengthTool.toolName);
        toolGroup.addTool(ProbeTool.toolName);
        toolGroup.addTool(RectangleROITool.toolName);
        toolGroup.addTool(EllipticalROITool.toolName);
        toolGroup.addTool(BidirectionalTool.toolName);
        toolGroup.addTool(AngleTool.toolName);

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

  // 윈도우 레벨 프리셋 적용
  const applyPreset = (preset: typeof WINDOW_LEVEL_PRESETS[keyof typeof WINDOW_LEVEL_PRESETS]) => {
    setWindowLevel(preset);
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
      <div className="bg-gray-800 border-b border-gray-700 px-4 py-2 flex items-center gap-2 flex-wrap">
        <Badge variant="outline" className="text-white border-gray-600">
          도구
        </Badge>
        <Button
          size="sm"
          variant={activeTool === WindowLevelTool.toolName ? 'default' : 'outline'}
          onClick={() => handleToolChange(WindowLevelTool.toolName)}
          className="h-8"
        >
          <Sun className="w-4 h-4 mr-1" />
          윈도우/레벨
        </Button>
        <Button
          size="sm"
          variant={activeTool === LengthTool.toolName ? 'default' : 'outline'}
          onClick={() => handleToolChange(LengthTool.toolName)}
          className="h-8"
        >
          <Ruler className="w-4 h-4 mr-1" />
          거리
        </Button>
        <Button
          size="sm"
          variant={activeTool === RectangleROITool.toolName ? 'default' : 'outline'}
          onClick={() => handleToolChange(RectangleROITool.toolName)}
          className="h-8"
        >
          <Square className="w-4 h-4 mr-1" />
          사각형
        </Button>
        <Button
          size="sm"
          variant={activeTool === EllipticalROITool.toolName ? 'default' : 'outline'}
          onClick={() => handleToolChange(EllipticalROITool.toolName)}
          className="h-8"
        >
          <Circle className="w-4 h-4 mr-1" />
          타원
        </Button>
        <Button
          size="sm"
          variant={activeTool === ProbeTool.toolName ? 'default' : 'outline'}
          onClick={() => handleToolChange(ProbeTool.toolName)}
          className="h-8"
        >
          <MousePointer2 className="w-4 h-4 mr-1" />
          프로브
        </Button>

        <div className="ml-auto flex items-center gap-2">
          <Badge variant="outline" className="text-white border-gray-600">
            프리셋
          </Badge>
          <Button
            size="sm"
            variant="outline"
            onClick={() => applyPreset(WINDOW_LEVEL_PRESETS.MRI_BRAIN)}
            className="h-8 text-xs"
          >
            뇌
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => applyPreset(WINDOW_LEVEL_PRESETS.CT_LUNG)}
            className="h-8 text-xs"
          >
            폐
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => applyPreset(WINDOW_LEVEL_PRESETS.CT_BONE)}
            className="h-8 text-xs"
          >
            뼈
          </Button>
        </div>
      </div>

      {/* 뷰포트 */}
      <div className="flex-1 relative">
        <div
          ref={viewportRef}
          className="w-full h-full"
          style={{ minHeight: '400px' }}
        />

        {/* 오버레이 정보 */}
        <div className="absolute top-4 left-4 flex flex-col gap-2 pointer-events-none">
          <Badge className="bg-black/60 backdrop-blur-md text-white border-none">
            슬라이스: {currentIndex + 1} / {instanceIds.length}
          </Badge>
          <Badge className="bg-blue-600/80 backdrop-blur-md text-white border-none">
            W: {windowLevel.windowWidth} / L: {windowLevel.windowCenter}
          </Badge>
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

