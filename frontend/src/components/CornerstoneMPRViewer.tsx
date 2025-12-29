/**
 * Cornerstone3D 4분할 MPR 뷰어 컴포넌트
 * Axial, Sagittal, Coronal, 3D 뷰를 동시에 표시
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
  WindowLevelTool,
  PanTool,
  ZoomTool,
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
  const volume3DRef = useRef<HTMLDivElement>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [windowLevel, setWindowLevel] = useState(WINDOW_LEVEL_PRESETS.MRI_BRAIN);
  const renderingEngineRef = useRef<RenderingEngine | null>(null);
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
      !volume3DRef.current ||
      instanceIds.length === 0
    ) {
      return;
    }

    const setupMPRViewports = async () => {
      try {
        const renderingEngineId = 'mprRenderingEngine';

        // 기존 렌더링 엔진 정리
        if (renderingEngineRef.current) {
          renderingEngineRef.current.destroy();
        }

        // 렌더링 엔진 생성
        const renderingEngine = new RenderingEngine(renderingEngineId);
        renderingEngineRef.current = renderingEngine;

        // 4개의 뷰포트 생성
        const viewportInputs = [
          {
            viewportId: 'AXIAL',
            type: Enums.ViewportType.ORTHOGRAPHIC,
            element: axialRef.current,
            defaultOptions: {
              orientation: Enums.OrientationAxis.AXIAL,
              background: [0, 0, 0] as Types.Point3,
            },
          },
          {
            viewportId: 'SAGITTAL',
            type: Enums.ViewportType.ORTHOGRAPHIC,
            element: sagittalRef.current,
            defaultOptions: {
              orientation: Enums.OrientationAxis.SAGITTAL,
              background: [0, 0, 0] as Types.Point3,
            },
          },
          {
            viewportId: 'CORONAL',
            type: Enums.ViewportType.ORTHOGRAPHIC,
            element: coronalRef.current,
            defaultOptions: {
              orientation: Enums.OrientationAxis.CORONAL,
              background: [0, 0, 0] as Types.Point3,
            },
          },
          {
            viewportId: 'VOLUME_3D',
            type: Enums.ViewportType.ORTHOGRAPHIC,
            element: volume3DRef.current,
            defaultOptions: {
              orientation: Enums.OrientationAxis.AXIAL,
              background: [0, 0, 0] as Types.Point3,
            },
          },
        ];

        // 뷰포트 활성화
        viewportInputs.forEach((input) => {
          if (input.element) {
            renderingEngine.enableElement(input as any);
          }
        });

        // 이미지 ID 생성
        const imageIds = instanceIds.map((id) =>
          createImageId(`/api/mri/orthanc/instances/${id}/file`)
        );

        // 각 뷰포트에 이미지 스택 설정
        const viewportIds = ['AXIAL', 'SAGITTAL', 'CORONAL', 'VOLUME_3D'];
        
        for (const viewportId of viewportIds) {
          const viewport = renderingEngine.getViewport(viewportId);
          if (viewport) {
            // @ts-ignore
            await viewport.setStack(imageIds, Math.floor(imageIds.length / 2));
            
            // 윈도우 레벨 설정
            // @ts-ignore - setProperties exists but types are incomplete
            viewport.setProperties({
              voiRange: {
                lower: windowLevel.windowCenter - windowLevel.windowWidth / 2,
                upper: windowLevel.windowCenter + windowLevel.windowWidth / 2,
              },
            });
            
            viewport.render();
          }
        }

        // 도구 그룹 설정
        setupTools(viewportIds);
      } catch (error) {
        console.error('Failed to setup MPR viewports:', error);
      }
    };

    setupMPRViewports();

    return () => {
      if (renderingEngineRef.current) {
        renderingEngineRef.current.destroy();
        renderingEngineRef.current = null;
      }
    };
  }, [isInitialized, instanceIds]);

  // 도구 설정
  const setupTools = (viewportIds: string[]) => {
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
      }
    } catch (error) {
      console.error('Failed to setup tools:', error);
    }
  };

  // 윈도우 레벨 프리셋 적용
  const applyPreset = (preset: typeof WINDOW_LEVEL_PRESETS[keyof typeof WINDOW_LEVEL_PRESETS]) => {
    setWindowLevel(preset);
    
    if (!renderingEngineRef.current) return;

    const viewportIds = ['AXIAL', 'SAGITTAL', 'CORONAL', 'VOLUME_3D'];
    viewportIds.forEach((viewportId) => {
      const viewport = renderingEngineRef.current!.getViewport(viewportId);
      if (viewport) {
        // @ts-ignore - setProperties exists but types are incomplete
        viewport.setProperties({
          voiRange: {
            lower: preset.windowCenter - preset.windowWidth / 2,
            upper: preset.windowCenter + preset.windowWidth / 2,
          },
        });
        viewport.render();
      }
    });
  };

  if (!isInitialized) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900 text-white">
        MPR 뷰어 초기화 중...
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* 도구 바 */}
      <div className="bg-gray-800 border-b border-gray-700 px-4 py-2 flex items-center gap-2 flex-wrap">
        <Badge variant="outline" className="text-white border-gray-600">
          4분할 MPR 뷰
        </Badge>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-white border-gray-600">
            프리셋
          </Badge>
          <Button
            size="sm"
            variant="outline"
            onClick={() => applyPreset(WINDOW_LEVEL_PRESETS.MRI_BREAST)}
            className="h-8 text-xs"
          >
            유방
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => applyPreset(WINDOW_LEVEL_PRESETS.MRI_T1)}
            className="h-8 text-xs"
          >
            T1
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => applyPreset(WINDOW_LEVEL_PRESETS.MRI_T2)}
            className="h-8 text-xs"
          >
            T2
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => applyPreset(WINDOW_LEVEL_PRESETS.DEFAULT)}
            className="h-8 text-xs"
          >
            기본
          </Button>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <Badge className="bg-blue-600/80 backdrop-blur-md text-white border-none">
            W: {windowLevel.windowWidth} / L: {windowLevel.windowCenter}
          </Badge>
          {onClose && (
            <Button
              size="sm"
              variant="outline"
              onClick={onClose}
              className="h-8"
            >
              <Grid3x3 className="w-4 h-4 mr-1" />
              단일 뷰로 전환
            </Button>
          )}
        </div>
      </div>

      {/* 4분할 뷰포트 그리드 */}
      <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-1 p-1">
        {/* Axial */}
        <div className="relative bg-black border border-gray-800 rounded-lg overflow-hidden">
          <div
            ref={axialRef}
            className="w-full h-full"
            style={{ minHeight: '300px' }}
          />
          <div className="absolute top-2 left-2 pointer-events-none">
            <Badge className="bg-black/60 backdrop-blur-md text-white border-none">
              Axial (횡단면)
            </Badge>
          </div>
        </div>

        {/* Sagittal */}
        <div className="relative bg-black border border-gray-800 rounded-lg overflow-hidden">
          <div
            ref={sagittalRef}
            className="w-full h-full"
            style={{ minHeight: '300px' }}
          />
          <div className="absolute top-2 left-2 pointer-events-none">
            <Badge className="bg-black/60 backdrop-blur-md text-white border-none">
              Sagittal (시상면)
            </Badge>
          </div>
        </div>

        {/* Coronal */}
        <div className="relative bg-black border border-gray-800 rounded-lg overflow-hidden">
          <div
            ref={coronalRef}
            className="w-full h-full"
            style={{ minHeight: '300px' }}
          />
          <div className="absolute top-2 left-2 pointer-events-none">
            <Badge className="bg-black/60 backdrop-blur-md text-white border-none">
              Coronal (관상면)
            </Badge>
          </div>
        </div>

        {/* 3D Volume */}
        <div className="relative bg-black border border-gray-800 rounded-lg overflow-hidden">
          <div
            ref={volume3DRef}
            className="w-full h-full"
            style={{ minHeight: '300px' }}
          />
          <div className="absolute top-2 left-2 pointer-events-none">
            <Badge className="bg-black/60 backdrop-blur-md text-white border-none">
              3D Volume
            </Badge>
          </div>
        </div>
      </div>
    </div>
  );
}

