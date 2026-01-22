import React, { useEffect, useRef, useState } from 'react';
import {
    RenderingEngine,
    type Types,
    Enums,
    volumeLoader,
    setVolumesForViewports,
    cache,
} from '@cornerstonejs/core';
import {
    addTool,
    ToolGroupManager,
    Enums as ToolEnums,
    ZoomTool,
    PanTool,
    TrackballRotateTool,
    StackScrollTool,
    segmentation,
} from '@cornerstonejs/tools';
import { initCornerstone, createImageId } from '@/lib/cornerstone';
import { Loader2 } from 'lucide-react';

interface Props {
    imageUrls: string[];
    maskUrls?: string[];
    patientId: string;
}

const Cornerstone3DMPRViewer: React.FC<Props> = ({ imageUrls, maskUrls = [], patientId }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const axialRef = useRef<HTMLDivElement>(null);
    const sagittalRef = useRef<HTMLDivElement>(null);
    const coronalRef = useRef<HTMLDivElement>(null);
    const volume3dRef = useRef<HTMLDivElement>(null);

    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const renderingEngineId = `engine_${patientId}`;
    const volumeId = `volume_${patientId}`;
    const segmentationId = `seg_${patientId}`;
    const toolGroupId = `toolGroup_${patientId}`;

    useEffect(() => {
        let isMounted = true;
        let engine: RenderingEngine | undefined;

        const init = async () => {
            if (imageUrls.length === 0) return;

            try {
                await initCornerstone();
                if (!isMounted) return;

                // Add tools
                addTool(ZoomTool);
                addTool(PanTool);
                addTool(TrackballRotateTool);
                addTool(StackScrollTool);

                // Tool Group
                let toolGroup = ToolGroupManager.getToolGroup(toolGroupId);
                if (!toolGroup) {
                    toolGroup = ToolGroupManager.createToolGroup(toolGroupId)!;
                    toolGroup.addTool(ZoomTool.toolName);
                    toolGroup.addTool(PanTool.toolName);
                    toolGroup.addTool(TrackballRotateTool.toolName);
                    toolGroup.addTool(StackScrollTool.toolName);

                    toolGroup.setToolActive(StackScrollTool.toolName);
                    toolGroup.setToolActive(PanTool.toolName, {
                        bindings: [{ mouseButton: ToolEnums.MouseBindings.Auxiliary }],
                    });
                    toolGroup.setToolActive(ZoomTool.toolName, {
                        bindings: [{ mouseButton: ToolEnums.MouseBindings.Secondary }],
                    });
                    toolGroup.setToolActive(TrackballRotateTool.toolName, {
                        bindings: [{ mouseButton: ToolEnums.MouseBindings.Primary }],
                    });
                }

                // Engine
                engine = new RenderingEngine(renderingEngineId);

                const viewportIds = {
                    AXIAL: `axial_${patientId}`,
                    SAGITTAL: `sagittal_${patientId}`,
                    CORONAL: `coronal_${patientId}`,
                    VOLUME_3D: `3d_${patientId}`,
                };

                const viewportInputs: Types.PublicViewportInput[] = [
                    {
                        viewportId: viewportIds.AXIAL,
                        type: Enums.ViewportType.ORTHOGRAPHIC,
                        element: axialRef.current!,
                        defaultOptions: { orientation: Enums.OrientationAxis.AXIAL },
                    },
                    {
                        viewportId: viewportIds.SAGITTAL,
                        type: Enums.ViewportType.ORTHOGRAPHIC,
                        element: sagittalRef.current!,
                        defaultOptions: { orientation: Enums.OrientationAxis.SAGITTAL },
                    },
                    {
                        viewportId: viewportIds.CORONAL,
                        type: Enums.ViewportType.ORTHOGRAPHIC,
                        element: coronalRef.current!,
                        defaultOptions: { orientation: Enums.OrientationAxis.CORONAL },
                    },
                    {
                        viewportId: viewportIds.VOLUME_3D,
                        type: Enums.ViewportType.VOLUME_3D,
                        element: volume3dRef.current!,
                    },
                ];

                engine.setViewports(viewportInputs);

                Object.values(viewportIds).forEach((id) => {
                    toolGroup!.addViewport(id, renderingEngineId);
                });

                // Load Volume
                const imageIds = imageUrls.map(createImageId);

                // Define volume type and load
                const volume = await volumeLoader.createAndCacheVolume(volumeId, {
                    imageIds,
                });

                // Load the volume data
                volume.load();

                await setVolumesForViewports(
                    engine,
                    [{ volumeId }],
                    Object.values(viewportIds)
                );

                // Handle Segmentations
                if (maskUrls.length > 0) {
                    const segImageIds = maskUrls.map(createImageId);

                    await segmentation.addSegmentations([
                        {
                            segmentationId,
                            representation: {
                                type: ToolEnums.SegmentationRepresentations.Labelmap,
                                data: {
                                    imageIds: segImageIds,
                                },
                            },
                        },
                    ]);

                    await segmentation.addSegmentationRepresentations(toolGroupId, [
                        {
                            segmentationId,
                            type: ToolEnums.SegmentationRepresentations.Labelmap,
                        },
                    ]);

                    // Add surface representation for 3D viewport
                    try {
                        await segmentation.addSegmentationRepresentations(toolGroupId, [
                            {
                                segmentationId,
                                type: ToolEnums.SegmentationRepresentations.Surface,
                            },
                        ]);
                    } catch (e) {
                        console.warn('Surface representation not supported or failed:', e);
                    }
                }

                engine.render();
                setIsLoading(false);
            } catch (err: any) {
                console.error('Cornerstone 3D Error:', err);
                setError(err.message || 'Error loading 3D visualization');
                setIsLoading(false);
            }
        };

        init();

        return () => {
            isMounted = false;
            if (engine) engine.destroy();
            cache.removeVolumeLoadObject(volumeId);
            segmentation.state.removeSegmentation(segmentationId);
        };
    }, [imageUrls, maskUrls]);

    if (error) {
        return (
            <div className="flex items-center justify-center h-[800px] bg-gray-900 text-red-500">
                <p>오류: {error}</p>
            </div>
        );
    }

    return (
        <div ref={containerRef} className="grid grid-cols-2 gap-2 h-[800px] bg-black p-2 rounded-lg overflow-hidden relative">
            {isLoading && (
                <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black/50 backdrop-blur-sm">
                    <Loader2 className="w-12 h-12 animate-spin text-blue-500 mb-4" />
                    <p className="text-white">3D 볼륨 구성 중...</p>
                </div>
            )}

            <div className="relative border border-gray-700 rounded overflow-hidden">
                <div className="absolute top-2 left-2 z-10 text-xs text-white bg-black/50 px-2 py-1 rounded">Axial</div>
                <div ref={axialRef} className="w-full h-full" />
            </div>

            <div className="relative border border-gray-700 rounded overflow-hidden">
                <div className="absolute top-2 left-2 z-10 text-xs text-white bg-black/50 px-2 py-1 rounded">Sagittal</div>
                <div ref={sagittalRef} className="w-full h-full" />
            </div>

            <div className="relative border border-gray-700 rounded overflow-hidden">
                <div className="absolute top-2 left-2 z-10 text-xs text-white bg-black/50 px-2 py-1 rounded">Coronal</div>
                <div ref={coronalRef} className="w-full h-full" />
            </div>

            <div className="relative border border-gray-700 rounded overflow-hidden">
                <div className="absolute top-2 left-2 z-10 text-xs text-white bg-black/50 px-2 py-1 rounded">3D Rendering</div>
                <div ref={volume3dRef} className="w-full h-full" />
            </div>
        </div>
    );
};

export default Cornerstone3DMPRViewer;
