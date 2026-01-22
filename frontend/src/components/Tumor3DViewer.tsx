/**
 * Plotly를 사용한 종양 전용 3D 렌더링 뷰어
 * Orthanc의 DICOM SEG 파일에서 종양만 추출하여 3D로 시각화
 */
import { useEffect, useRef, useState } from 'react';
import Plotly from 'plotly.js-dist-min';

interface Tumor3DViewerProps {
  segmentationInstanceId?: string; // 세그멘테이션 인스턴스 ID
}

export default function Tumor3DViewer({
  segmentationInstanceId,
}: Tumor3DViewerProps) {
  const plotlyRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!segmentationInstanceId || !plotlyRef.current) {
      if (!segmentationInstanceId) {
        setError('세그멘테이션 인스턴스 ID가 없습니다.');
        setIsLoading(false);
      }
      return;
    }

    const loadTumor3D = async () => {
      try {
        setIsLoading(true);
        setError(null);

        console.log('[Tumor3DViewer] 🎯 종양 3D 데이터 로드 시작...', {
          segmentationInstanceId,
        });

        // 1. 백엔드 API에서 종양 3D 데이터 가져오기
        const response = await fetch(
          `/api/mri/segmentation/instances/${segmentationInstanceId}/3d-data/`
        );
        const data = await response.json();

        if (!data.success) {
          throw new Error(data.error || '종양 3D 데이터 로드 실패');
        }

        console.log('[Tumor3DViewer] ✅ 종양 3D 데이터 로드 완료:', {
          num_voxels: data.num_voxels,
          dimensions: data.dimensions,
          spacing: data.spacing,
        });

        if (data.num_voxels === 0) {
          setError('종양 복셀이 없습니다.');
          setIsLoading(false);
          return;
        }

        // 2. Plotly로 3D 산점도 생성
        const trace = {
          x: data.x,
          y: data.y,
          z: data.z,
          mode: 'markers' as const,
          marker: {
            size: 3,
            color: 'red',
            opacity: 0.8,
            line: {
              width: 0,
            },
          },
          type: 'scatter3d' as const,
          name: '종양',
        };

        const layout = {
          title: {
            text: '종양 3D 시각화',
            font: { size: 18 },
          },
          scene: {
            xaxis: {
              title: `X (mm)`,
              backgroundcolor: 'rgb(20, 20, 20)',
              gridcolor: 'rgb(100, 100, 100)',
              showbackground: true,
            },
            yaxis: {
              title: `Y (mm)`,
              backgroundcolor: 'rgb(20, 20, 20)',
              gridcolor: 'rgb(100, 100, 100)',
              showbackground: true,
            },
            zaxis: {
              title: `Z (mm)`,
              backgroundcolor: 'rgb(20, 20, 20)',
              gridcolor: 'rgb(100, 100, 100)',
              showbackground: true,
            },
            aspectmode: 'data' as const,
            bgcolor: 'rgb(10, 10, 10)',
            camera: {
              eye: { x: 1.5, y: 1.5, z: 1.5 },
            },
          },
          margin: { l: 0, r: 0, t: 50, b: 0 },
          paper_bgcolor: 'rgb(10, 10, 10)',
          plot_bgcolor: 'rgb(10, 10, 10)',
          font: { color: 'white' },
        };

        const config = {
          displayModeBar: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['lasso2d', 'select2d'],
          responsive: true,
        };

        // 3. Plotly 그래프 생성
        await Plotly.newPlot(plotlyRef.current, [trace], layout, config);

        console.log('[Tumor3DViewer] ✅ Plotly 그래프 생성 완료');
        setIsLoading(false);
      } catch (err) {
        console.error('[Tumor3DViewer] ❌ 종양 3D 로드 실패:', err);
        setError(
          err instanceof Error ? err.message : '종양 3D 데이터 로드 실패'
        );
        setIsLoading(false);
      }
    };

    loadTumor3D();

    // Cleanup
    return () => {
      if (plotlyRef.current) {
        Plotly.purge(plotlyRef.current);
      }
    };
  }, [segmentationInstanceId]);

  return (
    <div className="relative w-full h-full bg-gray-950">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-950/80 z-10">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-red-500 mx-auto mb-4"></div>
            <p className="text-white">종양 3D 데이터 로딩 중...</p>
          </div>
        </div>
      )}

      {error && !isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-950/80 z-10">
          <div className="text-center bg-red-900/90 text-white p-6 rounded-lg max-w-md">
            <p className="text-lg font-bold mb-2">3D 뷰 로드 실패</p>
            <p className="text-sm">{error}</p>
          </div>
        </div>
      )}

      {!segmentationInstanceId && !isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-950/80 z-10">
          <div className="text-center text-gray-400">
            <p className="text-lg font-bold mb-2">세그멘테이션 데이터 없음</p>
            <p className="text-sm">AI 분석을 먼저 실행하여 세그멘테이션 데이터를 생성해주세요.</p>
          </div>
        </div>
      )}

      <div ref={plotlyRef} className="w-full h-full" />
    </div>
  );
}
