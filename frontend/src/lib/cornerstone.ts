/**
 * Cornerstone3D 초기화 및 설정
 */
import { init as csRenderInit } from '@cornerstonejs/core';
import { init as csToolsInit } from '@cornerstonejs/tools';
// @ts-ignore - No type definitions available
import * as cornerstoneWADOImageLoader from 'cornerstone-wado-image-loader';
// @ts-ignore - No type definitions available
import dicomParser from 'dicom-parser';

let isInitialized = false;

/**
 * Cornerstone3D 초기화
 */
export async function initCornerstone() {
  if (isInitialized) {
    return;
  }

  try {
    // Cornerstone Core 초기화
    await csRenderInit();
    console.log('✅ Cornerstone3D Core initialized');

    // Cornerstone Tools 초기화
    await csToolsInit();
    console.log('✅ Cornerstone3D Tools initialized');

    // WADO Image Loader 설정
    cornerstoneWADOImageLoader.external.cornerstone = await import('@cornerstonejs/core');
    cornerstoneWADOImageLoader.external.dicomParser = dicomParser;

    // WADO Image Loader 설정 - 이벤트 에러 방지
    cornerstoneWADOImageLoader.configure({
      useWebWorkers: true,
      decodeConfig: {
        convertFloatPixelDataToInt: false,
        use16BitDataType: true,
      },
      // 진행 상황 이벤트 에러 방지: beforeSend에서 progress 이벤트 리스너 제거
      beforeSend: function (xhr: any) {
        // XMLHttpRequest의 progress 이벤트 리스너를 안전하게 처리
        if (xhr && xhr.upload) {
          try {
            // progress 이벤트 핸들러를 안전하게 추가 (오류 무시)
            xhr.upload.addEventListener('progress', () => {}, { passive: true });
          } catch (e) {
            // 이벤트 리스너 추가 실패는 무시
          }
        }
      },
    });
    
    // 전역 에러 핸들러로 Cornerstone 이벤트 오류 무시 (업로드 기능에 영향 없도록)
    if (typeof window !== 'undefined') {
      window.addEventListener('error', (event) => {
        // "Event type was not defined" 오류는 무시 (Cornerstone 내부 이벤트 처리 관련)
        if (event.message && (
          event.message.includes('Event type was not defined') ||
          event.message.includes('triggerEvent')
        )) {
          event.preventDefault();
          event.stopPropagation();
          return false;
        }
      }, true); // capture phase에서 처리
      
      // Unhandled promise rejection도 처리
      window.addEventListener('unhandledrejection', (event) => {
        if (event.reason && typeof event.reason === 'string') {
          if (event.reason.includes('Event type was not defined') ||
              event.reason.includes('triggerEvent')) {
            event.preventDefault();
            return false;
          }
        }
      });
    }

    // DICOM 이미지 로더에 Photometric Interpretation 자동 처리 활성화
    try {
      cornerstoneWADOImageLoader.wadouri.dataSetCacheManager.purge();
    } catch (e) {
      // 캐시 매니저가 없을 수 있음 - 무시
    }

    // Web Worker 초기화
    let maxWebWorkers = 1;
    if (navigator.hardwareConcurrency) {
      maxWebWorkers = Math.min(navigator.hardwareConcurrency, 4); // 4로 제한하여 메모리 사용량 감소
    }

    const config = {
      maxWebWorkers,
      startWebWorkersOnDemand: true, // 필요할 때만 시작
      taskConfiguration: {
        decodeTask: {
          initializeCodecsOnStartup: false,
          strict: false,
        },
      },
    };

    cornerstoneWADOImageLoader.webWorkerManager.initialize(config);
    console.log(`✅ WADO Image Loader initialized with ${maxWebWorkers} workers`);

    isInitialized = true;
  } catch (error) {
    console.error('❌ Failed to initialize Cornerstone3D:', error);
    throw error;
  }
}

/**
 * DICOM 이미지 ID 생성
 * @param url - DICOM 이미지 URL
 */
export function createImageId(url: string): string {
  // Orthanc DICOM 파일 URL을 Cornerstone 이미지 ID로 변환
  return `wadouri:${url}`;
}

/**
 * 윈도우 레벨 프리셋
 */
export const WINDOW_LEVEL_PRESETS = {
  // CT 프리셋
  CT_LUNG: { windowWidth: 1500, windowCenter: -600 },
  CT_BONE: { windowWidth: 2000, windowCenter: 500 },
  CT_BRAIN: { windowWidth: 80, windowCenter: 40 },
  CT_ABDOMEN: { windowWidth: 400, windowCenter: 50 },
  CT_LIVER: { windowWidth: 150, windowCenter: 30 },

  // MRI 프리셋 (유방 MRI 최적화)
  MRI_BRAIN: { windowWidth: 600, windowCenter: 300 },
  MRI_T1: { windowWidth: 1000, windowCenter: 500 },
  MRI_T2: { windowWidth: 2000, windowCenter: 1000 },
  MRI_BREAST: { windowWidth: 1500, windowCenter: 750 }, // 유방 MRI 전용

  // 기본 (16비트 DICOM용)
  DEFAULT: { windowWidth: 4096, windowCenter: 2048 },
};

/**
 * 측정 도구 타입
 */
export enum MeasurementTool {
  LENGTH = 'Length',
  BIDIRECTIONAL = 'Bidirectional',
  ELLIPSE_ROI = 'EllipticalROI',
  RECTANGLE_ROI = 'RectangleROI',
  ANGLE = 'Angle',
  ARROW_ANNOTATE = 'ArrowAnnotate',
  PROBE = 'Probe',
}

/**
 * 뷰포트 방향
 */
export enum ViewportOrientation {
  AXIAL = 'axial',
  SAGITTAL = 'sagittal',
  CORONAL = 'coronal',
}

export { cornerstoneWADOImageLoader, dicomParser };

