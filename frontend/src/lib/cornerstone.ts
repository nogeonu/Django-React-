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

    // WADO Image Loader 설정
    cornerstoneWADOImageLoader.configure({
      useWebWorkers: true,
      decodeConfig: {
        convertFloatPixelDataToInt: false,
        use16BitDataType: true,
      },
    });
    
    // DICOM 이미지 로더에 Photometric Interpretation 자동 처리 활성화
    cornerstoneWADOImageLoader.wadouri.dataSetCacheManager.purge();

    // Web Worker 초기화
    let maxWebWorkers = 1;
    if (navigator.hardwareConcurrency) {
      maxWebWorkers = Math.min(navigator.hardwareConcurrency, 7);
    }

    const config = {
      maxWebWorkers,
      startWebWorkersOnDemand: false,
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
  
  // MRI 프리셋
  MRI_BRAIN: { windowWidth: 600, windowCenter: 300 },
  MRI_T1: { windowWidth: 400, windowCenter: 200 },
  MRI_T2: { windowWidth: 800, windowCenter: 400 },
  
  // 기본
  DEFAULT: { windowWidth: 256, windowCenter: 128 },
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

