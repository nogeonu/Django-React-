import { X, Download, AlertCircle, CheckCircle2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface Detection {
    bbox: [number, number, number, number];
    confidence: number;
    class_id: number;
    class_name: string;
}

interface AIAnalysisResult {
    success: boolean;
    instance_id: string;
    detections: Detection[];
    detection_count: number;
    image_with_detections: string;
    model_info: {
        name: string;
        confidence_threshold: number;
    };
    error?: string;
}

interface AIAnalysisModalProps {
    isOpen: boolean;
    isAnalyzing: boolean;
    progress: number; // 0-100
    result: AIAnalysisResult | null;
    onClose: () => void;
}

export default function AIAnalysisModal({
    isOpen,
    isAnalyzing,
    progress,
    result,
    onClose
}: AIAnalysisModalProps) {
    if (!isOpen) return null;

    // 분석 중일 때
    if (isAnalyzing) {
        const radius = 80;
        const circumference = 2 * Math.PI * radius;
        const offset = circumference - (progress / 100) * circumference;

        return (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-80 backdrop-blur-sm">
                <div className="relative flex flex-col items-center">
                    {/* Circular Progress */}
                    <div className="relative">
                        {/* Glow effect background */}
                        <div className="absolute inset-0 blur-2xl opacity-50">
                            <svg width="200" height="200" className="transform -rotate-90">
                                <circle
                                    cx="100"
                                    cy="100"
                                    r={radius}
                                    stroke="url(#gradient)"
                                    strokeWidth="8"
                                    fill="none"
                                    strokeDasharray={circumference}
                                    strokeDashoffset={offset}
                                    className="transition-all duration-300 ease-out"
                                />
                                <defs>
                                    <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                        <stop offset="0%" stopColor="#3b82f6" />
                                        <stop offset="100%" stopColor="#06b6d4" />
                                    </linearGradient>
                                </defs>
                            </svg>
                        </div>

                        {/* Main progress circle */}
                        <svg width="200" height="200" className="transform -rotate-90">
                            {/* Background circle */}
                            <circle
                                cx="100"
                                cy="100"
                                r={radius}
                                stroke="#1e293b"
                                strokeWidth="8"
                                fill="none"
                            />
                            {/* Progress circle */}
                            <circle
                                cx="100"
                                cy="100"
                                r={radius}
                                stroke="url(#gradient2)"
                                strokeWidth="8"
                                fill="none"
                                strokeDasharray={circumference}
                                strokeDashoffset={offset}
                                strokeLinecap="round"
                                className="transition-all duration-300 ease-out drop-shadow-[0_0_10px_rgba(59,130,246,0.8)]"
                            />
                            <defs>
                                <linearGradient id="gradient2" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stopColor="#3b82f6" />
                                    <stop offset="50%" stopColor="#06b6d4" />
                                    <stop offset="100%" stopColor="#8b5cf6" />
                                </linearGradient>
                            </defs>
                        </svg>

                        {/* Percentage text */}
                        <div className="absolute inset-0 flex items-center justify-center">
                            <span className="text-6xl font-bold text-white drop-shadow-[0_0_20px_rgba(59,130,246,0.8)]">
                                {Math.round(progress)}%
                            </span>
                        </div>
                    </div>

                    {/* Status message */}
                    <div className="mt-8 text-center">
                        <p className="text-2xl font-bold text-white mb-2 drop-shadow-[0_0_10px_rgba(59,130,246,0.6)]">
                            AI 의료영상 판독중
                        </p>
                        <div className="flex items-center justify-center gap-1">
                            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" style={{ animationDelay: '0ms' }}></div>
                            <div className="w-2 h-2 bg-cyan-500 rounded-full animate-pulse" style={{ animationDelay: '200ms' }}></div>
                            <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" style={{ animationDelay: '400ms' }}></div>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    // 분석 완료 - 결과 표시
    if (result) {
        const handleDownload = () => {
            if (result.image_with_detections) {
                const link = document.createElement('a');
                link.href = result.image_with_detections;
                link.download = `ai_detection_${result.instance_id}.png`;
                link.click();
            }
        };

        return (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-80 backdrop-blur-sm p-4">
                <div className="bg-gray-900 rounded-2xl shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-hidden border border-gray-700">
                    {/* Header */}
                    <div className="bg-gradient-to-r from-blue-600 to-cyan-600 px-6 py-4 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            {result.success ? (
                                <CheckCircle2 className="w-6 h-6 text-white" />
                            ) : (
                                <AlertCircle className="w-6 h-6 text-yellow-300" />
                            )}
                            <h2 className="text-2xl font-bold text-white">
                                AI 정밀 분석 결과
                            </h2>
                        </div>
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={onClose}
                            className="text-white hover:bg-white/20"
                        >
                            <X className="w-5 h-5" />
                        </Button>
                    </div>

                    {/* Content */}
                    <div className="p-6 overflow-y-auto max-h-[calc(90vh-80px)]">
                        {result.success ? (
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                {/* 왼쪽: 디텍션 이미지 */}
                                <div className="space-y-4">
                                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                        <h3 className="text-lg font-semibold text-white mb-3">
                                            디텍션 결과 이미지
                                        </h3>
                                        <img
                                            src={result.image_with_detections}
                                            alt="AI Detection Result"
                                            className="w-full rounded-lg border-2 border-blue-500/30"
                                        />
                                        <Button
                                            onClick={handleDownload}
                                            className="w-full mt-4 bg-blue-600 hover:bg-blue-700"
                                        >
                                            <Download className="w-4 h-4 mr-2" />
                                            이미지 다운로드
                                        </Button>
                                    </div>
                                </div>

                                {/* 오른쪽: 디텍션 정보 */}
                                <div className="space-y-4">
                                    {/* 모델 정보 */}
                                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                        <h3 className="text-lg font-semibold text-white mb-3">
                                            모델 정보
                                        </h3>
                                        <div className="space-y-2 text-sm">
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">모델명:</span>
                                                <span className="text-white font-medium">
                                                    {result.model_info.name}
                                                </span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">신뢰도 임계값:</span>
                                                <span className="text-white font-medium">
                                                    {(result.model_info.confidence_threshold * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">검출된 객체:</span>
                                                <Badge variant="default" className="bg-blue-600">
                                                    {result.detection_count}개
                                                </Badge>
                                            </div>
                                        </div>
                                    </div>

                                    {/* 디텍션 목록 */}
                                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                                        <h3 className="text-lg font-semibold text-white mb-3">
                                            검출 목록
                                        </h3>
                                        {result.detections.length > 0 ? (
                                            <div className="space-y-3 max-h-96 overflow-y-auto">
                                                {result.detections.map((detection, index) => (
                                                    <div
                                                        key={index}
                                                        className="bg-gray-700/50 rounded-lg p-3 border border-gray-600"
                                                    >
                                                        <div className="flex items-center justify-between mb-2">
                                                            <span className="text-white font-medium">
                                                                #{index + 1} {detection.class_name}
                                                            </span>
                                                            <Badge
                                                                variant={
                                                                    detection.confidence > 0.7
                                                                        ? 'default'
                                                                        : detection.confidence > 0.5
                                                                            ? 'secondary'
                                                                            : 'outline'
                                                                }
                                                                className={
                                                                    detection.confidence > 0.7
                                                                        ? 'bg-green-600'
                                                                        : detection.confidence > 0.5
                                                                            ? 'bg-yellow-600'
                                                                            : 'bg-gray-600'
                                                                }
                                                            >
                                                                {(detection.confidence * 100).toFixed(1)}%
                                                            </Badge>
                                                        </div>
                                                        <div className="text-xs text-gray-400 space-y-1">
                                                            <div>
                                                                위치: ({Math.round(detection.bbox[0])}, {Math.round(detection.bbox[1])}) →
                                                                ({Math.round(detection.bbox[2])}, {Math.round(detection.bbox[3])})
                                                            </div>
                                                            <div>
                                                                크기: {Math.round(detection.bbox[2] - detection.bbox[0])} × {Math.round(detection.bbox[3] - detection.bbox[1])} px
                                                            </div>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : (
                                            <div className="text-center py-8 text-gray-400">
                                                <AlertCircle className="w-12 h-12 mx-auto mb-2 opacity-50" />
                                                <p>검출된 객체가 없습니다</p>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>
                        ) : (
                            // 에러 표시
                            <div className="text-center py-12">
                                <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
                                <h3 className="text-xl font-semibold text-white mb-2">
                                    분석 실패
                                </h3>
                                <p className="text-gray-400 mb-4">
                                    {result.error || '알 수 없는 오류가 발생했습니다'}
                                </p>
                                <Button onClick={onClose} variant="outline">
                                    닫기
                                </Button>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    }

    return null;
}
