import { useEffect } from 'react';

interface AIAnalysisModalProps {
    isOpen: boolean;
    progress: number; // 0-100
    onComplete?: () => void;
}

export default function AIAnalysisModal({ isOpen, progress, onComplete }: AIAnalysisModalProps) {
    useEffect(() => {
        if (progress >= 100 && onComplete) {
            onComplete();
        }
    }, [progress, onComplete]);

    if (!isOpen) return null;

    // Calculate circle parameters
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
