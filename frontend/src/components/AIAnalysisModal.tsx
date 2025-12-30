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
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-90 backdrop-blur-md">
            <div className="relative flex flex-col items-center">
                {/* Outer glow rings */}
                <div className="absolute inset-0 flex items-center justify-center">
                    {/* Pulsing outer ring 1 */}
                    <div className="absolute w-[280px] h-[280px] rounded-full bg-gradient-to-r from-blue-500/20 via-cyan-500/20 to-pink-500/20 blur-3xl animate-pulse" />
                    {/* Pulsing outer ring 2 */}
                    <div className="absolute w-[240px] h-[240px] rounded-full bg-gradient-to-r from-pink-500/30 via-purple-500/30 to-blue-500/30 blur-2xl animate-pulse" style={{ animationDelay: '0.5s' }} />
                    {/* Pulsing outer ring 3 */}
                    <div className="absolute w-[200px] h-[200px] rounded-full bg-gradient-to-r from-cyan-400/40 via-blue-400/40 to-pink-400/40 blur-xl animate-pulse" style={{ animationDelay: '1s' }} />
                </div>

                {/* Main progress container */}
                <div className="relative z-10">
                    {/* Background decorative rings */}
                    <svg width="220" height="220" className="absolute -inset-2.5 transform -rotate-90 opacity-30">
                        <circle
                            cx="110"
                            cy="110"
                            r="95"
                            stroke="url(#bgGradient1)"
                            strokeWidth="2"
                            fill="none"
                            className="animate-spin"
                            style={{ animationDuration: '8s' }}
                        />
                        <circle
                            cx="110"
                            cy="110"
                            r="100"
                            stroke="url(#bgGradient2)"
                            strokeWidth="1"
                            fill="none"
                            className="animate-spin"
                            style={{ animationDuration: '12s', animationDirection: 'reverse' }}
                        />
                        <defs>
                            <linearGradient id="bgGradient1" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stopColor="#ec4899" />
                                <stop offset="50%" stopColor="#06b6d4" />
                                <stop offset="100%" stopColor="#3b82f6" />
                            </linearGradient>
                            <linearGradient id="bgGradient2" x1="100%" y1="0%" x2="0%" y2="100%">
                                <stop offset="0%" stopColor="#8b5cf6" />
                                <stop offset="50%" stopColor="#06b6d4" />
                                <stop offset="100%" stopColor="#ec4899" />
                            </linearGradient>
                        </defs>
                    </svg>

                    {/* Intense glow effect */}
                    <div className="absolute inset-0 blur-3xl opacity-60">
                        <svg width="200" height="200" className="transform -rotate-90">
                            <circle
                                cx="100"
                                cy="100"
                                r={radius}
                                stroke="url(#glowGradient)"
                                strokeWidth="12"
                                fill="none"
                                strokeDasharray={circumference}
                                strokeDashoffset={offset}
                                className="transition-all duration-300 ease-out"
                            />
                            <defs>
                                <linearGradient id="glowGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stopColor="#ec4899" />
                                    <stop offset="33%" stopColor="#06b6d4" />
                                    <stop offset="66%" stopColor="#3b82f6" />
                                    <stop offset="100%" stopColor="#8b5cf6" />
                                </linearGradient>
                            </defs>
                        </svg>
                    </div>

                    {/* Main progress circle */}
                    <svg width="200" height="200" className="transform -rotate-90 relative z-10">
                        {/* Dark background circle */}
                        <circle
                            cx="100"
                            cy="100"
                            r={radius}
                            stroke="#0f172a"
                            strokeWidth="10"
                            fill="none"
                        />
                        {/* Inner shadow circle */}
                        <circle
                            cx="100"
                            cy="100"
                            r={radius - 5}
                            stroke="#1e293b"
                            strokeWidth="2"
                            fill="none"
                            opacity="0.5"
                        />
                        {/* Main progress circle with vibrant gradient */}
                        <circle
                            cx="100"
                            cy="100"
                            r={radius}
                            stroke="url(#mainGradient)"
                            strokeWidth="10"
                            fill="none"
                            strokeDasharray={circumference}
                            strokeDashoffset={offset}
                            strokeLinecap="round"
                            className="transition-all duration-300 ease-out"
                            style={{
                                filter: 'drop-shadow(0 0 8px rgba(236, 72, 153, 0.8)) drop-shadow(0 0 16px rgba(6, 182, 212, 0.6)) drop-shadow(0 0 24px rgba(59, 130, 246, 0.4))'
                            }}
                        />
                        {/* Outer highlight ring */}
                        <circle
                            cx="100"
                            cy="100"
                            r={radius + 5}
                            stroke="url(#outerGradient)"
                            strokeWidth="2"
                            fill="none"
                            strokeDasharray={circumference}
                            strokeDashoffset={offset}
                            strokeLinecap="round"
                            className="transition-all duration-300 ease-out opacity-60"
                        />
                        <defs>
                            <linearGradient id="mainGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stopColor="#ec4899" />
                                <stop offset="25%" stopColor="#06b6d4" />
                                <stop offset="50%" stopColor="#3b82f6" />
                                <stop offset="75%" stopColor="#8b5cf6" />
                                <stop offset="100%" stopColor="#ec4899" />
                            </linearGradient>
                            <linearGradient id="outerGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stopColor="#fbbf24" />
                                <stop offset="50%" stopColor="#06b6d4" />
                                <stop offset="100%" stopColor="#ec4899" />
                            </linearGradient>
                        </defs>
                    </svg>

                    {/* Percentage text with enhanced glow */}
                    <div className="absolute inset-0 flex items-center justify-center">
                        <span
                            className="text-7xl font-black text-white"
                            style={{
                                textShadow: '0 0 20px rgba(236, 72, 153, 0.8), 0 0 40px rgba(6, 182, 212, 0.6), 0 0 60px rgba(59, 130, 246, 0.4), 0 0 80px rgba(139, 92, 246, 0.3)'
                            }}
                        >
                            {Math.round(progress)}%
                        </span>
                    </div>
                </div>

                {/* Status message with enhanced styling */}
                <div className="mt-12 text-center relative z-10">
                    <p
                        className="text-3xl font-black text-white mb-3 tracking-wide"
                        style={{
                            textShadow: '0 0 10px rgba(236, 72, 153, 0.8), 0 0 20px rgba(6, 182, 212, 0.6), 0 0 30px rgba(59, 130, 246, 0.4)'
                        }}
                    >
                        AI 의료영상 판독중
                    </p>
                    <div className="flex items-center justify-center gap-2">
                        <div className="w-3 h-3 bg-pink-500 rounded-full animate-pulse shadow-lg shadow-pink-500/50" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-3 h-3 bg-cyan-500 rounded-full animate-pulse shadow-lg shadow-cyan-500/50" style={{ animationDelay: '200ms' }}></div>
                        <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse shadow-lg shadow-blue-500/50" style={{ animationDelay: '400ms' }}></div>
                        <div className="w-3 h-3 bg-purple-500 rounded-full animate-pulse shadow-lg shadow-purple-500/50" style={{ animationDelay: '600ms' }}></div>
                    </div>
                </div>

                {/* Floating particles */}
                <div className="absolute inset-0 pointer-events-none">
                    {[...Array(12)].map((_, i) => (
                        <div
                            key={i}
                            className="absolute w-1 h-1 bg-white rounded-full animate-pulse"
                            style={{
                                left: `${Math.random() * 100}%`,
                                top: `${Math.random() * 100}%`,
                                animationDelay: `${Math.random() * 2}s`,
                                opacity: Math.random() * 0.5 + 0.3,
                                boxShadow: '0 0 4px rgba(255, 255, 255, 0.8)'
                            }}
                        />
                    ))}
                </div>
            </div>
        </div>
    );
}
