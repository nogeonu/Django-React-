import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { Activity, Home, Plus, Calendar } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/context/AuthContext";
import NotificationDropdown from "@/components/NotificationDropdown";
import UserProfileDropdown from "@/components/UserProfileDropdown";
import doctorProfile from "@/assets/doctor-profile.png";
import { useQuery } from "@tanstack/react-query";
import { apiRequest } from "@/lib/api";

interface MedicalLayoutProps {
    children: React.ReactNode;
}

export default function MedicalLayout({ children }: MedicalLayoutProps) {
    const { user } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();

    const { data: dashboardStats } = useQuery({
        queryKey: ["dashboard-statistics"],
        queryFn: async () => {
            try {
                const response = await apiRequest("GET", "/api/lung_cancer/medical-records/dashboard_statistics/");
                return response;
            } catch (err) {
                console.error("Layout - 통계 데이터 조회 오류:", err);
                return {
                    waiting_count: 0,
                };
            }
        },
        refetchInterval: 30000,
    });

    const isDashboard = location.pathname === '/medical_staff' || location.pathname === '/admin_staff';
    const dashboardHref = user?.role === 'medical_staff' ? '/medical_staff' : '/admin_staff';

    return (
        <div className="flex-1 flex flex-col min-h-screen bg-gray-50">
            {/* Global Header */}
            <header className="bg-white/80 backdrop-blur-md sticky top-0 z-30 border-b border-gray-100">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center h-16">
                        <div className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
                            <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">System Online</span>
                        </div>

                        <div className="flex items-center gap-4">
                            <div className="hidden md:flex relative group">
                                <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
                                    <Activity className="h-4 w-4 text-gray-400" />
                                </div>
                                <input
                                    type="text"
                                    placeholder="Quick patient search..."
                                    className="bg-gray-100 border-none rounded-full py-2 pl-10 pr-4 text-xs w-64 focus:ring-2 focus:ring-blue-500/20 transition-all"
                                />
                            </div>

                            <div className="flex items-center gap-2 border-l pl-4 border-gray-100">
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="rounded-full text-gray-500 hover:text-blue-600 hover:bg-blue-50 transition-colors"
                                    onClick={() => navigate(dashboardHref)}
                                    title="Home"
                                >
                                    <Home className="h-5 w-5" />
                                </Button>
                                <NotificationDropdown />

                                <div className="ml-2">
                                    <UserProfileDropdown />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
                {/* Persistent Hero Banner Section - Only on Dashboard */}
                {isDashboard && (
                    <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-[#1e3a8a] via-[#1e40af] to-[#1d4ed8] p-6 md:p-8 mb-6 shadow-xl shadow-blue-900/10 transition-all duration-500">
                        <div className="absolute inset-0 opacity-10" style={{ backgroundImage: 'radial-gradient(circle, white 1px, transparent 1px)', backgroundSize: '24px 24px' }}></div>
                        <div className="absolute -right-20 -top-20 w-64 h-64 bg-white/5 rounded-full blur-3xl"></div>

                        <div className="relative z-10 flex flex-col md:flex-row items-center gap-6">
                            <div className="shrink-0">
                                <div className="relative w-28 h-28 md:w-32 md:h-32 rounded-2xl overflow-hidden border-4 border-white/20 shadow-2xl group transition-transform hover:scale-[1.02]">
                                    <img src={doctorProfile} alt="Doctor Profile" className="w-full h-full object-cover" />
                                    <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                                </div>
                            </div>

                            <div className="flex-1 text-center md:text-left text-white">
                                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-400/20 border border-blue-400/30 text-[10px] font-medium uppercase tracking-wider mb-3 animate-fade-in">
                                    <Activity className="w-3 h-3" />
                                    의료 통계 활성화
                                </div>

                                <h1 className="text-2xl md:text-3xl font-bold mb-2 tracking-tight leading-tight">
                                    안녕하세요,<br />
                                    <span className="text-blue-200">
                                        {user ? `${user.last_name || ''} ${user.first_name || user.username}`.trim() : "Guest"}
                                    </span> 선생님
                                </h1>

                                <p className="text-blue-100/80 mb-4 font-medium">
                                    오늘 <span className="text-white border-b border-white font-bold">{dashboardStats?.waiting_count || 0}명의 환자</span>가 진료 예정입니다.
                                </p>

                                <div className="flex flex-wrap items-center justify-center md:justify-start gap-3">
                                    <Button
                                        className="bg-white text-blue-900 hover:bg-blue-50 font-bold px-5 py-2.5 rounded-xl shadow-lg shadow-black/10 transition-all hover:scale-[1.05] active:scale-95 h-11 text-sm flex items-center gap-2"
                                        onClick={() => navigate('/medical-registration')}
                                    >
                                        <Plus className="w-4 h-4" /> 신규 진료
                                    </Button>
                                    <Button
                                        variant="ghost"
                                        className="bg-blue-400/10 hover:bg-blue-400/20 text-white border border-white/20 font-bold px-5 py-2.5 rounded-xl transition-all hover:bg-blue-500/20 h-11 text-sm flex items-center gap-2"
                                        onClick={() => navigate('/reservation-info')}
                                    >
                                        <Calendar className="w-4 h-4" /> 일정 보기
                                    </Button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Page Content */}
                <div className="animate-in fade-in duration-500">
                    {children}
                </div>
            </main>
        </div>
    );
}
