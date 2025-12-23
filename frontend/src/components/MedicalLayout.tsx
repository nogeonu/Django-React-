import React from "react";
import { useNavigate } from "react-router-dom";
import { Activity, Home } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/context/AuthContext";
import NotificationDropdown from "@/components/NotificationDropdown";
import UserProfileDropdown from "@/components/UserProfileDropdown";


interface MedicalLayoutProps {
    children: React.ReactNode;
}

export default function MedicalLayout({ children }: MedicalLayoutProps) {
    const { user } = useAuth();
    const navigate = useNavigate();

    const dashboardHref = user?.role === 'medical_staff' ? '/medical_staff' : '/admin_staff';

    return (
        <div className="flex-1 flex flex-col min-h-screen bg-gray-50">
            {/* Global Header */}
            <header className="bg-white/80 backdrop-blur-md sticky top-0 z-30 border-b border-gray-100">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center h-16">
                        <div className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
                            <span className="text-[10px] font-bold text-gray-400 uppercase tracking-widest">시스템 정상 (System Online)</span>
                        </div>

                        <div className="flex items-center gap-4">
                            <div className="hidden md:flex relative group">
                                <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
                                    <Activity className="h-4 w-4 text-gray-400" />
                                </div>
                                <input
                                    type="text"
                                    placeholder="빠른 환자 검색..."
                                    className="bg-gray-100 border-none rounded-full py-2 pl-10 pr-4 text-xs w-64 focus:ring-2 focus:ring-blue-500/20 transition-all"
                                />
                            </div>

                            <div className="flex items-center gap-2 border-l pl-4 border-gray-100">
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="rounded-full text-gray-500 hover:text-blue-600 hover:bg-blue-50 transition-colors"
                                    onClick={() => navigate(dashboardHref)}
                                    title="홈 (Home)"
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
                {/* Page Content */}
                <div className="animate-in fade-in duration-500">
                    {children}
                </div>
            </main>
        </div>
    );
}
