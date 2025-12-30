import { NavLink, useNavigate, useLocation } from 'react-router-dom';
import { Activity, Users, BarChart3, Stethoscope, TrendingUp, ClipboardList, BookOpen, CalendarDays, LogOut, Scan } from 'lucide-react';
import { useAuth } from '@/context/AuthContext';
import doctorProfile from "@/assets/doctor-profile.png";

const departmentNavigation = {
  호흡기내과: [
    { name: "폐암 예측", href: "/lung-cancer", icon: Stethoscope },
    { name: "폐암 통계", href: "/lung-cancer-stats", icon: TrendingUp },
    { name: "지식 허브", href: "/knowledge-hub", icon: BookOpen },
  ],
  방사선과: [
    { name: "영상 업로드", href: "/mri-viewer", icon: Scan },
  ],
  영상의학과: [
    { name: "영상 판독", href: "/mri-viewer", icon: Scan },
  ],
  외과: [
    { name: "영상 판독", href: "/mri-viewer", icon: Scan },
    { name: "지식 허브", href: "/knowledge-hub", icon: BookOpen },
  ],
};

const adminNavigation = [
  { name: "환자 정보", href: "/patients", icon: Users },
  { name: "진료 접수", href: "/medical-registration", icon: ClipboardList },
  { name: "예약 정보", href: "/reservation-info", icon: CalendarDays },
];

export default function Sidebar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/', { replace: true });
    } catch {
      navigate('/', { replace: true });
    }
  };

  const dashboardHref = user
    ? (user.role === 'medical_staff' ? '/medical_staff' : '/admin_staff')
    : '/';

  // 역할 및 진료과에 따라 다른 메뉴 구성
  let departmentMenuItems: Array<{ name: string; href: string; icon: any }> = [];

  if (user) {
    if (user.role === 'admin_staff') {
      departmentMenuItems = adminNavigation;
    } else if (user.role === 'medical_staff') {
      // 진료과별 메뉴 가져오기
      const deptMenu = departmentNavigation[user.department as keyof typeof departmentNavigation];
      departmentMenuItems = deptMenu || departmentNavigation['외과']; // 기본값: 외과 메뉴
    }
  } else {
    // 로그인하지 않은 경우 외과 메뉴 표시 (fallback)
    departmentMenuItems = departmentNavigation['외과'];
  }

  const menuItems = [
    { name: '대시보드', href: dashboardHref, icon: BarChart3 },
    ...departmentMenuItems
  ];

  return (
    <div className="bg-white w-64 h-screen border-r border-gray-100 fixed left-0 top-0 z-40 flex flex-col shadow-sm">
      {/* Brand Logo */}
      <div className="px-8 py-8 flex flex-col gap-1">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-200">
            <Activity className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-black text-gray-900 leading-none tracking-tight">CDSSentials</h1>
            <p className="text-[10px] font-bold text-blue-600 uppercase tracking-widest mt-0.5">의료 통합 플랫폼</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 px-4 py-2 overflow-y-auto">
        <div className="mb-4 px-4">
          <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-4">메인 메뉴</p>
          <div className="space-y-1">
            {menuItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.href;
              return (
                <NavLink
                  key={item.href}
                  to={item.href}
                  className={({ isActive }) =>
                    `group flex items-center px-4 py-3 text-sm font-semibold rounded-2xl transition-all duration-200 ${isActive
                      ? "bg-blue-600 text-white shadow-lg shadow-blue-200"
                      : "text-gray-500 hover:bg-gray-50 hover:text-gray-900"
                    }`
                  }
                >
                  <Icon
                    className={`mr-3 h-5 w-5 transition-colors ${isActive ? "text-white" : "text-gray-400 group-hover:text-gray-600"
                      }`}
                  />
                  {item.name}
                  {isActive && (
                    <div className="ml-auto w-1.5 h-1.5 rounded-full bg-white opacity-80" />
                  )}
                </NavLink>
              );
            })}
          </div>
        </div>
      </nav>

      {/* User Session Footer */}
      <div className="p-4 border-t border-gray-50">
        <div className="bg-gray-50/50 rounded-3xl p-4 flex flex-col gap-4 border border-gray-100/50">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-full bg-blue-100 border-2 border-white overflow-hidden">
                <img src={doctorProfile} alt="User" className="w-full h-full object-cover" />
              </div>
              <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-green-500 border-2 border-white rounded-full"></div>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-bold text-gray-900 truncate">
                {user ? `${user.last_name || ''} ${user.first_name || user.username}`.trim() : "게스트 사용자"}
              </p>
              <div className="flex flex-col">
                <p className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                  {user?.role === 'medical_staff' ? '원격 판독 의료진' : user?.role === 'admin_staff' ? '원무 관리자' : '게스트'}
                </p>
                <button
                  onClick={handleLogout}
                  className="flex items-center gap-1 text-xs font-bold text-red-500 hover:text-red-600 transition-colors mt-1"
                >
                  <LogOut className="w-3 h-3" />
                  로그아웃
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
