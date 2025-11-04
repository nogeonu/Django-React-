import { NavLink, useNavigate, useLocation } from 'react-router-dom';
import { Activity, Users, FileImage, BarChart3, Stethoscope, TrendingUp, ClipboardList, BookOpen, Calendar, User as UserIcon, LogOut } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/context/AuthContext';

const baseNavigation = {
  medical_staff: [
    { name: "환자 정보", href: "/patients", icon: Users },
    { name: "진료 접수", href: "/medical-registration", icon: ClipboardList },
    { name: "일정관리", href: "/schedule", icon: Calendar },
    { name: "의료 이미지", href: "/images", icon: FileImage },
    { name: "폐암 예측", href: "/lung-cancer", icon: Stethoscope },
    { name: "폐암 통계", href: "/lung-cancer-stats", icon: TrendingUp },
    { name: "지식 허브", href: "/knowledge-hub", icon: BookOpen },
  ],
  admin_staff: [
    { name: "환자 정보", href: "/patients", icon: Users },
    { name: "진료 접수", href: "/medical-registration", icon: ClipboardList },
    { name: "일정관리", href: "/schedule", icon: Calendar },
  ],
};

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

  return (
    <div className="bg-white w-64 min-h-screen border-r border-gray-200 fixed left-0 top-0 z-40 flex flex-col">
      <div className="flex items-center px-6 py-4 border-b">
        <Activity className="text-blue-600 text-2xl mr-3" />
        <h1 className="text-lg font-bold text-gray-900">병원 관리 시스템</h1>
      </div>
      
      <nav className="mt-6 flex-1">
        <div className="px-3">
          {(() => {
            const dashboardHref = user
              ? (user.role === 'medical_staff' ? '/medical_staff' : '/admin_staff')
              : '/';
            
            // 역할에 따라 다른 메뉴 표시
            const navigationItems = user 
              ? [
                  { name: '대시보드', href: dashboardHref, icon: BarChart3 },
                  ...(baseNavigation[user.role as keyof typeof baseNavigation] || baseNavigation.admin_staff)
                ]
              : [
                  { name: '대시보드', href: dashboardHref, icon: BarChart3 },
                  ...baseNavigation.medical_staff
                ];
            
            return navigationItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.name}
                  to={item.href}
                  className={({ isActive }) =>
                    `group flex items-center px-3 py-2 text-sm font-medium rounded-md mb-1 transition-colors ${
                      isActive
                        ? "bg-blue-100 text-blue-700"
                        : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                      }`
                  }
                  data-testid={`nav-${item.href.slice(1) || 'dashboard'}`}
                >
                  <Icon
                    className={`mr-3 h-5 w-5 ${
                      false ? "text-blue-500" : "text-gray-400 group-hover:text-gray-500"
                    }`}
                  />
                  {item.name}
                </NavLink>
              );
            });
          })()}
        </div>
      </nav>
      <div className="p-3 border-t space-y-2">
        {['/', '/login', '/signup'].includes(location.pathname) ? null : user ? (
          <div className="space-y-3">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center">
                <UserIcon className="w-6 h-6 text-gray-400" />
              </div>
              <div className="min-w-0">
                <div className="text-sm font-medium text-gray-900 truncate">
                  {(user.last_name || '') + (user.first_name ? ' ' + user.first_name : '')}
                </div>
                <div className="text-xs text-gray-500 truncate">
                  {user.role === 'medical_staff' ? '의료진' : user.role === 'admin_staff' ? '원무과' : user.role}
                </div>
              </div>
            </div>
            <Button
              variant="outline"
              className="w-full justify-center rounded-xl bg-gray-50 hover:bg-gray-100 border-0"
              onClick={handleLogout}
            >
              <LogOut className="w-4 h-4 mr-2" />
              로그아웃
            </Button>
          </div>
        ) : (() => {
          const hideOnPaths = [
            '/',
            '/login',
            '/signup',
            '/patients',
            '/medical-registration',
            '/images',
            '/lung-cancer',
            '/lung-cancer-stats',
          ];
          if (hideOnPaths.includes(location.pathname)) return null;
          return (
            <Button className="w-full" onClick={() => navigate('/login')}>
              로그인
            </Button>
          );
        })()}
      </div>
    </div>
  );
}
