import { NavLink, useNavigate, useLocation } from 'react-router-dom';
import { Activity, Users, FileImage, BarChart3, Stethoscope, TrendingUp, ClipboardList } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/context/AuthContext';

const baseNavigation = [
  { name: "환자 관리", href: "/patients", icon: Users },
  { name: "진료 접수", href: "/medical-registration", icon: ClipboardList },
  { name: "의료 이미지", href: "/images", icon: FileImage },
  { name: "폐암 예측", href: "/lung-cancer", icon: Stethoscope },
  { name: "폐암 통계", href: "/lung-cancer-stats", icon: TrendingUp },
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
            const navigation = [
              { name: '대시보드', href: dashboardHref, icon: BarChart3 },
              ...baseNavigation,
            ];
            return navigation.map((item) => {
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
        {user ? (
          <>
            <div className="text-sm text-gray-700">
              {user.role === 'medical_staff' ? '의료진' : user.role === 'admin_staff' ? '원무과' : '관리자'} {' '}
              {user.last_name || ''}{user.first_name ? ' ' + user.first_name : ''}
            </div>
            <Button variant="outline" className="w-full" onClick={handleLogout}>
              로그아웃
            </Button>
          </>
        ) : location.pathname !== '/' ? (
          <Button className="w-full" onClick={() => navigate('/login')}>
            로그인
          </Button>
        ) : null}
      </div>
    </div>
  );
}
