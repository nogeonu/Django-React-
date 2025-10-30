import { NavLink } from 'react-router-dom';
import { Activity, Users, FileImage, BarChart3, Stethoscope, TrendingUp, ClipboardList } from 'lucide-react';

const navigation = [
  { name: "대시보드", href: "/", icon: BarChart3 },
  { name: "환자 관리", href: "/patients", icon: Users },
  { name: "진료 접수", href: "/medical-registration", icon: ClipboardList },
  { name: "의료 이미지", href: "/images", icon: FileImage },
  { name: "폐암 예측", href: "/lung-cancer", icon: Stethoscope },
  { name: "폐암 통계", href: "/lung-cancer-stats", icon: TrendingUp },
];

export default function Sidebar() {
  return (
    <div className="bg-white w-64 min-h-screen border-r border-gray-200 fixed left-0 top-0 z-40">
      <div className="flex items-center px-6 py-4 border-b">
        <Activity className="text-blue-600 text-2xl mr-3" />
        <h1 className="text-lg font-bold text-gray-900">병원 관리 시스템</h1>
      </div>
      
      <nav className="mt-6">
        <div className="px-3">
          {navigation.map((item) => {
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
          })}
        </div>
      </nav>
    </div>
  );
}
