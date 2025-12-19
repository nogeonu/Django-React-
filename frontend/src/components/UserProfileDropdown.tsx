import { Settings, LogOut, UserCircle, Shield } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useAuth } from '@/context/AuthContext';
import { useNavigate } from 'react-router-dom';

export default function UserProfileDropdown() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await logout();
      navigate('/login');
    } catch (error) {
      console.error('로그아웃 오류:', error);
    }
  };

  const getUserDisplayName = () => {
    if (!user) return '사용자';
    if (user.first_name || user.last_name) {
      return `${user.last_name || ''}${user.first_name || ''}`.trim() || user.username;
    }
    return user.username;
  };

  const getRoleLabel = (role: string) => {
    switch (role) {
      case 'medical_staff':
        return '의료진';
      case 'admin_staff':
        return '원무과';
      case 'superuser':
        return '관리자';
      default:
        return role;
    }
  };

  const getUserInitial = () => {
    if (!user) return 'U';
    if (user.first_name) {
      return user.first_name.charAt(0).toUpperCase();
    }
    if (user.last_name) {
      return user.last_name.charAt(0).toUpperCase();
    }
    if (user.username) {
      return user.username.charAt(0).toUpperCase();
    }
    return 'U';
  };

  if (!user) {
    return null;
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button 
          variant="ghost" 
          size="icon"
          className="relative rounded-full hover:bg-gray-100"
        >
          <div className="w-10 h-10 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0 border-2 border-white shadow-sm">
            <span className="text-white font-bold text-sm">{getUserInitial()}</span>
          </div>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel>
          <div className="flex flex-col space-y-1">
            <p className="text-sm font-medium leading-none">{getUserDisplayName()}</p>
            <p className="text-xs leading-none text-gray-500">
              {getRoleLabel(user.role)}
            </p>
            {user.department && (
              <p className="text-xs leading-none text-gray-500">
                {user.department}
              </p>
            )}
          </div>
        </DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={() => navigate('/profile')}>
          <UserCircle className="mr-2 h-4 w-4" />
          <span>마이페이지</span>
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => navigate('/settings')}>
          <Settings className="mr-2 h-4 w-4" />
          <span>설정</span>
        </DropdownMenuItem>
        {user.role === 'superuser' && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => navigate('/admin')}>
              <Shield className="mr-2 h-4 w-4" />
              <span>관리자 페이지</span>
            </DropdownMenuItem>
          </>
        )}
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={handleLogout} className="text-red-600">
          <LogOut className="mr-2 h-4 w-4" />
          <span>로그아웃</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

