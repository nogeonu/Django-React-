import { Navigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Loader2 } from 'lucide-react';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: 'medical_staff' | 'admin_staff' | 'superuser';
  allowedRoles?: ('medical_staff' | 'admin_staff' | 'superuser')[];
}

export default function ProtectedRoute({ children, requiredRole, allowedRoles }: ProtectedRouteProps) {
  const { user, loading } = useAuth();

  // 로딩 중
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  // 로그인하지 않은 경우
  if (!user) {
    return <Navigate to="/login" replace />;
  }

  // 특정 역할만 허용하는 경우
  if (allowedRoles && !allowedRoles.includes(user.role)) {
    // 권한이 없는 경우 대시보드로 리다이렉트
    const dashboardPath = user.role === 'medical_staff' ? '/medical_staff' : '/admin_staff';
    return <Navigate to={dashboardPath} replace />;
  }

  // 특정 역할 필수인 경우
  if (requiredRole && user.role !== requiredRole) {
    const dashboardPath = user.role === 'medical_staff' ? '/medical_staff' : '/admin_staff';
    return <Navigate to={dashboardPath} replace />;
  }

  return <>{children}</>;
}

