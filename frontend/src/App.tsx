import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import Dashboard from '@/pages/Dashboard';
import Patients from '@/pages/Patients';
import MedicalImages from '@/pages/MedicalImages';
import LungCancerPrediction from '@/pages/LungCancerPrediction';
import LungCancerStats from '@/pages/LungCancerStats';
import MedicalRegistration from '@/pages/MedicalRegistration';
import KnowledgeHub from '@/pages/KnowledgeHub';
import NotFound from '@/pages/NotFound';
import Sidebar from '@/components/Sidebar';
import Login from '@/pages/Login';
import Signup from '@/pages/Signup';
import ProtectedRoute from '@/components/ProtectedRoute';
import { AuthProvider } from '@/context/AuthContext';

const queryClient = new QueryClient();

function AppContent() {
  const location = useLocation();
  const isPublicPage = ['/', '/login', '/signup'].includes(location.pathname);
  
  return (
    <div className="flex min-h-screen bg-gray-50">
      {!isPublicPage && <Sidebar />}
      <div className={isPublicPage ? "flex-1" : "flex-1 ml-64"}>
        <Routes>
          {/* 공개 라우트 */}
          <Route path="/" element={<Login />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          
          {/* 로그인 필수 라우트 */}
          <Route path="/medical_staff" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          <Route path="/admin_staff" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          
          {/* 공통 접근 가능 페이지 */}
          <Route path="/patients" element={<ProtectedRoute><Patients /></ProtectedRoute>} />
          <Route path="/medical-registration" element={<ProtectedRoute><MedicalRegistration /></ProtectedRoute>} />
          
          {/* 의료진만 접근 가능한 페이지 */}
          <Route path="/images" element={<ProtectedRoute allowedRoles={['medical_staff', 'superuser']}><MedicalImages /></ProtectedRoute>} />
          <Route path="/lung-cancer" element={<ProtectedRoute allowedRoles={['medical_staff', 'superuser']}><LungCancerPrediction /></ProtectedRoute>} />
          <Route path="/lung-cancer-stats" element={<ProtectedRoute allowedRoles={['medical_staff', 'superuser']}><LungCancerStats /></ProtectedRoute>} />
          <Route path="/knowledge-hub" element={<ProtectedRoute allowedRoles={['medical_staff', 'superuser']}><KnowledgeHub /></ProtectedRoute>} />
          
          <Route path="*" element={<NotFound />} />
        </Routes>
      </div>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <AuthProvider>
          <Router>
            <AppContent />
            <Toaster />
          </Router>
        </AuthProvider>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
