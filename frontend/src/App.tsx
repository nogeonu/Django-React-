import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
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
import { AuthProvider } from '@/context/AuthContext';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <AuthProvider>
          <Router>
            <div className="flex min-h-screen bg-gray-50">
              <Sidebar />
              <div className="flex-1 ml-64">
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/medical_staff" element={<Dashboard />} />
                  <Route path="/admin_staff" element={<Dashboard />} />
                  <Route path="/login" element={<Login />} />
                  <Route path="/signup" element={<Signup />} />
                  <Route path="/patients" element={<Patients />} />
                  <Route path="/images" element={<MedicalImages />} />
                  <Route path="/lung-cancer" element={<LungCancerPrediction />} />
                  <Route path="/lung-cancer-stats" element={<LungCancerStats />} />
                  <Route path="/medical-registration" element={<MedicalRegistration />} />
                  <Route path="/knowledge-hub" element={<KnowledgeHub />} />
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </div>
            </div>
            <Toaster />
          </Router>
        </AuthProvider>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
