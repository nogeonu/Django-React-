import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import { Activity, Users, FileImage, BarChart3, Stethoscope, TrendingUp } from 'lucide-react';
import Dashboard from '@/pages/Dashboard';
import Patients from '@/pages/Patients';
import MedicalImages from '@/pages/MedicalImages';
import LungCancerPrediction from '@/pages/LungCancerPrediction';
import LungCancerStats from '@/pages/LungCancerStats';
import MedicalRegistration from '@/pages/MedicalRegistration';
import NotFound from '@/pages/NotFound';
import Sidebar from '@/components/Sidebar';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Router>
          <div className="flex min-h-screen bg-gray-50">
            <Sidebar />
            <div className="flex-1 ml-64">
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/patients" element={<Patients />} />
                <Route path="/images" element={<MedicalImages />} />
                <Route path="/lung-cancer" element={<LungCancerPrediction />} />
                <Route path="/lung-cancer-stats" element={<LungCancerStats />} />
                <Route path="/medical-registration" element={<MedicalRegistration />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </div>
          </div>
          <Toaster />
        </Router>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
