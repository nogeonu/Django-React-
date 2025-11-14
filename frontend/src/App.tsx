import {
  BrowserRouter as Router,
  Routes,
  Route,
  useLocation,
} from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import Dashboard from "@/pages/Dashboard";
import Patients from "@/pages/Patients";
import MedicalImages from "@/pages/MedicalImages";
import LungCancerPrediction from "@/pages/LungCancerPrediction";
import LungCancerStats from "@/pages/LungCancerStats";
import MedicalRegistration from "@/pages/MedicalRegistration";
import KnowledgeHub from "@/pages/KnowledgeHub";
import ReservationInfo from "@/pages/ReservationInfo";
import NotFound from "@/pages/NotFound";
import Sidebar from "@/components/Sidebar";
import Login from "@/pages/Login";
import Signup from "@/pages/Signup";
import PatientLogin from "@/pages/PatientLogin";
import PatientSignup from "@/pages/PatientSignup";
import ProtectedRoute from "@/components/ProtectedRoute";
import { AuthProvider } from "@/context/AuthContext";
import { CalendarProvider } from "@/context/CalendarContext";
import Home from "@/pages/Home";
import PatientMyPage from "@/pages/PatientMyPage";
import PatientMedicalRecords from "@/pages/PatientMedicalRecords";
import PatientDoctors from "@/pages/PatientDoctors";
import AppDownload from "@/pages/AppDownload";

const queryClient = new QueryClient();

function AppContent() {
  const location = useLocation();
  const isPublicPage = [
    "/",
    "/login",
    "/signup",
    "/patient/login",
    "/patient/signup",
    "/patient/mypage",
    "/patient/records",
    "/patient/doctors",
    "/app-download",
  ].includes(location.pathname);

  return (
    <div className="flex min-h-screen bg-gray-50">
      {!isPublicPage && <Sidebar />}
      <div className={isPublicPage ? "flex-1" : "flex-1 ml-64"}>
        <Routes>
          {/* 공개 라우트 */}
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />
          <Route path="/patient/login" element={<PatientLogin />} />
          <Route path="/patient/signup" element={<PatientSignup />} />
          <Route path="/patient/mypage" element={<PatientMyPage />} />
          <Route path="/patient/records" element={<PatientMedicalRecords />} />
          <Route path="/patient/doctors" element={<PatientDoctors />} />
          <Route path="/app-download" element={<AppDownload />} />

          {/* 로그인 필수 라우트 */}
          <Route
            path="/medical_staff"
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin_staff"
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            }
          />

          {/* 공통 접근 가능 페이지 */}
          <Route
            path="/patients"
            element={
              <ProtectedRoute>
                <Patients />
              </ProtectedRoute>
            }
          />
          <Route
            path="/medical-registration"
            element={
              <ProtectedRoute>
                <MedicalRegistration />
              </ProtectedRoute>
            }
          />
          <Route
            path="/reservation-info"
            element={
              <ProtectedRoute allowedRoles={["medical_staff", "admin_staff", "superuser"]}>
                <ReservationInfo />
              </ProtectedRoute>
            }
          />

          {/* 의료진만 접근 가능한 페이지 */}
          <Route
            path="/images"
            element={
              <ProtectedRoute allowedRoles={["medical_staff", "superuser"]}>
                <MedicalImages />
              </ProtectedRoute>
            }
          />
          <Route
            path="/lung-cancer"
            element={
              <ProtectedRoute allowedRoles={["medical_staff", "superuser"]}>
                <LungCancerPrediction />
              </ProtectedRoute>
            }
          />
          <Route
            path="/lung-cancer-stats"
            element={
              <ProtectedRoute allowedRoles={["medical_staff", "superuser"]}>
                <LungCancerStats />
              </ProtectedRoute>
            }
          />
          <Route
            path="/knowledge-hub"
            element={
              <ProtectedRoute allowedRoles={["medical_staff", "superuser"]}>
                <KnowledgeHub />
              </ProtectedRoute>
            }
          />

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
          <CalendarProvider>
            <Router>
              <AppContent />
              <Toaster />
            </Router>
          </CalendarProvider>
        </AuthProvider>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
