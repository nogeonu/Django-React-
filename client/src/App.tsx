import { Switch, Route, Link, useLocation } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Activity, Users, FileImage, BarChart3 } from "lucide-react";
import Dashboard from "@/pages/dashboard";
import Patients from "@/pages/patients";
import MedicalImages from "@/pages/medical-images";
import NotFound from "@/pages/not-found";

function Sidebar() {
  const [location] = useLocation();
  
  const navigation = [
    { name: "대시보드", href: "/", icon: BarChart3 },
    { name: "환자 관리", href: "/patients", icon: Users },
    { name: "의료 이미지", href: "/images", icon: FileImage },
  ];

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
            const isActive = location === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`group flex items-center px-3 py-2 text-sm font-medium rounded-md mb-1 transition-colors ${
                  isActive
                    ? "bg-blue-100 text-blue-700"
                    : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                }`}
                data-testid={`nav-${item.href.slice(1) || 'dashboard'}`}
              >
                <Icon
                  className={`mr-3 h-5 w-5 ${
                    isActive ? "text-blue-500" : "text-gray-400 group-hover:text-gray-500"
                  }`}
                />
                {item.name}
              </Link>
            );
          })}
        </div>
      </nav>
    </div>
  );
}

function Router() {
  return (
    <div className="flex">
      <Sidebar />
      <div className="flex-1 ml-64">
        <Switch>
          <Route path="/" component={Dashboard} />
          <Route path="/patients" component={Patients} />
          <Route path="/images" component={MedicalImages} />
          <Route component={NotFound} />
        </Switch>
      </div>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Router />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
