import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { 
  Users, 
  Calendar, 
  FileImage, 
  Activity, 
  Plus,
  Search,
  Filter,
  CheckCircle
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { apiRequest } from "@/lib/api";
import PatientRegistrationModal from "@/components/PatientRegistrationModal";

interface MedicalRecord {
  id: number;
  patient_id: string;
  name: string;
  department: string;
  status: string;
  notes: string;
  reception_start_time: string;
  treatment_end_time?: string;
  is_treatment_completed: boolean;
}

export default function Dashboard() {
  const [searchTerm, setSearchTerm] = useState("");
  const [isPatientModalOpen, setIsPatientModalOpen] = useState(false);
  const queryClient = useQueryClient();

  const { data: waitingPatients = [], isLoading, error } = useQuery({
    queryKey: ["waiting-patients"],
    queryFn: async () => {
      try {
        console.log("대시보드 - 대기 중인 환자 데이터 조회 시작...");
        const response = await apiRequest("GET", "/api/lung_cancer/api/medical-records/waiting_patients/");
        console.log("대시보드 - API 응답:", response);
        const result = response || [];
        console.log("대시보드 - 대기 중인 환자 수:", result.length);
        return result;
      } catch (err) {
        console.error("대시보드 - 대기 중인 환자 데이터 조회 오류:", err);
        throw err;
      }
    },
    refetchInterval: 30000, // 30초마다 자동 새로고침
  });

  const recentPatients = (waitingPatients as MedicalRecord[])
    .sort((a, b) => new Date(a.reception_start_time).getTime() - new Date(b.reception_start_time).getTime())
    .slice(0, 5);
  const totalPatients = (waitingPatients as MedicalRecord[]).length;
  const todayExams = 3; // 임시 데이터
  const pendingAnalysis = totalPatients; // 대기 중인 환자 수

  const stats = [
    {
      title: "총 환자 수",
      value: totalPatients,
      icon: Users,
      color: "text-blue-600",
      bgColor: "bg-blue-100"
    },
    {
      title: "오늘 예약 검사",
      value: todayExams,
      icon: Calendar,
      color: "text-red-600",
      bgColor: "bg-red-100"
    },
    {
      title: "진료 완료 환자",
      value: 0,
      icon: CheckCircle,
      color: "text-green-600",
      bgColor: "bg-green-100"
    },
    {
      title: "진료 대기 중 환자",
      value: pendingAnalysis,
      icon: Activity,
      color: "text-orange-600",
      bgColor: "bg-orange-100"
    }
  ];

  // 오류 처리
  if (error) {
    console.error("대시보드 로딩 오류:", error);
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-red-600 mb-4">오류가 발생했습니다</h2>
          <p className="text-gray-600 mb-4">대시보드를 불러오는 중 오류가 발생했습니다.</p>
          <Button onClick={() => window.location.reload()}>새로고침</Button>
        </div>
      </div>
    );
  }

  console.log("대시보드 렌더링 중 - waitingPatients:", waitingPatients, "isLoading:", isLoading);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Activity className="text-blue-600 text-2xl mr-3" />
              <h1 className="text-xl font-bold text-gray-900">병원 환자관리 시스템</h1>
            </div>
            <div className="flex items-center space-x-4">
              <Button 
                size="sm" 
                data-testid="button-add-patient"
                onClick={() => setIsPatientModalOpen(true)}
              >
                <Plus className="w-4 h-4 mr-2" />
                환자 등록
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {stats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <Card key={index} className="hover:shadow-md transition-shadow">
                <CardContent className="flex items-center p-6">
                  <div className={`${stat.bgColor} p-3 rounded-lg mr-4`}>
                    <Icon className={`w-6 h-6 ${stat.color}`} />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                    <p className="text-2xl font-bold text-gray-900" data-testid={`stat-${index}`}>
                      {stat.value}
                    </p>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Recent Patients */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                대기 중인 환자(순번 기준)
                <Button variant="outline" size="sm" data-testid="button-view-all-patients">
                  전체 보기
                </Button>
              </CardTitle>
              <CardDescription>
                현재 대기 중인 환자 목록입니다
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="space-y-3">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="animate-pulse">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-gray-200 rounded-full"></div>
                        <div className="flex-1">
                          <div className="h-4 bg-gray-200 rounded w-3/4 mb-1"></div>
                          <div className="h-3 bg-gray-200 rounded w-1/2"></div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : recentPatients.length > 0 ? (
                <div className="space-y-3">
                  {recentPatients.map((record: MedicalRecord, index: number) => (
                    <div 
                      key={record.id} 
                      className="flex items-center space-x-3 p-3 hover:bg-gray-50 rounded-lg cursor-pointer"
                      data-testid={`patient-item-${record.id}`}
                    >
                      <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                        <span className="text-blue-600 font-semibold">
                          {index + 1}
                        </span>
                      </div>
                      <div className="flex-1">
                        <p className="font-medium text-gray-900" data-testid={`text-patient-name-${record.id}`}>
                          {record.name}
                        </p>
                        <p className="text-sm text-gray-500" data-testid={`text-patient-number-${record.id}`}>
                          {record.patient_id} | {record.department}
                        </p>
                        <p className="text-xs text-gray-400">
                          {record.notes}
                        </p>
                      </div>
                      <div className="text-sm text-gray-400">
                        {new Date(record.reception_start_time).toLocaleTimeString('ko-KR', {
                          hour: '2-digit',
                          minute: '2-digit'
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-6 text-gray-500">
                  대기 중인 환자가 없습니다
                </div>
              )}
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card>
            <CardHeader>
              <CardTitle>빠른 작업</CardTitle>
              <CardDescription>
                자주 사용하는 기능들을 빠르게 실행할 수 있습니다
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button 
                className="w-full justify-start" 
                variant="outline"
                data-testid="button-register-patient"
                onClick={() => setIsPatientModalOpen(true)}
              >
                <Plus className="w-4 h-4 mr-2" />
                새 환자 등록
              </Button>
              
              <Button 
                className="w-full justify-start" 
                variant="outline"
                data-testid="button-upload-image"
              >
                <FileImage className="w-4 h-4 mr-2" />
                의료 이미지 업로드
              </Button>
              
              <Button 
                className="w-full justify-start" 
                variant="outline"
                data-testid="button-new-examination"
              >
                <Calendar className="w-4 h-4 mr-2" />
                예약 검사 등록
              </Button>
              
              <Button 
                className="w-full justify-start" 
                variant="outline"
                data-testid="button-ai-analysis"
                onClick={() => window.location.href = '/medical-registration'}
              >
                <Activity className="w-4 h-4 mr-2" />
                진료 접수 
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* Search Section */}
        <Card className="mt-8">
          <CardHeader>
            <CardTitle>환자 검색</CardTitle>
            <CardDescription>
              환자 이름이나 번호로 검색하세요
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex space-x-4">
              <div className="flex-1">
                <Input
                  placeholder="환자 이름 또는 번호를 입력하세요..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  data-testid="input-patient-search"
                />
              </div>
              <Button data-testid="button-search">
                <Search className="w-4 h-4 mr-2" />
                검색
              </Button>
              <Button variant="outline" data-testid="button-filter">
                <Filter className="w-4 h-4 mr-2" />
                필터
              </Button>
            </div>
          </CardContent>
        </Card>
      </main>

      {/* Patient Registration Modal */}
      <PatientRegistrationModal
        isOpen={isPatientModalOpen}
        onClose={() => setIsPatientModalOpen(false)}
        onSuccess={() => {
          setIsPatientModalOpen(false);
          // 환자 목록 새로고침
          queryClient.invalidateQueries({ queryKey: ["patients"] });
        }}
      />
    </div>
  );
}
