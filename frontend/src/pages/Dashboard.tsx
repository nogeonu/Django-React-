import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { 
  Users, 
  Calendar, 
  FileImage, 
  Activity, 
  UserPlus,
  Search,
  Filter,
  CheckCircle
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/api";
 
import { useNavigate, useLocation } from "react-router-dom";

interface Patient {
  id: string;
  name: string;
  birth_date: string;
  gender: string;
  phone?: string;
  address?: string;
  emergency_contact?: string;
  blood_type?: string;
  age: number;
  created_at: string;
  updated_at: string;
}

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
  const [showAllWaiting, setShowAllWaiting] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<MedicalRecord | null>(null);
  const [isCompleteDialogOpen, setIsCompleteDialogOpen] = useState(false);
  const [examinationResult, setExaminationResult] = useState("");
  const [treatmentNote, setTreatmentNote] = useState("");
  const [isCompleting, setIsCompleting] = useState(false);
  
  const navigate = useNavigate();
  const location = useLocation();
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: waitingPatients = [], isLoading } = useQuery({
    queryKey: ["waiting-patients"],
    queryFn: async () => {
      try {
        console.log("대시보드 - 대기 중인 환자 데이터 조회 시작...");
        const response = await apiRequest("GET", "/api/lung_cancer/medical-records/waiting_patients/");
        console.log("대시보드 - API 응답:", response);
        const result = response || [];
        console.log("대시보드 - 대기 중인 환자 수:", result.length);
        return result;
      } catch (err) {
        console.error("대시보드 - 대기 중인 환자 데이터 조회 오류:", err);
        return [];
      }
    },
    refetchInterval: 30000, // 30초마다 자동 새로고침
  });

  // 환자 목록(환자관리 페이지와 동일 엔드포인트/형태)
  const { data: patients = [] } = useQuery({
    queryKey: ["patients"],
    queryFn: async () => {
      try {
        const response = await apiRequest("GET", "/api/lung_cancer/patients/");
        return response.results || [];
      } catch (err) {
        console.error("대시보드 - 환자 목록 조회 오류:", err);
        return [];
      }
    },
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });

  // 대시보드 통계 데이터
  const { data: dashboardStats } = useQuery({
    queryKey: ["dashboard-statistics"],
    queryFn: async () => {
      try {
        const response = await apiRequest("GET", "/api/lung_cancer/medical-records/dashboard_statistics/");
        return response;
      } catch (err) {
        console.error("대시보드 - 통계 데이터 조회 오류:", err);
        return {
          total_records: 0,
          waiting_count: 0,
          completed_count: 0,
          today_exams: 0,
        };
      }
    },
    refetchInterval: 30000, // 30초마다 자동 새로고침
  });

  const sortedWaitingPatients = [...(waitingPatients as MedicalRecord[])]
    .sort((a, b) => new Date(a.reception_start_time).getTime() - new Date(b.reception_start_time).getTime());
  const recentPatients = sortedWaitingPatients.slice(0, 5);

  const stats = [
    {
      title: "총 환자 수",
      value: dashboardStats?.total_records || 0,
      icon: Users,
      color: "text-blue-600",
      bgColor: "bg-blue-100"
    },
    {
      title: "오늘 예약 검사",
      value: dashboardStats?.today_exams || 0,
      icon: Calendar,
      color: "text-red-600",
      bgColor: "bg-red-100"
    },
    {
      title: "진료 완료 환자",
      value: dashboardStats?.completed_count || 0,
      icon: CheckCircle,
      color: "text-green-600",
      bgColor: "bg-green-100"
    },
    {
      title: "진료 대기 중 환자",
      value: dashboardStats?.waiting_count || 0,
      icon: Activity,
      color: "text-orange-600",
      bgColor: "bg-orange-100"
    }
  ];

  console.log("대시보드 렌더링 중 - waitingPatients:", waitingPatients, "isLoading:", isLoading);

  const filteredPatients = (patients as Patient[]).filter((p) =>
    p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    p.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleCompleteTreatment = async () => {
    if (!selectedRecord) return;

    setIsCompleting(true);
    try {
      await apiRequest('POST', `/api/lung_cancer/medical-records/${selectedRecord.id}/complete_treatment/`, {
        examination_result: examinationResult,
        treatment_note: treatmentNote,
      });
      
      toast({
        title: "진료 완료",
        description: "진료가 성공적으로 완료되었습니다.",
      });

      // 모달 닫기 및 상태 초기화
      setIsCompleteDialogOpen(false);
      setSelectedRecord(null);
      setExaminationResult("");
      setTreatmentNote("");

      // 대기 환자 목록 및 통계 새로고침
      queryClient.invalidateQueries({ queryKey: ["waiting-patients"] });
      queryClient.invalidateQueries({ queryKey: ["patients"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-statistics"] });
    } catch (error: any) {
      console.error('진료 완료 오류:', error);
      toast({
        title: "오류 발생",
        description: error?.response?.data?.error || "진료 완료 처리 중 오류가 발생했습니다.",
        variant: "destructive",
      });
    } finally {
      setIsCompleting(false);
    }
  };

  const openCompleteDialog = (record: MedicalRecord) => {
    setSelectedRecord(record);
    setIsCompleteDialogOpen(true);
  };

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
            <div className="flex items-center space-x-2">
              {location.pathname === '/' && (
                <>
                  <Button size="sm" variant="outline" onClick={() => navigate('/login')}>
                    로그인
                  </Button>
                  <Button size="sm" onClick={() => navigate('/signup')}>
                    회원가입
                  </Button>
                </>
              )}
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
                <Button 
                  variant="outline" 
                  size="sm" 
                  data-testid="button-view-all-patients"
                  onClick={() => setShowAllWaiting((v) => !v)}
                >
                  {showAllWaiting ? "접기" : "전체 보기"}
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
              ) : (showAllWaiting ? sortedWaitingPatients : recentPatients).length > 0 ? (
                <div className="space-y-3">
                  {(showAllWaiting ? sortedWaitingPatients : recentPatients).map((record: MedicalRecord, index: number) => (
                    <div 
                      key={record.id} 
                      className="flex items-center space-x-3 p-3 hover:bg-gray-50 rounded-lg"
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
                      <div className="flex flex-col items-end space-y-1">
                        <div className="text-sm text-gray-400">
                          {new Date(record.reception_start_time).toLocaleTimeString('ko-KR', {
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </div>
                        <Button 
                          size="sm" 
                          variant="outline"
                          onClick={(e) => {
                            e.stopPropagation();
                            openCompleteDialog(record);
                          }}
                          className="text-xs"
                        >
                          <CheckCircle className="w-3 h-3 mr-1" />
                          진료 완료
                        </Button>
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
                onClick={() => navigate('/patients')}
              >
                <UserPlus className="w-4 h-4 mr-2" />
                환자 등록
              </Button>
              
              <Button 
                className="w-full justify-start" 
                variant="outline"
                data-testid="button-upload-image"
                onClick={() => navigate('/images')}
              >
                <FileImage className="w-4 h-4 mr-2" />
                의료 이미지 업로드
              </Button>
              
              <Button 
                className="w-full justify-start" 
                variant="outline"
                data-testid="button-new-examination"
                onClick={() => navigate('/reservation-info')}
              >
                <Calendar className="w-4 h-4 mr-2" />
                예약 검사 등록
              </Button>
              
              <Button 
                className="w-full justify-start" 
                variant="outline"
                data-testid="button-ai-analysis"
                onClick={() => navigate('/medical-registration')}
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

        {/* 검색 결과 테이블 (환자관리 페이지와 동일 필터 방식) */}
        {searchTerm.trim() && (
          <Card className="mt-6">
            <CardHeader>
              <CardTitle>환자 목록</CardTitle>
              <CardDescription>검색 결과 {filteredPatients.length}명</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b text-gray-600">
                      <th className="py-2">환자정보</th>
                      <th className="py-2">성별/나이</th>
                      <th className="py-2">연락처</th>
                      <th className="py-2">혈액형</th>
                      <th className="py-2">등록일</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredPatients.map((p) => (
                      <tr key={p.id} className="border-b hover:bg-gray-50">
                        <td className="py-2">
                          <div className="flex items-center space-x-3">
                            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                              <span className="text-blue-600 text-sm font-semibold">{p.name.charAt(0)}</span>
                            </div>
                            <div>
                              <div className="text-gray-900 font-medium">{p.name}</div>
                              <div className="text-xs text-gray-500">{p.id}</div>
                            </div>
                          </div>
                        </td>
                        <td className="py-2 text-sm text-gray-700">{p.gender} / {p.age}세</td>
                        <td className="py-2 text-sm text-gray-700">{p.phone || "-"}</td>
                        <td className="py-2 text-sm text-gray-700">{p.blood_type || "-"}</td>
                        <td className="py-2 text-sm text-gray-700">{new Date(p.created_at).toLocaleDateString("ko-KR")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}
      </main>

      {/* 진료 완료 모달 */}
      <Dialog open={isCompleteDialogOpen} onOpenChange={setIsCompleteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>진료 완료</DialogTitle>
            <DialogDescription>
              {selectedRecord && `${selectedRecord.name} (${selectedRecord.patient_id})`} 환자의 진료를 완료합니다.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div>
              <Label htmlFor="examination-result">검사 결과</Label>
              <Input
                id="examination-result"
                placeholder="검사 결과를 입력하세요 (예: 정상, 이상소견 등)"
                value={examinationResult}
                onChange={(e) => setExaminationResult(e.target.value)}
              />
            </div>
            
            <div>
              <Label htmlFor="treatment-note">진료 메모</Label>
              <textarea
                id="treatment-note"
                className="w-full min-h-[100px] p-2 border border-gray-300 rounded-md resize-none"
                placeholder="진료 관련 메모를 입력하세요"
                value={treatmentNote}
                onChange={(e) => setTreatmentNote(e.target.value)}
              />
            </div>
          </div>
          
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setIsCompleteDialogOpen(false);
                setExaminationResult("");
                setTreatmentNote("");
                setSelectedRecord(null);
              }}
              disabled={isCompleting}
            >
              취소
            </Button>
            <Button
              onClick={handleCompleteTreatment}
              disabled={isCompleting}
            >
              {isCompleting ? "처리 중..." : "진료 완료"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
