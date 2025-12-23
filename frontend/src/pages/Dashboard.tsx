import { useState, useRef, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import {
  Users,
  Calendar,
  Activity,
  UserPlus,
  Search,
  Filter,
  CheckCircle,
  Clock,
  ChevronRight,
  TrendingUp,
  FileText,
  Plus,
  Mail,
  Phone,
  Layout,
  Monitor,
  Database,
  ShieldCheck
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/api";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import doctorProfile from "@/assets/doctor-profile.png";
import {
  PieChart, Pie, Cell, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip
} from 'recharts';

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

const PIE_COLORS = ['#3b82f6', '#f472b6', '#fbbf24', '#ef4444', '#10b981'];

export default function Dashboard() {
  const { user } = useAuth();
  const [searchTerm, setSearchTerm] = useState("");
  const [showAllWaiting, setShowAllWaiting] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<MedicalRecord | null>(null);
  const [isCompleteDialogOpen, setIsCompleteDialogOpen] = useState(false);
  const [examinationResult, setExaminationResult] = useState("");
  const [treatmentNote, setTreatmentNote] = useState("");
  const [isCompleting, setIsCompleting] = useState(false);
  const searchResultRef = useRef<HTMLDivElement>(null);

  const navigate = useNavigate();
  const { toast } = useToast();
  const queryClient = useQueryClient();

  useEffect(() => {
    if (searchTerm.trim() && searchResultRef.current) {
      searchResultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [searchTerm]);

  const { data: waitingPatients = [], isLoading } = useQuery({
    queryKey: ["waiting-patients"],
    queryFn: async () => {
      try {
        const response = await apiRequest("GET", "/api/lung_cancer/medical-records/waiting_patients/");
        return response || [];
      } catch (err) {
        console.error("Dashboard - Waiting patients fetch error:", err);
        return [];
      }
    },
    refetchInterval: 30000,
  });

  const { data: patients = [] } = useQuery({
    queryKey: ["patients"],
    queryFn: async () => {
      try {
        const response = await apiRequest("GET", "/api/lung_cancer/patients/");
        return response.results || [];
      } catch (err) {
        return [];
      }
    },
    staleTime: Infinity,
  });

  const { data: dashboardStats } = useQuery({
    queryKey: ["dashboard-statistics"],
    queryFn: async () => {
      try {
        const response = await apiRequest("GET", "/api/lung_cancer/medical-records/dashboard_statistics/");
        return response;
      } catch (err) {
        return {
          total_records: 0,
          waiting_count: 0,
          completed_count: 0,
          today_exams: 0,
        };
      }
    },
    refetchInterval: 30000,
  });

  const sortedWaitingPatients = [...(waitingPatients as MedicalRecord[])]
    .sort((a, b) => new Date(a.reception_start_time).getTime() - new Date(b.reception_start_time).getTime());
  const recentPatients = sortedWaitingPatients.slice(0, 5);

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
      toast({ title: "진료 완료", description: "진료가 성공적으로 완료되었습니다." });
      setIsCompleteDialogOpen(false);
      setSelectedRecord(null);
      setExaminationResult("");
      setTreatmentNote("");
      queryClient.invalidateQueries({ queryKey: ["waiting-patients"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-statistics"] });
    } catch (error: any) {
      toast({ title: "오류", description: "처리 중 오류가 발생했습니다.", variant: "destructive" });
    } finally {
      setIsCompleting(false);
    }
  };

  // Mock Data for Charts
  const browserData = [
    { name: 'Chrome', value: 450 },
    { name: 'Safari', value: 150 },
    { name: 'Edge', value: 100 },
    { name: 'Firefox', value: 80 },
    { name: 'Etc', value: 50 },
  ];

  const deviceData = [
    { name: 'PC', value: 700 },
    { name: 'Mobile', value: 250 },
    { name: 'Tablet', value: 100 },
  ];

  const visitorTrend = [
    { date: '21.07.21', count: 18 },
    { date: '21.07.22', count: 0 },
    { date: '21.07.23', count: 9 },
    { date: '21.07.24', count: 24 },
    { date: '21.07.25', count: 32 },
    { date: '21.07.26', count: 22 },
    { date: '21.07.27', count: 20 },
    { date: '21.07.28', count: 12 },
    { date: '21.07.29', count: 11 },
    { date: '21.07.30', count: 18 },
    { date: '21.08.01', count: 18 },
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1 }
  };

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={containerVariants}
      className="space-y-6"
    >
      {/* Top Header Section */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h2 className="text-xl font-bold text-gray-900">오늘</h2>
          <ChevronRight className="w-4 h-4 text-gray-300" />
        </div>
        <Button size="sm" variant="outline" className="rounded-xl h-8 text-[10px] font-bold bg-green-50 text-green-600 border-green-100 hover:bg-green-600 hover:text-white">
          엑셀 다운로드
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Side (Charts & Stats) */}
        <div className="lg:col-span-9 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {/* Small Stats Stack */}
            <div className="space-y-4">
              <Card className="border-none shadow-sm rounded-3xl overflow-hidden bg-white hover:shadow-md transition-shadow">
                <CardContent className="p-6 flex items-center justify-between">
                  <div>
                    <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1">방문자 수</p>
                    <p className="text-3xl font-black text-gray-900">{dashboardStats?.total_records || 243}</p>
                  </div>
                  <div className="w-12 h-12 bg-orange-50 rounded-2xl flex items-center justify-center">
                    <Users className="w-6 h-6 text-orange-500" />
                  </div>
                </CardContent>
              </Card>
              <Card className="border-none shadow-sm rounded-3xl overflow-hidden bg-white hover:shadow-md transition-shadow">
                <CardContent className="p-6 flex items-center justify-between">
                  <div>
                    <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-1">사이트 클릭 수</p>
                    <p className="text-3xl font-black text-gray-900">1,513</p>
                  </div>
                  <div className="w-12 h-12 bg-purple-50 rounded-2xl flex items-center justify-center">
                    <Activity className="w-6 h-6 text-purple-500" />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Pie Chart 1 */}
            <Card className="border-none shadow-sm rounded-3xl bg-white p-6 relative">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-sm font-black text-gray-900 tracking-tight">브라우저 메뉴 통계</h3>
                <Select defaultValue="7days">
                  <option value="7days">7일</option>
                </Select>
              </div>
              <div className="h-[180px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={browserData}
                      innerRadius={45}
                      outerRadius={65}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {browserData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} strokeWidth={0} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* Pie Chart 2 */}
            <Card className="border-none shadow-sm rounded-3xl bg-white p-6 relative">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-sm font-black text-gray-900 tracking-tight">디바이스 통계</h3>
                <Select defaultValue="7days">
                  <option value="7days">7일</option>
                </Select>
              </div>
              <div className="h-[180px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={deviceData}
                      innerRadius={45}
                      outerRadius={65}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {deviceData.map((_, index) => (
                        <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} strokeWidth={0} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* Quick List (Optional items from image) */}
            <Card className="border-none shadow-sm rounded-3xl bg-white p-6">
              <h3 className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-4">오늘 가장 많이 클릭한 메뉴</h3>
              <div className="space-y-3">
                {[
                  { num: 1, label: "인시던스 컨트롤러", color: "bg-orange-100 text-orange-600" },
                  { num: 2, label: "인시던스 유지보수", color: "bg-blue-100 text-blue-600" },
                  { num: 3, label: "인시던스 UI/UX Framework", color: "bg-orange-100 text-orange-600" },
                ].map((item, idx) => (
                  <div key={idx} className="flex items-center gap-3">
                    <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-black ${item.color}`}>
                      {item.num}
                    </div>
                    <span className="text-xs font-bold text-gray-600 truncate">{item.label}</span>
                  </div>
                ))}
              </div>
            </Card>
          </div>

          {/* Large Bar Chart Section */}
          <Card className="border-none shadow-sm rounded-3xl bg-white p-8">
            <div className="flex items-center gap-2 mb-8">
              <h3 className="text-lg font-black text-gray-900">사용자 방문자 수 통계</h3>
              <div className="w-5 h-5 rounded-full border border-gray-200 flex items-center justify-center text-[10px] text-gray-400">?</div>
            </div>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={visitorTrend}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                  <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                  <Tooltip cursor={{ fill: '#f8fafc' }} contentStyle={{ borderRadius: '16px', border: 'none', boxShadow: '0 10px 30px rgba(0,0,0,0.1)' }} />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {visitorTrend.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>

        {/* Right Side Column (Profile Card) */}
        <div className="lg:col-span-3 space-y-6">
          <Card className="border-none shadow-2xl shadow-blue-900/10 rounded-[2.5rem] overflow-hidden bg-gradient-to-b from-blue-600 to-blue-700 text-white relative h-fit">
            <div className="absolute top-0 right-0 w-32 h-32 bg-white/10 rounded-full blur-3xl -mr-16 -mt-16"></div>
            <div className="absolute bottom-0 left-0 w-24 h-24 bg-blue-400/20 rounded-full blur-2xl -ml-12 -mb-12"></div>

            <CardContent className="p-8 pt-10 flex flex-col items-center text-center relative z-10">
              <div className="relative mb-6">
                <div className="w-32 h-32 rounded-full border-4 border-white/20 p-1 shadow-2xl relative overflow-hidden group">
                  <img src={doctorProfile} alt="Profile" className="w-full h-full object-cover rounded-full" />
                  <div className="absolute top-1 right-1 bg-yellow-400 text-white w-6 h-6 rounded-full flex items-center justify-center shadow-lg transform rotate-12 border-2 border-blue-600">
                    <span className="text-[10px]">👑</span>
                  </div>
                </div>
              </div>

              <div className="mb-6 space-y-1">
                <h2 className="text-2xl font-black tracking-tight">
                  {user ? `${user.last_name || ''} ${user.first_name || user.username}`.trim() : "홍길동"}
                </h2>
                <Badge className="bg-white/20 hover:bg-white/30 text-white border-none py-1 px-4 rounded-full font-bold text-[10px] uppercase tracking-wider">
                  {user?.role === 'medical_staff' ? '원격 판독 의료진' : '총괄 시스템 관리자'}
                </Badge>
              </div>

              <div className="w-full space-y-4 pt-6 border-t border-white/10">
                <div className="flex items-center justify-between group">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-xl bg-white/10 flex items-center justify-center">
                      <Database className="w-4 h-4" />
                    </div>
                    <span className="text-xs font-bold text-blue-100">아이디</span>
                  </div>
                  <span className="text-xs font-black">{user?.username || "inseq123"}</span>
                </div>
                <div className="flex items-center justify-between group">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-xl bg-white/10 flex items-center justify-center">
                      <Layout className="w-4 h-4" />
                    </div>
                    <span className="text-xs font-bold text-blue-100">소속부서</span>
                  </div>
                  <span className="text-xs font-black">건양대학교 병원</span>
                </div>
                <div className="flex items-center justify-between group">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-xl bg-white/10 flex items-center justify-center">
                      <Mail className="w-4 h-4" />
                    </div>
                    <span className="text-xs font-bold text-blue-100">이메일</span>
                  </div>
                  <span className="text-xs font-black">{user?.email || "doctor@konyang.ac.kr"}</span>
                </div>
                <div className="flex items-center justify-between group">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-xl bg-white/10 flex items-center justify-center">
                      <Phone className="w-4 h-4" />
                    </div>
                    <span className="text-xs font-bold text-blue-100">전화번호</span>
                  </div>
                  <span className="text-xs font-black">02-1234-5678</span>
                </div>
              </div>

              <div className="mt-8 pt-4 border-t border-white/10 w-full flex items-center justify-between text-[10px] font-bold text-white/50">
                <span>최근 접속 시간</span>
                <span className="text-white/80">{new Date().toLocaleString('ko-KR')}</span>
              </div>
            </CardContent>
          </Card>

          <Card className="border-none shadow-sm rounded-3xl bg-white p-6 h-fit">
            <h3 className="text-sm font-black text-gray-900 mb-6 tracking-tight">나의 접속환경</h3>
            <div className="space-y-4">
              <div className="p-4 rounded-3xl bg-gray-50 flex items-center justify-between group hover:bg-blue-50 transition-colors cursor-default">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-white rounded-2xl flex items-center justify-center shadow-sm">
                    <Monitor className="w-6 h-6 text-blue-600" />
                  </div>
                  <div>
                    <p className="text-[10px] font-bold text-gray-400 uppercase">Browser</p>
                    <p className="text-sm font-black text-gray-900">Chrome</p>
                  </div>
                </div>
                <div className="w-8 h-8 rounded-full flex items-center justify-center bg-blue-100/50 group-hover:bg-blue-600 transition-colors">
                  <Activity className="w-4 h-4 text-blue-600 group-hover:text-white" />
                </div>
              </div>
              <div className="p-4 rounded-3xl bg-gray-50 flex items-center justify-between group hover:bg-purple-50 transition-colors cursor-default">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-white rounded-2xl flex items-center justify-center shadow-sm">
                    <ShieldCheck className="w-6 h-6 text-purple-600" />
                  </div>
                  <div>
                    <p className="text-[10px] font-bold text-gray-400 uppercase">OS</p>
                    <p className="text-sm font-black text-gray-900">Windows 11</p>
                  </div>
                </div>
                <div className="w-8 h-8 rounded-full flex items-center justify-center bg-purple-100/50 group-hover:bg-purple-600 transition-colors">
                  <Activity className="w-4 h-4 text-purple-600 group-hover:text-white" />
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Waiting Patients List */}
        <motion.div variants={itemVariants} className="lg:col-span-2">
          <Card className="border-none shadow-sm h-full bg-white rounded-3xl overflow-hidden">
            <CardHeader className="border-b border-gray-50 pb-6">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-xl font-bold text-gray-900">대기 중인 환자</CardTitle>
                  <CardDescription className="text-xs font-medium text-gray-400">현재 총 {waitingPatients.length}명이 대기 중입니다.</CardDescription>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="rounded-xl text-blue-600 font-bold text-xs hover:bg-blue-50"
                  onClick={() => setShowAllWaiting(!showAllWaiting)}
                >
                  {showAllWaiting ? "간략히" : "전체 보기"}
                  <ChevronRight className={`ml-1 w-4 h-4 transition-transform ${showAllWaiting ? 'rotate-90' : ''}`} />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div className="divide-y divide-gray-50 max-h-[500px] overflow-y-auto custom-scrollbar">
                {isLoading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="p-6 animate-pulse flex items-center gap-4">
                      <div className="w-12 h-12 bg-gray-100 rounded-2xl" />
                      <div className="flex-1 space-y-2">
                        <div className="h-4 bg-gray-100 rounded w-1/4" />
                        <div className="h-3 bg-gray-100 rounded w-1/2" />
                      </div>
                    </div>
                  ))
                ) : (showAllWaiting ? sortedWaitingPatients : recentPatients).length > 0 ? (
                  <AnimatePresence mode="popLayout">
                    {(showAllWaiting ? sortedWaitingPatients : recentPatients).map((record: MedicalRecord, index: number) => (
                      <motion.div
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 10 }}
                        layout
                        key={record.id}
                        className="group flex items-center gap-4 p-5 hover:bg-blue-50/30 transition-colors"
                      >
                        <div className="w-12 h-12 flex-shrink-0 bg-white shadow-sm border border-gray-100 rounded-2xl flex items-center justify-center text-blue-600 font-black text-lg group-hover:bg-blue-600 group-hover:text-white transition-all">
                          {index + 1}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-0.5">
                            <span className="font-bold text-gray-900">{record.name}</span>
                            <Badge variant="outline" className="text-[10px] py-0 border-gray-200 text-gray-500">
                              {record.patient_id}
                            </Badge>
                          </div>
                          <div className="flex items-center gap-3 text-xs font-medium text-gray-400">
                            <span className="flex items-center gap-1">
                              <Activity className="w-3 h-3" /> {record.department}
                            </span>
                            <span className="flex items-center gap-1">
                              <Calendar className="w-3 h-3" /> {new Date(record.reception_start_time).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => {
                              setSelectedRecord(record);
                              setIsCompleteDialogOpen(true);
                            }}
                            className="rounded-xl bg-emerald-50 text-emerald-600 hover:bg-emerald-600 hover:text-white font-bold text-xs"
                          >
                            <CheckCircle className="w-3.5 h-3.5 mr-1" />
                            진료 완료
                          </Button>
                        </div>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                ) : (
                  <div className="p-12 text-center">
                    <div className="w-16 h-16 bg-gray-50 rounded-full flex items-center justify-center mx-auto mb-4">
                      <Clock className="w-8 h-8 text-gray-300" />
                    </div>
                    <p className="text-sm font-bold text-gray-400">대기 중인 환자가 없습니다.</p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Action Quick Panel */}
        <div className="space-y-6">
          <motion.div variants={itemVariants}>
            <Card className="border-none shadow-sm bg-blue-600 text-white rounded-3xl p-1 overflow-hidden relative group">
              <div className="absolute inset-0 bg-blue-700 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <div className="absolute -right-8 -top-8 w-32 h-32 bg-white/10 rounded-full blur-2xl"></div>
              <CardContent className="p-6 relative z-10">
                <div className="bg-white/20 w-10 h-10 rounded-xl flex items-center justify-center mb-4 backdrop-blur-md">
                  <TrendingUp className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-lg font-black mb-2 tracking-tight">지식 허브 바로가기</h3>
                <p className="text-blue-100/80 text-xs font-medium mb-4 leading-relaxed">
                  최신 의학 논문 및 치료 가이드라인을 확인하세요.
                </p>
                <Button
                  onClick={() => navigate('/knowledge-hub')}
                  className="w-full bg-white text-blue-600 hover:bg-blue-50 font-black rounded-xl"
                >
                  지식 허브 열기
                </Button>
              </CardContent>
            </Card>
          </motion.div>

          {/* Rapid Access Buttons */}
          <Card className="border-none shadow-sm bg-white rounded-3xl p-6">
            <h3 className="text-sm font-black text-gray-900 mb-6 flex items-center gap-2 tracking-tight">
              <Plus className="w-4 h-4 text-blue-600" />
              빠른 작업 (Quick Actions)
            </h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: "환자 등록", icon: UserPlus, path: "/patients", color: "text-blue-600", bg: "bg-blue-50" },
                { label: "MRI 조회", icon: Activity, path: "/mri-viewer", color: "text-purple-600", bg: "bg-purple-50" },
                { label: "전체 예약", icon: Calendar, path: "/reservation-info", color: "text-emerald-600", bg: "bg-emerald-50" },
                { label: "진료 접수", icon: FileText, path: "/medical-registration", color: "text-amber-600", bg: "bg-amber-50" },
              ].map((action, i) => (
                <button
                  key={i}
                  onClick={() => navigate(action.path)}
                  className="flex flex-col items-center justify-center p-4 rounded-3xl border border-gray-50 hover:bg-gray-50 hover:scale-105 transition-all group"
                >
                  <div className={`${action.bg} ${action.color} p-3 rounded-2xl mb-2 group-hover:shadow-sm`}>
                    <action.icon className="w-5 h-5" />
                  </div>
                  <span className="text-[10px] font-bold text-gray-500 uppercase tracking-tighter">{action.label}</span>
                </button>
              ))}
            </div>
          </Card>
        </div>
      </div>

      {/* Patient Search Section */}
      <motion.div variants={itemVariants}>
        <Card className="border-none shadow-sm bg-white rounded-3xl p-8">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-8">
            <div>
              <h3 className="text-xl font-black text-gray-900 tracking-tight">통합 환자 검색</h3>
              <p className="text-xs font-medium text-gray-400">데이터베이스 내 모든 환자 정보를 빠르고 상세하게 검색합니다.</p>
            </div>
            <div className="flex items-center gap-2 flex-1 md:max-w-md">
              <div className="relative flex-1 group">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 group-focus-within:text-blue-600 transition-colors" />
                <Input
                  placeholder="환자 이름 또는 고유 번호 입력..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-11 h-12 bg-gray-50 border-none rounded-2xl focus-visible:ring-2 focus-visible:ring-blue-600/20"
                />
              </div>
              <Button size="icon" className="h-12 w-12 rounded-2xl bg-gray-900 hover:bg-black group">
                <Filter className="w-4 h-4 group-hover:scale-110 transition-transform" />
              </Button>
            </div>
          </div>

          <AnimatePresence>
            {searchTerm.trim() && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="overflow-x-auto"
                ref={searchResultRef}
              >
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="text-[10px] font-black text-gray-400 uppercase tracking-widest border-b border-gray-50">
                      <th className="py-4 px-4">환자 프로필</th>
                      <th className="py-4 px-4">성별 / 나이</th>
                      <th className="py-4 px-4">연락처</th>
                      <th className="py-4 px-4">등록일</th>
                      <th className="py-4 px-4 text-right">관리</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-50">
                    {filteredPatients.map((p) => (
                      <tr key={p.id} className="group hover:bg-gray-50/50 transition-colors">
                        <td className="py-4 px-4">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-blue-50 text-blue-600 rounded-2xl flex items-center justify-center font-black text-lg">
                              {p.name.charAt(0)}
                            </div>
                            <div>
                              <div className="font-bold text-gray-900">{p.name}</div>
                              <div className="text-[10px] font-bold text-gray-400">ID: {p.id}</div>
                            </div>
                          </div>
                        </td>
                        <td className="py-4 px-4 font-medium text-gray-600">{p.gender} / {p.age}세</td>
                        <td className="py-4 px-4 font-medium text-gray-600">{p.phone || "-"}</td>
                        <td className="py-4 px-4 font-medium text-gray-600">{new Date(p.created_at).toLocaleDateString("ko-KR")}</td>
                        <td className="py-4 px-4 text-right">
                          <Button
                            variant="ghost"
                            size="icon"
                            className="rounded-xl hover:bg-blue-50 hover:text-blue-600"
                            onClick={() => navigate('/patients')}
                          >
                            <ChevronRight className="w-4 h-4" />
                          </Button>
                        </td>
                      </tr>
                    ))}
                    {filteredPatients.length === 0 && (
                      <tr>
                        <td colSpan={5} className="py-12 text-center text-gray-400 font-bold">검색 결과가 없습니다.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </motion.div>
            )}
          </AnimatePresence>
        </Card>
      </motion.div>

      {/* Treatment Completion Dialog */}
      <Dialog open={isCompleteDialogOpen} onOpenChange={setIsCompleteDialogOpen}>
        <DialogContent className="rounded-3xl border-none p-0 overflow-hidden max-w-md">
          <div className="bg-emerald-600 p-8 text-white relative overflow-hidden">
            <div className="absolute -right-8 -top-8 w-32 h-32 bg-white/10 rounded-full blur-2xl"></div>
            <DialogHeader className="relative z-10">
              <div className="bg-white/20 w-12 h-12 rounded-2xl flex items-center justify-center mb-4 backdrop-blur-md">
                <CheckCircle className="w-6 h-6 text-white" />
              </div>
              <DialogTitle className="text-2xl font-black tracking-tight">진료 완료 승인</DialogTitle>
              <DialogDescription className="text-emerald-100 font-medium">
                {selectedRecord && `${selectedRecord.name} (${selectedRecord.patient_id})`} 환자의 진료 기록을 최종 저장하고 대기열에서 제외합니다.
              </DialogDescription>
            </DialogHeader>
          </div>

          <div className="p-8 space-y-6 bg-white">
            <div className="space-y-2">
              <Label className="text-[10px] font-black uppercase tracking-widest text-gray-400">검사 결과 및 소견</Label>
              <Input
                placeholder="예: 특이 소견 없음, 추가 MRI 필요 등"
                value={examinationResult}
                onChange={(e) => setExaminationResult(e.target.value)}
                className="h-12 bg-gray-50 border-none rounded-xl focus-visible:ring-2 focus-visible:ring-emerald-600/20"
              />
            </div>

            <div className="space-y-2">
              <Label className="text-[10px] font-black uppercase tracking-widest text-gray-400">의료진 처방 메모</Label>
              <textarea
                className="w-full min-h-[120px] p-4 bg-gray-50 border-none rounded-2xl focus:ring-2 focus:ring-emerald-600/20 outline-none resize-none text-sm placeholder:text-gray-300 font-medium"
                placeholder="환자에게 전달할 주의사항이나 내부 기록용 메모를 입력하세요."
                value={treatmentNote}
                onChange={(e) => setTreatmentNote(e.target.value)}
              />
            </div>

            <div className="flex gap-3">
              <Button
                variant="ghost"
                className="flex-1 h-12 rounded-xl font-black text-gray-400 hover:bg-gray-50"
                onClick={() => setIsCompleteDialogOpen(false)}
                disabled={isCompleting}
              >
                닫기
              </Button>
              <Button
                className="flex-1 h-12 rounded-xl bg-emerald-600 hover:bg-emerald-700 font-black shadow-lg shadow-emerald-600/20"
                onClick={handleCompleteTreatment}
                disabled={isCompleting}
              >
                {isCompleting ? "처리 중..." : "최종 완료"}
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </motion.div>
  );
}

function Select({ children, defaultValue }: { children: React.ReactNode, defaultValue: string }) {
  return (
    <select defaultValue={defaultValue} className="text-[10px] font-bold text-gray-400 bg-gray-50 border-none rounded-lg px-2 py-1 outline-none">
      {children}
    </select>
  );
}
