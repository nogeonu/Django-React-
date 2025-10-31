import { useState, useEffect, useCallback } from "react";
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
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { apiRequest } from "@/lib/api";
import PatientRegistrationModal from "@/components/PatientRegistrationModal";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

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
  const [isPatientModalOpen, setIsPatientModalOpen] = useState(false);
  const [isCalendarOpen, setIsCalendarOpen] = useState(false);
  const [selectedDate, setSelectedDate] = useState<Date | null>(null);
  const [rSearchQuery, setRSearchQuery] = useState("");
  const [rPatients, setRPatients] = useState<Patient[]>([]);
  const [rIsLoading, setRIsLoading] = useState(false);
  const [rShowSuggestions, setRShowSuggestions] = useState(false);
  const [rSelectedIndex, setRSelectedIndex] = useState(-1);
  const [rSelectedPatient, setRSelectedPatient] = useState<Patient | null>(null);
  const [reservations, setReservations] = useState<Record<string, { id: string; name: string; time: string; memo?: string }[]>>({});
  const [selectedTime, setSelectedTime] = useState<string>("");
  const [reserveMemo, setReserveMemo] = useState<string>("");
  const [selectedHour, setSelectedHour] = useState<string>("");
  const [selectedMinute, setSelectedMinute] = useState<string>("");
  const queryClient = useQueryClient();

  useEffect(() => {
    try {
      const saved = localStorage.getItem("reservations");
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed && typeof parsed === 'object') setReservations(parsed);
      }
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("reservations", JSON.stringify(reservations));
    } catch { /* ignore */ }
  }, [reservations]);

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

  const sortedWaitingPatients = [...(waitingPatients as MedicalRecord[])]
    .sort((a, b) => new Date(a.reception_start_time).getTime() - new Date(b.reception_start_time).getTime());
  const recentPatients = sortedWaitingPatients.slice(0, 5);
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
      value: 2,
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
  // 환자 목록(환자관리 페이지와 동일 엔드포인트/형태)
  const { data: patients = [] } = useQuery({
    queryKey: ["patients"],
    queryFn: async () => {
      try {
        const response = await apiRequest("GET", "/api/lung_cancer/api/patients/");
        return response.results || [];
      } catch (err) {
        console.error("대시보드 - 환자 목록 조회 오류:", err);
        throw err;
      }
    },
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });

  const filteredPatients = (patients as Patient[]).filter((p) =>
    p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    p.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // 4주 캘린더 데이터 생성 (당일 포함 주부터 시작, 주 시작: 월요일)
  const testReservation = (): Date[] => {
    const day = new Date().getDay();
    const diffToMonday = (day === 0 ? -6 : 1 - day);
    const start = new Date();
    start.setDate(start.getDate() + diffToMonday);

    const days: Date[] = [];
    for (let i = 0; i < 28; i++) {
      const d = new Date(start);
      d.setDate(start.getDate() + i);
      days.push(d);
    }
    return days;
  };

  const startOfDay = (d: Date) => {
    const x = new Date(d);
    x.setHours(0,0,0,0);
    return x;
  };

  const initSuggestedTime = (date: Date) => {
    const now = new Date();
    const isToday = startOfDay(now).getTime() === startOfDay(date).getTime();
    if (!isToday) return "09:00";
    let hh = now.getHours();
    let mm = now.getMinutes();
    const rounded = Math.ceil(mm / 15) * 15; // 15분 단위 반올림(올림)
    if (rounded === 60) { hh += 1; mm = 0; } else { mm = rounded; }
    if (hh < 9) hh = 9;
    if (hh > 17) hh = 17;
    if (hh === 17 && mm > 45) { hh = 17; mm = 45; }
    const h = String(hh).padStart(2, '0');
    const m = String(mm).padStart(2, '0');
    return `${h}:${m}`;
  };

  const handleReserveSearch = useCallback(async (query: string) => {
    if (!query.trim()) {
      setRPatients([]);
      return;
    }
    setRIsLoading(true);
    try {
      const encodedQuery = encodeURIComponent(query.trim());
      const response = await apiRequest("GET", `/api/lung_cancer/api/medical-records/search_patients/?q=${encodedQuery}`);
      setRPatients(response.patients || []);
    } catch (e) {
      setRPatients([]);
    } finally {
      setRIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!isCalendarOpen) return;
    const t = setTimeout(() => {
      if (rSearchQuery.trim().length >= 1 && !rSelectedPatient) {
        handleReserveSearch(rSearchQuery);
        setRShowSuggestions(true);
      } else if (!rSearchQuery.trim()) {
        setRPatients([]);
        setRShowSuggestions(false);
      }
    }, 300);
    return () => clearTimeout(t);
  }, [rSearchQuery, rSelectedPatient, handleReserveSearch, isCalendarOpen]);

  const handleReserveKeyDown = (e: React.KeyboardEvent) => {
    if (!rShowSuggestions || rPatients.length === 0) {
      if (e.key === 'Enter') {
        handleReserveSearch(rSearchQuery);
      }
      return;
    }
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setRSelectedIndex(prev => (prev < rPatients.length - 1 ? prev + 1 : 0));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setRSelectedIndex(prev => (prev > 0 ? prev - 1 : rPatients.length - 1));
        break;
      case 'Enter':
        e.preventDefault();
        if (rSelectedIndex >= 0 && rSelectedIndex < rPatients.length) {
          const p = rPatients[rSelectedIndex];
          setRSelectedPatient(p);
          setRSearchQuery(p.name);
          setRPatients([]);
          setRShowSuggestions(false);
          setRSelectedIndex(-1);
        }
        break;
      case 'Escape':
        setRShowSuggestions(false);
        setRSelectedIndex(-1);
        break;
    }
  };

  const handleReservePatientSelect = (p: Patient) => {
    setRSelectedPatient(p);
    setRSearchQuery(p.name);
    setRPatients([]);
    setRShowSuggestions(false);
    setRSelectedIndex(-1);
  };

  const handleReserveSubmit = () => {
    if (!selectedDate || !rSelectedPatient || !selectedTime) return;
    const key = selectedDate.toISOString().slice(0, 10);
    setReservations((prev) => {
      const list = prev[key] || [];
      return {
        ...prev,
        [key]: [
          ...list,
          { id: rSelectedPatient.id, name: rSelectedPatient.name, time: selectedTime, memo: reserveMemo }
        ],
      };
    });
    setRSearchQuery("");
    setRPatients([]);
    setRShowSuggestions(false);
    setRSelectedIndex(-1);
    setRSelectedPatient(null);
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
                  <div className="md:col-span-1">
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
                onClick={() => setIsCalendarOpen(true)}
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

      {/* 4주 캘린더 모달 */}
      <Dialog open={isCalendarOpen} onOpenChange={setIsCalendarOpen}>
        <DialogContent className="max-w-5xl">
          <DialogHeader>
            <DialogTitle>예약 날짜 선택</DialogTitle>
          </DialogHeader>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="md:col-span-2">
              <div className="grid grid-cols-7 gap-2">
                {["월","화","수","목","금","토","일"].map((d) => (
                  <div key={d} className="text-center text-xs text-gray-500 py-1">{d}</div>
                ))}
                {testReservation().map((d) => {
                  const isToday = new Date().toDateString() === d.toDateString();
                  const isSelected = selectedDate && selectedDate.toDateString() === d.toDateString();
                  const key = d.toISOString().slice(0, 10);
                  const dayReservations = reservations[key] || [];
                  return (
                    <button
                      key={d.toISOString()}
                      className={`border rounded-md p-3 text-sm transition text-left ${
                        isSelected
                          ? 'border-blue-600 ring-2 ring-blue-300 bg-blue-50'
                          : dayReservations.length > 0
                            ? 'border-blue-200 bg-blue-50 hover:bg-blue-100'
                            : isToday
                              ? 'border-blue-500 ring-1 ring-blue-200 hover:bg-gray-50'
                              : 'border-gray-200 hover:bg-gray-50'
                      }`}
                      onClick={() => {
                        setSelectedDate(d);
                        const t = initSuggestedTime(d);
                        setSelectedTime(t);
                        const [h, m] = t.split(":");
                        setSelectedHour(h);
                        setSelectedMinute(m);
                        setRSearchQuery("");
                        setRPatients([]);
                        setRSelectedIndex(-1);
                        setRSelectedPatient(null);
                        setRShowSuggestions(false);
                        setReserveMemo("");
                      }}
                    >
                      {(() => {
                        const todayOnly = startOfDay(new Date());
                        const dayOnly = startOfDay(d);
                        const isPast = dayOnly.getTime() < todayOnly.getTime();
                        return (
                          <div className={`flex items-start justify-between ${isPast ? 'opacity-40' : ''}`}>
                            <span className="font-medium text-base">{d.getDate()}</span>
                            {isToday && <span className="text-xxs text-blue-600">오늘</span>}
                          </div>
                        );
                      })()}
                    </button>
                  );
                })}
              </div>
              {selectedDate && (
                <div className="mt-4 border-t pt-3">
                  <div className="text-sm font-medium text-gray-700 mb-2">해당 날짜 예약자</div>
                  {(() => {
                    const key = selectedDate.toISOString().slice(0, 10);
                    const list = [...(reservations[key] || [])].sort((a, b) => a.time.localeCompare(b.time));
                    if (list.length === 0) return <div className="text-sm text-gray-500">예약자가 없습니다</div>;
                    return (
                      <div className="space-y-2">
                        {list.map((p) => (
                          <div key={`${p.id}-${p.time}`} className="text-sm text-gray-800">- {p.name} (<span className="text-xs text-gray-500">{p.id}</span>) · {p.time}</div>
                        ))}
                      </div>
                    );
                  })()}
                </div>
              )}
            </div>
            <div>
              {!selectedDate ? (
                <div className="h-full flex items-center justify-center border rounded-md p-6 text-sm text-gray-500">
                  날짜를 클릭하면 환자 검색창이 여기에 표시됩니다.
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="text-sm text-gray-600">
                    선택 날짜: {`${selectedDate.getFullYear()}.${String(selectedDate.getMonth()+1).padStart(2,'0')}.${String(selectedDate.getDate()).padStart(2,'0')}`}
                  </div>
                  <div className="relative">
                    <Label htmlFor="reserve-search">환자 검색</Label>
                    <Input
                      id="reserve-search"
                      placeholder="환자 이름을 입력하세요"
                      value={rSearchQuery}
                      onChange={(e) => {
                        setRSearchQuery(e.target.value);
                        if (rSelectedPatient) setRSelectedPatient(null);
                        setRSelectedIndex(-1);
                      }}
                      onKeyDown={handleReserveKeyDown}
                      onFocus={() => { if (rPatients.length > 0) setRShowSuggestions(true); }}
                      onBlur={() => { setTimeout(() => setRShowSuggestions(false), 200); }}
                    />
                    {rShowSuggestions && rPatients.length > 0 && (
                      <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-y-auto">
                        {rPatients.map((p, idx) => (
                          <div
                            key={p.id}
                            className={`p-3 cursor-pointer border-b last:border-b-0 ${idx === rSelectedIndex ? 'bg-blue-50 border-blue-200' : 'hover:bg-gray-50'}`}
                            onClick={() => handleReservePatientSelect(p)}
                          >
                            <div className="font-medium">{p.name}</div>
                            <div className="text-sm text-gray-500">{p.gender} | {p.age}세 | {p.phone}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <Label htmlFor="reserve-time">시간</Label>
                      <div className="flex gap-2">
                        <Select
                          value={selectedHour}
                          onValueChange={(v) => {
                            setSelectedHour(v);
                            const mm = selectedMinute || '00';
                            setSelectedTime(`${v}:${mm}`);
                          }}
                        >
                          <SelectTrigger className="w-[100px]"><SelectValue placeholder="시" /></SelectTrigger>
                          <SelectContent>
                            {Array.from({ length: 9 }).map((_, i) => {
                              const h = String(9 + i).padStart(2, '0');
                              return <SelectItem key={h} value={h}>{h}</SelectItem>;
                            })}
                          </SelectContent>
                        </Select>
                        <Select
                          value={selectedMinute}
                          onValueChange={(v) => {
                            setSelectedMinute(v);
                            const hh = selectedHour || '09';
                            setSelectedTime(`${hh}:${v}`);
                          }}
                        >
                          <SelectTrigger className="w-[100px]"><SelectValue placeholder="분" /></SelectTrigger>
                          <SelectContent>
                            {['00','15','30','45'].map((m) => (
                              <SelectItem key={m} value={m}>{m}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <div>
                      <Label htmlFor="reserve-memo">메모</Label>
                      <textarea id="reserve-memo" className="w-full border rounded-md p-2 h-[38px] text-sm" placeholder="메모를 입력하세요" value={reserveMemo} onChange={(e) => setReserveMemo(e.target.value)} />
                    </div>
                  </div>
                  {rSelectedPatient && (
                    <div className="bg-blue-50 p-3 rounded-md">
                      <div className="text-sm font-medium">선택 환자</div>
                      <div className="mt-1 text-sm">{rSelectedPatient.name} · {rSelectedPatient.gender} · {rSelectedPatient.age}세</div>
                      <div className="mt-1 text-xs text-gray-500">ID: {rSelectedPatient.id} · {rSelectedPatient.phone}</div>
                    </div>
                  )}
                  <div className="pt-2">
                    <Button className="w-full" disabled={!selectedDate || !rSelectedPatient || rIsLoading} onClick={handleReserveSubmit}>
                      {rIsLoading ? '처리 중...' : '이 날짜로 예약 등록'}
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>

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
