import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { 
  Plus, 
  Search, 
  Filter, 
  Edit, 
  Eye,
  Calendar,
  FileImage,
  Trash2
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { 
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { apiRequest } from "@/lib/api";
import PatientRegistrationModal from "@/components/PatientRegistrationModal";

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

export default function Patients() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [isRegistrationModalOpen, setIsRegistrationModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [editingPatient, setEditingPatient] = useState<Patient | null>(null);
  const [deletingPatient, setDeletingPatient] = useState<Patient | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isDetailModalOpen, setIsDetailModalOpen] = useState(false);
  const [detailPatient, setDetailPatient] = useState<Patient | null>(null);
  const [medicalRecords, setMedicalRecords] = useState<MedicalRecord[]>([]);
  const [isLoadingRecords, setIsLoadingRecords] = useState(false);
  const [isReservationModalOpen, setIsReservationModalOpen] = useState(false);
  const [reservationPatient, setReservationPatient] = useState<Patient | null>(null);
  const [selectedReserveDate, setSelectedReserveDate] = useState<Date | null>(null);
  const [reservations, setReservations] = useState<Record<string, { id: string; name: string; time: string; memo?: string }[]>>({});

  const { data: patients = [], isLoading, refetch, error } = useQuery({
    queryKey: ["patients"],
    queryFn: async () => {
      try {
        const response = await apiRequest("GET", "/api/lung_cancer/api/patients/");
        console.log("API 응답:", response);
        return response.results || [];
      } catch (err) {
        console.error("환자 목록 조회 오류:", err);
        throw err;
      }
    },
  });

  const handleRegistrationSuccess = () => {
    refetch(); // 환자 목록 새로고침
  };

  const handleDeleteClick = (patient: Patient) => {
    setDeletingPatient(patient);
    setIsDeleteModalOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!deletingPatient) return;

    try {
      await apiRequest("DELETE", `/api/lung_cancer/api/patients/${deletingPatient.id}/`);
      alert("환자가 성공적으로 삭제되었습니다.");
      refetch(); // 환자 목록 새로고침
      setIsDeleteModalOpen(false);
      setDeletingPatient(null);
    } catch (error: any) {
      console.error("환자 삭제 오류:", error);
      alert(`환자 삭제 중 오류가 발생했습니다: ${error.response?.data?.error || error.message}`);
    }
  };

  const handleDeleteCancel = () => {
    setIsDeleteModalOpen(false);
    setDeletingPatient(null);
  };

  const handleViewDetails = async (patient: Patient) => {
    setDetailPatient(patient);
    setIsDetailModalOpen(true);
    setIsLoadingRecords(true);
    
    try {
      const response = await apiRequest('GET', `/api/lung_cancer/api/patients/${patient.id}/medical_records/`);
      setMedicalRecords(response.medical_records || []);
    } catch (error: any) {
      console.error('진료 기록 조회 오류:', error);
      alert(`진료 기록 조회 중 오류가 발생했습니다: ${error.response?.data?.error || error.message}`);
      setMedicalRecords([]);
    } finally {
      setIsLoadingRecords(false);
    }
  };

  const handleDetailClose = () => {
    setIsDetailModalOpen(false);
    setDetailPatient(null);
    setMedicalRecords([]);
  };

  const filteredPatients = (patients as Patient[]).filter((patient: Patient) =>
    patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    patient.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const formatDate = (dateString: string | Date | null) => {
    if (!dateString) return "-";
    return new Date(dateString).toLocaleDateString('ko-KR');
  };

  const startOfDay = (d: Date) => {
    const x = new Date(d);
    x.setHours(0,0,0,0);
    return x;
  };

  const fourWeekDates = (): Date[] => {
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


  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div>
              <h1 className="text-xl font-bold text-gray-900">환자 관리</h1>
              <p className="text-sm text-gray-500">등록된 환자 정보를 관리합니다</p>
            </div>
            <Button 
              data-testid="button-add-patient"
              onClick={() => setIsRegistrationModalOpen(true)}
            >
              <Plus className="w-4 h-4 mr-2" />
              환자 등록
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Search and Filter */}
        <Card className="mb-6">
          <CardContent className="pt-6">
            <div className="flex space-x-4">
              <div className="flex-1">
                <Input
                  placeholder="환자 이름 또는 번호로 검색..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  data-testid="input-search-patients"
                />
              </div>
              <Button variant="outline" data-testid="button-search-patients">
                <Search className="w-4 h-4 mr-2" />
                검색
              </Button>
              <Button variant="outline" data-testid="button-filter-patients">
                <Filter className="w-4 h-4 mr-2" />
                필터
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Patient List */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              환자 목록
              <Badge variant="secondary" data-testid="text-patient-count">
                총 {filteredPatients.length}명
              </Badge>
            </CardTitle>
            <CardDescription>
              등록된 모든 환자의 기본 정보를 확인할 수 있습니다
            </CardDescription>
          </CardHeader>
          <CardContent>
            {error ? (
              <div className="text-center py-12">
                <div className="text-red-500 mb-4">환자 목록을 불러오는 중 오류가 발생했습니다.</div>
                <div className="text-sm text-gray-500 mb-4">
                  오류: {error.message || "알 수 없는 오류"}
                </div>
                <Button onClick={() => refetch()}>
                  다시 시도
                </Button>
              </div>
            ) : isLoading ? (
              <div className="text-center py-12">
                <div className="text-gray-500 mb-4">환자 목록을 불러오는 중...</div>
                <div className="space-y-4">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="animate-pulse">
                      <div className="flex items-center space-x-4">
                        <div className="w-12 h-12 bg-gray-200 rounded-full"></div>
                        <div className="flex-1">
                          <div className="h-4 bg-gray-200 rounded w-1/4 mb-2"></div>
                          <div className="h-3 bg-gray-200 rounded w-1/3"></div>
                        </div>
                        <div className="w-20 h-8 bg-gray-200 rounded"></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : filteredPatients.length > 0 ? (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>환자정보</TableHead>
                      <TableHead>성별/나이</TableHead>
                      <TableHead>연락처</TableHead>
                      <TableHead>혈액형</TableHead>
                      <TableHead>등록일</TableHead>
                      <TableHead>작업</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredPatients.map((patient: Patient) => (
                      <TableRow key={patient.id} className="hover:bg-gray-50">
                        <TableCell>
                          <div className="flex items-center space-x-3">
                            <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                              <span className="text-blue-600 font-semibold text-sm">
                                {patient.name.charAt(0)}
                              </span>
                            </div>
                            <div>
                              <p className="font-medium text-gray-900" data-testid={`text-patient-name-${patient.id}`}>
                                {patient.name}
                              </p>
                              <p className="text-sm text-gray-500" data-testid={`text-patient-number-${patient.id}`}>
                                {patient.id}
                              </p>
                            </div>
                          </div>
                        </TableCell>
                        <TableCell>
                          <div>
                            <p className="text-sm text-gray-900">{patient.gender}</p>
                            <p className="text-sm text-gray-500">
                              {patient.age}세
                            </p>
                          </div>
                        </TableCell>
                        <TableCell>
                          <div>
                            <p className="text-sm text-gray-900">{patient.phone || "-"}</p>
                            <p className="text-sm text-gray-500">{patient.emergency_contact || "-"}</p>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge variant="outline">{patient.blood_type || "-"}</Badge>
                        </TableCell>
                        <TableCell>
                          <p className="text-sm text-gray-900">
                            {formatDate(patient.created_at)}
                          </p>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center space-x-2">
                            <Button 
                              size="sm" 
                              variant="outline"
                              onClick={() => handleViewDetails(patient)}
                              data-testid={`button-view-patient-${patient.id}`}
                            >
                              <Eye className="w-4 h-4" />
                            </Button>
                            <Button 
                              size="sm" 
                              variant="outline"
                              onClick={() => {
                                setEditingPatient(patient);
                                setIsEditModalOpen(true);
                              }}
                              data-testid={`button-edit-patient-${patient.id}`}
                            >
                              <Edit className="w-4 h-4" />
                            </Button>
                            <Button 
                              size="sm" 
                              variant="outline"
                              data-testid={`button-exams-patient-${patient.id}`}
                              onClick={() => {
                                setReservationPatient(patient);
                                setSelectedReserveDate(null);
                                try {
                                  const saved = localStorage.getItem('reservations');
                                  if (saved) {
                                    const parsed = JSON.parse(saved);
                                    if (parsed && typeof parsed === 'object') setReservations(parsed);
                                  } else {
                                    setReservations({});
                                  }
                                } catch {
                                  setReservations({});
                                }
                                setIsReservationModalOpen(true);
                              }}
                            >
                              <Calendar className="w-4 h-4" />
                            </Button>
                            <Button 
                              size="sm" 
                              variant="outline"
                              data-testid={`button-images-patient-${patient.id}`}
                            >
                              <FileImage className="w-4 h-4" />
                            </Button>
                            <Button 
                              size="sm" 
                              variant="outline"
                              onClick={() => handleDeleteClick(patient)}
                              data-testid={`button-delete-patient-${patient.id}`}
                              className="text-red-600 hover:text-red-700 hover:bg-red-50"
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="text-gray-500 mb-4">
                  {searchTerm ? "검색 결과가 없습니다" : "등록된 환자가 없습니다"}
                </div>
                {!searchTerm && (
                  <Button 
                    data-testid="button-add-first-patient"
                    onClick={() => setIsRegistrationModalOpen(true)}
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    첫 번째 환자 등록
                  </Button>
                )}

      {isReservationModalOpen && reservationPatient && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => { setIsReservationModalOpen(false); setReservationPatient(null); }}
          data-testid="modal-patient-reservations"
        >
          <div 
            className="bg-white rounded-lg w-full max-w-5xl max-h-[90vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6 border-b flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">예약 보기</h3>
                <p className="text-sm text-gray-500">{reservationPatient.name} · ID: {reservationPatient.id}</p>
              </div>
              <Button variant="outline" onClick={() => { setIsReservationModalOpen(false); setReservationPatient(null); }}>닫기</Button>
            </div>
            <div className="p-6 grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="md:col-span-2">
                <div className="grid grid-cols-7 gap-2">
                  {["월","화","수","목","금","토","일"].map((d) => (
                    <div key={d} className="text-center text-xs text-gray-500 py-1">{d}</div>
                  ))}
                  {fourWeekDates().map((d) => {
                    const isToday = new Date().toDateString() === d.toDateString();
                    const isSelected = selectedReserveDate && selectedReserveDate.toDateString() === d.toDateString();
                    const key = d.toISOString().slice(0, 10);
                    const dayList = (reservations[key] || []).filter(r => r.id === reservationPatient.id);
                    return (
                      <button
                        key={d.toISOString()}
                        className={`border rounded-md p-3 text-sm transition text-left ${
                          isSelected
                            ? 'border-blue-600 ring-2 ring-blue-300 bg-blue-50'
                            : dayList.length > 0
                              ? 'border-blue-200 bg-blue-50 hover:bg-blue-100'
                              : isToday
                                ? 'border-blue-500 ring-1 ring-blue-200 hover:bg-gray-50'
                                : 'border-gray-200 hover:bg-gray-50'
                        }`}
                        onClick={() => setSelectedReserveDate(d)}
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
              </div>
              <div>
                {!selectedReserveDate ? (
                  <div className="h-full flex items-center justify-center border rounded-md p-6 text-sm text-gray-500">
                    날짜를 선택하면 예약 목록이 표시됩니다.
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="text-sm text-gray-600">
                      선택 날짜: {`${selectedReserveDate.getFullYear()}.${String(selectedReserveDate.getMonth()+1).padStart(2,'0')}.${String(selectedReserveDate.getDate()).padStart(2,'0')}`}
                    </div>
                    {(() => {
                      const key = selectedReserveDate.toISOString().slice(0, 10);
                      const list = [...((reservations[key] || []).filter(r => r.id === reservationPatient.id))].sort((a,b)=>a.time.localeCompare(b.time));
                      if (list.length === 0) return <div className="text-sm text-gray-500">예약이 없습니다</div>;
                      return (
                        <div className="space-y-2">
                          {list.map((r, idx) => (
                            <div key={`${r.id}-${r.time}-${idx}`} className="text-sm text-gray-800">- {r.time}{r.memo ? ` - ${r.memo}` : ""}</div>
                          ))}
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
              </div>
            )}
          </CardContent>
        </Card>
      </main>

      {/* Patient Registration Modal */}
      <PatientRegistrationModal
        isOpen={isRegistrationModalOpen}
        onClose={() => setIsRegistrationModalOpen(false)}
        onSuccess={handleRegistrationSuccess}
      />

      {/* Patient Edit Modal */}
      <PatientRegistrationModal
        isOpen={isEditModalOpen}
        onClose={() => {
          setIsEditModalOpen(false);
          setEditingPatient(null);
        }}
        onSuccess={() => {
          setIsEditModalOpen(false);
          setEditingPatient(null);
          handleRegistrationSuccess();
        }}
        patient={editingPatient}
        isEdit={true}
      />

      {/* Patient Detail Modal - 추후 구현 */}
      {selectedPatient && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setSelectedPatient(null)}
          data-testid="modal-patient-detail"
        >
          <div 
            className="bg-white rounded-lg w-full max-w-2xl max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <h3 className="text-lg font-semibold mb-4">환자 상세 정보</h3>
              <p>환자: {selectedPatient.name}</p>
              <p>환자번호: {selectedPatient.id}</p>
              {/* 추가 환자 정보 표시 */}
              <Button 
                className="mt-4" 
                onClick={() => setSelectedPatient(null)}
                data-testid="button-close-patient-detail"
              >
                닫기
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {isDeleteModalOpen && deletingPatient && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={handleDeleteCancel}
          data-testid="modal-delete-confirmation"
        >
          <div 
            className="bg-white rounded-lg w-full max-w-md"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6">
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center mr-3">
                  <Trash2 className="w-5 h-5 text-red-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900">환자 삭제 확인</h3>
              </div>
              
              <div className="mb-6">
                <p className="text-gray-600 mb-2">
                  다음 환자를 삭제하시겠습니까?
                </p>
                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="font-medium text-gray-900">{deletingPatient.name}</p>
                  <p className="text-sm text-gray-500">환자번호: {deletingPatient.id}</p>
                </div>
                <p className="text-sm text-red-600 mt-2">
                  ⚠️ 이 작업은 되돌릴 수 없습니다.
                </p>
              </div>

              <div className="flex justify-end space-x-2">
                <Button 
                  variant="outline" 
                  onClick={handleDeleteCancel}
                  data-testid="button-cancel-delete"
                >
                  취소
                </Button>
                <Button 
                  variant="destructive" 
                  onClick={handleDeleteConfirm}
                  data-testid="button-confirm-delete"
                >
                  삭제
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Patient Detail Modal */}
      {isDetailModalOpen && detailPatient && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={handleDetailClose}
          data-testid="modal-patient-detail"
        >
          <div 
            className="bg-white rounded-lg w-full max-w-4xl max-h-[90vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="p-6 border-b">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center">
                    <span className="text-blue-600 font-semibold text-lg">
                      {detailPatient.name.charAt(0)}
                    </span>
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold text-gray-900">{detailPatient.name}</h3>
                    <p className="text-sm text-gray-500">환자 ID: {detailPatient.id}</p>
                  </div>
                </div>
                <Button 
                  variant="outline" 
                  onClick={handleDetailClose}
                  data-testid="button-close-detail-modal"
                >
                  닫기
                </Button>
              </div>
            </div>

            <div className="p-6 overflow-y-auto max-h-[70vh]">
              {/* 환자 기본 정보 */}
              <div className="mb-6">
                <h4 className="text-lg font-semibold text-gray-900 mb-4">환자 정보</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-500">성별</p>
                    <p className="text-sm text-gray-900">{detailPatient.gender}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">나이</p>
                    <p className="text-sm text-gray-900">{detailPatient.age}세</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">전화번호</p>
                    <p className="text-sm text-gray-900">{detailPatient.phone || "-"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">응급연락처</p>
                    <p className="text-sm text-gray-900">{detailPatient.emergency_contact || "-"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">혈액형</p>
                    <p className="text-sm text-gray-900">{detailPatient.blood_type || "-"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500">등록일</p>
                    <p className="text-sm text-gray-900">{formatDate(detailPatient.created_at)}</p>
                  </div>
                </div>
              </div>

              {/* 진료 기록 */}
              <div>
                <h4 className="text-lg font-semibold text-gray-900 mb-4">진료 기록</h4>
                {isLoadingRecords ? (
                  <div className="text-center py-8">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
                    <p className="text-sm text-gray-500 mt-2">진료 기록을 불러오는 중...</p>
                  </div>
                ) : medicalRecords.length > 0 ? (
                  <div className="space-y-4">
                    {medicalRecords.map((record) => (
                      <div key={record.id} className="border rounded-lg p-4 hover:bg-gray-50">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <Badge variant={record.status === '진료완료' ? 'default' : 'secondary'}>
                              {record.status}
                            </Badge>
                            <span className="text-sm text-gray-500">{record.department}</span>
                          </div>
                          <span className="text-sm text-gray-500">
                            {new Date(record.reception_start_time).toLocaleString('ko-KR')}
                          </span>
                        </div>
                        <p className="text-sm text-gray-900">{record.notes}</p>
                        {record.treatment_end_time && (
                          <p className="text-xs text-gray-500 mt-1">
                            진료 완료: {new Date(record.treatment_end_time).toLocaleString('ko-KR')}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    <Calendar className="w-12 h-12 mx-auto mb-2 text-gray-300" />
                    <p>진료 기록이 없습니다</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
