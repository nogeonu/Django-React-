import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { 
  Plus, 
  Search, 
  Filter, 
  Edit, 
  Eye,
  Calendar,
  FileImage
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

export default function Patients() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [isRegistrationModalOpen, setIsRegistrationModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [editingPatient, setEditingPatient] = useState<Patient | null>(null);

  const { data: patients = [], isLoading, refetch, error } = useQuery({
    queryKey: ["patients"],
    queryFn: async () => {
      try {
        const response = await apiRequest("GET", "/api/lung_cancer/api/patients/");
        console.log("API 응답:", response.data);
        return response.data.results || [];
      } catch (err) {
        console.error("환자 목록 조회 오류:", err);
        throw err;
      }
    },
  });

  const handleRegistrationSuccess = () => {
    refetch(); // 환자 목록 새로고침
  };

  const filteredPatients = (patients as Patient[]).filter((patient: Patient) =>
    patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    patient.id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const formatDate = (dateString: string | Date | null) => {
    if (!dateString) return "-";
    return new Date(dateString).toLocaleDateString('ko-KR');
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
                              onClick={() => setSelectedPatient(patient)}
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
    </div>
  );
}
