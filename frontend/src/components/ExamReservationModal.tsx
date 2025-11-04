import { useCallback, useEffect, useState } from "react";
import { Calendar } from "lucide-react";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { useCalendar } from "@/context/CalendarContext";
import { useNavigate } from "react-router-dom";
import { apiRequest } from "@/lib/api";

export interface Patient {
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

interface ExamReservationModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  patients: Patient[];
  navigateToSchedule?: boolean; // 등록 후 일정관리 페이지로 이동할지 여부
}

export default function ExamReservationModal({
  open,
  onOpenChange,
  patients: _patients,
  navigateToSchedule = true,
}: ExamReservationModalProps) {
  const { toast } = useToast();
  const { addEvent } = useCalendar();
  const navigate = useNavigate();

  const [reserveTitle, setReserveTitle] = useState("");
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [patientSearchTerm, setPatientSearchTerm] = useState("");
  const [patientSuggestions, setPatientSuggestions] = useState<Patient[]>([]);
  const [isPatientLoading, setIsPatientLoading] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [reserveDate, setReserveDate] = useState<string>(new Date().toISOString().slice(0, 10));
  const [startAmPm, setStartAmPm] = useState<'AM' | 'PM'>("AM");
  const [startHour, setStartHour] = useState<string>("09");
  const [startMinute, setStartMinute] = useState<string>("00");
  const [endAmPm, setEndAmPm] = useState<'AM' | 'PM'>("AM");
  const [endHour, setEndHour] = useState<string>("10");
  const [endMinute, setEndMinute] = useState<string>("00");
  const [reserveType, setReserveType] = useState<'검진' | '회의' | '내근' | '외근'>("검진");

  const resetForm = () => {
    setReserveTitle("");
    setSelectedPatient(null);
    setPatientSearchTerm("");
    setPatientSuggestions([]);
    setShowSuggestions(false);
    setReserveDate(new Date().toISOString().slice(0, 10));
    setStartAmPm("AM");
    setStartHour("09");
    setStartMinute("00");
    setEndAmPm("AM");
    setEndHour("10");
    setEndMinute("00");
    setReserveType("검진");
  };

  const handleSubmit = () => {
    try {
      if (!reserveTitle.trim()) {
        toast({ title: "제목을 입력해 주세요", variant: "destructive" });
        return;
      }

      const to24h = (ampm: 'AM' | 'PM', h: string, m: string) => {
        let hourNum = parseInt(h, 10) % 12;
        if (ampm === 'PM') hourNum += 12;
        return `${String(hourNum).padStart(2, '0')}:${m}`;
      };

      const startStr = to24h(startAmPm, startHour, startMinute);
      const endStr = to24h(endAmPm, endHour, endMinute);
      const start = new Date(`${reserveDate}T${startStr}:00`);
      const end = new Date(`${reserveDate}T${endStr}:00`);

      if (end <= start) {
        toast({ title: "종료 시간이 시작 시간보다 늦어야 합니다", variant: "destructive" });
        return;
      }

      const startIso = start.toISOString();
      const endIso = end.toISOString();
      const titleFinal = reserveTitle.trim();

      addEvent({
        title: titleFinal,
        start: startIso,
        end: endIso,
        type: reserveType,
        patientId: selectedPatient?.id,
        patientName: selectedPatient?.name,
        patientGender: selectedPatient?.gender,
        patientAge: selectedPatient?.age,
      });

      onOpenChange(false);
      resetForm();
      toast({ title: "예약이 등록되었습니다", description: "일정관리 캘린더에서 확인할 수 있습니다." });
      
      if (navigateToSchedule) {
        navigate('/schedule');
      }
    } catch (e) {
      toast({ title: "등록 실패", description: "다시 시도해 주세요.", variant: "destructive" });
    }
  };

  const handlePatientSelect = (patient: Patient) => {
    setSelectedPatient(patient);
    setPatientSearchTerm(patient.name);
    setPatientSuggestions([]);
    setShowSuggestions(false);
  };

  const fetchPatientSuggestions = useCallback(async (keyword: string) => {
    if (!keyword.trim()) {
      setPatientSuggestions([]);
      return;
    }

    setIsPatientLoading(true);
    try {
      const encoded = encodeURIComponent(keyword.trim());
      const response = await apiRequest("GET", `/api/lung_cancer/medical-records/search_patients/?q=${encoded}`);
      setPatientSuggestions(response.patients || []);
    } catch (error) {
      console.error("환자 검색 자동완성 오류:", error);
      setPatientSuggestions([]);
    } finally {
      setIsPatientLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;

    const trimmed = patientSearchTerm.trim();
    if (trimmed.length >= 1 && !selectedPatient) {
      const timeoutId = window.setTimeout(() => {
        fetchPatientSuggestions(trimmed);
      }, 300);
      setShowSuggestions(true);
      return () => window.clearTimeout(timeoutId);
    }

    setPatientSuggestions([]);
    setShowSuggestions(false);
  }, [patientSearchTerm, selectedPatient, fetchPatientSuggestions, open]);

  useEffect(() => {
    if (!open) {
      setPatientSearchTerm("");
      setPatientSuggestions([]);
      setShowSuggestions(false);
    }
  }, [open]);

  return (
    <Dialog open={open} onOpenChange={(v) => {
      onOpenChange(v);
      if (!v) resetForm();
    }}>
      <DialogContent className="max-w-lg max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Calendar className="w-5 h-5" />
            예약 검사 등록
          </DialogTitle>
          <DialogDescription>
            예약 정보를 입력하면 일정관리 캘린더에 표시됩니다.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          <div className="space-y-2">
            <Label htmlFor="reserve-title">제목</Label>
            <Input
              id="reserve-title"
              value={reserveTitle}
              onChange={(e) => setReserveTitle(e.target.value)}
              placeholder="예: 흉부 CT 검사"
            />
          </div>

          <div className="space-y-2">
            <Label>환자명 (선택사항)</Label>
            <div className="relative">
              <Input
                value={patientSearchTerm}
                placeholder="환자 이름을 입력하세요"
                autoComplete="off"
                onFocus={() => {
                  if (patientSearchTerm.trim().length >= 1 && !selectedPatient) {
                    setShowSuggestions(true);
                  }
                }}
                onBlur={() => {
                  window.setTimeout(() => setShowSuggestions(false), 150);
                }}
                onChange={(e) => {
                  const value = e.target.value;
                  setPatientSearchTerm(value);
                  setSelectedPatient(null);
                }}
              />
              {showSuggestions && (
                <div className="absolute z-20 mt-2 w-full max-h-56 overflow-y-auto rounded-md border bg-white shadow">
                  {isPatientLoading ? (
                    <div className="px-3 py-2 text-sm text-gray-500">검색 중...</div>
                  ) : patientSuggestions.length > 0 ? (
                    patientSuggestions.map((patient) => (
                      <button
                        key={patient.id}
                        type="button"
                        className="w-full px-3 py-2 text-left text-sm hover:bg-blue-50 focus:bg-blue-50"
                        onMouseDown={(e) => {
                          e.preventDefault();
                          handlePatientSelect(patient);
                        }}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="font-medium text-gray-900">{patient.name}</div>
                            <div className="text-xs text-gray-500">{patient.id}</div>
                          </div>
                          <div className="text-right text-xs text-gray-500">
                            <div>{patient.age}세</div>
                            <div>{patient.gender}</div>
                          </div>
                        </div>
                      </button>
                    ))
                  ) : (
                    <div className="px-3 py-2 text-sm text-gray-500">검색 결과가 없습니다.</div>
                  )}
                </div>
              )}
            </div>
            {selectedPatient && (
              <div className="rounded-md bg-blue-50 px-3 py-2 text-sm text-blue-700">
                선택된 환자: {selectedPatient.name} ({selectedPatient.id})
              </div>
            )}
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="reserve-date">날짜</Label>
              <Input
                id="reserve-date"
                type="date"
                value={reserveDate}
                onChange={(e) => setReserveDate(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label>시작 시간</Label>
              <div className="grid grid-cols-3 gap-2">
                <Select value={startAmPm} onValueChange={(v: any) => setStartAmPm(v)}>
                  <SelectTrigger className="h-10"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="AM">오전</SelectItem>
                    <SelectItem value="PM">오후</SelectItem>
                  </SelectContent>
                </Select>
                <Select value={startHour} onValueChange={(v: any) => setStartHour(v)}>
                  <SelectTrigger className="h-10"><SelectValue placeholder="시" /></SelectTrigger>
                  <SelectContent>
                    {Array.from({ length: 12 }, (_, i) => String(i + 1).padStart(2, '0')).map(h => (
                      <SelectItem key={h} value={h}>{h}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Select value={startMinute} onValueChange={(v: any) => setStartMinute(v)}>
                  <SelectTrigger className="h-10"><SelectValue placeholder="분" /></SelectTrigger>
                  <SelectContent>
                    {['00', '10', '20', '30', '40', '50'].map(m => (
                      <SelectItem key={m} value={m}>{m}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label>종료 시간</Label>
              <div className="grid grid-cols-3 gap-2">
                <Select value={endAmPm} onValueChange={(v: any) => setEndAmPm(v)}>
                  <SelectTrigger className="h-10"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="AM">오전</SelectItem>
                    <SelectItem value="PM">오후</SelectItem>
                  </SelectContent>
                </Select>
                <Select value={endHour} onValueChange={(v: any) => setEndHour(v)}>
                  <SelectTrigger className="h-10"><SelectValue placeholder="시" /></SelectTrigger>
                  <SelectContent>
                    {Array.from({ length: 12 }, (_, i) => String(i + 1).padStart(2, '0')).map(h => (
                      <SelectItem key={h} value={h}>{h}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Select value={endMinute} onValueChange={(v: any) => setEndMinute(v)}>
                  <SelectTrigger className="h-10"><SelectValue placeholder="분" /></SelectTrigger>
                  <SelectContent>
                    {['00', '10', '20', '30', '40', '50'].map(m => (
                      <SelectItem key={m} value={m}>{m}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <Label>유형</Label>
            <Select value={reserveType} onValueChange={(v: any) => setReserveType(v)}>
              <SelectTrigger>
                <SelectValue placeholder="유형 선택" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="검진">검진</SelectItem>
                <SelectItem value="회의">회의</SelectItem>
                <SelectItem value="내근">내근</SelectItem>
                <SelectItem value="외근">외근</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>취소</Button>
          <Button onClick={handleSubmit}>등록</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
