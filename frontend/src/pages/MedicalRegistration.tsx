import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, User, Stethoscope, Calendar as CalendarIcon, ChevronLeft, ChevronRight } from "lucide-react";
import { apiRequest } from "@/lib/api";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface Patient {
  id: string;
  name: string;
  birth_date: string;
  gender: string;
  phone: string;
  age: number;
}

const MedicalRegistration: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [department, setDepartment] = useState('');
  const [notes, setNotes] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);

  // 예약검사 등록 모달/상태
  const [isCalendarOpen, setIsCalendarOpen] = useState(false);
  const [selectedDate, setSelectedDate] = useState<Date | null>(null);
  const [currentMonth, setCurrentMonth] = useState<Date>(() => {
    const d = new Date();
    d.setDate(1);
    d.setHours(0, 0, 0, 0);
    return d;
  });
  // 예약 전용 환자 검색 상태 (메인 검색과 충돌 방지)
  const [rSearchQuery, setRSearchQuery] = useState('');
  const [rPatients, setRPatients] = useState<Patient[]>([]);
  const [rIsLoading, setRIsLoading] = useState(false);
  const [rShowSuggestions, setRShowSuggestions] = useState(false);
  const [rSelectedIndex, setRSelectedIndex] = useState(-1);
  const [rSelectedPatient, setRSelectedPatient] = useState<Patient | null>(null);
  const [reservations, setReservations] = useState<Record<string, { id: string; name: string; time: string; memo?: string }[]>>({});
  const [selectedTime, setSelectedTime] = useState<string>('');
  const [reserveMemo, setReserveMemo] = useState<string>('');
  const [selectedHour, setSelectedHour] = useState<string>('');
  const [selectedMinute, setSelectedMinute] = useState<string>('');

  // 환자 검색
  const handleSearch = useCallback(async (query: string) => {
    console.log('검색 시작:', query);
    if (!query.trim()) {
      setPatients([]);
      return;
    }

    setIsLoading(true);
    try {
      const encodedQuery = encodeURIComponent(query.trim());
      console.log('인코딩된 검색어:', encodedQuery);
      const response = await apiRequest('GET', `/api/lung_cancer/api/medical-records/search_patients/?q=${encodedQuery}`);
      console.log('검색 결과:', response);
      console.log('환자 수:', response.patients?.length || 0);
      setPatients(response.patients || []);
    } catch (error) {
      console.error('환자 검색 오류:', error);
      alert('환자 검색 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // reservations localStorage 로드/저장
  useEffect(() => {
    try {
      const saved = localStorage.getItem('reservations');
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed && typeof parsed === 'object') setReservations(parsed);
      }
    } catch {}
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem('reservations', JSON.stringify(reservations));
    } catch {}
  }, [reservations]);

  // 자동완성을 위한 디바운스 검색
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (searchQuery.trim().length >= 1 && !selectedPatient) {
        handleSearch(searchQuery);
        setShowSuggestions(true);
      } else if (!searchQuery.trim()) {
        setPatients([]);
        setShowSuggestions(false);
      }
    }, 300); // 300ms 디바운스

    return () => clearTimeout(timeoutId);
  }, [searchQuery, selectedPatient, handleSearch]);

  // 환자 선택
  const handlePatientSelect = (patient: Patient) => {
    setSelectedPatient(patient);
    setSearchQuery(patient.name);
    setPatients([]);
    setShowSuggestions(false);
    setSelectedIndex(-1);
  };

  // 진료기록 생성
  const handleSubmit = async () => {
    if (!selectedPatient) {
      alert('환자를 선택해주세요.');
      return;
    }
    if (!department) {
      alert('진료과를 선택해주세요.');
      return;
    }

    setIsSubmitting(true);
    try {
      await apiRequest('POST', '/api/lung_cancer/api/medical-records/', {
        patient_id: selectedPatient.id,
        name: selectedPatient.name,
        department: department,
        notes: notes
      });
      
      alert('진료기록이 성공적으로 생성되었습니다.');
      
      // 폼 초기화
      setSelectedPatient(null);
      setSearchQuery('');
      setDepartment('');
      setNotes('');
    } catch (error: any) {
      console.error('진료기록 생성 오류:', error);
      alert(`진료기록 생성 중 오류가 발생했습니다: ${error.response?.data?.error || error.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  // 키보드 네비게이션
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!showSuggestions || patients.length === 0) {
      if (e.key === 'Enter') {
        handleSearch(searchQuery);
      }
      return;
    }

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(prev => 
          prev < patients.length - 1 ? prev + 1 : 0
        );
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(prev => 
          prev > 0 ? prev - 1 : patients.length - 1
        );
        break;
      case 'Enter':
        e.preventDefault();
        if (selectedIndex >= 0 && selectedIndex < patients.length) {
          handlePatientSelect(patients[selectedIndex]);
        }
        break;
      case 'Escape':
        setShowSuggestions(false);
        setSelectedIndex(-1);
        break;
    }
  };

  useEffect(() => {
    if (isCalendarOpen) {
      const d = new Date();
      d.setDate(1);
      d.setHours(0, 0, 0, 0);
      setCurrentMonth(d);
    }
  }, [isCalendarOpen]);

  const addMonths = (date: Date, months: number) => {
    const d = new Date(date);
    d.setMonth(d.getMonth() + months, 1);
    d.setHours(0, 0, 0, 0);
    return d;
  };

  const getMonthGridDays = (month: Date): Date[] => {
    const firstOfMonth = new Date(month);
    firstOfMonth.setDate(1);
    firstOfMonth.setHours(0, 0, 0, 0);
    const weekDay = firstOfMonth.getDay(); // 0(일)~6(토)
    const diffToMonday = (weekDay === 0 ? -6 : 1 - weekDay);
    const start = new Date(firstOfMonth);
    start.setDate(firstOfMonth.getDate() + diffToMonday);
    const days: Date[] = [];
    for (let i = 0; i < 42; i++) {
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
    if (!isToday) return '09:00';
    let hh = now.getHours();
    let mm = now.getMinutes();
    const rounded = Math.ceil(mm / 15) * 15;
    if (rounded === 60) { hh += 1; mm = 0; } else { mm = rounded; }
    if (hh < 9) hh = 9;
    if (hh > 17) hh = 17;
    if (hh === 17 && mm > 45) { hh = 17; mm = 45; }
    return `${String(hh).padStart(2,'0')}:${String(mm).padStart(2,'0')}`;
  };

  const openPatientPanelForDate = (date: Date) => {
    setSelectedDate(date);
    // 초기화
    setRSearchQuery('');
    setRPatients([]);
    setRSelectedIndex(-1);
    setRSelectedPatient(null);
    setRShowSuggestions(false);
    const t = initSuggestedTime(date);
    setSelectedTime(t);
    const [h, m] = t.split(':');
    setSelectedHour(h);
    setSelectedMinute(m);
    setReserveMemo('');
  };

  // 예약 전용 환자 검색
  const handleReserveSearch = useCallback(async (query: string) => {
    if (!query.trim()) {
      setRPatients([]);
      return;
    }
    setRIsLoading(true);
    try {
      const encodedQuery = encodeURIComponent(query.trim());
      const response = await apiRequest('GET', `/api/lung_cancer/api/medical-records/search_patients/?q=${encodedQuery}`);
      setRPatients(response.patients || []);
    } catch (error) {
      console.error('예약용 환자 검색 오류:', error);
      alert('예약용 환자 검색 중 오류가 발생했습니다.');
    } finally {
      setRIsLoading(false);
    }
  }, []);

  // 예약 검색 자동완성 디바운스
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

  // 예약 검색 키보드 네비게이션
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
    if (!selectedDate) {
      alert('날짜를 먼저 선택하세요.');
      return;
    }
    if (!rSelectedPatient) {
      alert('환자를 선택하세요.');
      return;
    }
    if (!selectedTime) {
      alert('시간을 선택하세요.');
      return;
    }
    const key = selectedDate.toISOString().slice(0, 10);
    // 동일 시간 예약 인원 제한: 최대 2명까지 허용
    const existing = reservations[key] || [];
    const sameTimeCount = existing.filter(r => r.time === selectedTime).length;
    if (sameTimeCount >= 2) {
      alert('해당 시간대는 이미 2명 예약되어 있습니다. 다른 시간을 선택하세요.');
      return;
    }
    setReservations(prev => {
      const list = prev[key] || [];
      return {
        ...prev,
        [key]: [...list, { id: rSelectedPatient.id, name: rSelectedPatient.name, time: selectedTime, memo: reserveMemo }]
      };
    });
    // 검색 상태만 초기화 (모달 유지)
    setRSearchQuery('');
    setRPatients([]);
    setRShowSuggestions(false);
    setRSelectedIndex(-1);
    setRSelectedPatient(null);
  };

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">진료 접수</h1>
        <p className="text-gray-600">환자를 검색하고 진료과를 선택하여 접수하세요.</p>
      </div>

      <div className="grid gap-6">
        {/* 예약검사 등록: 4주 캘린더 모달 트리거 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CalendarIcon className="w-5 h-5" />
              예약검사 등록
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Button onClick={() => setIsCalendarOpen(true)} data-testid="button-open-reservation-calendar">
              캘린더 열기
            </Button>
          </CardContent>
        </Card>

        {/* 환자 검색 */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="w-5 h-5" />
              환자 검색
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              <div className="flex-1 relative">
                <Label htmlFor="search">환자 이름</Label>
                <Input
                  id="search"
                  placeholder="환자 이름을 입력하세요"
                  value={searchQuery}
                  onChange={(e) => {
                    setSearchQuery(e.target.value);
                    if (selectedPatient) {
                      setSelectedPatient(null);
                    }
                    setSelectedIndex(-1);
                  }}
                  onKeyDown={handleKeyDown}
                  onFocus={() => {
                    if (patients.length > 0) {
                      setShowSuggestions(true);
                    }
                  }}
                  onBlur={() => {
                    // 약간의 지연을 두어 클릭 이벤트가 먼저 실행되도록 함
                    setTimeout(() => setShowSuggestions(false), 200);
                  }}
                  data-testid="input-patient-search"
                />
                
                {/* 자동완성 드롭다운 */}
                {showSuggestions && patients.length > 0 && (
                  <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-y-auto">
                    {patients.map((patient, index) => (
                      <div
                        key={patient.id}
                        className={`p-3 cursor-pointer border-b last:border-b-0 ${
                          index === selectedIndex 
                            ? 'bg-blue-50 border-blue-200' 
                            : 'hover:bg-gray-50'
                        }`}
                        onClick={() => handlePatientSelect(patient)}
                        data-testid={`patient-option-${patient.id}`}
                      >
                        <div className="flex items-center gap-3">
                          <User className="w-4 h-4 text-gray-500" />
                          <div>
                            <div className="font-medium">{patient.name}</div>
                            <div className="text-sm text-gray-500">
                              {patient.gender} | {patient.age}세 | {patient.phone}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <div className="flex items-end">
                <Button 
                  onClick={() => handleSearch(searchQuery)} 
                  disabled={isLoading || !searchQuery.trim()}
                  data-testid="button-search-patients"
                >
                  {isLoading ? '검색 중...' : '검색'}
                </Button>
              </div>
            </div>

            {/* 검색 결과 없음 메시지 */}
            {searchQuery.trim().length > 0 && patients.length === 0 && !isLoading && !showSuggestions && (
              <div className="p-4 text-center text-gray-500 border rounded-lg">
                '{searchQuery}'에 대한 검색 결과가 없습니다.
              </div>
            )}
          </CardContent>
        </Card>

        {/* 선택된 환자 정보 */}
        {selectedPatient && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <User className="w-5 h-5" />
                선택된 환자
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="bg-blue-50 p-4 rounded-lg">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label className="text-sm font-medium text-gray-600">환자명</Label>
                    <div className="text-lg font-semibold">{selectedPatient.name}</div>
                  </div>
                  <div>
                    <Label className="text-sm font-medium text-gray-600">환자번호</Label>
                    <div className="text-lg font-semibold">{selectedPatient.id}</div>
                  </div>
                  <div>
                    <Label className="text-sm font-medium text-gray-600">성별/나이</Label>
                    <div className="text-lg font-semibold">{selectedPatient.gender} / {selectedPatient.age}세</div>
                  </div>
                  <div>
                    <Label className="text-sm font-medium text-gray-600">전화번호</Label>
                    <div className="text-lg font-semibold">{selectedPatient.phone}</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* 진료 정보 입력 */}
        {selectedPatient && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Stethoscope className="w-5 h-5" />
                진료 정보
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="department">진료과</Label>
                <Select value={department} onValueChange={setDepartment}>
                  <SelectTrigger>
                    <SelectValue placeholder="진료과를 선택하세요" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="호흡기내과">호흡기내과</SelectItem>
                    <SelectItem value="외과">외과</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="notes">메모 (선택사항)</Label>
                <textarea
                  id="notes"
                  className="w-full p-3 border border-gray-300 rounded-md resize-none"
                  rows={3}
                  placeholder="진료 관련 메모를 입력하세요"
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                />
              </div>

              <div className="flex gap-2 pt-4">
                <Button 
                  onClick={handleSubmit}
                  disabled={isSubmitting || !department}
                  className="flex-1"
                  data-testid="button-submit-registration"
                >
                  {isSubmitting ? '접수 중...' : '접수하기'}
                </Button>
                <Button 
                  variant="outline" 
                  onClick={() => {
                    setSelectedPatient(null);
                    setSearchQuery('');
                    setDepartment('');
                    setNotes('');
                  }}
                  data-testid="button-reset-form"
                >
                  초기화
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* 4주 캘린더 모달 */}
      <Dialog open={isCalendarOpen} onOpenChange={setIsCalendarOpen}>
        <DialogContent className="max-w-5xl">
          <DialogHeader>
            <DialogTitle>예약 날짜 선택</DialogTitle>
          </DialogHeader>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Left: Calendar */}
            <div className="md:col-span-2">
              <div className="flex items-center justify-between mb-2">
                <Button variant="outline" size="icon" onClick={() => setCurrentMonth(prev => addMonths(prev, -1))}>
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <div className="text-sm font-medium">
                  {`${currentMonth.getFullYear()}년 ${String(currentMonth.getMonth() + 1).padStart(2, '0')}월`}
                </div>
                <Button variant="outline" size="icon" onClick={() => setCurrentMonth(prev => addMonths(prev, 1))}>
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
              <div className="grid grid-cols-7 gap-2">
                {['월','화','수','목','금','토','일'].map((d) => (
                  <div key={d} className="text-center text-xs text-gray-500 py-1">{d}</div>
                ))}
                {getMonthGridDays(currentMonth).map((d) => {
                  const isToday = new Date().toDateString() === d.toDateString();
                  const isSelected = selectedDate && selectedDate.toDateString() === d.toDateString();
                  const isOutside = d.getMonth() !== currentMonth.getMonth();
                  const key = d.toISOString().slice(0,10);
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
                      } ${isOutside ? 'opacity-40' : ''} disabled:opacity-50 disabled:cursor-not-allowed`}
                      disabled={startOfDay(d).getTime() < startOfDay(new Date()).getTime()}
                      onClick={() => {
                        if (startOfDay(d).getTime() < startOfDay(new Date()).getTime()) return;
                        openPatientPanelForDate(d);
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
              <div className="mt-4 border-t pt-3">
                <div className="text-sm font-medium text-gray-700 mb-2">해당 날짜 예약자</div>
                {(() => {
                  if (!selectedDate) return <div className="text-sm text-gray-500">날짜를 선택하세요</div>;
                  const key = selectedDate.toISOString().slice(0,10);
                  const list = [...(reservations[key] || [])].sort((a,b)=>a.time.localeCompare(b.time));
                  if (list.length === 0) return <div className="text-sm text-gray-500">예약자가 없습니다</div>;
                  return (
                    <div className="space-y-2">
                      {list.map((p, idx) => (
                        <div key={`${p.id}-${p.time}-${idx}`} className="text-sm text-gray-800">- {p.name} (<span className="text-xs text-gray-500">{p.id}</span>) · {p.time}</div>
                      ))}
                    </div>
                  );
                })()}
              </div>
            </div>

            {/* Right: Patient search panel */}
            <div className="md:col-span-1">
              <div className="space-y-3">
                {!selectedDate ? (
                  <div className="h-full flex items-center justify-center border rounded-md p-6 text-sm text-gray-500">
                    날짜를 클릭하면 환자 검색창이 여기에 표시됩니다.
                  </div>
                ) : (
                  <>
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
                              <div className="flex items-center gap-3">
                                <User className="w-4 h-4 text-gray-500" />
                                <div>
                                  <div className="font-medium">{p.name}</div>
                                  <div className="text-sm text-gray-500">{p.gender} | {p.age}세 | {p.phone}</div>
                                </div>
                              </div>
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
                      <Button className="w-full" disabled={!selectedDate || !rSelectedPatient || !selectedTime || rIsLoading} onClick={handleReserveSubmit}>
                        {rIsLoading ? '처리 중...' : '이 날짜로 예약 등록'}
                      </Button>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default MedicalRegistration;
