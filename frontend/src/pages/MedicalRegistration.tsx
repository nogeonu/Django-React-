import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search, User, Stethoscope } from "lucide-react";
import { apiRequest } from "@/lib/api";

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
      const response = await apiRequest('GET', `/api/lung_cancer/medical-records/search_patients/?q=${encodedQuery}`);
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
      await apiRequest('POST', '/api/lung_cancer/medical-records/', {
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

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">진료 접수</h1>
        <p className="text-gray-600">환자를 검색하고 진료과를 선택하여 접수하세요.</p>
      </div>

      <div className="grid gap-6">
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
    </div>
  );
};

export default MedicalRegistration;
