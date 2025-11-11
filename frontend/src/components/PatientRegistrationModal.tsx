import { useState, useEffect } from "react";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { apiRequest } from "@/lib/api";

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

interface PatientRegistrationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  patient?: Patient | null;
  isEdit?: boolean;
}

interface PatientFormData {
  name: string;
  birth_date: string;
  gender: string;
  phone: string;
  address: string;
  emergency_contact: string;
  blood_type: string;
}

export default function PatientRegistrationModal({ isOpen, onClose, onSuccess, patient, isEdit = false }: PatientRegistrationModalProps) {
  const [formData, setFormData] = useState<PatientFormData>({
    name: "",
    birth_date: "",
    gender: "",
    phone: "",
    address: "",
    emergency_contact: "",
    blood_type: "",
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  // 수정 모드일 때 기존 환자 데이터로 폼 초기화
  useEffect(() => {
    if (isEdit && patient) {
      setFormData({
        name: patient.name || "",
        birth_date: patient.birth_date || "",
        gender: patient.gender === "M" ? "남성" : patient.gender === "F" ? "여성" : patient.gender || "",
        phone: patient.phone || "",
        address: patient.address || "",
        emergency_contact: patient.emergency_contact || "",
        blood_type: patient.blood_type || "",
      });
    } else {
      // 등록 모드일 때 폼 초기화
      setFormData({
        name: "",
        birth_date: "",
        gender: "",
        phone: "",
        address: "",
        emergency_contact: "",
        blood_type: "",
      });
    }
  }, [isEdit, patient]);

  const handleInputChange = (field: keyof PatientFormData, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");

    try {
      console.log(`${isEdit ? '환자 수정' : '환자 등록'} 요청 데이터:`, formData);
      const payload = {
        ...formData,
        gender: formData.gender === "남성" ? "M" : formData.gender === "여성" ? "F" : formData.gender,
      };

      let result;
      if (isEdit && patient) {
        // 수정 모드: PUT 요청
        result = await apiRequest("PUT", `/api/lung_cancer/patients/${patient.id}/`, payload);
      } else {
        // 등록 모드: POST 요청
        result = await apiRequest("POST", "/api/lung_cancer/patients/register/", payload);
      }
      
      console.log("응답 데이터:", result);
      
      // apiRequest는 성공 시 data만 반환하므로 여기까지 오면 성공으로 간주
      alert(`환자가 성공적으로 ${isEdit ? '수정' : '등록'}되었습니다!${!isEdit ? `\n환자 ID: ${result?.patient_id ?? ''}` : ''}`);
      onSuccess();
      onClose();
      setFormData({
        name: "",
        birth_date: "",
        gender: "",
        phone: "",
        address: "",
        emergency_contact: "",
        blood_type: "",
      });
    } catch (err: any) {
      console.error("네트워크 오류 상세:", err);
      if (err.response) {
        // 서버에서 응답을 받았지만 오류 상태
        console.error("서버 응답 오류:", err.response.status, err.response.data);
        setError(`서버 오류 (${err.response.status}): ${err.response.data?.error || err.response.data?.message || "알 수 없는 오류"}`);
      } else if (err.request) {
        // 요청은 보냈지만 응답을 받지 못함
        console.error("네트워크 요청 실패:", err.request);
        setError("서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.");
      } else {
        // 기타 오류
        console.error("기타 오류:", err.message);
        setError(`오류가 발생했습니다: ${err.message}`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold">{isEdit ? '환자 정보 수정' : '환자 등록'}</h2>
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="h-8 w-8 p-0"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                {error}
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="name">환자명 *</Label>
                <Input
                  id="name"
                  value={formData.name}
                  onChange={(e) => handleInputChange("name", e.target.value)}
                  placeholder="환자명을 입력하세요"
                  required
                />
              </div>

              <div>
                <Label htmlFor="birth_date">생년월일 *</Label>
                <Input
                  id="birth_date"
                  type="date"
                  value={formData.birth_date}
                  onChange={(e) => handleInputChange("birth_date", e.target.value)}
                  required
                />
              </div>

              <div>
                <Label htmlFor="gender">성별 *</Label>
                <Select value={formData.gender} onValueChange={(value) => handleInputChange("gender", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="성별을 선택하세요" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="남성">남성</SelectItem>
                    <SelectItem value="여성">여성</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="phone">전화번호</Label>
                <Input
                  id="phone"
                  value={formData.phone}
                  onChange={(e) => handleInputChange("phone", e.target.value)}
                  placeholder="전화번호를 입력하세요"
                />
              </div>

              <div>
                <Label htmlFor="blood_type">혈액형</Label>
                <Select value={formData.blood_type} onValueChange={(value) => handleInputChange("blood_type", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="혈액형을 선택하세요" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="A+">A+</SelectItem>
                    <SelectItem value="A-">A-</SelectItem>
                    <SelectItem value="B+">B+</SelectItem>
                    <SelectItem value="B-">B-</SelectItem>
                    <SelectItem value="AB+">AB+</SelectItem>
                    <SelectItem value="AB-">AB-</SelectItem>
                    <SelectItem value="O+">O+</SelectItem>
                    <SelectItem value="O-">O-</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label htmlFor="emergency_contact">응급연락처</Label>
                <Input
                  id="emergency_contact"
                  value={formData.emergency_contact}
                  onChange={(e) => handleInputChange("emergency_contact", e.target.value)}
                  placeholder="응급연락처를 입력하세요"
                />
              </div>
            </div>

            <div>
              <Label htmlFor="address">주소</Label>
              <Textarea
                id="address"
                value={formData.address}
                onChange={(e) => handleInputChange("address", e.target.value)}
                placeholder="주소를 입력하세요"
                rows={3}
              />
            </div>

            <div className="flex justify-end space-x-2 pt-4">
              <Button type="button" variant="outline" onClick={onClose}>
                취소
              </Button>
              <Button type="submit" disabled={isLoading}>
                {isLoading ? "등록 중..." : "환자 등록"}
              </Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
