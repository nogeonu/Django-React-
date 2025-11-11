import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/context/AuthContext";
import { getPatientProfile, updatePatientProfile } from "@/lib/api";

type ProfileState = {
  name: string;
  birth_date: string;
  gender: string;
  phone: string;
  blood_type: string;
  emergency_contact: string;
  address: string;
  medical_history: string;
  allergies: string;
};

const initialProfile: ProfileState = {
  name: "",
  birth_date: "",
  gender: "",
  phone: "",
  blood_type: "",
  emergency_contact: "",
  address: "",
  medical_history: "",
  allergies: "",
};

function PatientMyPage() {
  const { patientUser, setPatientUser } = useAuth();
  const { toast } = useToast();
  const navigate = useNavigate();
  const [profile, setProfile] = useState<ProfileState>(initialProfile);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);

  const genderOptions = useMemo(
    () => [
      { value: "M", label: "남성" },
      { value: "F", label: "여성" },
    ],
    [],
  );

  const bloodTypeOptions = useMemo(
    () => ["A", "B", "AB", "O", "RH+", "RH-"],
    [],
  );

  useEffect(() => {
    if (!patientUser) {
      return;
    }
    setLoading(true);
    getPatientProfile(patientUser.account_id)
      .then((data) => {
        setProfile({
          name: data.name ?? "",
          birth_date: data.birth_date ?? "",
          gender: data.gender ?? "",
          phone: data.phone ?? "",
          blood_type: data.blood_type ?? "",
          emergency_contact: data.emergency_contact ?? "",
          address: data.address ?? "",
          medical_history: data.medical_history ?? "",
          allergies: data.allergies ?? "",
        });
      })
      .catch(() => {
        toast({
          title: "정보를 불러오지 못했습니다.",
          description: "잠시 후 다시 시도해주세요.",
          variant: "destructive",
        });
      })
      .finally(() => setLoading(false));
  }, [patientUser, toast]);

  const handleChange = (key: keyof ProfileState) => (value: string) => {
    setProfile((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!patientUser) {
      toast({
        title: "로그인이 필요합니다.",
        description: "환자 로그인을 먼저 진행해주세요.",
        variant: "destructive",
      });
      navigate("/patient/login");
      return;
    }

    setSaving(true);
    try {
      const payload = {
        ...profile,
        birth_date: profile.birth_date || null,
        gender: profile.gender || null,
        blood_type: profile.blood_type || null,
      };
      const updated = await updatePatientProfile(patientUser.account_id, payload);
      setProfile((prev) => ({
        ...prev,
        name: updated.name ?? prev.name,
        phone: updated.phone ?? prev.phone,
      }));
      setPatientUser({
        ...patientUser,
        name: updated.name ?? patientUser.name,
        phone: updated.phone ?? patientUser.phone,
      });
      toast({
        title: "저장되었습니다.",
        description: "환자 정보가 안전하게 업데이트되었습니다.",
      });
    } catch (error: any) {
      const message = error?.response?.data?.detail ?? "저장 중 오류가 발생했습니다.";
      toast({
        title: "저장 실패",
        description: message,
        variant: "destructive",
      });
    } finally {
      setSaving(false);
    }
  };

  if (!patientUser) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-50 px-4">
        <Card className="max-w-md p-6 text-center">
          <CardHeader>
            <CardTitle className="text-xl font-semibold">환자 로그인 필요</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm text-slate-600">
            <p>환자 마이페이지에 접근하려면 먼저 로그인해주세요.</p>
            <Button asChild className="w-full">
              <Link to="/patient/login">환자 로그인 이동</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-50 pb-16">
      <header className="border-b bg-white">
        <div className="mx-auto flex max-w-5xl flex-col gap-2 px-6 py-6 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-primary">MY PAGE</p>
            <h1 className="text-2xl font-bold text-slate-800">{patientUser.name}님 환자 정보 관리</h1>
            <p className="text-sm text-slate-500">필요한 정보를 최신 상태로 유지해주세요.</p>
          </div>
          <div className="text-xs text-slate-500">
            환자 번호 <span className="font-semibold text-slate-700">{patientUser.patient_id}</span>
          </div>
        </div>
      </header>

      <main className="mx-auto mt-10 max-w-5xl px-6">
        <Card className="border-2 border-primary/10 shadow-sm">
          <CardHeader className="border-b bg-white/80">
            <CardTitle className="text-lg font-semibold text-slate-800">기본 정보</CardTitle>
            <p className="text-sm text-slate-500">온라인 예약과 안내를 위해 정확한 정보를 입력해주세요.</p>
          </CardHeader>
          <CardContent className="px-6 py-8">
            {loading ? (
              <div className="py-10 text-center text-sm text-slate-500">정보를 불러오는 중입니다...</div>
            ) : (
              <form className="space-y-8" onSubmit={handleSubmit}>
                <div className="grid gap-6 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="name">환자명 *</Label>
                    <Input
                      id="name"
                      value={profile.name}
                      onChange={(event) => handleChange("name")(event.target.value)}
                      placeholder="환자명을 입력하세요"
                      required
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="birth_date">생년월일 *</Label>
                    <Input
                      id="birth_date"
                      type="date"
                      value={profile.birth_date}
                      onChange={(event) => handleChange("birth_date")(event.target.value)}
                      placeholder="연도. 월. 일을 선택하세요."
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>성별 *</Label>
                    <Select value={profile.gender} onValueChange={handleChange("gender")}>
                      <SelectTrigger>
                        <SelectValue placeholder="성별을 선택하세요" />
                      </SelectTrigger>
                      <SelectContent>
                        {genderOptions.map((option) => (
                          <SelectItem key={option.value} value={option.value}>
                            {option.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="phone">전화번호 *</Label>
                    <Input
                      id="phone"
                      value={profile.phone}
                      onChange={(event) => handleChange("phone")(event.target.value)}
                      placeholder="전화번호를 입력하세요"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>혈액형</Label>
                    <Select value={profile.blood_type} onValueChange={handleChange("blood_type")}>
                      <SelectTrigger>
                        <SelectValue placeholder="혈액형을 선택하세요" />
                      </SelectTrigger>
                      <SelectContent>
                        {bloodTypeOptions.map((type) => (
                          <SelectItem key={type} value={type}>
                            {type}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="emergency_contact">응급연락처</Label>
                    <Input
                      id="emergency_contact"
                      value={profile.emergency_contact}
                      onChange={(event) => handleChange("emergency_contact")(event.target.value)}
                      placeholder="응급연락처를 입력하세요"
                    />
                  </div>
                </div>

                <div className="grid gap-6 md:grid-cols-2">
                  <div className="space-y-2 md:col-span-2">
                    <Label htmlFor="address">주소</Label>
                    <Textarea
                      id="address"
                      value={profile.address}
                      onChange={(event) => handleChange("address")(event.target.value)}
                      placeholder="주소를 입력하세요"
                      rows={3}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="medical_history">과거 병력</Label>
                    <Textarea
                      id="medical_history"
                      value={profile.medical_history}
                      onChange={(event) => handleChange("medical_history")(event.target.value)}
                      placeholder="과거 병력을 입력하세요"
                      rows={3}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="allergies">알레르기</Label>
                    <Textarea
                      id="allergies"
                      value={profile.allergies}
                      onChange={(event) => handleChange("allergies")(event.target.value)}
                      placeholder="알레르기 정보를 입력하세요"
                      rows={3}
                    />
                  </div>
                </div>

                <div className="flex items-center justify-end gap-3">
                  <Button type="button" variant="outline" onClick={() => navigate(-1)}>
                    취소
                  </Button>
                  <Button type="submit" disabled={saving}>
                    {saving ? "저장 중..." : "정보 저장"}
                  </Button>
                </div>
              </form>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  );
}

export default PatientMyPage;

