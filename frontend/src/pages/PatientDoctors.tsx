import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { CalendarDays, Heart, Mail, Search, User } from "lucide-react";
import { getDoctorsApi } from "@/lib/api";

interface Doctor {
  id: number;
  username: string;
  email: string;
  first_name?: string;
  last_name?: string;
  department: string;
  doctor_id?: string | null;
}

const DEPARTMENTS = [
  { id: "respiratory", label: "호흡기내과" },
  { id: "surgery", label: "외과" },
];

const DEPARTMENT_LABELS: Record<string, string> = {
  respiratory: "호흡기내과",
  surgery: "외과",
};

function getDisplayName(doctor: Doctor) {
  const hasFirst = Boolean(doctor.first_name && doctor.first_name.trim());
  const hasLast = Boolean(doctor.last_name && doctor.last_name.trim());

  if (hasFirst && hasLast) {
    const first = doctor.first_name!.trim();
    const last = doctor.last_name!.trim();
    const containsKorean = /[ㄱ-ㅎ가-힣]/.test(first + last);
    if (containsKorean) {
      return `${last}${first}`; // 한국식: 성 + 이름 (공백 없이)
    }
    return `${first} ${last}`; // 영문 등은 First Last
  }

  if (hasFirst) {
    return doctor.first_name!.trim();
  }

  if (hasLast) {
    return doctor.last_name!.trim();
  }

  return doctor.username;
}

function getInitials(name: string) {
  return name
    .split(/\s+/)
    .map((part) => part.charAt(0))
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

export default function PatientDoctors() {
  const [department, setDepartment] = useState<string>(DEPARTMENTS[0].id);
  const [searchTerm, setSearchTerm] = useState("");
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDoctors = async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await getDoctorsApi(department);
        setDoctors(data.doctors ?? []);
      } catch (err) {
        console.error("의료진 조회 오류", err);
        setError(
          "의료진 정보를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.",
        );
        setDoctors([]);
      } finally {
        setLoading(false);
      }
    };

    fetchDoctors();
  }, [department]);

  const filteredDoctors = useMemo(() => {
    const keyword = searchTerm.trim().toLowerCase();
    if (!keyword) {
      return doctors;
    }
    return doctors.filter((doctor) => {
      const displayName = getDisplayName(doctor).toLowerCase();
      return (
        displayName.includes(keyword) ||
        (doctor.email ?? "").toLowerCase().includes(keyword) ||
        (doctor.doctor_id ?? "").toLowerCase().includes(keyword)
      );
    });
  }, [doctors, searchTerm]);

  const selectedDepartment = DEPARTMENTS.find((dept) => dept.id === department);

  return (
    <div className="min-h-screen bg-slate-50 pb-16">
      <header className="border-b bg-white">
        <div className="mx-auto flex max-w-5xl flex-col gap-3 px-6 py-8 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.35em] text-primary">
              MEDICAL STAFF
            </p>
            <h1 className="text-2xl font-bold text-slate-800">
              진료과 · 의료진 검색
            </h1>
            <p className="text-sm text-slate-500">
              진료과를 선택하고 전문 의료진의 프로필과 연락처를 확인하세요.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
            <Link
              to="/"
              className="rounded-full border border-slate-200 px-3 py-1.5 font-semibold text-slate-600 hover:border-primary/40 hover:text-primary"
            >
              환자 홈으로 돌아가기
            </Link>
          </div>
        </div>
      </header>

      <main className="mx-auto mt-10 max-w-5xl space-y-8 px-6">
        <Card>
          <CardHeader className="border-b bg-white/80">
            <CardTitle className="text-base font-semibold text-slate-800">
              원하는 진료과와 의료진을 찾아보세요
            </CardTitle>
            <CardDescription>
              호흡기내과와 외과에 소속된 의료진 정보를 실시간으로 확인할 수
              있습니다.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 py-6 md:grid-cols-[240px,1fr] md:items-center">
            <div className="space-y-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-700">
                진료과 선택
              </span>
              <Select value={department} onValueChange={setDepartment}>
                <SelectTrigger className="bg-white">
                  <SelectValue placeholder="진료과를 선택하세요" />
                </SelectTrigger>
                <SelectContent>
                  {DEPARTMENTS.map((dept) => (
                    <SelectItem key={dept.id} value={dept.id}>
                      {dept.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {selectedDepartment && (
                <p className="text-xs text-slate-500">
                  {selectedDepartment.label} 전문 의료진과 상담 가능 일정을
                  확인할 수 있습니다.
                </p>
              )}
            </div>
            <div className="space-y-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-700">
                의료진 검색
              </span>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                <Input
                  className="pl-9"
                  placeholder="의료진 이름, 이메일 또는 doctor_id로 검색하세요."
                  value={searchTerm}
                  onChange={(event) => setSearchTerm(event.target.value)}
                />
              </div>
              <p className="text-xs text-slate-500">
                예) 홍길동, resp001, doctor@example.com 등 키워드로 검색해
                보세요.
              </p>
            </div>
          </CardContent>
        </Card>

        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <section className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-slate-800">
              {selectedDepartment?.label ?? ""} 의료진 ({filteredDoctors.length}
              명)
            </h2>
            <Button asChild variant="ghost" size="sm" className="text-primary">
              <Link to="#appointments">진료 예약 안내 보기</Link>
            </Button>
          </div>

          {loading ? (
            <Card className="border-dashed border-slate-200 bg-white/70 text-center text-slate-600">
              <CardContent className="space-y-4 py-12">
                <Search className="mx-auto h-10 w-10 animate-spin text-primary/60" />
                <p className="text-base font-semibold">
                  의료진 정보를 불러오는 중입니다.
                </p>
              </CardContent>
            </Card>
          ) : filteredDoctors.length === 0 ? (
            <Card className="border-dashed border-slate-200 bg-white/70 text-center text-slate-600">
              <CardContent className="space-y-4 py-12">
                <Search className="mx-auto h-10 w-10 text-primary/60" />
                <p className="text-base font-semibold">검색 결과가 없습니다.</p>
                <p className="text-sm text-slate-500">
                  검색어를 다르게 입력하거나 다른 진료과를 선택해 주세요.
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-6 md:grid-cols-2">
              {filteredDoctors.map((doctor) => {
                const name = getDisplayName(doctor);
                const departmentLabel =
                  DEPARTMENT_LABELS[doctor.department] ?? doctor.department;
                return (
                  <Card
                    key={doctor.id}
                    className="border-slate-200 bg-white shadow-sm"
                  >
                    <CardHeader className="flex flex-row items-center gap-4 border-b bg-white/70">
                      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 text-lg font-semibold text-primary">
                        {getInitials(name)}
                      </div>
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <CardTitle className="text-lg font-semibold text-slate-800">
                            {name}
                          </CardTitle>
                          <Badge
                            variant="secondary"
                            className="border border-primary/30 bg-primary/10 text-primary"
                          >
                            {departmentLabel}
                          </Badge>
                        </div>
                        <p className="text-xs text-slate-500">
                          Doctor ID: {doctor.doctor_id ?? "발급 준비중"}
                        </p>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4 py-6 text-sm text-slate-600">
                      <div className="flex items-center gap-2 text-xs text-slate-500">
                        <User className="h-4 w-4 text-primary" />
                        <span>아이디: {doctor.username}</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-slate-500">
                        <Mail className="h-4 w-4 text-primary" />
                        <span>이메일: {doctor.email || "-"}</span>
                      </div>
                      <div className="flex flex-wrap items-center gap-3 pt-2">
                        <Button variant="outline" size="sm" className="gap-2">
                          <Search className="h-4 w-4" /> 상세소개 준비중
                        </Button>
                        <Button
                          size="sm"
                          className="gap-2 bg-primary text-white hover:bg-primary/90"
                        >
                          <CalendarDays className="h-4 w-4" /> 진료예약 문의
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="gap-2 text-primary hover:bg-primary/10"
                        >
                          <Heart className="h-4 w-4" /> 감사해요
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
