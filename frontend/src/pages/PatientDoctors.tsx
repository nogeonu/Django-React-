import { useMemo, useState } from "react";
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
import { Search, CalendarDays, Heart } from "lucide-react";

interface Doctor {
  id: string;
  name: string;
  title: string;
  department: string;
  specialties: string[];
  description: string;
  schedule: string;
}

const DOCTORS: Doctor[] = [
  {
    id: "resp-01",
    name: "이상철",
    title: "교수",
    department: "호흡기내과",
    specialties: ["고지혈증", "심근질환", "심장판막증", "협심증"],
    description:
      "심혈관 질환과 콜레스테롤 질환의 통합 관리에 주력하고 있습니다.",
    schedule: "월/수/금 오전, 화 오후",
  },
  {
    id: "resp-02",
    name: "이경호",
    title: "교수",
    department: "호흡기내과",
    specialties: ["만성신질환", "신장이식", "고혈압성 신질환"],
    description: "만성 신장질환 및 합병증 치료, 신장이식 사후 관리 전문.",
    schedule: "화/목 오전, 금 오후",
  },
  {
    id: "surg-01",
    name: "김도윤",
    title: "교수",
    department: "외과",
    specialties: ["간담췌외과", "담낭염", "간암", "췌장암"],
    description: "간담췌 암 환자의 로봇 수술과 맞춤형 항암 치료를 담당합니다.",
    schedule: "월/화/목 오전, 수 오후",
  },
  {
    id: "surg-02",
    name: "박서현",
    title: "교수",
    department: "외과",
    specialties: ["유방외과", "갑상선", "유방암", "갑상선결절"],
    description:
      "유방암·갑상선 질환 환자의 수술 및 재활 치료를 중점 관리합니다.",
    schedule: "월/수/금 오후, 화 오전",
  },
];

const DEPARTMENTS = [
  { id: "호흡기내과", label: "호흡기내과" },
  { id: "외과", label: "외과" },
];

function getInitials(name: string) {
  return name
    .split(" ")
    .map((part) => part.charAt(0))
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

export default function PatientDoctors() {
  const [department, setDepartment] = useState<string>(DEPARTMENTS[0].id);
  const [searchTerm, setSearchTerm] = useState("");

  const doctors = useMemo(() => {
    const lower = searchTerm.trim().toLowerCase();
    return DOCTORS.filter(
      (doctor) =>
        doctor.department === department &&
        (lower.length === 0 ||
          doctor.name.toLowerCase().includes(lower) ||
          doctor.specialties.some((spec) =>
            spec.toLowerCase().includes(lower),
          )),
    );
  }, [department, searchTerm]);

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
              진료과를 선택하고 전문 의료진의 진료 분야와 일정을 확인하세요.
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
              호흡기내과와 외과 의료진 정보를 기반으로 예약 상담을 도와드립니다.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 py-6 md:grid-cols-[240px,1fr] md:items-center">
            <div className="space-y-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
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
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                의료진 또는 질환 검색
              </span>
              <div className="flex items-center gap-2">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                  <Input
                    className="pl-9"
                    placeholder="의료진 이름 또는 질환명을 검색해 주세요."
                    value={searchTerm}
                    onChange={(event) => setSearchTerm(event.target.value)}
                  />
                </div>
                <Button
                  variant="outline"
                  className="hidden border-primary text-primary hover:bg-primary/10 md:inline-flex"
                >
                  검색
                </Button>
              </div>
              <p className="text-xs text-slate-500">
                예) 천식, 고혈압, 유방암, 신장이식 등 관심 있는 질환명으로
                검색해 보세요.
              </p>
            </div>
          </CardContent>
        </Card>

        <section className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-slate-800">
              {selectedDepartment?.label ?? ""} 의료진 ({doctors.length}명)
            </h2>
            <Button asChild variant="ghost" size="sm" className="text-primary">
              <Link to="#appointments">진료 예약 안내 보기</Link>
            </Button>
          </div>

          {doctors.length === 0 ? (
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
              {doctors.map((doctor) => (
                <Card
                  key={doctor.id}
                  className="border-slate-200 bg-white shadow-sm"
                >
                  <CardHeader className="flex flex-row items-center gap-4 border-b bg-white/70">
                    <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 text-lg font-semibold text-primary">
                      {getInitials(doctor.name)}
                    </div>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <CardTitle className="text-lg font-semibold text-slate-800">
                          {doctor.name} {doctor.title}
                        </CardTitle>
                        <Badge
                          variant="secondary"
                          className="border border-primary/30 bg-primary/10 text-primary"
                        >
                          {doctor.department}
                        </Badge>
                      </div>
                      <p className="text-xs font-medium text-slate-500">
                        전문 진료 분야
                      </p>
                      <p className="text-sm text-slate-600">
                        {doctor.specialties.join(", ")}
                      </p>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4 py-6 text-sm text-slate-600">
                    <p>{doctor.description}</p>
                    <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
                      <CalendarDays className="h-4 w-4 text-primary" />
                      <span>진료 일정: {doctor.schedule}</span>
                    </div>
                    <div className="flex flex-wrap items-center gap-3 pt-2">
                      <Button variant="outline" size="sm" className="gap-2">
                        <Search className="h-4 w-4" /> 상세소개
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
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
