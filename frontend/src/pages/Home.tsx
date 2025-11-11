import { Link } from "react-router-dom";
import heroImage from "@/assets/doctor-bg.jpg";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Calendar,
  Camera,
  Clock,
  Info,
  MapPin,
  ParkingCircle,
  PhoneCall,
  Pill,
  Stethoscope,
} from "lucide-react";

const navItems = [
  { label: "예약 / 조회 / 발급", link: "#reservations" },
  { label: "의료진 / 진료과", link: "#doctors" },
  { label: "이용안내", link: "#guide" },
  { label: "건강정보", link: "#health" },
  { label: "병원소개", link: "#about" },
];

const heroQuickLinks = [
  { icon: MapPin, label: "찾아오시는길", link: "#map" },
  { icon: PhoneCall, label: "전화번호 안내", link: "#contact" },
  { icon: Info, label: "의무기록발급", link: "#records" },
  { icon: Calendar, label: "증명서발급", link: "#certificate" },
  { icon: Camera, label: "홍보영상", link: "#promo" },
  { icon: Stethoscope, label: "의료진 소개", link: "#doctors" },
  { icon: Clock, label: "진료시간 안내", link: "#schedule" },
  { icon: Pill, label: "검진센터", link: "#checkup" },
];

const quickCallouts = [
  {
    title: "전화예약(초진)",
    content: "051-797-3500",
    highlight: true,
  },
  {
    title: "진료과 · 의료진 검색",
    content: "전문의 정보를 확인하고 예약하세요",
    link: "#doctors",
  },
  {
    title: "온라인 진료예약",
    content: "모바일로 빠르게 진료 예약하기",
    link: "#appointments",
  },
  {
    title: "진료예약 조회",
    content: "예약 내역과 대기 순번을 실시간 확인",
    link: "#appointments",
  },
  {
    title: "비자(VISA)검진 예약",
    content: "국제 환자를 위한 전문 검진센터",
    link: "#visa",
  },
];

const quickServices = [
  {
    icon: MapPin,
    title: "병원 길찾기",
    description: "층별 실시간 지도와 시설 안내",
    link: "#map",
  },
  {
    icon: Stethoscope,
    title: "진료과 · 의료진 검색",
    description: "전문의 정보를 확인하고 예약하세요",
    link: "#doctors",
  },
  {
    icon: Calendar,
    title: "온라인 진료예약",
    description: "모바일로 빠르게 진료 예약하기",
    link: "#appointments",
  },
  {
    icon: Clock,
    title: "대기 순번 확인",
    description: "현재 진료 대기 현황 실시간 확인",
    link: "#waiting",
  },
  {
    icon: Pill,
    title: "약국 위치 안내",
    description: "원내·주변 약국 위치와 운영 정보",
    link: "#pharmacy",
  },
  {
    icon: ParkingCircle,
    title: "주차장 안내",
    description: "주차 가능 구역과 혼잡도 정보",
    link: "#parking",
  },
  {
    icon: Camera,
    title: "AR 내비게이션",
    description: "증강현실로 병원 내부 안내를 받아보세요",
    link: "#ar",
  },
];

const notices = [
  {
    title: "2025년 독감 예방접종 안내",
    content: "10월부터 12월까지 예방접종 클리닉에서 접종 가능합니다.",
  },
  {
    title: "면회 시간 안내",
    content: "환자 안전을 위해 면회는 매일 18:00~20:00에만 가능합니다.",
  },
  {
    title: "주차장 이용 변경",
    content: "신관 주차장이 증축 공사로 인해 임시 폐쇄됩니다.",
  },
];

function Home() {
  return (
    <div className="min-h-screen bg-slate-50">
      <header>
        <div className="border-b bg-slate-100">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-2 text-xs text-slate-600">
            <div className="flex items-center gap-2">
              <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-semibold text-primary">
                NOTICE
              </span>
              <span>방문인원은 환자와 나의 건강을 위하여 자제하여 주세요.</span>
            </div>
            <div className="flex items-center gap-4 font-semibold">
              <Link to="/patient/login" className="hover:text-primary">
                로그인
              </Link>
              <Link to="/patient/signup" className="hover:text-primary">
                회원가입
              </Link>
              <Link
                to="/login"
                className="rounded-full bg-primary px-3 py-1 text-[11px] text-white hover:bg-primary/90"
              >
                의료진 플랫폼
              </Link>
            </div>
          </div>
        </div>
        <div className="border-b bg-white">
          <div className="mx-auto flex max-w-6xl flex-col gap-4 px-6 py-4 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-6">
              <div>
                <p className="text-xs font-semibold uppercase tracking-tight text-primary">
                  Konyang University Hospital
                </p>
                <h1 className="text-xl font-bold text-slate-900">
                  건양대학교 병원
                </h1>
              </div>
            </div>
            <nav className="flex flex-wrap items-center gap-4 text-sm font-semibold text-slate-600 lg:gap-6">
              {navItems.map((item) => (
                <Link key={item.label} to={item.link} className="hover:text-primary">
                  {item.label}
                </Link>
              ))}
            </nav>
          </div>
        </div>
      </header>

      <main>
        <section
          className="relative overflow-hidden"
          style={{
            backgroundImage: `linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.45)), url(${heroImage})`,
            backgroundSize: "cover",
            backgroundPosition: "center",
          }}
        >
          <div className="mx-auto grid max-w-6xl gap-8 px-6 py-14 text-white md:grid-cols-[2fr,1fr]">
            <div className="space-y-5">
              <h2 className="text-sm font-semibold uppercase tracking-[0.2em] text-white/80">
                생명존중 · 인간사랑을 실천하는
              </h2>
              <p className="text-3xl font-bold leading-snug md:text-4xl">
                세계속의 건양대학교병원
              </p>
              <p className="max-w-xl text-base text-white/90 md:text-lg">
                환자 안전과 편의를 최우선으로 생각하는 스마트 병원 서비스. 온라인 예약부터 증명서 발급까지
                One-Stop으로 경험해보세요.
              </p>
              <div className="flex flex-wrap gap-3">
                <Link
                  to="#appointments"
                  className="rounded-full bg-primary px-6 py-2 text-sm font-semibold text-white shadow hover:bg-primary/90"
                >
                  진료예약 바로가기
                </Link>
                <Link
                  to="#guide"
                  className="rounded-full border border-white/60 px-6 py-2 text-sm font-semibold text-white hover:bg-white/10"
                >
                  이용안내 보기
                </Link>
              </div>
            </div>
            <Card className="bg-white/95 text-slate-900 backdrop-blur">
              <CardHeader>
                <CardTitle className="text-lg font-bold">간편 안내</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-3">
                <div className="rounded-lg bg-primary/5 px-3 py-2 text-sm font-semibold text-primary">
                  전화예약(초진) 051-797-3500
                </div>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <Link
                    to="#appointments"
                    className="rounded-md border border-slate-200 px-3 py-2 font-medium text-slate-700 hover:border-primary/50 hover:text-primary"
                  >
                    온라인 예약
                  </Link>
                  <Link
                    to="#appointments"
                    className="rounded-md border border-slate-200 px-3 py-2 font-medium text-slate-700 hover:border-primary/50 hover:text-primary"
                  >
                    예약 조회
                  </Link>
                  <Link
                    to="#records"
                    className="rounded-md border border-slate-200 px-3 py-2 font-medium text-slate-700 hover:border-primary/50 hover:text-primary"
                  >
                    의무기록 발급
                  </Link>
                  <Link
                    to="/login"
                    className="rounded-md border border-slate-200 px-3 py-2 font-medium text-slate-700 hover:border-primary/50 hover:text-primary"
                  >
                    의료진 로그인
                  </Link>
                </div>
                <p className="rounded-md bg-slate-100 px-3 py-2 text-xs text-slate-600">
                  모바일 앱에서 진료과·의료진을 선택하고 희망 날짜를 지정하세요. 접수 현황과 대기 순번을 실시간으로
                  확인할 수 있습니다.
                </p>
              </CardContent>
            </Card>
          </div>
          <div className="bg-white/90 py-6">
            <div className="mx-auto grid max-w-6xl grid-cols-2 gap-3 px-6 md:grid-cols-4 lg:grid-cols-8">
              {heroQuickLinks.map((item) => (
                <Link
                  key={item.label}
                  to={item.link}
                  className="flex flex-col items-center justify-center gap-2 rounded-md border border-white/0 bg-slate-50 py-4 text-center text-xs font-semibold text-slate-700 transition hover:-translate-y-1 hover:border-primary/40 hover:bg-white hover:text-primary"
                >
                  <span className="rounded-full bg-primary/10 p-2 text-primary">
                    <item.icon className="h-5 w-5" />
                  </span>
                  {item.label}
                </Link>
              ))}
            </div>
          </div>
        </section>

        <section className="bg-white" id="reservations">
          <div className="mx-auto max-w-6xl px-6 py-10">
            <div className="grid gap-4 md:grid-cols-5">
              {quickCallouts.map((item) => (
                <Link
                  key={item.title}
                  to={item.link ?? "#"}
                  className={`flex h-full flex-col justify-between rounded-lg border px-4 py-5 text-sm transition hover:-translate-y-1 hover:border-primary/40 hover:shadow ${
                    item.highlight
                      ? "border-primary/30 bg-primary/10 text-primary"
                      : "border-slate-200 bg-slate-50 text-slate-700"
                  }`}
                >
                  <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                    Quick
                  </span>
                  <div className="mt-2 space-y-2">
                    <p className="text-base font-bold">{item.title}</p>
                    <p
                      className={`text-sm ${
                        item.highlight ? "font-semibold" : "text-slate-600"
                      }`}
                    >
                      {item.content}
                    </p>
                  </div>
                </Link>
              ))}
            </div>
          </div>
        </section>

        <section id="quick-services" className="bg-white">
          <div className="mx-auto max-w-6xl px-6 pb-12">
            <div className="mb-8 flex items-center gap-3">
              <Info className="h-6 w-6 text-primary" />
              <h3 className="text-2xl font-semibold text-slate-800">
                주요 서비스 바로가기
              </h3>
            </div>
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {quickServices.map((service) => (
                <Card key={service.title} className="h-full border-slate-200">
                  <CardContent className="space-y-4 pt-6">
                    <div className="flex items-center gap-3">
                      <div className="rounded-xl bg-primary/10 p-3 text-primary">
                        <service.icon className="h-6 w-6" />
                      </div>
                      <div>
                        <h4 className="text-lg font-semibold text-slate-800">
                          {service.title}
                        </h4>
                        <p className="text-sm text-slate-600">
                          {service.description}
                        </p>
                      </div>
                    </div>
                    <Link
                      to={service.link}
                      className="inline-flex items-center text-sm font-semibold text-primary hover:underline"
                    >
                      자세히 보기 →
                    </Link>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </section>

        <section id="notices" className="bg-slate-50">
          <div className="mx-auto grid max-w-6xl gap-8 px-6 py-12 md:grid-cols-[2fr_3fr]">
            <Card className="border-slate-200">
              <CardHeader>
                <CardTitle className="text-xl font-semibold text-slate-800">
                  공지사항
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {notices.map((notice) => (
                  <div key={notice.title} className="rounded-lg border border-slate-200 p-4">
                    <h5 className="text-sm font-semibold text-slate-800">
                      {notice.title}
                    </h5>
                    <p className="mt-2 text-sm text-slate-600">{notice.content}</p>
                  </div>
                ))}
              </CardContent>
            </Card>
            <div className="grid gap-6 md:grid-cols-2">
              <Card
                id="map"
                className="flex flex-col justify-between border-slate-200 bg-gradient-to-br from-primary/10 via-white to-white"
              >
                <CardHeader>
                  <CardTitle className="text-xl font-semibold text-slate-800">
                    실시간 병원 지도
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm text-slate-600">
                  <p>
                    층별 주요 시설과 이동 동선을 한 눈에 확인하세요. 비콘 기반 위치
                    추적과 연동되어 안내 정확도가 높습니다.
                  </p>
                  <Link to="#" className="text-primary hover:underline">
                    지도 열기 →
                  </Link>
                </CardContent>
              </Card>
              <Card
                id="doctors"
                className="flex flex-col justify-between border-slate-200 bg-gradient-to-br from-cyan-100 via-white to-white"
              >
                <CardHeader>
                  <CardTitle className="text-xl font-semibold text-slate-800">
                    진료과 · 의료진 안내
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm text-slate-600">
                  <p>
                    진료과별 전문 의료진 정보를 확인하고, 담당 교수진의 진료 일정과
                    전문 분야를 미리 살펴보세요.
                  </p>
                  <Link to="/patient/login" className="text-primary hover:underline">
                    의료진 정보 보기 →
                  </Link>
                </CardContent>
              </Card>
              <Card
                id="waiting"
                className="flex flex-col justify-between border-slate-200 bg-gradient-to-br from-slate-100 via-white to-white"
              >
                <CardHeader>
                  <CardTitle className="text-xl font-semibold text-slate-800">
                    진료 대기 현황
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm text-slate-600">
                  <p>
                    접수한 진료과의 현재 대기 순서를 실시간으로 확인하고, 입실
                    알림까지 받아보세요.
                  </p>
                  <Link to="#" className="text-primary hover:underline">
                    대기 현황 보기 →
                  </Link>
                </CardContent>
              </Card>
              <Card
                id="appointments"
                className="flex flex-col justify-between border-slate-200 bg-gradient-to-br from-amber-100 via-white to-white"
              >
                <CardHeader>
                  <CardTitle className="text-xl font-semibold text-slate-800">
                    온라인 진료 예약
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm text-slate-600">
                  <p>
                    모바일로 원하는 진료과와 의료진을 선택하여 예약하세요. 예약 내역을
                    앱에서 바로 확인할 수 있습니다.
                  </p>
                  <Link to="/patient/login" className="text-primary hover:underline">
                    예약 방법 안내 →
                  </Link>
                </CardContent>
              </Card>
              <Card
                id="pharmacy"
                className="flex flex-col justify-between border-slate-200 bg-gradient-to-br from-emerald-50 via-white to-white"
              >
                <CardHeader>
                  <CardTitle className="text-xl font-semibold text-slate-800">
                    약국 위치 안내
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm text-slate-600">
                  <p>
                    원내 조제실과 주변 협력 약국의 운영 시간, 혼잡도를 확인할 수
                    있습니다.
                  </p>
                  <Link to="#" className="text-primary hover:underline">
                    약국 정보 보기 →
                  </Link>
                </CardContent>
              </Card>
              <Card
                id="ar"
                className="flex flex-col justify-between border-slate-200 bg-gradient-to-br from-violet-100 via-white to-white"
              >
                <CardHeader>
                  <CardTitle className="text-xl font-semibold text-slate-800">
                    AR 내비게이션
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm text-slate-600">
                  <p>
                    스마트폰 카메라를 통해 실시간으로 길 안내를 제공합니다. ARCore /
                    ARKit 기반의 정밀 안내를 경험해보세요.
                  </p>
                  <Link to="#" className="text-primary hover:underline">
                    AR 안내 시작 →
                  </Link>
                </CardContent>
              </Card>
              <Card
                id="parking"
                className="flex flex-col justify-between border-slate-200 bg-gradient-to-br from-sky-100 via-white to-white"
              >
                <CardHeader>
                  <CardTitle className="text-xl font-semibold text-slate-800">
                    주차장 이용 안내
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm text-slate-600">
                  <p>
                    주차 가능 구역과 혼잡도를 확인하고, 전용 QR코드로 요금 정산을 빠르게
                    진행하세요.
                  </p>
                  <Link to="#parking" className="text-primary hover:underline">
                    주차 정보 보기 →
                  </Link>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t bg-white">
        <div className="mx-auto flex max-w-6xl flex-col gap-4 px-6 py-8 text-sm text-slate-600 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="font-semibold text-slate-700">
              건양대학교병원 스마트 케어 내비게이션 센터
            </p>
            <p>대전광역시 서구 관저동 196-5 · 대표전화 051-797-3500</p>
          </div>
          <div className="flex flex-wrap gap-4 text-xs font-semibold uppercase tracking-wide text-slate-500">
            <Link to="/login" className="hover:text-primary">
              의료진 플랫폼
            </Link>
            <Link to="/patient/login" className="hover:text-primary">
              환자 서비스 안내
            </Link>
            <Link to="#" className="hover:text-primary">
              개인정보 처리방침
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default Home;

