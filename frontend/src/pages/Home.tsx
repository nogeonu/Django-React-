import { Link } from "react-router-dom";
import heroImage from "@/assets/doctor-bg.jpg";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  MapPin,
  Stethoscope,
  Clock,
  Pill,
  ParkingCircle,
  Camera,
  PhoneCall,
  Calendar,
  Info,
} from "lucide-react";

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
    <div className="min-h-screen bg-gray-50">
      <header className="border-b bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="rounded-full bg-primary/10 p-2 text-primary">
              <Stethoscope className="h-8 w-8" />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-primary">
                최신 의료 정보를 제공하는 병원
              </p>
              <h1 className="text-xl font-bold text-slate-800">
                건양대학교병원
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-4 text-sm text-slate-600">
            <div className="flex items-center gap-2">
              <PhoneCall className="h-4 w-4 text-primary" />
              <span>대표전화 051-797-3500</span>
            </div>
            <Link
              to="/staff"
              className="rounded-full bg-primary px-4 py-2 text-xs font-semibold uppercase tracking-wide text-white shadow hover:bg-primary/90"
            >
              의료진 플랫폼 바로가기
            </Link>
          </div>
        </div>
      </header>

      <main>
        <section
          className="relative"
          style={{
            backgroundImage: `linear-gradient(rgba(0,0,0,0.35), rgba(0,0,0,0.35)), url(${heroImage})`,
            backgroundSize: "cover",
            backgroundPosition: "center",
          }}
        >
          <div className="mx-auto flex max-w-6xl flex-col gap-6 px-6 py-16 text-white md:flex-row md:items-center md:justify-between">
            <div className="max-w-xl space-y-4">
              <span className="inline-flex items-center rounded-full bg-white/10 px-3 py-1 text-xs font-medium uppercase tracking-wide">
                환자 중심 · 정밀 치료 · 디지털 전환
              </span>
              <h2 className="text-3xl font-semibold leading-snug md:text-4xl">
                환자와 보호자가 쉬게 찾는 <br /> 병원 내 위치 기반 스마트 서비스
              </h2>
              <p className="text-base text-white/90 md:text-lg">
                실시간 지도와 대기 현황, AR 내비게이션까지 원스톱으로 제공하여
                병원 이용 경험을 한 단계 높여드립니다.
              </p>
              <div className="flex flex-wrap gap-3">
                <Link
                  to="#map"
                  className="rounded-full bg-white px-6 py-2 text-sm font-semibold text-primary shadow"
                >
                  병원 지도 바로가기
                </Link>
                <Link
                  to="#appointments"
                  className="rounded-full border border-white/70 px-6 py-2 text-sm font-semibold text-white hover:bg-white/10"
                >
                  진료 예약 안내
                </Link>
              </div>
            </div>
            <Card className="w-full max-w-sm bg-white/90 text-slate-800 backdrop-blur">
              <CardHeader>
                <CardTitle className="text-xl font-bold text-slate-800">
                  진료 예약 간편 안내
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="rounded-lg bg-slate-100 p-3 text-sm text-slate-700">
                  전화 예약(초진) <strong className="text-primary">051-797-3500</strong>
                </div>
                <div className="rounded-lg border border-slate-200 p-3 text-sm text-slate-600">
                  모바일 앱에서 진료과·의료진을 선택하고 희망 날짜를 지정하세요.
                  접수 현황과 대기 순번을 실시간으로 확인할 수 있습니다.
                </div>
                <Link
                  to="/signup"
                  className="block rounded-md bg-primary px-4 py-2 text-center text-sm font-semibold text-white hover:bg-primary/90"
                >
                  환자용 서비스 안내 보기
                </Link>
              </CardContent>
            </Card>
          </div>
        </section>

        <section id="quick-services" className="bg-white">
          <div className="mx-auto max-w-6xl px-6 py-12">
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
                  <Link to="/signup" className="text-primary hover:underline">
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
                  <Link to="/signup" className="text-primary hover:underline">
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
            <Link to="/staff" className="hover:text-primary">
              의료진 플랫폼
            </Link>
            <Link to="/signup" className="hover:text-primary">
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

