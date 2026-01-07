import { Link } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/context/AuthContext";
import {
  ArrowRight,
  Activity,
  Brain,
  ShieldCheck,
  Stethoscope,
  CalendarCheck,
  Search,
  FileText,
  Bot,
  MapPin,
  Info,
  Clock,
  Pill,
  ParkingCircle,
} from "lucide-react";

const navItems = [
  { label: "예약 / 조회", link: "#reservations" },
  { label: "의료진 / 진료과", link: "/patient/doctors" },
  { label: "이용안내", link: "#guide" },
  { label: "병원소개", link: "#about" },
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
    link: "/patient/doctors",
  },
  {
    icon: CalendarCheck,
    title: "온라인 진료예약",
    description: "모바일로 빠르게 진료 예약하기",
    link: "/patient/login",
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
  const { patientUser, setPatientUser } = useAuth();
  const { toast } = useToast();

  const handlePatientLogout = () => {
    setPatientUser(null);
    toast({
      title: "로그아웃 되었습니다.",
      description: "안전하게 로그아웃 처리되었습니다.",
    });
  };

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header>
        <div className="border-b bg-slate-100">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-2 text-xs text-slate-600">
            <div className="flex items-center gap-2">
              <span className="rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-semibold text-primary">
                NOTICE
              </span>
              <span>방문인원은 환자와 나의 건강을 위하여 자제하여 주세요.</span>
            </div>
            {patientUser ? (
              <div className="flex items-center gap-3 text-xs font-semibold text-slate-700">
                <span className="hidden sm:inline">
                  {patientUser.name}님 환영합니다.
                </span>
                <span className="sm:hidden">환자 로그인 중</span>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 rounded-full border-slate-300 px-3 text-xs"
                  onClick={handlePatientLogout}
                >
                  로그아웃
                </Button>
              </div>
            ) : (
              <div className="flex items-center gap-4 font-semibold">
                <Link to="/patient/login" className="hover:text-primary">
                  환자 로그인
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
            )}
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
                <Link
                  key={item.label}
                  to={item.link}
                  className="hover:text-primary"
                >
                  {item.label}
                </Link>
              ))}
            </nav>
          </div>
        </div>
      </header>

      <main>
        {/* Hero Section - 새로운 디자인 */}
        <section className="relative min-h-[90vh] flex items-center overflow-hidden bg-slate-50">
          {/* Background Image with Overlay */}
          <div className="absolute inset-0 z-0">
            <img 
              src="/images/hero-bg.jpg" 
              alt="Futuristic Hospital Lobby" 
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-r from-white/90 via-white/70 to-transparent"></div>
          </div>

          <div className="container mx-auto relative z-10 pt-20 px-6">
            <div className="max-w-2xl animate-in slide-in-from-left-10 duration-1000 fade-in">
              <Badge className="mb-6 bg-primary/10 text-primary hover:bg-primary/20 border-primary/20 px-4 py-1.5 text-sm backdrop-blur-sm">
                <Activity className="w-3.5 h-3.5 mr-2 animate-pulse" />
                Next Generation Medical AI
              </Badge>
              <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight text-slate-900 mb-6 leading-[1.1]">
                미래 의료의 기준,<br/>
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-accent">CDSS 메디컬 센터</span>
              </h1>
              <p className="text-xl text-slate-600 mb-10 leading-relaxed max-w-lg">
                최첨단 AI 진단 시스템과 전문 의료진의 협진으로<br/>
                당신의 건강한 삶을 위한 가장 정확한 답을 제시합니다.
              </p>
              <div className="flex flex-col sm:flex-row gap-4">
                <Link to="/patient/login">
                  <Button size="lg" className="h-14 px-8 text-lg bg-primary hover:bg-primary/90 shadow-lg shadow-primary/20 rounded-full">
                    진료 예약하기 <ArrowRight className="ml-2 w-5 h-5" />
                  </Button>
                </Link>
                <Link to="/patient/doctors">
                  <Button size="lg" variant="outline" className="h-14 px-8 text-lg bg-white/50 backdrop-blur-sm border-slate-300 hover:bg-white rounded-full">
                    의료진 찾기 <Search className="ml-2 w-5 h-5" />
                  </Button>
                </Link>
              </div>
            </div>
          </div>

          {/* Floating Stats Card */}
          <div className="absolute bottom-10 right-10 hidden lg:block animate-in slide-in-from-bottom-10 duration-1000 delay-300 fade-in">
            <div className="glass-panel p-6 rounded-2xl max-w-xs backdrop-blur-md bg-white/90 shadow-xl">
              <div className="flex items-center gap-4 mb-4">
                <div className="w-12 h-12 rounded-full bg-accent/10 flex items-center justify-center text-accent">
                  <Brain className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-sm text-slate-600 font-medium">AI 진단 정확도</p>
                  <p className="text-2xl font-bold text-slate-900">99.8%</p>
                </div>
              </div>
              <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                <div className="h-full bg-accent w-[99.8%]"></div>
              </div>
            </div>
          </div>
        </section>

        {/* Quick Access Menu */}
        <section className="relative z-20 -mt-20 pb-20">
          <div className="container mx-auto px-6">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              {[
                { icon: CalendarCheck, title: "간편 예약", desc: "모바일로 쉽고 빠르게", link: "/patient/login" },
                { icon: Search, title: "진료과 찾기", desc: "증상별 맞춤 진료과", link: "/patient/doctors" },
                { icon: FileText, title: "제증명 발급", desc: "온라인 즉시 발급", link: "#certificate" },
                { icon: Stethoscope, title: "건강검진", desc: "나만을 위한 맞춤 검진", link: "#checkup" }
              ].map((item, idx) => (
                <Link key={idx} to={item.link}>
                  <Card className="border-none shadow-xl hover:shadow-2xl transition-all duration-300 hover:-translate-y-1 bg-white overflow-hidden group cursor-pointer">
                    <CardContent className="p-8 flex flex-col items-center text-center relative">
                      <div className="absolute top-0 right-0 w-24 h-24 bg-primary/5 rounded-bl-full -mr-4 -mt-4 group-hover:bg-primary/10 transition-colors"></div>
                      <div className="w-16 h-16 rounded-2xl bg-primary/10 text-primary flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                        <item.icon className="w-8 h-8" />
                      </div>
                      <h3 className="text-xl font-bold mb-2 text-slate-900">{item.title}</h3>
                      <p className="text-slate-600">{item.desc}</p>
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>
          </div>
        </section>

        {/* AI Diagnosis Section */}
        <section className="py-24 bg-slate-50 overflow-hidden">
          <div className="container mx-auto px-6">
            <div className="flex flex-col lg:flex-row items-center gap-16">
              <div className="lg:w-1/2 relative">
                <div className="relative rounded-3xl overflow-hidden shadow-2xl border-8 border-white">
                  <img 
                    src="/images/ai-diagnosis.jpg" 
                    alt="AI Diagnosis System" 
                    className="w-full h-auto hover:scale-105 transition-transform duration-700"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent flex items-end p-8">
                    <div className="text-white">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge className="bg-accent text-white border-none">CDSS v2.0</Badge>
                        <span className="text-sm font-medium text-white/80">실시간 분석 중</span>
                      </div>
                      <p className="font-medium">환자 데이터 실시간 동기화 및 분석</p>
                    </div>
                  </div>
                </div>
                {/* Decorative Elements */}
                <div className="absolute -top-10 -left-10 w-40 h-40 bg-primary/10 rounded-full blur-3xl"></div>
                <div className="absolute -bottom-10 -right-10 w-40 h-40 bg-accent/10 rounded-full blur-3xl"></div>
              </div>
              
              <div className="lg:w-1/2">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-bold mb-6">
                  <Brain className="w-4 h-4" />
                  <span>Intelligent Healthcare</span>
                </div>
                <h2 className="text-4xl md:text-5xl font-bold mb-6 leading-tight text-slate-900">
                  AI가 더하는<br/>
                  <span className="text-primary">정확함의 깊이</span>
                </h2>
                <p className="text-lg text-slate-600 mb-8 leading-relaxed">
                  CDSS(Clinical Decision Support System)는 수만 건의 임상 데이터를 학습한 AI가 의료진의 진단을 보조하여, 오진율을 획기적으로 낮추고 최적의 치료 계획을 수립하도록 돕습니다.
                </p>
                
                <div className="space-y-6">
                  {[
                    { title: "실시간 데이터 분석", desc: "환자의 생체 신호를 실시간으로 모니터링하고 이상 징후를 즉시 감지합니다." },
                    { title: "정밀 영상 판독", desc: "MRI, CT 등 의료 영상을 AI가 픽셀 단위로 분석하여 미세한 병변까지 찾아냅니다." },
                    { title: "맞춤형 치료 제안", desc: "유전체 정보와 생활 습관을 분석하여 개인별 최적의 치료법을 제시합니다." }
                  ].map((feature, idx) => (
                    <div key={idx} className="flex gap-4 p-4 rounded-xl hover:bg-white hover:shadow-md transition-all duration-300 border border-transparent hover:border-slate-100">
                      <div className="mt-1 w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center text-primary shrink-0">
                        <ShieldCheck className="w-5 h-5" />
                      </div>
                      <div>
                        <h4 className="text-lg font-bold mb-1 text-slate-900">{feature.title}</h4>
                        <p className="text-slate-600">{feature.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Medical Staff Section */}
        <section className="py-24 bg-white">
          <div className="container mx-auto px-6">
            <div className="text-center max-w-3xl mx-auto mb-16">
              <h2 className="text-4xl font-bold mb-4 text-slate-900">최고의 의료진</h2>
              <p className="text-lg text-slate-600">
                각 분야 최고의 전문의들이 첨단 시스템과 함께 당신의 건강을 지킵니다.
              </p>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {[
                { name: "김태훈 교수", dept: "순환기내과", img: "/images/doctor-1.jpg", desc: "심장질환 AI 진단 권위자" },
                { name: "이서연 교수", dept: "신경외과", img: "/images/doctor-2.jpg", desc: "뇌혈관 정밀 수술 전문" },
                { name: "박준형 교수", dept: "정형외과", img: "/images/doctor-1.jpg", desc: "로봇 인공관절 수술 전문" },
                { name: "최지민 교수", dept: "소아청소년과", img: "/images/doctor-2.jpg", desc: "소아 희귀질환 전문" }
              ].map((doctor, idx) => (
                <div key={idx} className="group relative overflow-hidden rounded-2xl bg-slate-50">
                  <div className="aspect-[4/5] overflow-hidden">
                    <img 
                      src={doctor.img} 
                      alt={doctor.name} 
                      className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-60 group-hover:opacity-80 transition-opacity"></div>
                  </div>
                  <div className="absolute bottom-0 left-0 right-0 p-6 text-white transform translate-y-2 group-hover:translate-y-0 transition-transform duration-300">
                    <p className="text-accent font-medium text-sm mb-1">{doctor.dept}</p>
                    <h3 className="text-2xl font-bold mb-2">{doctor.name}</h3>
                    <p className="text-white/80 text-sm opacity-0 group-hover:opacity-100 transition-opacity duration-300 delay-100">
                      {doctor.desc}
                    </p>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="text-center mt-12">
              <Link to="/patient/doctors">
                <Button variant="outline" size="lg" className="rounded-full px-8 border-primary text-primary hover:bg-primary hover:text-white">
                  의료진 전체보기
                </Button>
              </Link>
            </div>
          </div>
        </section>

        {/* CDSS Platform Access Banner */}
        <section className="py-20 relative overflow-hidden">
          <div className="absolute inset-0">
            <img 
              src="/images/medical-tech.jpg" 
              alt="Medical Tech Background" 
              className="w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-primary/90 mix-blend-multiply"></div>
          </div>
          
          <div className="container mx-auto relative z-10 text-center text-white px-6">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">의료진 전용 플랫폼</h2>
            <p className="text-xl text-white/80 mb-10 max-w-2xl mx-auto">
              CDSS(Clinical Decision Support System)는 의료진을 위한 통합 진료 지원 시스템입니다.<br/>
              권한이 있는 의료진만 접속 가능합니다.
            </p>
            <div className="flex flex-col sm:flex-row justify-center gap-4">
              <Link to="/login">
                <Button size="lg" className="bg-white text-primary hover:bg-white/90 h-14 px-8 text-lg rounded-full font-bold">
                  CDSS 시스템 접속
                </Button>
              </Link>
              <Button size="lg" variant="outline" className="border-white text-white hover:bg-white/10 h-14 px-8 text-lg rounded-full">
                사용자 매뉴얼 다운로드
              </Button>
            </div>
          </div>
        </section>

        {/* Quick Services */}
        <section id="quick-services" className="bg-white py-12">
          <div className="container mx-auto px-6">
            <div className="mb-8 flex items-center gap-3">
              <Info className="h-6 w-6 text-primary" />
              <h3 className="text-2xl font-semibold text-slate-800">
                주요 서비스 바로가기
              </h3>
            </div>
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {quickServices.map((service) => (
                <Link key={service.title} to={service.link}>
                  <Card className="h-full border-slate-200 hover:shadow-md transition-shadow">
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
                      <span className="inline-flex items-center text-sm font-semibold text-primary hover:underline">
                        자세히 보기 →
                      </span>
                    </CardContent>
                  </Card>
                </Link>
              ))}
            </div>
          </div>
        </section>

        {/* Notices */}
        <section id="notices" className="bg-slate-50 py-12">
          <div className="container mx-auto px-6">
            <Card className="border-slate-200">
              <CardContent className="p-6 space-y-4">
                <h3 className="text-xl font-semibold text-slate-800 mb-4">
                  공지사항
                </h3>
                {notices.map((notice) => (
                  <div
                    key={notice.title}
                    className="rounded-lg border border-slate-200 p-4 hover:bg-slate-50 transition-colors"
                  >
                    <h5 className="text-sm font-semibold text-slate-800">
                      {notice.title}
                    </h5>
                    <p className="mt-2 text-sm text-slate-600">
                      {notice.content}
                    </p>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Chatbot Button */}
        <Button
          onClick={() =>
            toast({ title: "챗봇 상담", description: "LLM 챗봇 서비스가 곧 제공될 예정입니다." })
          }
          className="fixed bottom-6 right-6 z-40 flex items-center gap-2 rounded-full bg-blue-600 px-5 py-4 text-white shadow-lg transition hover:bg-blue-700"
        >
          <Bot className="h-5 w-5" />
          <span className="text-sm font-semibold">챗봇 상담</span>
        </Button>
      </main>

      {/* Footer */}
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
            <span className="hover:text-primary cursor-pointer">
              개인정보 처리방침
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default Home;
