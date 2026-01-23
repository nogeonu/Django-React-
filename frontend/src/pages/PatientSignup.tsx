import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { patientSignupApi } from "@/lib/api";

function PatientSignup() {
  const { toast } = useToast();
  const navigate = useNavigate();
  const [form, setForm] = useState({
    accountId: "",
    name: "",
    email: "",
    phone: "",
    password: "",
    confirmPassword: "",
  });
  const [loading, setLoading] = useState(false);

  const handleChange =
    (key: keyof typeof form) => (event: React.ChangeEvent<HTMLInputElement>) => {
      setForm((prev) => ({ ...prev, [key]: event.target.value }));
    };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!form.accountId || !form.name || !form.email || !form.phone || !form.password) {
      toast({
        title: "필수 정보를 입력해주세요.",
        description: "계정 ID, 이름, 이메일, 연락처, 비밀번호는 필수 항목입니다.",
        variant: "destructive",
      });
      return;
    }

    if (form.password !== form.confirmPassword) {
      toast({
        title: "비밀번호가 일치하지 않습니다.",
        description: "비밀번호와 비밀번호 확인이 동일해야 합니다.",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    try {
      const response = await patientSignupApi({
        account_id: form.accountId.trim(),
        name: form.name.trim(),
        email: form.email.trim(),
        phone: form.phone.trim(),
        password: form.password,
      });
      toast({
        title: "회원가입이 완료되었습니다.",
        description: `환자 번호: ${response.patient_id} / 로그인 ID: ${response.account_id}`,
      });
      setForm({ accountId: "", name: "", email: "", phone: "", password: "", confirmPassword: "" });
      setTimeout(() => navigate("/patient/login"), 1200);
    } catch (error: any) {
      const data = error?.response?.data;
      let message =
        data?.detail ||
        data?.non_field_errors?.[0];

      if (!message && data && typeof data === "object") {
        const firstKey = Object.keys(data)[0];
        const value = (data as Record<string, unknown>)[firstKey];
        if (Array.isArray(value) && value.length > 0) {
          message = String(value[0]);
        }
      }

      if (!message) {
        message = "회원가입 중 오류가 발생했습니다.";
      }
      toast({
        title: "회원가입 실패",
        description: message,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-sky-50 via-white to-emerald-50">
      <div className="mx-auto flex min-h-screen max-w-5xl flex-col justify-center px-4 py-12">
        <div className="mb-10 flex items-center justify-center md:justify-start">
          <Link
            to="/"
            className="flex flex-col items-center text-center md:flex-row md:items-center md:gap-4 md:text-left"
          >
            <div className="rounded-full bg-primary/10 px-4 py-2 text-lg font-bold text-primary shadow-sm">
              CDSS
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.25em] text-primary">
                CDSS Medical Center
              </p>
              <p className="text-base font-bold text-slate-700 md:text-lg">CDSS 메디컬 센터 환자 포털</p>
            </div>
          </Link>
        </div>

        <Card className="border-2 border-primary/30 bg-white/90 shadow-2xl backdrop-blur md:rounded-[32px]">
          <CardContent className="mx-auto flex w-full max-w-[520px] flex-col gap-8 px-3.5 py-6 md:px-5 md:py-9">
            <div className="space-y-3 text-center md:text-left">
              <h2 className="text-3xl font-bold text-slate-800">환자 회원가입</h2>
              <p className="text-sm text-slate-500">
                진료 예약과 건강 기록 확인을 위한 환자 계정을 지금 생성하세요.
              </p>
            </div>

            <form className="space-y-6" onSubmit={handleSubmit}>
              <div className="grid gap-5 md:grid-cols-2">
                <div className="md:col-span-2 space-y-2">
                  <Label htmlFor="accountId" className="text-sm font-medium text-slate-700">
                    계정 ID
                  </Label>
                  <Input
                    id="accountId"
                    value={form.accountId}
                    onChange={handleChange("accountId")}
                    placeholder="로그인에 사용할 아이디"
                    className="h-11 rounded-xl border border-sky-100 bg-slate-50 text-base placeholder:text-slate-400 focus:border-primary focus:ring-2 focus:ring-primary/40"
                    required
                  />
                </div>
                <div className="md:col-span-2 space-y-2">
                  <Label htmlFor="name" className="text-sm font-medium text-slate-700">
                    이름
                  </Label>
                  <Input
                    id="name"
                    value={form.name}
                    onChange={handleChange("name")}
                    placeholder="홍길동"
                    className="h-11 rounded-xl border border-sky-100 bg-slate-50 text-base placeholder:text-slate-400 focus:border-primary focus:ring-2 focus:ring-primary/40"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="email" className="text-sm font-medium text-slate-700">
                    이메일
                  </Label>
                  <Input
                    id="email"
                    type="email"
                    value={form.email}
                    onChange={handleChange("email")}
                    placeholder="patient@example.com"
                    autoComplete="email"
                    className="h-11 rounded-xl border border-sky-100 bg-slate-50 text-base placeholder:text-slate-400 focus:border-primary focus:ring-2 focus:ring-primary/40"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="phone" className="text-sm font-medium text-slate-700">
                    연락처
                  </Label>
                  <Input
                    id="phone"
                    value={form.phone}
                    onChange={handleChange("phone")}
                    placeholder="010-0000-0000"
                    autoComplete="tel"
                    className="h-11 rounded-xl border border-sky-100 bg-slate-50 text-base placeholder:text-slate-400 focus:border-primary focus:ring-2 focus:ring-primary/40"
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="password" className="text-sm font-medium text-slate-700">
                    비밀번호
                  </Label>
                  <Input
                    id="password"
                    type="password"
                    value={form.password}
                    onChange={handleChange("password")}
                    placeholder="6자 이상 입력하세요"
                    autoComplete="new-password"
                    className="h-11 rounded-xl border border-sky-100 bg-slate-50 text-base placeholder:text-slate-400 focus:border-primary focus:ring-2 focus:ring-primary/40"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="confirmPassword" className="text-sm font-medium text-slate-700">
                    비밀번호 확인
                  </Label>
                  <Input
                    id="confirmPassword"
                    type="password"
                    value={form.confirmPassword}
                    onChange={handleChange("confirmPassword")}
                    placeholder="비밀번호를 다시 입력하세요"
                    autoComplete="new-password"
                    className="h-11 rounded-xl border border-sky-100 bg-slate-50 text-base placeholder:text-slate-400 focus:border-primary focus:ring-2 focus:ring-primary/40"
                  />
                </div>
              </div>

              <Button
                type="submit"
                className="h-11 w-full rounded-full bg-primary text-base font-semibold text-white shadow-lg shadow-primary/30 hover:bg-primary/90"
                disabled={loading}
              >
                {loading ? "가입 처리 중..." : "회원가입"}
              </Button>
            </form>

            <div className="text-center text-sm text-slate-600 md:text-left">
              이미 계정이 있으신가요?{" "}
              <Link to="/patient/login" className="font-semibold text-primary hover:underline">
                환자 로그인
              </Link>
            </div>
          </CardContent>
        </Card>

        <div className="mt-8 text-center text-xs text-slate-400 md:text-left">
          © {new Date().getFullYear()} CDSS Medical Center Patient Portal. All rights reserved.
        </div>
      </div>
    </div>
  );
}

export default PatientSignup;


