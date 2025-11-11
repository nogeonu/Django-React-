import { FormEvent, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { patientLoginApi } from "@/lib/api";

function PatientLogin() {
  const { toast } = useToast();
  const [accountId, setAccountId] = useState("");
  const [password, setPassword] = useState("");
  const [rememberId, setRememberId] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const savedAccount = localStorage.getItem("patient_account_id");
    if (savedAccount) {
      setAccountId(savedAccount);
      setRememberId(true);
    }
  }, []);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    if (!accountId || !password) {
      toast({
        title: "입력 값을 확인해주세요.",
        description: "계정 ID와 비밀번호를 모두 입력해주세요.",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    try {
      const response = await patientLoginApi({
        account_id: accountId.trim(),
        password,
      });

      if (rememberId) {
        localStorage.setItem("patient_account_id", accountId.trim());
      } else {
        localStorage.removeItem("patient_account_id");
      }

      toast({
        title: "환자 로그인",
        description: `${response.name}님 환영합니다.`,
      });
      setPassword("");
    } catch (error: any) {
      const data = error?.response?.data;
      let message = data?.detail;

      if (!message && data && typeof data === "object") {
        const firstKey = Object.keys(data)[0];
        const value = (data as Record<string, unknown>)[firstKey];
        if (Array.isArray(value) && value.length > 0) {
          message = String(value[0]);
        }
      }

      if (!message) {
        message = "로그인 중 오류가 발생했습니다.";
      }

      toast({
        title: "로그인 실패",
        description: message,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-sky-50">
      <div className="mx-auto flex min-h-screen max-w-5xl flex-col justify-center px-4 py-12">
        <div className="mb-10 flex items-center justify-center md:justify-start">
          <Link to="/" className="flex flex-col items-center text-center md:flex-row md:items-center md:gap-4 md:text-left">
            <div className="rounded-full bg-primary/10 px-4 py-2 text-lg font-bold text-primary shadow-sm">
              KYUH
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.25em] text-primary">
                Konyang University Hospital
              </p>
              <p className="text-base font-bold text-slate-700 md:text-lg">건양대학교병원 환자 포털</p>
            </div>
          </Link>
        </div>

        <Card className="border-2 border-primary/30 bg-white/95 shadow-2xl backdrop-blur md:rounded-[32px]">
          <CardContent className="mx-auto flex w-full max-w-[480px] flex-col gap-8 px-3.5 py-6 md:px-5 md:py-9">
              <div className="space-y-3 text-center md:text-left">
                <h2 className="text-3xl font-bold text-slate-800">환자 로그인</h2>
                <p className="text-sm text-slate-500">
                  온라인 진료예약, 예약 조회, 증명서 발급을 위해 환자 계정으로 로그인하세요.
                </p>
              </div>

            <form className="space-y-6" onSubmit={handleSubmit}>
                <div className="space-y-2">
                  <Label htmlFor="accountId" className="text-sm font-medium text-slate-700">
                    계정 ID
                  </Label>
                  <Input
                    id="accountId"
                    type="text"
                    value={accountId}
                    onChange={(event) => setAccountId(event.target.value)}
                    placeholder="아이디를 입력하세요."
                    autoComplete="username"
                    className="h-11 rounded-xl border border-sky-100 bg-slate-50 text-base placeholder:text-slate-400 focus:border-primary focus:ring-2 focus:ring-primary/40"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="password" className="text-sm font-medium text-slate-700">
                    비밀번호
                  </Label>
                  <Input
                    id="password"
                    type="password"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    placeholder="비밀번호를 입력하세요."
                    autoComplete="current-password"
                    className="h-11 rounded-xl border border-sky-100 bg-slate-50 text-base placeholder:text-slate-400 focus:border-primary focus:ring-2 focus:ring-primary/40"
                  />
                </div>

                <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-slate-600">
                  <label className="flex items-center gap-2 font-medium">
                    <Checkbox
                      id="remember"
                      checked={rememberId}
                      onCheckedChange={(checked) => setRememberId(Boolean(checked))}
                    />
                    아이디 저장
                  </label>
                  <Link to="#" className="font-semibold text-primary hover:underline">
                    비밀번호 찾기
                  </Link>
                </div>

                <Button
                  type="submit"
                  className="h-11 w-full rounded-full bg-primary text-base font-semibold text-white shadow-lg shadow-primary/30 hover:bg-primary/90"
                  disabled={loading}
                >
                  {loading ? "로그인 처리 중..." : "로그인"}
                </Button>
              </form>

              <div className="flex flex-wrap items-center justify-center gap-3 text-xs font-semibold text-slate-500 md:justify-start">
                <Link to="#" className="hover:text-primary hover:underline">
                  아이디 찾기
                </Link>
                <span className="text-slate-300">|</span>
                <Link to="#" className="hover:text-primary hover:underline">
                  비밀번호 찾기
                </Link>
                <span className="text-slate-300">|</span>
                <Link to="/patient/signup" className="hover:text-primary hover:underline">
                  회원가입
                </Link>
              </div>
          </CardContent>
        </Card>

        <div className="mt-8 text-center text-xs text-slate-400 md:text-left">
          © {new Date().getFullYear()} Konyang University Hospital Patient Portal. All rights reserved.
        </div>
      </div>
    </div>
  );
}

export default PatientLogin;

