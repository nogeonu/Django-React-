import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Loader2,
  Info,
  Search,
  Stethoscope,
  Activity,
  ArrowRight,
  User,
  Heart
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/api";

interface Patient {
  id: number;
  patient_id: string;
  name: string;
  birth_date: string;
  gender: string;
  age: number;
}

interface PredictionResult {
  patient_id: string;
  prediction: "YES" | "NO";
  probability: number;
  risk_level: "낮음" | "중간" | "높음";
  risk_message: string;
  external_db_saved: boolean;
  symptoms: Record<string, any>;
}

export default function LungCancerPrediction() {
  const [formData, setFormData] = useState({
    patient_id: "",
    name: "",
    gender: "",
    age: "",
    smoking: "",
    yellow_fingers: "",
    anxiety: "",
    peer_pressure: "",
    chronic_disease: "",
    fatigue: "",
    allergy: "",
    wheezing: "",
    alcohol_consuming: "",
    coughing: "",
    shortness_of_breath: "",
    swallowing_difficulty: "",
    chest_pain: "",
  });

  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [patientsLoading, setPatientsLoading] = useState(false);
  const [patientSearchOpen, setPatientSearchOpen] = useState(false);
  const [patientSearchTerm, setPatientSearchTerm] = useState("");
  const { toast } = useToast();

  useEffect(() => {
    const fetchPatients = async () => {
      setPatientsLoading(true);
      try {
        const response = await apiRequest(
          "GET",
          "/api/lung_cancer/patients/prediction_candidates/",
        );
        const respiratoryPatients = response.patients || response.results || [];
        setPatients(respiratoryPatients);
      } catch (error) {
        console.error("환자 목록 조회 오류:", error);
      } finally {
        setPatientsLoading(false);
      }
    };
    fetchPatients();
  }, []);

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handlePatientSelect = (patient: Patient) => {
    setFormData((prev) => ({
      ...prev,
      patient_id: patient.patient_id,
      name: patient.name,
      gender:
        patient.gender === "M" ||
          patient.gender === "남성" ||
          patient.gender === "남"
          ? "1"
          : patient.gender === "F" ||
            patient.gender === "여성" ||
            patient.gender === "여"
            ? "0"
            : "",
      age: patient.age.toString(),
    }));
    setPatientSearchOpen(false);
    setPatientSearchTerm("");
  };

  const filteredPatients = patients.filter(
    (patient) =>
      patient.name.toLowerCase().includes(patientSearchTerm.toLowerCase()) ||
      patient.patient_id.toLowerCase().includes(patientSearchTerm.toLowerCase()),
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const age = parseInt(formData.age);
      const birthYear = new Date().getFullYear() - age;
      const birth_date = `${birthYear}-01-01`;

      const response = await apiRequest(
        "POST",
        "/api/lung_cancer/patients/predict/",
        {
          patient_id: formData.patient_id,
          name: formData.name,
          birth_date: birth_date,
          gender: formData.gender === "1" ? "M" : "F",
          age: age,
          smoking: formData.smoking === "2",
          yellow_fingers: formData.yellow_fingers === "2",
          anxiety: formData.anxiety === "2",
          peer_pressure: formData.peer_pressure === "2",
          chronic_disease: formData.chronic_disease === "2",
          fatigue: formData.fatigue === "2",
          allergy: formData.allergy === "2",
          wheezing: formData.wheezing === "2",
          alcohol_consuming: formData.alcohol_consuming === "2",
          coughing: formData.coughing === "2",
          shortness_of_breath: formData.shortness_of_breath === "2",
          swallowing_difficulty: formData.swallowing_difficulty === "2",
          chest_pain: formData.chest_pain === "2",
        },
      );

      setResult(response);
      toast({
        title: "예측 완료",
        description: "폐암 예측이 성공적으로 완료되었습니다.",
      });
    } catch (error: any) {
      const errorMessage =
        error?.response?.data?.error ||
        error?.response?.data?.detail ||
        error?.message ||
        "예측 중 오류가 발생했습니다.";
      toast({
        title: "오류 발생",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case "높음": return "bg-red-50 text-red-600 border-red-100";
      case "중간": return "bg-amber-50 text-amber-600 border-amber-100";
      case "낮음": return "bg-emerald-50 text-emerald-600 border-emerald-100";
      default: return "bg-gray-50 text-gray-500 border-gray-100";
    }
  };

  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { staggerChildren: 0.1 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: { opacity: 1, y: 0 }
  };

  return (
    <motion.div
      initial="hidden" animate="visible" variants={containerVariants}
      className="space-y-8"
    >
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="bg-blue-600 p-2 rounded-xl shadow-lg shadow-blue-200">
              <Stethoscope className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-3xl font-black text-gray-900 tracking-tight">폐암 예측 시스템</h1>
          </div>
          <p className="text-sm font-medium text-gray-400 max-w-2xl">
            AI 기반 알고리즘을 활용하여 환자의 증상과 생활 습관 데이터를 분석하고 폐암 발병 위험도를 정밀 예측합니다.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
        {/* 입력 폼 - XL에서 7컬럼 차지 */}
        <motion.div variants={itemVariants} className="xl:col-span-7 space-y-6">
          <Card className="border-none shadow-sm rounded-3xl bg-white overflow-hidden">
            <CardHeader className="bg-gray-50/50 border-b border-gray-100 p-8">
              <div className="flex items-center gap-2 mb-1">
                <User className="w-4 h-4 text-blue-600" />
                <CardTitle className="text-lg font-bold text-gray-900">환자 데이터 입력</CardTitle>
              </div>
              <CardDescription className="text-[11px] font-bold text-gray-400 uppercase tracking-widest">
                정확한 예측을 위해 모든 항목을 검토 후 입력해 주세요.
              </CardDescription>
            </CardHeader>
            <CardContent className="p-8">
              <form onSubmit={handleSubmit} className="space-y-8">
                {/* 기본 정보 섹션 */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label className="text-[10px] font-black text-gray-400 uppercase tracking-widest ml-1">대상 환자 선택</Label>
                    <Popover open={patientSearchOpen} onOpenChange={setPatientSearchOpen}>
                      <PopoverTrigger asChild>
                        <Button
                          variant="outline"
                          className="w-full h-12 rounded-2xl bg-gray-50 border-none justify-between font-bold text-sm px-4 focus:ring-2 focus:ring-blue-600/20"
                        >
                          {formData.name || "환자를 조회하세요"}
                          <Search className="ml-2 h-4 w-4 opacity-50 text-blue-600" />
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-[300px] p-0 rounded-2xl border-none shadow-2xl" align="start">
                        <Command className="rounded-2xl">
                          <div className="flex items-center border-b border-gray-50 px-3 py-2">
                            <Search className="mr-2 h-4 w-4 shrink-0 opacity-50" />
                            <CommandInput
                              placeholder="이름 또는 ID 입력..."
                              value={patientSearchTerm}
                              onValueChange={setPatientSearchTerm}
                              className="h-10 border-none shadow-none focus-visible:ring-0"
                            />
                          </div>
                          <CommandList className="max-h-[300px] custom-scrollbar">
                            {patientsLoading ? (
                              <div className="p-8 text-center"><Loader2 className="h-6 w-6 animate-spin mx-auto text-blue-600" /></div>
                            ) : filteredPatients.length > 0 ? (
                              <CommandGroup>
                                {filteredPatients.map((p) => (
                                  <CommandItem
                                    key={p.id}
                                    onSelect={() => handlePatientSelect(p)}
                                    className="p-3 cursor-pointer hover:bg-blue-50/50 rounded-xl m-1"
                                  >
                                    <div className="flex items-center gap-3">
                                      <div className="w-8 h-8 rounded-lg bg-blue-100 flex items-center justify-center font-black text-blue-600 text-xs">{p.name.charAt(0)}</div>
                                      <div>
                                        <p className="font-bold text-sm">{p.name}</p>
                                        <p className="text-[10px] text-gray-400">ID: {p.id}</p>
                                      </div>
                                    </div>
                                  </CommandItem>
                                ))}
                              </CommandGroup>
                            ) : (
                              <CommandEmpty className="p-4 text-center text-xs font-bold text-gray-400">검색 결과가 없습니다.</CommandEmpty>
                            )}
                          </CommandList>
                        </Command>
                      </PopoverContent>
                    </Popover>
                  </div>
                  <div className="space-y-2">
                    <Label className="text-[10px] font-black text-gray-400 uppercase tracking-widest ml-1">나이 (시스템 자동계산)</Label>
                    <Input
                      type="number"
                      value={formData.age}
                      onChange={(e) => handleInputChange("age", e.target.value)}
                      className="h-12 rounded-2xl bg-gray-50 border-none font-bold focus-visible:ring-2 focus-visible:ring-blue-600/20"
                    />
                  </div>
                </div>

                {/* 성별 선택 */}
                <div className="space-y-2">
                  <Label className="text-[10px] font-black text-gray-400 uppercase tracking-widest ml-1">성별</Label>
                  <div className="flex gap-2">
                    {["1", "0"].map((v) => (
                      <Button
                        key={v}
                        type="button"
                        onClick={() => handleInputChange("gender", v)}
                        className={`flex-1 h-12 rounded-2xl font-black text-xs transition-all ${formData.gender === v
                          ? "bg-blue-600 text-white shadow-lg shadow-blue-100"
                          : "bg-gray-50 text-gray-400 hover:bg-gray-100"
                          }`}
                      >
                        {v === "1" ? "남성 (Male)" : "여성 (Female)"}
                      </Button>
                    ))}
                  </div>
                </div>

                {/* 상세 증상 섹션 */}
                <div className="space-y-4 pt-4 border-t border-gray-50">
                  <div className="flex items-center gap-2 mb-4">
                    <Heart className="w-4 h-4 text-red-500" />
                    <h3 className="text-sm font-black text-gray-900 uppercase tracking-tight">주요 증상 및 생활 습관</h3>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-4">
                    {[
                      { key: "smoking", label: "흡연 여부" },
                      { key: "alcohol_consuming", label: "음주 여부" },
                      { key: "coughing", label: "기침 증상" },
                      { key: "shortness_of_breath", label: "호흡 곤란" },
                      { key: "chest_pain", label: "가슴 통증" },
                      { key: "fatigue", label: "피로감" },
                      { key: "wheezing", label: "천명 (쌕쌕거림)" },
                      { key: "swallowing_difficulty", label: "삼킴 곤란" },
                      { key: "chronic_disease", label: "만성 질환" },
                      { key: "allergy", label: "알레르기" },
                      { key: "yellow_fingers", label: "손가락 황색 변색" },
                      { key: "anxiety", label: "불안감" },
                      { key: "peer_pressure", label: "주변 압박" },
                    ].map(({ key, label }) => (
                      <div key={key} className="flex items-center justify-between group py-1">
                        <Label className="text-xs font-bold text-gray-500 group-hover:text-gray-900 transition-colors uppercase tracking-tight">
                          {label}
                        </Label>
                        <Select
                          value={formData[key as keyof typeof formData]}
                          onValueChange={(value) => handleInputChange(key, value)}
                        >
                          <SelectTrigger className="w-28 h-9 rounded-xl bg-gray-50 border-none font-bold text-[10px] uppercase">
                            <SelectValue placeholder="선택" />
                          </SelectTrigger>
                          <SelectContent className="rounded-xl border-none shadow-xl">
                            <SelectItem value="2" className="text-xs font-bold py-2">있음 / YES</SelectItem>
                            <SelectItem value="1" className="text-xs font-bold py-2">없음 / NO</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    ))}
                  </div>
                </div>

                <Button
                  type="submit"
                  disabled={loading}
                  className="w-full h-14 rounded-2xl bg-gray-900 hover:bg-black text-white font-black text-sm shadow-xl shadow-gray-200 transition-all hover:scale-[1.01] active:scale-[0.99]"
                >
                  {loading ? (
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-5 h-5 animate-spin" />
                      데이터 분석 중...
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      AI 예측 실행하기
                      <ArrowRight className="w-4 h-4" />
                    </div>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </motion.div>

        {/* 결과 표시 - XL에서 5컬럼 차지 */}
        <motion.div variants={itemVariants} className="xl:col-span-5 h-full">
          <Card className="border-none shadow-sm rounded-3xl bg-white overflow-hidden h-full flex flex-col">
            <CardHeader className="p-8">
              <div className="flex items-center gap-2 mb-1">
                <Activity className="w-4 h-4 text-purple-600" />
                <CardTitle className="text-lg font-bold text-gray-900">예측 대시보드</CardTitle>
              </div>
              <CardDescription className="text-[11px] font-bold text-gray-400 uppercase tracking-widest">
                AI 분석 엔진의 최종 판단 결과입니다.
              </CardDescription>
            </CardHeader>
            <CardContent className="p-8 flex-1 flex flex-col">
              <AnimatePresence mode="wait">
                {result ? (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="space-y-8 flex-1 flex flex-col"
                  >
                    <div className="bg-gray-50/50 rounded-[2rem] p-10 text-center relative overflow-hidden border border-gray-100">
                      <div className="absolute top-0 right-0 w-32 h-32 bg-blue-600/5 rounded-full -mr-16 -mt-16 blur-3xl"></div>
                      <div className="absolute bottom-0 left-0 w-32 h-32 bg-purple-600/5 rounded-full -ml-16 -mb-16 blur-3xl"></div>

                      <p className="text-[10px] font-black text-gray-400 uppercase tracking-[0.3em] mb-4">폐암 발병 확률</p>
                      <div className="relative inline-block">
                        <span className={`text-7xl font-black tracking-tighter ${result.probability > 70 ? 'text-red-600' : result.probability > 40 ? 'text-amber-500' : 'text-emerald-500'
                          }`}>
                          {result.probability}
                        </span>
                        <span className="text-xl font-black text-gray-300 ml-1">%</span>
                      </div>

                      <div className="mt-6">
                        <Badge
                          className={`px-6 py-2 rounded-full text-xs font-black uppercase tracking-widest border ${getRiskColor(result.risk_level)}`}
                        >
                          {result.risk_level} 위험도군
                        </Badge>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div className={`p-6 rounded-3xl border-2 flex flex-col items-center justify-center gap-2 ${result.prediction === "YES" ? "bg-red-50 border-red-100" : "bg-emerald-50 border-emerald-100"
                        }`}>
                        <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest">최종 판정</p>
                        <h4 className={`text-xl font-black ${result.prediction === "YES" ? "text-red-600" : "text-emerald-600"}`}>
                          {result.prediction === "YES" ? "양성 (Positive)" : "음성 (Negative)"}
                        </h4>
                      </div>
                      <div className="p-6 rounded-3xl bg-gray-50 border border-gray-100 flex flex-col items-center justify-center gap-2">
                        <p className="text-[10px] font-black text-gray-400 uppercase tracking-widest">데이터 정합성</p>
                        <h4 className="text-xl font-black text-gray-900">
                          {result.external_db_saved ? "검증됨" : "확인 중"}
                        </h4>
                      </div>
                    </div>

                    <div className="bg-blue-50/50 rounded-3xl p-6 border border-blue-100/50 flex-1">
                      <div className="flex gap-4">
                        <div className="bg-blue-600 p-2 rounded-xl h-fit">
                          <Info className="w-5 h-5 text-white" />
                        </div>
                        <div className="space-y-2">
                          <p className="text-[10px] font-black text-blue-600 uppercase tracking-widest">의료진 가이드라인</p>
                          <p className="text-sm font-bold text-blue-900/80 leading-relaxed italic">
                            "{result.risk_message}"
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="pt-4 border-t border-gray-50 space-y-3">
                      <div className="flex justify-between items-center text-[10px] font-bold uppercase tracking-widest text-gray-400">
                        <span>Patient ID</span>
                        <span className="text-gray-900">{result.patient_id}</span>
                      </div>
                      <Button
                        variant="ghost"
                        className="w-full rounded-2xl h-12 text-xs font-black text-gray-400 hover:text-blue-600 hover:bg-blue-50 transition-all"
                        onClick={() => setResult(null)}
                      >
                        결과 초기화 및 재입력
                      </Button>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex-1 flex flex-col items-center justify-center text-center p-12"
                  >
                    <div className="w-24 h-24 bg-gray-50 rounded-[2rem] flex items-center justify-center mb-6">
                      <Activity className="w-10 h-10 text-gray-200" />
                    </div>
                    <h4 className="text-lg font-bold text-gray-900 mb-2">예측 결과 대기 중</h4>
                    <p className="text-xs font-medium text-gray-400 leading-relaxed">
                      좌측 폼에 환자 데이터를 입력하고<br />
                      예측 버튼을 클릭하면 결과가 여기에 표시됩니다.
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </motion.div>
  );
}
