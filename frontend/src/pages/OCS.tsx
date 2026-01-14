import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/context/AuthContext";
import {
  Plus,
  Search,
  Send,
  CheckCircle,
  XCircle,
  Clock,
  AlertTriangle,
  FileText,
  Pill,
  FlaskConical,
  Scan,
  RefreshCw,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import {
  getOrdersApi,
  createOrderApi,
  sendOrderApi,
  startProcessingOrderApi,
  completeOrderApi,
  cancelOrderApi,
  getOrderStatisticsApi,
  getMyOrdersApi,
  getPendingOrdersApi,
  searchPatientsApi,
  createImagingAnalysisApi,
} from "@/lib/api";
import { format } from "date-fns";
import { useNavigate } from "react-router-dom";

interface Order {
  id: string;
  order_type: "prescription" | "lab_test" | "imaging";
  patient: string;
  patient_name: string;
  patient_number: string;
  doctor: number;
  doctor_name: string;
  status: "pending" | "sent" | "processing" | "completed" | "cancelled";
  priority: "routine" | "urgent" | "stat" | "emergency";
  order_data: any;
  target_department: string;
  validation_passed: boolean;
  validation_notes: string;
  created_at: string;
  due_time?: string;
  completed_at?: string;
  notes?: string;
  drug_interaction_checks?: any[];
  allergy_checks?: any[];
  imaging_analysis?: {
    id: string;
    findings: string;
    recommendations: string;
    confidence_score: number;
  };
}

const ORDER_TYPE_LABELS = {
  prescription: "처방전",
  lab_test: "검사",
  imaging: "영상촬영",
};

const STATUS_LABELS = {
  pending: "대기중",
  sent: "전달됨",
  processing: "처리중",
  completed: "완료",
  cancelled: "취소",
};

const PRIORITY_LABELS = {
  routine: "일반",
  urgent: "긴급",
  stat: "즉시",
  emergency: "응급",
};

const STATUS_COLORS = {
  pending: "bg-yellow-100 text-yellow-800",
  sent: "bg-blue-100 text-blue-800",
  processing: "bg-purple-100 text-purple-800",
  completed: "bg-green-100 text-green-800",
  cancelled: "bg-red-100 text-red-800",
};

const PRIORITY_COLORS = {
  routine: "bg-gray-100 text-gray-800",
  urgent: "bg-orange-100 text-orange-800",
  stat: "bg-red-100 text-red-800",
  emergency: "bg-red-200 text-red-900",
};

export default function OCS() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const [selectedPatient, setSelectedPatient] = useState<any>(null);
  const [patientSearchTerm, setPatientSearchTerm] = useState("");
  const [viewMode, setViewMode] = useState<"all" | "my" | "pending">("all");

  // 주문 목록 조회 (역할별 자동 필터링)
  const { data: ordersData, isLoading } = useQuery({
    queryKey: ["ocs-orders", statusFilter, typeFilter, viewMode],
    queryFn: () => {
      if (viewMode === "my") {
        return getMyOrdersApi();
      } else if (viewMode === "pending") {
        // 부서별로 자동 필터링됨
        return getPendingOrdersApi(user?.department || undefined);
      } else {
        return getOrdersApi({ 
          status: statusFilter !== "all" ? statusFilter : undefined, 
          order_type: typeFilter !== "all" ? typeFilter : undefined 
        });
      }
    },
  });

  // 통계 조회
  const { data: statistics } = useQuery({
    queryKey: ["ocs-statistics"],
    queryFn: getOrderStatisticsApi,
  });

  // 환자 검색
  const { data: patients, isLoading: isSearchingPatients } = useQuery({
    queryKey: ["search-patients", patientSearchTerm],
    queryFn: () => searchPatientsApi(patientSearchTerm),
    enabled: patientSearchTerm.length > 0,
  });

  // 주문 생성
  const createOrderMutation = useMutation({
    mutationFn: createOrderApi,
    onSuccess: () => {
      toast({
        title: "주문 생성 완료",
        description: "주문이 성공적으로 생성되었습니다.",
      });
      queryClient.invalidateQueries({ queryKey: ["ocs-orders"] });
      queryClient.invalidateQueries({ queryKey: ["ocs-statistics"] });
      setIsCreateDialogOpen(false);
      setSelectedPatient(null);
    },
    onError: (error: any) => {
      console.error("주문 생성 에러:", error);
      console.error("에러 응답:", error.response?.data);
      
      let errorMessage = "주문 생성에 실패했습니다.";
      
      if (error.response?.data) {
        const data = error.response.data;
        
        // details 객체가 있는 경우 (serializer validation errors)
        if (data.details) {
          const details = data.details;
          const errorMessages: string[] = [];
          
          // 각 필드별 에러 메시지 수집
          Object.keys(details).forEach((key) => {
            const fieldErrors = details[key];
            if (Array.isArray(fieldErrors)) {
              errorMessages.push(`${key}: ${fieldErrors.join(", ")}`);
            } else if (typeof fieldErrors === "string") {
              errorMessages.push(`${key}: ${fieldErrors}`);
            } else {
              errorMessages.push(`${key}: ${JSON.stringify(fieldErrors)}`);
            }
          });
          
          errorMessage = errorMessages.length > 0 
            ? errorMessages.join("\n")
            : data.error || data.detail || errorMessage;
        } else if (data.error) {
          errorMessage = typeof data.error === "string" ? data.error : JSON.stringify(data.error);
        } else if (data.detail) {
          errorMessage = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
        } else if (data.message) {
          errorMessage = data.message;
        } else {
          // 첫 번째 필드의 에러 메시지 사용
          const firstKey = Object.keys(data)[0];
          if (firstKey) {
            const val = data[firstKey];
            if (Array.isArray(val)) {
              errorMessage = `${firstKey}: ${val.join(", ")}`;
            } else if (typeof val === "string") {
              errorMessage = `${firstKey}: ${val}`;
            } else {
              errorMessage = JSON.stringify(data);
            }
          }
        }
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      toast({
        title: "주문 생성 실패",
        description: errorMessage,
        variant: "destructive",
      });
    },
  });

  // 주문 전달
  const sendOrderMutation = useMutation({
    mutationFn: sendOrderApi,
    onSuccess: () => {
      toast({
        title: "주문 전달 완료",
        description: "주문이 성공적으로 전달되었습니다.",
      });
      queryClient.invalidateQueries({ queryKey: ["ocs-orders"] });
    },
    onError: (error: any) => {
      toast({
        title: "주문 전달 실패",
        description: error.response?.data?.error || "주문 전달에 실패했습니다.",
        variant: "destructive",
      });
    },
  });

  // 주문 처리 시작
  const startProcessingMutation = useMutation({
    mutationFn: startProcessingOrderApi,
    onSuccess: () => {
      toast({
        title: "처리 시작",
        description: "주문 처리를 시작했습니다.",
      });
      queryClient.invalidateQueries({ queryKey: ["ocs-orders"] });
    },
  });

  // 주문 완료
  const completeOrderMutation = useMutation({
    mutationFn: completeOrderApi,
    onSuccess: () => {
      toast({
        title: "주문 완료",
        description: "주문이 완료 처리되었습니다.",
      });
      queryClient.invalidateQueries({ queryKey: ["ocs-orders"] });
    },
  });

  // 주문 취소
  const cancelOrderMutation = useMutation({
    mutationFn: ({ id, reason }: { id: string; reason?: string }) => cancelOrderApi(id, reason),
    onSuccess: () => {
      toast({
        title: "주문 취소 완료",
        description: "주문이 취소되었습니다.",
      });
      queryClient.invalidateQueries({ queryKey: ["ocs-orders"] });
    },
  });

  const orders = ordersData?.results || ordersData || [];

  const filteredOrders = orders.filter((order: Order) => {
    if (searchTerm) {
      const searchLower = searchTerm.toLowerCase();
      return (
        order.patient_name?.toLowerCase().includes(searchLower) ||
        order.patient_number?.toLowerCase().includes(searchLower) ||
        order.doctor_name?.toLowerCase().includes(searchLower)
      );
    }
    return true;
  });

  const handleCreateOrder = (formData: any) => {
    if (!selectedPatient) {
      toast({
        title: "환자 선택 필요",
        description: "환자를 선택해주세요.",
        variant: "destructive",
      });
      return;
    }

    // pk (숫자 ID) 또는 id (문자열 patient_id) 사용
    // pk가 있으면 숫자 ID로, 없으면 patient_id 문자열로 전달
    const orderData = {
      ...formData,
    };
    
    if (selectedPatient.pk !== undefined) {
      // 숫자 primary key가 있는 경우
      orderData.patient = selectedPatient.pk;
    } else if (selectedPatient.id && typeof selectedPatient.id === 'number') {
      // 숫자 id가 있는 경우
      orderData.patient = selectedPatient.id;
    } else {
      // 문자열 patient_id인 경우
      const patientId = selectedPatient.patient_id || selectedPatient.id;
      orderData.patient_id = patientId;
    }

    console.log("주문 생성 데이터:", orderData);
    console.log("선택된 환자:", selectedPatient);
    createOrderMutation.mutate(orderData);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* 헤더 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">처방전달시스템 (OCS)</h1>
          <p className="text-muted-foreground mt-1">
            {user?.department && (
              <span className="font-medium">{user.department}</span>
            )}
            {user?.department && " | "}
            처방전, 검사, 영상촬영 주문을 관리합니다.
          </p>
        </div>
        <div className="flex gap-2">
          {/* 부서별로 주문 생성 버튼 표시 */}
          {(() => {
            // 원무과, 영상의학과, 방사선과는 주문 생성 불가
            if (user?.department === "원무과" || 
                user?.department === "영상의학과" || 
                user?.department === "방사선과") {
              return null;
            }
            // 의료진(외과, 호흡기내과 등) 또는 superuser만 생성 가능
            if (user?.role === "medical_staff" || user?.role === "superuser") {
              return (
                <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
                  <DialogTrigger asChild>
                    <Button>
                      <Plus className="mr-2 h-4 w-4" />
                      주문 생성
                    </Button>
                  </DialogTrigger>
          <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>새 주문 생성</DialogTitle>
              <DialogDescription>
                처방전, 검사, 또는 영상촬영 주문을 생성합니다.
              </DialogDescription>
            </DialogHeader>
            <CreateOrderForm
              selectedPatient={selectedPatient}
              setSelectedPatient={setSelectedPatient}
              patientSearchTerm={patientSearchTerm}
              setPatientSearchTerm={setPatientSearchTerm}
              patients={patients || []}
              isSearchingPatients={isSearchingPatients}
              onSubmit={handleCreateOrder}
              isLoading={createOrderMutation.isPending}
            />
          </DialogContent>
        </Dialog>
              );
            }
            return null;
          })()}
        </div>
      </div>

      {/* 뷰 모드 선택 (의료진만) */}
      {user?.role === "medical_staff" && (
        <Card>
          <CardContent className="pt-6">
            <div className="flex gap-2">
              <Button
                variant={viewMode === "all" ? "default" : "outline"}
                size="sm"
                onClick={() => setViewMode("all")}
              >
                전체 주문
              </Button>
              <Button
                variant={viewMode === "my" ? "default" : "outline"}
                size="sm"
                onClick={() => setViewMode("my")}
              >
                내 주문
              </Button>
              <Button
                variant={viewMode === "pending" ? "default" : "outline"}
                size="sm"
                onClick={() => setViewMode("pending")}
              >
                대기 중인 주문
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 통계 카드 */}
      {statistics && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">오늘 주문</CardTitle>
              <FileText className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{statistics.total_orders_today || 0}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">대기 중</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {statistics.orders_by_status?.find((s: any) => s.status === "pending")?.count || 0}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">긴급 주문</CardTitle>
              <AlertTriangle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{statistics.urgent_orders_pending || 0}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">검증 실패</CardTitle>
              <XCircle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{statistics.validation_failed_orders || 0}</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* 필터 및 검색 */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="환자명, 환자번호, 의사명으로 검색..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-8"
                />
              </div>
            </div>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="상태 필터" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">전체 상태</SelectItem>
                <SelectItem value="pending">대기중</SelectItem>
                <SelectItem value="sent">전달됨</SelectItem>
                <SelectItem value="processing">처리중</SelectItem>
                <SelectItem value="completed">완료</SelectItem>
                <SelectItem value="cancelled">취소</SelectItem>
              </SelectContent>
            </Select>
            <Select value={typeFilter} onValueChange={setTypeFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="유형 필터" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">전체 유형</SelectItem>
                <SelectItem value="prescription">처방전</SelectItem>
                <SelectItem value="lab_test">검사</SelectItem>
                <SelectItem value="imaging">영상촬영</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* 주문 목록 */}
      <div className="space-y-4">
        {isLoading ? (
          <Card>
            <CardContent className="py-12 text-center">
              <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-muted-foreground" />
              <p className="text-muted-foreground">로딩 중...</p>
            </CardContent>
          </Card>
        ) : filteredOrders.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center">
              <FileText className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
              <p className="text-muted-foreground">주문이 없습니다.</p>
            </CardContent>
          </Card>
        ) : (
          filteredOrders.map((order: Order) => (
            <OrderCard
              key={order.id}
              order={order}
              user={user}
              onSend={() => sendOrderMutation.mutate(order.id)}
              onStartProcessing={() => startProcessingMutation.mutate(order.id)}
              onComplete={() => completeOrderMutation.mutate(order.id)}
              onCancel={(reason) => cancelOrderMutation.mutate({ id: order.id, reason })}
              isSending={sendOrderMutation.isPending}
              isCompleting={completeOrderMutation.isPending}
              onCreateAnalysis={createImagingAnalysisApi}
              onViewAnalysis={(analysisId) => navigate(`/ocs/imaging-analysis/${analysisId}?order=${order.id}`)}
            />
          ))
        )}
      </div>
    </div>
  );
}

function OrderCard({
  order,
  user,
  onSend,
  onStartProcessing,
  onComplete,
  onCancel,
  isSending,
  isCompleting,
  onCreateAnalysis,
  onViewAnalysis,
}: {
  order: Order;
  user: any;
  onSend: () => void;
  onStartProcessing?: () => void;
  onComplete: () => void;
  onCancel: (reason: string) => void;
  isSending: boolean;
  isCompleting: boolean;
  onCreateAnalysis?: (data: any) => Promise<any>;
  onViewAnalysis?: (analysisId: string) => void;
}) {
  const [showAnalysisDialog, setShowAnalysisDialog] = useState(false);
  const [findings, setFindings] = useState("");
  const [recommendations, setRecommendations] = useState("");
  const [confidenceScore, setConfidenceScore] = useState(0.95);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const handleCreateAnalysis = async () => {
    if (!onCreateAnalysis) return;
    
    try {
      await onCreateAnalysis({
        order: order.id,
        findings,
        recommendations,
        confidence_score: confidenceScore,
        analysis_result: {},
      });
      toast({
        title: "분석 결과 생성 완료",
        description: "의사에게 알림이 전송되었습니다.",
      });
      queryClient.invalidateQueries({ queryKey: ["ocs-orders"] });
      setShowAnalysisDialog(false);
      setFindings("");
      setRecommendations("");
    } catch (error: any) {
      toast({
        title: "분석 결과 생성 실패",
        description: error.response?.data?.detail || "분석 결과 생성에 실패했습니다.",
        variant: "destructive",
      });
    }
  };
  const getOrderIcon = () => {
    switch (order.order_type) {
      case "prescription":
        return <Pill className="h-5 w-5" />;
      case "lab_test":
        return <FlaskConical className="h-5 w-5" />;
      case "imaging":
        return <Scan className="h-5 w-5" />;
      default:
        return <FileText className="h-5 w-5" />;
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-4">
            <div className="p-2 bg-primary/10 rounded-lg">{getOrderIcon()}</div>
            <div>
              <CardTitle className="text-lg">
                {ORDER_TYPE_LABELS[order.order_type]} - {order.patient_name}
              </CardTitle>
              <CardDescription className="mt-1">
                환자번호: {order.patient_number} | 의사: {order.doctor_name}
              </CardDescription>
            </div>
          </div>
          <div className="flex gap-2">
            <Badge className={STATUS_COLORS[order.status]}>
              {STATUS_LABELS[order.status]}
            </Badge>
            <Badge className={PRIORITY_COLORS[order.priority]}>
              {PRIORITY_LABELS[order.priority]}
            </Badge>
            {!order.validation_passed && (
              <Badge variant="destructive">검증 실패</Badge>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* 주문 내용 */}
          <div>
            <h4 className="text-sm font-medium mb-2">주문 내용</h4>
            <div className="text-sm text-muted-foreground">
              {order.order_type === "prescription" && (
                <div>
                  약물: {order.order_data?.medications?.map((m: any) => m.name).join(", ") || "없음"}
                </div>
              )}
              {order.order_type === "lab_test" && (
                <div>
                  검사 항목: {order.order_data?.test_items?.map((t: any) => t.name).join(", ") || "없음"}
                </div>
              )}
              {order.order_type === "imaging" && (
                <div>
                  촬영 유형: {order.order_data?.imaging_type || "없음"} | 부위: {order.order_data?.body_part || "없음"}
                  {order.order_data?.contrast && (
                    <Badge variant="outline" className="ml-2">조영제 사용</Badge>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* 검증 결과 */}
          {order.validation_notes && (
            <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <div className="flex items-start gap-2">
                <AlertTriangle className="h-4 w-4 text-yellow-600 mt-0.5" />
                <div className="text-sm">
                  <p className="font-medium text-yellow-800 dark:text-yellow-200">검증 메모</p>
                  <p className="text-yellow-700 dark:text-yellow-300">{order.validation_notes}</p>
                </div>
              </div>
            </div>
          )}

          {/* 약물 상호작용 경고 */}
          {order.drug_interaction_checks && order.drug_interaction_checks.length > 0 && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <div className="flex items-start gap-2">
                <AlertTriangle className="h-4 w-4 text-red-600 mt-0.5" />
                <div className="text-sm">
                  <p className="font-medium text-red-800 dark:text-red-200">약물 상호작용 경고</p>
                  {order.drug_interaction_checks.map((check: any, idx: number) => (
                    <p key={idx} className="text-red-700 dark:text-red-300">
                      {check.severity === "severe" && "⚠️ 심각: "}
                      {check.interactions?.map((i: any) => `${i.drug1} + ${i.drug2}: ${i.description}`).join(", ")}
                    </p>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* 알레르기 경고 */}
          {order.allergy_checks && order.allergy_checks.some((check: any) => check.has_allergy_risk) && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <div className="flex items-start gap-2">
                <AlertTriangle className="h-4 w-4 text-red-600 mt-0.5" />
                <div className="text-sm">
                  <p className="font-medium text-red-800 dark:text-red-200">알레르기 위험</p>
                  {order.allergy_checks
                    .filter((check: any) => check.has_allergy_risk)
                    .map((check: any, idx: number) => (
                      <p key={idx} className="text-red-700 dark:text-red-300">
                        {check.warnings?.map((w: any) => w.description).join(", ")}
                      </p>
                    ))}
                </div>
              </div>
            </div>
          )}

          {/* 일시 정보 */}
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Clock className="h-4 w-4" />
              생성: {format(new Date(order.created_at), "yyyy-MM-dd HH:mm")}
            </div>
            {order.due_time && (
              <div className="flex items-center gap-1">
                <Clock className="h-4 w-4" />
                기한: {format(new Date(order.due_time), "yyyy-MM-dd HH:mm")}
              </div>
            )}
            {order.completed_at && (
              <div className="flex items-center gap-1">
                <CheckCircle className="h-4 w-4" />
                완료: {format(new Date(order.completed_at), "yyyy-MM-dd HH:mm")}
              </div>
            )}
          </div>

          {/* 영상 분석 결과 */}
          {order.order_type === "imaging" && order.imaging_analysis && (
            <div className="p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <p className="font-medium text-green-800 dark:text-green-200 mb-1">
                    영상 분석 완료
                  </p>
                  <p className="text-sm text-green-700 dark:text-green-300">
                    {order.imaging_analysis.findings?.substring(0, 100) || "분석 결과가 있습니다."}
                    {order.imaging_analysis.findings?.length > 100 && "..."}
                  </p>
                </div>
                {onViewAnalysis && order.imaging_analysis?.id && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => onViewAnalysis(order.imaging_analysis!.id)}
                  >
                    상세 보기
                  </Button>
                )}
              </div>
            </div>
          )}

          {/* 액션 버튼 (역할별 제한) */}
          <div className="flex gap-2 pt-2 flex-wrap">
            {/* 의사: 자신이 생성한 주문만 전달 가능 */}
            {order.status === "pending" && order.validation_passed && (
              <Button onClick={onSend} disabled={isSending} size="sm">
                <Send className="mr-2 h-4 w-4" />
                전달
              </Button>
            )}
            {/* 부서 담당자: 전달된 주문 처리 시작 및 완료 */}
            {order.status === "sent" && (
              <>
                <Button onClick={onStartProcessing} disabled={isCompleting} size="sm" variant="outline">
                  <Clock className="mr-2 h-4 w-4" />
                  처리 시작
                </Button>
                <Button onClick={onComplete} disabled={isCompleting} size="sm" variant="default">
                  <CheckCircle className="mr-2 h-4 w-4" />
                  완료 처리
                </Button>
              </>
            )}
            {order.status === "processing" && (
              <Button onClick={onComplete} disabled={isCompleting} size="sm" variant="default">
                <CheckCircle className="mr-2 h-4 w-4" />
                완료 처리
              </Button>
            )}
            {/* 영상의학과: 영상 분석 결과 입력 */}
            {order.order_type === "imaging" && 
             order.status === "completed" && 
             !order.imaging_analysis &&
             user?.department === "영상의학과" && (
              <Button
                onClick={() => setShowAnalysisDialog(true)}
                size="sm"
                variant="default"
              >
                <Scan className="mr-2 h-4 w-4" />
                분석 결과 입력
              </Button>
            )}
            {/* 주문 생성자 또는 원무과만 취소 가능 */}
            {(order.status === "pending" || order.status === "sent") && (
              <Button
                onClick={() => {
                  const reason = prompt("취소 사유를 입력하세요:");
                  if (reason) onCancel(reason);
                }}
                size="sm"
                variant="destructive"
              >
                <XCircle className="mr-2 h-4 w-4" />
                취소
              </Button>
            )}
          </div>
        </div>
      </CardContent>

      {/* 영상 분석 결과 입력 다이얼로그 */}
      {showAnalysisDialog && (
        <Dialog open={showAnalysisDialog} onOpenChange={setShowAnalysisDialog}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>영상 분석 결과 입력</DialogTitle>
              <DialogDescription>
                {order.patient_name}님의 {order.order_data?.imaging_type} 영상 분석 결과를 입력하세요.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div>
                <Label>소견</Label>
                <Textarea
                  value={findings}
                  onChange={(e) => setFindings(e.target.value)}
                  placeholder="영상 분석 소견을 입력하세요..."
                  rows={5}
                />
              </div>
              <div>
                <Label>권고사항</Label>
                <Textarea
                  value={recommendations}
                  onChange={(e) => setRecommendations(e.target.value)}
                  placeholder="권고사항을 입력하세요..."
                  rows={3}
                />
              </div>
              <div>
                <Label>신뢰도: {(confidenceScore * 100).toFixed(1)}%</Label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={confidenceScore}
                  onChange={(e) => setConfidenceScore(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setShowAnalysisDialog(false)}>
                취소
              </Button>
              <Button onClick={handleCreateAnalysis} disabled={!findings.trim()}>
                분석 결과 저장
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      )}
    </Card>
  );
}

function CreateOrderForm({
  selectedPatient,
  setSelectedPatient,
  patientSearchTerm,
  setPatientSearchTerm,
  patients,
  isSearchingPatients,
  onSubmit,
  isLoading,
}: {
  selectedPatient: any;
  setSelectedPatient: (patient: any) => void;
  patientSearchTerm: string;
  setPatientSearchTerm: (term: string) => void;
  patients: any[];
  isSearchingPatients: boolean;
  onSubmit: (data: any) => void;
  isLoading: boolean;
}) {
  const [orderType, setOrderType] = useState<string>("prescription");
  const [priority, setPriority] = useState<string>("routine");
  const [targetDepartment, setTargetDepartment] = useState<string>("pharmacy");
  const [medications, setMedications] = useState([{ name: "", dosage: "", frequency: "", duration: "" }]);
  const [testItems, setTestItems] = useState([{ name: "", priority: "routine" }]);
  const [imagingData, setImagingData] = useState({ imaging_type: "", body_part: "", contrast: false });
  const [notes, setNotes] = useState("");
  const [dueTime, setDueTime] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    let orderData: any = {};
    let department = targetDepartment;

    if (orderType === "prescription") {
      const validMedications = medications.filter((m) => m.name.trim());
      if (validMedications.length === 0) {
        alert("최소 하나의 약물 정보를 입력해주세요.");
        return;
      }
      orderData = { medications: validMedications };
      department = "pharmacy";
    } else if (orderType === "lab_test") {
      const validTestItems = testItems.filter((t) => t.name.trim());
      if (validTestItems.length === 0) {
        alert("최소 하나의 검사 항목을 입력해주세요.");
        return;
      }
      orderData = { test_items: validTestItems };
      department = "lab";
    } else if (orderType === "imaging") {
      // 촬영 유형 필수 체크
      if (!imagingData.imaging_type || !imagingData.imaging_type.trim()) {
        alert("촬영 유형을 선택해주세요.");
        return;
      }
      // 촬영 부위 필수 체크
      if (!imagingData.body_part || !imagingData.body_part.trim()) {
        alert("촬영 부위를 입력해주세요.");
        return;
      }
      orderData = {
        imaging_type: imagingData.imaging_type,
        body_part: imagingData.body_part,
        contrast: imagingData.contrast || false,
      };
      department = "radiology";
    }

    onSubmit({
      order_type: orderType,
      order_data: orderData,
      target_department: department,
      priority,
      notes,
      due_time: dueTime || undefined,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* 환자 선택 */}
      <div className="space-y-2">
        <Label>환자 검색</Label>
        <Input
          placeholder="환자명 또는 환자번호로 검색..."
          value={patientSearchTerm}
          onChange={(e) => setPatientSearchTerm(e.target.value)}
        />
        {isSearchingPatients && <p className="text-sm text-muted-foreground">검색 중...</p>}
        {patients && patients.length > 0 && (
          <div className="border rounded-lg p-2 max-h-40 overflow-y-auto">
            {patients.map((patient: any) => (
              <div
                key={patient.id || patient.patient_id}
                className="p-2 hover:bg-accent rounded cursor-pointer"
                onClick={() => {
                  setSelectedPatient(patient);
                  setPatientSearchTerm("");
                }}
              >
                {patient.name} ({patient.patient_id || patient.patient_number})
              </div>
            ))}
          </div>
        )}
        {selectedPatient && (
          <div className="p-2 bg-accent rounded">
            선택된 환자: {selectedPatient.name} ({selectedPatient.patient_id || selectedPatient.patient_number})
          </div>
        )}
      </div>

      {/* 주문 유형 */}
      <div className="space-y-2">
        <Label>주문 유형</Label>
        <Select value={orderType} onValueChange={(value) => {
          setOrderType(value);
          if (value === "prescription") setTargetDepartment("pharmacy");
          else if (value === "lab_test") setTargetDepartment("lab");
          else if (value === "imaging") setTargetDepartment("radiology");
        }}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {/* 의료진(외과, 호흡기내과 등)은 모든 주문 유형 생성 가능 */}
            <SelectItem value="prescription">처방전</SelectItem>
            <SelectItem value="lab_test">검사</SelectItem>
            <SelectItem value="imaging">영상촬영</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* 주문 내용 */}
      {orderType === "prescription" && (
        <div className="space-y-2">
          <Label>약물 정보</Label>
          {medications.map((med, idx) => (
            <div key={idx} className="grid grid-cols-4 gap-2">
              <Input
                placeholder="약물명"
                value={med.name}
                onChange={(e) => {
                  const newMeds = [...medications];
                  newMeds[idx].name = e.target.value;
                  setMedications(newMeds);
                }}
              />
              <Input
                placeholder="용량"
                value={med.dosage}
                onChange={(e) => {
                  const newMeds = [...medications];
                  newMeds[idx].dosage = e.target.value;
                  setMedications(newMeds);
                }}
              />
              <Input
                placeholder="용법"
                value={med.frequency}
                onChange={(e) => {
                  const newMeds = [...medications];
                  newMeds[idx].frequency = e.target.value;
                  setMedications(newMeds);
                }}
              />
              <Input
                placeholder="기간"
                value={med.duration}
                onChange={(e) => {
                  const newMeds = [...medications];
                  newMeds[idx].duration = e.target.value;
                  setMedications(newMeds);
                }}
              />
            </div>
          ))}
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setMedications([...medications, { name: "", dosage: "", frequency: "", duration: "" }])}
          >
            <Plus className="mr-2 h-4 w-4" />
            약물 추가
          </Button>
        </div>
      )}

      {orderType === "lab_test" && (
        <div className="space-y-2">
          <Label>검사 항목</Label>
          {testItems.map((item, idx) => (
            <div key={idx} className="flex gap-2">
              <Input
                placeholder="검사명"
                value={item.name}
                onChange={(e) => {
                  const newItems = [...testItems];
                  newItems[idx].name = e.target.value;
                  setTestItems(newItems);
                }}
                className="flex-1"
              />
              <Select
                value={item.priority}
                onValueChange={(value) => {
                  const newItems = [...testItems];
                  newItems[idx].priority = value;
                  setTestItems(newItems);
                }}
              >
                <SelectTrigger className="w-[120px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="routine">일반</SelectItem>
                  <SelectItem value="urgent">긴급</SelectItem>
                </SelectContent>
              </Select>
            </div>
          ))}
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setTestItems([...testItems, { name: "", priority: "routine" }])}
          >
            <Plus className="mr-2 h-4 w-4" />
            검사 항목 추가
          </Button>
        </div>
      )}

      {orderType === "imaging" && (
        <div className="space-y-2">
          <Label>촬영 정보</Label>
          <Select
            value={imagingData.imaging_type}
            onValueChange={(value) => setImagingData({ ...imagingData, imaging_type: value })}
          >
            <SelectTrigger>
              <SelectValue placeholder="촬영 유형" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="MRI">MRI</SelectItem>
              <SelectItem value="CT">CT</SelectItem>
              <SelectItem value="X-RAY">X-RAY</SelectItem>
              <SelectItem value="ULTRASOUND">초음파</SelectItem>
            </SelectContent>
          </Select>
          <Input
            placeholder="촬영 부위"
            value={imagingData.body_part}
            onChange={(e) => setImagingData({ ...imagingData, body_part: e.target.value })}
          />
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="contrast"
              checked={imagingData.contrast}
              onChange={(e) => setImagingData({ ...imagingData, contrast: e.target.checked })}
            />
            <Label htmlFor="contrast">조영제 사용</Label>
          </div>
        </div>
      )}

      {/* 우선순위 */}
      <div className="space-y-2">
        <Label>우선순위</Label>
        <Select value={priority} onValueChange={setPriority}>
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="routine">일반</SelectItem>
            <SelectItem value="urgent">긴급</SelectItem>
            <SelectItem value="stat">즉시</SelectItem>
            <SelectItem value="emergency">응급</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* 완료 기한 */}
      <div className="space-y-2">
        <Label>완료 기한 (선택)</Label>
        <Input
          type="datetime-local"
          value={dueTime}
          onChange={(e) => setDueTime(e.target.value)}
        />
      </div>

      {/* 메모 */}
      <div className="space-y-2">
        <Label>메모</Label>
        <Textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="추가 메모를 입력하세요..."
        />
      </div>

      <div className="flex justify-end gap-2">
        <Button type="submit" disabled={isLoading || !selectedPatient}>
          {isLoading ? "생성 중..." : "주문 생성"}
        </Button>
      </div>
    </form>
  );
}
