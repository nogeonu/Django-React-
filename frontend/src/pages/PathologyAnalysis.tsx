import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { 
  Scan, 
  Brain,
  Loader2,
  Search,
  User,
  Calendar,
  CheckCircle2
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { getOrdersApi } from '@/lib/api';

interface Order {
  id: string;
  patient_name: string;
  patient_id: string;
  patient_number?: string;
  order_type: 'prescription' | 'lab_test' | 'imaging' | 'tissue_exam';
  order_data?: {
    imaging_type?: string;
    body_part?: string;
  };
  status: string;
  created_at: string;
}

// 교육원 워커 wsi/ 폴더에 있는 사용 가능한 파일 목록
const AVAILABLE_WSI_FILES = [
  { value: 'tumor_083.tif', label: 'tumor_083.tif (종양)' },
  { value: 'normal_059.tif', label: 'normal_059.tif (정상)' },
  { value: 'normal_103.tif', label: 'normal_103.tif (정상)' },
] as const;

export default function PathologyAnalysis() {
  const { toast } = useToast();
  const [orders, setOrders] = useState<Order[]>([]);
  const [filteredOrders, setFilteredOrders] = useState<Order[]>([]);
  const [selectedOrder, setSelectedOrder] = useState<Order | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [loadingOrders, setLoadingOrders] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [selectedFilename, setSelectedFilename] = useState<string>('tumor_083.tif'); // 기본값

  useEffect(() => {
    loadOrders();
  }, []);

  useEffect(() => {
    if (searchTerm.trim()) {
      const filtered = orders.filter(order => 
        order.patient_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        order.patient_id?.toLowerCase().includes(searchTerm.toLowerCase()) ||
        order.patient_number?.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredOrders(filtered);
    } else {
      setFilteredOrders(orders);
    }
  }, [searchTerm, orders]);

  const loadOrders = async () => {
    setLoadingOrders(true);
    try {
      // 조직검사 주문 중 처리 중(processing) 상태인 것만 가져오기
      // 검사실에서 처리 시작을 누른 주문만 표시
      const data = await getOrdersApi({
        order_type: 'tissue_exam',
        target_department: 'lab',
        status: 'processing',  // 처리 중 상태만
      });
      
      // 결과를 배열로 변환 (data.results 또는 data 자체가 배열)
      const orders = data.results || data || [];
      
      setOrders(orders);
      setFilteredOrders(orders);
    } catch (error: any) {
      console.error('주문 로드 실패:', error);
      toast({
        title: "주문 로드 실패",
        description: error.response?.data?.error || "주문 목록을 불러올 수 없습니다.",
        variant: "destructive",
      });
    } finally {
      setLoadingOrders(false);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedOrder) {
      toast({
        title: "주문 선택 필요",
        description: "분석할 주문을 선택해주세요.",
        variant: "destructive",
      });
      return;
    }

    setAnalyzing(true);
    try {
      // OCS 주문에서 환자 정보 가져오기
      const patientId = selectedOrder.patient_id || selectedOrder.patient_number;
      
      if (!patientId) {
        throw new Error('환자 ID를 찾을 수 없습니다.');
      }

      // 사용자가 선택한 파일명 사용 (교육원 워커 wsi/ 폴더에 있는 파일)
      const filename = selectedFilename;
      
      if (!filename) {
        throw new Error('분석할 파일을 선택해주세요.');
      }
      
      // instance_id는 참고용으로만 사용 (실제 파일은 교육원 워커가 wsi/ 폴더에서 찾음)
      const instanceId = `pathology_${selectedOrder.id}`;
      
      toast({
        title: "병리 이미지 분석 시작",
        description: "교육원 워커로 분석 요청을 전송했습니다. (약 2-5분 소요)",
      });

      // 교육원 워커로 API 신호 전송
      // 타임아웃: 50분 (3000초) - 병리 이미지 분석은 시간이 오래 걸림
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000 * 1000); // 50분
      
      try {
        const response = await fetch('/api/pathology/analyze/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include', // 쿠키 포함 (인증 정보)
          signal: controller.signal, // 타임아웃 제어
          body: JSON.stringify({
            instance_id: instanceId || `pathology_${selectedOrder.id}`, // 참고용
            filename: filename // 교육원 워커가 wsi/ 폴더에서 찾을 파일명
          }),
        });
        
        clearTimeout(timeoutId);

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.error || `서버 오류 (${response.status})`);
        }
      } catch (error: any) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') {
          throw new Error('요청 시간 초과 (50분)');
        }
        throw error;
      }

      toast({
        title: "분석 요청 완료",
        description: "교육원 워커에서 분석을 진행 중입니다. 결과는 나중에 확인할 수 있습니다.",
      });

      // 주문 목록 새로고침
      loadOrders();

    } catch (error: any) {
      console.error('병리 이미지 분석 오류:', error);
      toast({
        title: "분석 요청 실패",
        description: error.message || "분석 요청 중 오류가 발생했습니다.",
        variant: "destructive",
      });
    } finally {
      setAnalyzing(false);
    }
  };

  const getStatusBadge = (status: string) => {
    const statusConfig: Record<string, { label: string; variant: "default" | "secondary" | "destructive" | "outline" }> = {
      'pending': { label: '대기중', variant: 'outline' },
      'sent': { label: '전달됨', variant: 'secondary' },
      'processing': { label: '처리중', variant: 'default' },
      'completed': { label: '완료', variant: 'default' },
      'cancelled': { label: '취소됨', variant: 'destructive' },
    };
    const config = statusConfig[status] || { label: status, variant: 'outline' as const };
    return <Badge variant={config.variant}>{config.label}</Badge>;
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">병리 이미지 분석</h1>
          <p className="text-muted-foreground mt-1">
            OCS 주문 기반 병리 이미지 AI 분석 (교육원 워커 연동)
          </p>
        </div>
        {selectedOrder && (
          <Button 
            onClick={handleAnalyze}
            disabled={analyzing}
            className="bg-primary hover:bg-primary/90"
          >
            {analyzing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                분석 중...
              </>
            ) : (
              <>
                <Brain className="mr-2 h-4 w-4" />
                AI 분석 시작
              </>
            )}
          </Button>
        )}
      </div>

      {/* Search Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            병리 이미지 주문 검색
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="환자 이름 또는 환자번호로 검색..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Orders List */}
      <Card>
        <CardHeader>
          <CardTitle>병리 이미지 주문 목록 ({filteredOrders.length}개)</CardTitle>
        </CardHeader>
        <CardContent>
          {loadingOrders ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : filteredOrders.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              병리 이미지 주문이 없습니다.
            </div>
          ) : (
            <div className="space-y-2">
              {filteredOrders.map((order) => (
                <div
                  key={order.id}
                  onClick={() => setSelectedOrder(order)}
                  className={`p-4 border rounded-lg cursor-pointer transition-all ${
                    selectedOrder?.id === order.id
                      ? 'border-primary bg-primary/5'
                      : 'hover:bg-accent'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <User className="h-4 w-4 text-muted-foreground" />
                        <span className="font-semibold">{order.patient_name}</span>
                        <span className="text-sm text-muted-foreground">
                          ({order.patient_id || order.patient_number})
                        </span>
                        {getStatusBadge(order.status)}
                      </div>
                      <div className="flex items-center gap-4 text-sm text-muted-foreground">
                        <div className="flex items-center gap-1">
                          <Scan className="h-3 w-3" />
                          <span>병리 이미지</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Calendar className="h-3 w-3" />
                          <span>{new Date(order.created_at).toLocaleDateString('ko-KR')}</span>
                        </div>
                      </div>
                    </div>
                    {selectedOrder?.id === order.id && (
                      <CheckCircle2 className="h-5 w-5 text-primary" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Selected Order Info */}
      {selectedOrder && (
        <Card>
          <CardHeader>
            <CardTitle>선택된 주문 정보</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">환자명</p>
                <p className="font-semibold">{selectedOrder.patient_name}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">환자 ID</p>
                <p className="font-semibold">{selectedOrder.patient_id || selectedOrder.patient_number}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">주문 상태</p>
                {getStatusBadge(selectedOrder.status)}
              </div>
              <div>
                <p className="text-sm text-muted-foreground">주문일시</p>
                <p className="font-semibold">
                  {new Date(selectedOrder.created_at).toLocaleString('ko-KR')}
                </p>
              </div>
            </div>
            <div className="pt-4 border-t space-y-4">
              <div>
                <Label className="mb-2 block">분석할 파일 선택 (교육원 워커 wsi/ 폴더)</Label>
                <Select
                  value={selectedFilename}
                  onValueChange={(value) => setSelectedFilename(value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="파일 선택" />
                  </SelectTrigger>
                  <SelectContent>
                    {AVAILABLE_WSI_FILES.map((file) => (
                      <SelectItem key={file.value} value={file.value}>
                        {file.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground mt-1">
                  교육원 워커가 wsi/ 폴더에서 찾을 파일명입니다.
                </p>
              </div>
              <Button 
                onClick={handleAnalyze}
                disabled={analyzing || selectedOrder.status === 'completed' || !selectedFilename}
                className="w-full bg-primary hover:bg-primary/90"
              >
                {analyzing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    교육원 워커로 분석 요청 중...
                  </>
                ) : (
                  <>
                    <Brain className="mr-2 h-4 w-4" />
                    AI 분석 시작 (교육원 워커)
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
