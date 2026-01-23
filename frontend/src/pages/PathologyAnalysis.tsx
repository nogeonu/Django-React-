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
  const [pendingRequestId, setPendingRequestId] = useState<string | null>(null); // 진행 중인 요청 ID
  const [analysisResult, setAnalysisResult] = useState<any>(null); // 분석 결과

  useEffect(() => {
    loadOrders();
  }, []);

  // 결과 폴링 함수
  const startPollingResult = (requestId: string) => {
    const maxAttempts = 1500; // 50분 = 3000초 / 2초 간격
    let attempts = 0;
    let timeoutId: NodeJS.Timeout | null = null;
    let isCancelled = false;
    
    const poll = async () => {
      if (isCancelled || attempts >= maxAttempts) {
        if (attempts >= maxAttempts && !isCancelled) {
          toast({
            title: "분석 시간 초과",
            description: "분석이 50분을 초과했습니다. 나중에 다시 확인해주세요.",
            variant: "destructive",
          });
          setPendingRequestId(null);
        }
        return;
      }
      
      try {
        const response = await fetch(`/api/pathology/result/${requestId}/`, {
          credentials: 'include',
        });
        
        if (!response.ok) {
          throw new Error('결과 조회 실패');
        }
        
        const data = await response.json();
        
        if (data.status === 'completed') {
          setAnalysisResult(data.result);
          setPendingRequestId(null);
          
          toast({
            title: "분석 완료!",
            description: `결과: ${data.result.class_name} (신뢰도: ${(data.result.confidence * 100).toFixed(2)}%)`,
          });
          
          // 주문 목록 새로고침
          loadOrders();
        } else if (data.status === 'failed') {
          setPendingRequestId(null);
          toast({
            title: "분석 실패",
            description: data.error || "분석 중 오류가 발생했습니다.",
            variant: "destructive",
          });
        } else {
          // pending 또는 processing 상태면 계속 폴링
          attempts++;
          if (!isCancelled) {
            timeoutId = setTimeout(poll, 2000); // 2초마다 확인
          }
        }
      } catch (error: any) {
        console.error('결과 조회 오류:', error);
        attempts++;
        if (attempts < maxAttempts && !isCancelled) {
          timeoutId = setTimeout(poll, 2000);
        }
      }
    };
    
    // 첫 폴링 시작
    timeoutId = setTimeout(poll, 2000);
    
    // 정리 함수 반환
    return () => {
      isCancelled = true;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  };
  
  // 컴포넌트 언마운트 시 폴링 정리
  useEffect(() => {
    return () => {
      // 컴포넌트 언마운트 시 폴링 중지
      setPendingRequestId(null);
    };
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
      // 백엔드는 즉시 응답 반환 (202 Accepted) - 타임아웃 불필요
      const response = await fetch('/api/pathology/analyze/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include', // 쿠키 포함 (인증 정보)
        body: JSON.stringify({
          instance_id: instanceId || `pathology_${selectedOrder.id}`, // 참고용
          filename: filename // 교육원 워커가 wsi/ 폴더에서 찾을 파일명
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `서버 오류 (${response.status})`);
      }

      // request_id 저장하고 폴링 시작
      if (data.request_id) {
        setPendingRequestId(data.request_id);
        setAnalysisResult(null);
        
        toast({
          title: "분석 요청 완료",
          description: "교육원 워커에서 분석을 진행 중입니다. 결과를 확인하는 중...",
        });
        
        // 결과 폴링 시작
        startPollingResult(data.request_id);
      } else {
        toast({
          title: "분석 요청 완료",
          description: "교육원 워커에서 분석을 진행 중입니다.",
        });
      }

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
              
              {/* 분석 결과 표시 */}
              {analysisResult && (
                <div className="p-4 bg-green-50 border-2 border-green-300 rounded-lg shadow-sm">
                  <div className="flex items-center gap-2 mb-3">
                    <CheckCircle2 className="h-5 w-5 text-green-600" />
                    <h4 className="font-bold text-lg text-green-800">분석 결과</h4>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-gray-700 min-w-[80px]">결과:</span>
                      <Badge variant={analysisResult.class_name === 'Tumor' ? 'destructive' : 'default'} className="text-base px-3 py-1">
                        {analysisResult.class_name === 'Tumor' ? '종양 (Tumor)' : '정상 (Normal)'}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-gray-700 min-w-[80px]">신뢰도:</span>
                      <span className="text-base font-bold text-green-700">
                        {(analysisResult.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                    {analysisResult.num_patches && (
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-gray-700 min-w-[80px]">패치 수:</span>
                        <span className="text-base">{analysisResult.num_patches.toLocaleString()}개</span>
                      </div>
                    )}
                    {analysisResult.elapsed_time_seconds && (
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-gray-700 min-w-[80px]">소요 시간:</span>
                        <span className="text-base">
                          {Math.floor(analysisResult.elapsed_time_seconds / 60)}분 {Math.floor(analysisResult.elapsed_time_seconds % 60)}초
                        </span>
                      </div>
                    )}
                    {analysisResult.image_url && (
                      <div className="mt-3 pt-3 border-t border-green-200">
                        <p className="font-semibold text-gray-700 mb-2">결과 이미지:</p>
                        <div className="flex gap-2">
                          <a 
                            href={analysisResult.image_url} 
                            target="_blank" 
                            rel="noopener noreferrer" 
                            className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
                          >
                            <Scan className="h-4 w-4" />
                            결과 이미지 보기
                          </a>
                          {analysisResult.viewer_url && (
                            <a 
                              href={analysisResult.viewer_url} 
                              target="_blank" 
                              rel="noopener noreferrer" 
                              className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                            >
                              뷰어에서 보기
                            </a>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
              
              {/* 진행 중 표시 */}
              {pendingRequestId && (
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                    <p className="text-sm text-blue-800">교육원 워커에서 분석 중입니다... (약 50분 소요)</p>
                  </div>
                </div>
              )}
              
              <Button 
                onClick={handleAnalyze}
                disabled={analyzing || selectedOrder.status === 'completed' || !selectedFilename || !!pendingRequestId}
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
