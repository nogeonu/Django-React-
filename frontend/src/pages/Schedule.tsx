import { useMemo, useRef, useState, useEffect } from 'react';
import FullCalendar from '@fullcalendar/react';
import dayGridPlugin from '@fullcalendar/daygrid';
import timeGridPlugin from '@fullcalendar/timegrid';
import interactionPlugin, { DateClickArg } from '@fullcalendar/interaction';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Calendar as MiniCalendar } from '@/components/ui/calendar';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { format } from 'date-fns';
import { useCalendar } from '@/context/CalendarContext';
import { useNavigate, useLocation } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '@/lib/api';
import { useToast } from '@/hooks/use-toast';

interface Patient {
  id: string;
  name: string;
  age?: number;
  gender?: string;
  patient_id?: string;
}

export default function Schedule() {
  const calendarRef = useRef<FullCalendar | null>(null);
  const [selectedDate, setSelectedDate] = useState<Date | undefined>(new Date());
  const [currentTitle, setCurrentTitle] = useState<string>(format(new Date(), 'yyyy.MM'));
  const [activeView, setActiveView] = useState<'day' | 'week' | 'month'>('month');
  const { events, removeEvent, updateEvent, addEvent } = useCalendar();
  const navigate = useNavigate();
  const location = useLocation();
  const { toast } = useToast();
  
  // 예약 검사 등록 모달 상태
  const [isReserveOpen, setIsReserveOpen] = useState(false);
  const [reserveTitle, setReserveTitle] = useState("");
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [reserveDate, setReserveDate] = useState<string>(new Date().toISOString().slice(0,10));
  const [reserveStartAmPm, setReserveStartAmPm] = useState<'AM'|'PM'>("AM");
  const [reserveStartHour, setReserveStartHour] = useState<string>("09");
  const [reserveStartMinute, setReserveStartMinute] = useState<string>("00");
  const [reserveEndAmPm, setReserveEndAmPm] = useState<'AM'|'PM'>("AM");
  const [reserveEndHour, setReserveEndHour] = useState<string>("10");
  const [reserveEndMinute, setReserveEndMinute] = useState<string>("00");
  const [reserveType, setReserveType] = useState<'검진'|'회의'|'내근'|'외근'>("검진");
  
  // URL 쿼리로 예약 모달 자동 오픈
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    if (params.get('reserve') === '1') {
      setIsReserveOpen(true);
    }
  }, [location.search]);
  
  // 환자 목록 조회
  const { data: patients = [] } = useQuery({
    queryKey: ["patients"],
    queryFn: async () => {
      try {
        const response = await apiRequest("GET", "/api/patients/patients/");
        return response?.results || response || [];
      } catch (err) {
        console.error("환자 목록 조회 오류:", err);
        return [];
      }
    },
  });
  // Pastel palette (bg/border) similar to reference
  const colorByType: Record<string, { bg: string; border: string; text?: string }> = {
    '검진': { bg: '#DBEAFE', border: '#93C5FD', text: '#0F172A' },   // light blue
    '회의': { bg: '#DCFCE7', border: '#86EFAC', text: '#0F172A' },   // light green
    '내근': { bg: '#FEF3C7', border: '#FCD34D', text: '#0F172A' },   // light amber
    '외근': { bg: '#FFE4E6', border: '#FDA4AF', text: '#0F172A' },   // light rose
  };
  const fcEvents = useMemo(
    () => events.map((e) => {
      const c = colorByType[e.type || '검진'] || colorByType['검진'];
      return {
        id: e.id,
        title: e.title,
        start: e.start,
        end: e.end,
        backgroundColor: c.bg,
        borderColor: c.border,
        textColor: c.text || '#0F172A',
        extendedProps: {
          type: e.type || '검진',
          patientId: e.patientId,
          patientName: e.patientName,
          patientGender: e.patientGender,
          patientAge: e.patientAge,
        },
      } as any;
    }),
    [events]
  );

  const [openDetail, setOpenDetail] = useState(false);
  const [detail, setDetail] = useState<{id?: string; title:string; start:string; end?:string; type?:string; patientId?:string; patientName?:string; patientGender?:string; patientAge?:number} | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  // Edit fields for date/time only
  const [editDate, setEditDate] = useState<string>('');
  const [startAmPm, setStartAmPm] = useState<'AM'|'PM'>('AM');
  const [startHour, setStartHour] = useState<string>('09');
  const [startMinute, setStartMinute] = useState<string>('00');
  const [endAmPm, setEndAmPm] = useState<'AM'|'PM'>('AM');
  const [endHour, setEndHour] = useState<string>('10');
  const [endMinute, setEndMinute] = useState<string>('00');

  const initEditFromDetail = () => {
    if (!detail?.start) return;
    const s = new Date(detail.start);
    const e = detail.end ? new Date(detail.end) : new Date(s.getTime() + 30*60000);
    const to12h = (d: Date) => {
      let h = d.getHours();
      const ampm: 'AM'|'PM' = h >= 12 ? 'PM' : 'AM';
      h = h % 12; if (h === 0) h = 12;
      return { ampm, hour: String(h).padStart(2,'0'), minute: String(d.getMinutes()).padStart(2,'0') };
    };
    const s12 = to12h(s);
    const e12 = to12h(e);
    setEditDate(s.toISOString().slice(0,10));
    setStartAmPm(s12.ampm); setStartHour(s12.hour); setStartMinute(s12.minute);
    setEndAmPm(e12.ampm); setEndHour(e12.hour); setEndMinute(e12.minute);
  };

  const handlePrev = () => {
    const api = calendarRef.current?.getApi();
    api?.prev();
    setCurrentTitle(api ? format(api.getDate(), 'yyyy.MM') : currentTitle);
  };

  const handleNext = () => {
    const api = calendarRef.current?.getApi();
    api?.next();
    setCurrentTitle(api ? format(api.getDate(), 'yyyy.MM') : currentTitle);
  };

  const handleToday = () => {
    const api = calendarRef.current?.getApi();
    api?.today();
    setSelectedDate(new Date());
    setCurrentTitle(api ? format(api.getDate(), 'yyyy.MM') : currentTitle);
  };

  const onDateClick = (arg: DateClickArg) => {
    setSelectedDate(arg.date);
    // 날짜 클릭 시 예약 검사 등록 모달 열기
    const clickedDate = format(arg.date, 'yyyy-MM-dd');
    setReserveDate(clickedDate);
    setIsReserveOpen(true);
  };

  const onMiniSelect = (date?: Date) => {
    if (!date) return;
    const api = calendarRef.current?.getApi();
    api?.gotoDate(date);
    setSelectedDate(date);
    setCurrentTitle(format(date, 'yyyy.MM'));
  };

  // 실제 보기 전환 (일/주/월)
  const toFcView = (v: 'day'|'week'|'month') =>
    v === 'day' ? 'timeGridDay' : v === 'week' ? 'timeGridWeek' : 'dayGridMonth';
  const changeView = (v: 'day'|'week'|'month') => {
    setActiveView(v);
    const api = calendarRef.current?.getApi();
    api?.changeView(toFcView(v));
    setCurrentTitle(api ? format(api.getDate(), 'yyyy.MM') : currentTitle);
  };

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight">일정관리</h2>
        <div className="flex-1" />
      </div>

      <Card>
        <CardContent className="px-4 py-3">
          <div className="flex items-center">
            <div className="flex items-center gap-2">
              <span className="text-lg font-semibold">{currentTitle}</span>
              <Button variant="outline" size="sm" onClick={handleToday}>오늘</Button>
              <Button variant="outline" size="icon" onClick={handlePrev}>
                <ChevronLeft className="w-4 h-4" />
              </Button>
              <Button variant="outline" size="icon" onClick={handleNext}>
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
            <div className="flex-1" />
            <div className="inline-flex rounded-full border bg-white shadow-sm overflow-hidden">
              {(
                [
                  { key: 'day', label: '일간' },
                  { key: 'week', label: '주간' },
                  { key: 'month', label: '월간' },
                ] as const
              ).map((v) => (
                <button
                  key={v.key}
                  onClick={() => changeView(v.key)}
                  className={`px-4 py-1.5 text-sm transition-colors ${
                    activeView === v.key ? 'bg-gray-100 text-gray-900' : 'bg-white hover:bg-gray-50 text-gray-600'
                  }`}
                >
                  {v.label}
                </button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-12 gap-4">
        <div className="col-span-12 lg:col-span-2">
          <Card>
            <CardContent className="p-4">
              <Button className="w-full mb-3 bg-green-600 hover:bg-green-700 text-white rounded-md shadow-sm" onClick={() => navigate('/?reserve=1')}>
                일정추가
              </Button>
              <style>{`
                /* Halved mini calendar sizing */
                .mini-cal .rdp { font-size: 8px !important; }
                .mini-cal .rdp-caption { padding: 1px 0 !important; }
                .mini-cal .rdp-caption_label { font-size: 10px !important; }
                .mini-cal .rdp-nav_button { width: 9px !important; height: 9px !important; }
                .mini-cal .rdp-table { margin: 1px 0 !important; }
                .mini-cal .rdp-head_cell { padding: 0 !important; font-size: 8px !important; }
                .mini-cal .rdp-cell { padding: 0 !important; }
                .mini-cal .rdp-day, .mini-cal .rdp-day_button { width: 9px !important; height: 9px !important; line-height: 9px !important; font-size: 8px !important; }
                .mini-cal .rdp-day.rdp-day_selected, .mini-cal .rdp-day_button[aria-pressed="true"] { border-radius: 3px !important; }
              `}</style>
              <div className="mini-cal" style={{ maxWidth: 120 }}>
                <MiniCalendar 
                  mode="single" 
                  selected={selectedDate} 
                  onSelect={onMiniSelect} 
                  className="rounded-md p-1"
                  classNames={{
                    months: "flex flex-col",
                    month: "space-y-1",
                    caption: "flex justify-center pt-0 relative items-center",
                    caption_label: "text-[10px] font-medium",
                    nav: "space-x-1 flex items-center",
                    nav_button: "h-4 w-4 bg-transparent p-0 opacity-70 hover:opacity-100",
                    nav_button_previous: "absolute left-0",
                    nav_button_next: "absolute right-0",
                    table: "w-full border-collapse",
                    head_row: "flex",
                    head_cell: "text-muted-foreground rounded-md w-5 font-normal text-[9px]",
                    row: "flex w-full mt-1",
                    cell: "h-5 w-5 text-center p-0 relative",
                    day: "h-5 w-5 p-0 text-[10px] font-normal aria-selected:opacity-100",
                    day_range_end: "day-range-end",
                    day_selected: "bg-primary text-primary-foreground hover:bg-primary hover:text-primary-foreground focus:bg-primary focus:text-primary-foreground",
                    day_today: "bg-accent text-accent-foreground",
                    day_outside: "day-outside text-muted-foreground aria-selected:bg-accent/50 aria-selected:text-muted-foreground",
                    day_disabled: "text-muted-foreground opacity-50",
                    day_range_middle: "aria-selected:bg-accent aria-selected:text-accent-foreground",
                    day_hidden: "invisible",
                  }}
                />
              </div>
              <div className="mt-6 space-y-4">
                <div>
                  <div className="text-sm font-medium text-gray-700 mb-2">내 캘린더</div>
                  <div className="space-y-1 text-sm">
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: '#93C5FD' }} /> 검진
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: '#86EFAC' }} /> 회의
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: '#FCD34D' }} /> 내근
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: '#FDA4AF' }} /> 외근
                    </div>
                  </div>
                </div>
                <div>
                  <div className="text-sm font-medium text-gray-700 mb-2">다른 캘린더</div>
                  <div className="text-sm text-gray-500">-</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
        <div className="col-span-12 lg:col-span-10">
          <Card>
            <CardContent className="p-2 md:p-4">
              <style>{`
                .fc-slim .fc .fc-toolbar { margin-bottom: 8px; }
                .fc-slim .fc .fc-daygrid-day-top { flex-direction: row; justify-content: flex-start; padding: 4px 8px; }
                .fc-slim .fc .fc-daygrid-day-number{ font-size:12px; color:#64748b; }
                .fc-slim .fc .fc-daygrid-day-frame { padding: 4px 8px 10px; }
                .fc-slim .fc .fc-daygrid-event { border-radius: 8px; padding: 3px 8px; box-shadow: 0 1px 0 rgba(0,0,0,0.04); }
                .fc-slim .fc .fc-daygrid-event:hover { transform: translateY(-1px); box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
                .fc-slim .fc .fc-daygrid-event .fc-event-title { font-weight: 500; }
                .fc-slim .fc .fc-daygrid-event .fc-event-time { font-weight: 600; }
                .fc-slim .fc .fc-scrollgrid-section-liquid { height: auto; }
                .fc-slim .fc .fc-daygrid-day.fc-day-today { background: #eef2ff; }
                .fc-slim .fc .fc-daygrid-day.fc-day-today .fc-daygrid-day-number { color:#4338ca; font-weight:700; }
                .fc-slim .fc .fc-day-sat .fc-daygrid-day-frame, .fc-slim .fc .fc-day-sun .fc-daygrid-day-frame { background: #fafafa; }
                /* timeGrid polish */
                .fc-slim .fc .fc-timegrid-slot { height: 28px; }
                .fc-slim .fc .fc-timegrid-axis-cushion, .fc-slim .fc .fc-timegrid-slot-label { font-size: 10px; color:#94a3b8; }
                .fc-slim .fc .fc-timegrid-event { border-radius: 8px; padding: 4px 6px; box-shadow: 0 1px 0 rgba(0,0,0,0.04); }
                .fc-slim .fc .fc-timegrid-event:hover { transform: translateY(-1px); box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
              `}</style>
              <div className="fc-slim" style={{
                ['--fc-font-size' as any]: '11px',
                ['--fc-small-font-size' as any]: '10px',
                ['--fc-event-border-color' as any]: 'transparent',
                ['--fc-border-color' as any]: '#e5e7eb',
              }}>
              <FullCalendar
                ref={calendarRef as any}
                plugins={[dayGridPlugin, timeGridPlugin, interactionPlugin]}
                initialView={toFcView(activeView)}
                headerToolbar={false}
                height="auto"
                expandRows={true}
                dayMaxEventRows={3}
                eventDisplay="block"
                eventTimeFormat={{ hour: '2-digit', minute: '2-digit', meridiem: false }}
                events={fcEvents}
                dateClick={onDateClick}
                eventDidMount={(info) => {
                  const d = info.event;
                  const type = (d.extendedProps as any)?.type || '';
                  const name = (d.extendedProps as any)?.patientName || '';
                  const pid = (d.extendedProps as any)?.patientId || '';
                  const start = d.start ? new Date(d.start).toLocaleString('ko-KR') : '';
                  const end = d.end ? new Date(d.end).toLocaleString('ko-KR') : '';
                  const tooltip = `${d.title}${name ? `\n환자: ${name}${pid ? ` (${pid})` : ''}` : ''}${type ? `\n유형: ${type}` : ''}${start ? `\n시작: ${start}` : ''}${end ? `\n종료: ${end}` : ''}`;
                  info.el.setAttribute('title', tooltip);
                }}
                eventClick={(info) => {
                  const e = info.event;
                  setDetail({
                    id: e.id,
                    title: e.title,
                    start: e.startStr,
                    end: e.endStr || undefined,
                    type: (e.extendedProps as any)?.type,
                    patientId: (e.extendedProps as any)?.patientId,
                    patientName: (e.extendedProps as any)?.patientName,
                    patientGender: (e.extendedProps as any)?.patientGender,
                    patientAge: (e.extendedProps as any)?.patientAge,
                  });
                  setOpenDetail(true);
                  setIsEditing(false);
                }}
                locale="ko"
                firstDay={0}
              />
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <Dialog open={openDetail} onOpenChange={(v)=>{ setOpenDetail(v); if(!v) setIsEditing(false); }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="text-xl">{isEditing ? '일정 시간 변경' : detail?.title}</DialogTitle>
            <DialogDescription>
              {(detail?.type ? `유형: ${detail.type} · ` : '') +
               (detail?.start ? `시작: ${new Date(detail.start).toLocaleString('ko-KR')}` : '') +
               (detail?.end ? ` · 종료: ${new Date(detail.end).toLocaleString('ko-KR')}` : '')}
            </DialogDescription>
            {(!isEditing && (detail?.patientName || detail?.patientId)) && (
              <div className="mt-2 text-sm text-gray-700">
                환자: {detail?.patientName || '-'}{detail?.patientId ? ` (${detail.patientId})` : ''}
                {typeof detail?.patientAge === 'number' ? ` · ${detail?.patientAge}세` : ''}
                {detail?.patientGender ? ` · ${detail?.patientGender}` : ''}
              </div>
            )}
          </DialogHeader>
          {isEditing ? (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="edit-date">날짜</Label>
                <Input id="edit-date" type="date" value={editDate} onChange={(e)=> setEditDate(e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>시작 시간</Label>
                <div className="grid grid-cols-3 gap-2">
                  <Select value={startAmPm} onValueChange={(v:any)=> setStartAmPm(v)}>
                    <SelectTrigger className="h-10"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="AM">오전</SelectItem>
                      <SelectItem value="PM">오후</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select value={startHour} onValueChange={(v:any)=> setStartHour(v)}>
                    <SelectTrigger className="h-10"><SelectValue placeholder="시" /></SelectTrigger>
                    <SelectContent>
                      {Array.from({length:12},(_,i)=> String(i+1).padStart(2,'0')).map(h => (
                        <SelectItem key={h} value={h}>{h}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select value={startMinute} onValueChange={(v:any)=> setStartMinute(v)}>
                    <SelectTrigger className="h-10"><SelectValue placeholder="분" /></SelectTrigger>
                    <SelectContent>
                      {['00','10','20','30','40','50'].map(m => (
                        <SelectItem key={m} value={m}>{m}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="space-y-2">
                <Label>종료 시간</Label>
                <div className="grid grid-cols-3 gap-2">
                  <Select value={endAmPm} onValueChange={(v:any)=> setEndAmPm(v)}>
                    <SelectTrigger className="h-10"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="AM">오전</SelectItem>
                      <SelectItem value="PM">오후</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select value={endHour} onValueChange={(v:any)=> setEndHour(v)}>
                    <SelectTrigger className="h-10"><SelectValue placeholder="시" /></SelectTrigger>
                    <SelectContent>
                      {Array.from({length:12},(_,i)=> String(i+1).padStart(2,'0')).map(h => (
                        <SelectItem key={h} value={h}>{h}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select value={endMinute} onValueChange={(v:any)=> setEndMinute(v)}>
                    <SelectTrigger className="h-10"><SelectValue placeholder="분" /></SelectTrigger>
                    <SelectContent>
                      {['00','10','20','30','40','50'].map(m => (
                        <SelectItem key={m} value={m}>{m}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={()=> setIsEditing(false)}>취소</Button>
                <Button onClick={()=> {
                  if (!detail?.id || !editDate) return;
                  const to24h = (ampm:'AM'|'PM', h:string, m:string) => {
                    let hourNum = parseInt(h,10) % 12; if (ampm === 'PM') hourNum += 12;
                    return `${String(hourNum).padStart(2,'0')}:${m}`;
                  };
                  const startStr = to24h(startAmPm, startHour, startMinute);
                  const endStr = to24h(endAmPm, endHour, endMinute);
                  const s = new Date(`${editDate}T${startStr}:00`);
                  const e = new Date(`${editDate}T${endStr}:00`);
                  if (e <= s) return;
                  updateEvent(detail.id, { start: s.toISOString(), end: e.toISOString() });
                  setIsEditing(false);
                  setOpenDetail(false);
                }}>저장</Button>
              </div>
            </div>
          ) : (
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={()=> { setIsEditing(true); initEditFromDetail(); }}>편집</Button>
              <Button variant="destructive" onClick={()=>{
                if (!detail?.id) return;
                removeEvent(detail.id);
                setOpenDetail(false);
              }}>삭제</Button>
            </div>
          )}
        </DialogContent>
      </Dialog>
      
      {/* 예약 검사 등록 모달 */}
      <Dialog open={isReserveOpen} onOpenChange={setIsReserveOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>예약 검사 등록</DialogTitle>
            <DialogDescription>
              예약 정보를 입력하면 일정관리 캘린더에 표시됩니다.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-2">
            <div className="space-y-2">
              <Label htmlFor="reserve-title">제목</Label>
              <Input id="reserve-title" value={reserveTitle} onChange={(e)=>setReserveTitle(e.target.value)} placeholder="예: 흉부 CT 검사" />
            </div>
            <div className="space-y-2">
              <Label>환자명 (선택사항)</Label>
              <Select value={selectedPatient?.id || ''} onValueChange={(v)=>{
                const p = (patients as Patient[]).find((x)=> x.id === v) || null;
                setSelectedPatient(p);
              }}>
                <SelectTrigger>
                  <SelectValue placeholder="환자명을 선택하세요" />
                </SelectTrigger>
                <SelectContent>
                  {(patients as Patient[]).map((p)=> (
                    <SelectItem key={p.id} value={p.id} textValue={p.name}>
                      <div className="flex items-center justify-between w-full">
                        <div className="flex items-center gap-3">
                          <div className="w-7 h-7 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-sm font-semibold">
                            {p.name?.charAt(0) || '-'}
                          </div>
                          <div>
                            <div className="text-gray-900 font-medium">{p.name}</div>
                            <div className="text-xs text-gray-500">{p.patient_id || p.id}</div>
                          </div>
                        </div>
                        <div className="text-right">
                          {p.age && <div className="text-sm text-gray-700">{p.age}세</div>}
                          {p.gender && <div className="text-xs text-gray-500">{p.gender}</div>}
                        </div>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="reserve-date">날짜</Label>
                <Input id="reserve-date" type="date" value={reserveDate} onChange={(e)=>setReserveDate(e.target.value)} />
              </div>
              <div className="space-y-2">
                <Label>시작 시간</Label>
                <div className="grid grid-cols-3 gap-2">
                  <Select value={reserveStartAmPm} onValueChange={(v:any)=>setReserveStartAmPm(v)}>
                    <SelectTrigger className="h-10"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="AM">오전</SelectItem>
                      <SelectItem value="PM">오후</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select value={reserveStartHour} onValueChange={(v:any)=>setReserveStartHour(v)}>
                    <SelectTrigger className="h-10"><SelectValue placeholder="시" /></SelectTrigger>
                    <SelectContent>
                      {Array.from({length:12},(_,i)=>String(i+1).padStart(2,'0')).map(h=> (
                        <SelectItem key={h} value={h}>{h}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select value={reserveStartMinute} onValueChange={(v:any)=>setReserveStartMinute(v)}>
                    <SelectTrigger className="h-10"><SelectValue placeholder="분" /></SelectTrigger>
                    <SelectContent>
                      {['00','10','20','30','40','50'].map(m=> (
                        <SelectItem key={m} value={m}>{m}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="space-y-2">
                <Label>종료 시간</Label>
                <div className="grid grid-cols-3 gap-2">
                  <Select value={reserveEndAmPm} onValueChange={(v:any)=>setReserveEndAmPm(v)}>
                    <SelectTrigger className="h-10"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="AM">오전</SelectItem>
                      <SelectItem value="PM">오후</SelectItem>
                    </SelectContent>
                  </Select>
                  <Select value={reserveEndHour} onValueChange={(v:any)=>setReserveEndHour(v)}>
                    <SelectTrigger className="h-10"><SelectValue placeholder="시" /></SelectTrigger>
                    <SelectContent>
                      {Array.from({length:12},(_,i)=>String(i+1).padStart(2,'0')).map(h=> (
                        <SelectItem key={h} value={h}>{h}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select value={reserveEndMinute} onValueChange={(v:any)=>setReserveEndMinute(v)}>
                    <SelectTrigger className="h-10"><SelectValue placeholder="분" /></SelectTrigger>
                    <SelectContent>
                      {['00','10','20','30','40','50'].map(m=> (
                        <SelectItem key={m} value={m}>{m}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
            <div className="space-y-2">
              <Label>유형</Label>
              <Select value={reserveType} onValueChange={(v:any)=>setReserveType(v)}>
                <SelectTrigger>
                  <SelectValue placeholder="유형 선택" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="검진">검진</SelectItem>
                  <SelectItem value="회의">회의</SelectItem>
                  <SelectItem value="내근">내근</SelectItem>
                  <SelectItem value="외근">외근</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={()=>setIsReserveOpen(false)}>취소</Button>
            <Button onClick={() => {
              try {
                if (!reserveTitle.trim()) {
                  toast({ title: "제목을 입력해 주세요", variant: "destructive" });
                  return;
                }
                const to24h = (ampm:'AM'|'PM', h:string, m:string) => {
                  let hourNum = parseInt(h,10)%12;
                  if (ampm === 'PM') hourNum += 12;
                  return `${String(hourNum).padStart(2,'0')}:${m}`;
                };
                const startStr = to24h(reserveStartAmPm, reserveStartHour, reserveStartMinute);
                const endStr = to24h(reserveEndAmPm, reserveEndHour, reserveEndMinute);
                const start = new Date(`${reserveDate}T${startStr}:00`);
                const end = new Date(`${reserveDate}T${endStr}:00`);
                if (end <= start) {
                  toast({ title: "종료 시간이 시작 시간보다 늦어야 합니다", variant: "destructive" });
                  return;
                }
                const startIso = start.toISOString();
                const endIso = end.toISOString();
                const titleFinal = selectedPatient ? `${reserveTitle.trim()} (${selectedPatient.name})` : reserveTitle.trim();
                addEvent({ 
                  title: titleFinal, 
                  start: startIso, 
                  end: endIso, 
                  type: reserveType,
                  patientId: selectedPatient?.id,
                  patientName: selectedPatient?.name,
                  patientGender: selectedPatient?.gender,
                  patientAge: selectedPatient?.age,
                });
                setIsReserveOpen(false);
                setReserveTitle("");
                setSelectedPatient(null);
                toast({ title: "예약이 등록되었습니다", description: "일정관리 캘린더에서 확인할 수 있습니다." });
              } catch (e) {
                toast({ title: "등록 실패", description: "다시 시도해 주세요.", variant: "destructive" });
              }
            }}>등록</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
