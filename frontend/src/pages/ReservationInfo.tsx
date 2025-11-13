import { useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import FullCalendar from "@fullcalendar/react";
import dayGridPlugin from "@fullcalendar/daygrid";
import timeGridPlugin from "@fullcalendar/timegrid";
import interactionPlugin, { DateClickArg } from "@fullcalendar/interaction";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Calendar as MiniCalendar } from "@/components/ui/calendar";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { format } from "date-fns";
import { CalendarEvent, useCalendar } from "@/context/CalendarContext";

type DetailEvent = {
  id: string;
  title: string;
  start: string;
  end?: string;
  type?: string;
  patientId?: string;
  patientName?: string;
  patientGender?: string;
  patientAge?: number;
};

const colorByType: Record<string, { bg: string; border: string; text: string }> = {
  검진: { bg: "#DBEAFE", border: "#93C5FD", text: "#0F172A" },
  회의: { bg: "#DCFCE7", border: "#86EFAC", text: "#0F172A" },
  내근: { bg: "#FEF3C7", border: "#FCD34D", text: "#0F172A" },
  외근: { bg: "#FFE4E6", border: "#FDA4AF", text: "#0F172A" },
  예약: { bg: "#E0E7FF", border: "#A5B4FC", text: "#1E1B4B" },
};

const viewOptions: { key: "day" | "week" | "month"; label: string }[] = [
  { key: "day", label: "일간" },
  { key: "week", label: "주간" },
  { key: "month", label: "월간" },
];

const toFcView = (view: "day" | "week" | "month") =>
  view === "day" ? "timeGridDay" : view === "week" ? "timeGridWeek" : "dayGridMonth";

export default function ReservationInfo() {
  const calendarRef = useRef<FullCalendar | null>(null);
  const [selectedDate, setSelectedDate] = useState<Date | undefined>(new Date());
  const [currentTitle, setCurrentTitle] = useState<string>(format(new Date(), "yyyy.MM"));
  const [activeView, setActiveView] = useState<"day" | "week" | "month">("month");
  const [openDetail, setOpenDetail] = useState(false);
  const [detailEvent, setDetailEvent] = useState<DetailEvent | null>(null);
  const navigate = useNavigate();
  const { events } = useCalendar();

  const fcEvents = useMemo(
    () =>
      events.map((event: CalendarEvent) => {
        const palette = colorByType[event.type || "예약"] || colorByType["예약"];
        return {
          id: event.id,
          title: event.title,
          start: event.start,
          end: event.end,
          backgroundColor: palette.bg,
          borderColor: palette.border,
          textColor: palette.text,
          extendedProps: {
            type: event.type,
            patientId: event.patientId,
            patientName: event.patientName,
            patientGender: event.patientGender,
            patientAge: event.patientAge,
          },
        };
      }),
    [events],
  );

  const upcomingReservations = useMemo(() => {
    const today = new Date();
    return [...events]
      .filter((event) => new Date(event.start).getTime() >= today.setHours(0, 0, 0, 0))
      .sort((a, b) => new Date(a.start).getTime() - new Date(b.start).getTime())
      .slice(0, 5);
  }, [events]);

  const syncTitle = () => {
    const api = calendarRef.current?.getApi();
    if (api) setCurrentTitle(format(api.getDate(), "yyyy.MM"));
  };

  const handlePrev = () => {
    const api = calendarRef.current?.getApi();
    api?.prev();
    syncTitle();
  };

  const handleNext = () => {
    const api = calendarRef.current?.getApi();
    api?.next();
    syncTitle();
  };

  const handleToday = () => {
    const api = calendarRef.current?.getApi();
    api?.today();
    const now = new Date();
    setSelectedDate(now);
    syncTitle();
  };

  const changeView = (view: "day" | "week" | "month") => {
    setActiveView(view);
    const api = calendarRef.current?.getApi();
    api?.changeView(toFcView(view));
    syncTitle();
  };

  const onDateClick = (arg: DateClickArg) => {
    setSelectedDate(arg.date);
  };

  const onMiniSelect = (date?: Date) => {
    if (!date) return;
    const api = calendarRef.current?.getApi();
    api?.gotoDate(date);
    setSelectedDate(date);
    syncTitle();
  };

  const colorLegend = [
    { label: "검진", color: colorByType["검진"]?.border },
    { label: "회의", color: colorByType["회의"]?.border },
    { label: "내근", color: colorByType["내근"]?.border },
    { label: "외근", color: colorByType["외근"]?.border },
  ];

  return (
    <div className="p-6 space-y-5">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">예약 정보</h2>
          <p className="text-sm text-muted-foreground">의료 일정과 예약 현황을 한눈에 확인하세요.</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="inline-flex rounded-full border bg-white shadow-sm overflow-hidden">
            <Button variant="ghost" size="sm" className="rounded-none border-r" onClick={handleToday}>
              오늘
            </Button>
            <Button variant="ghost" size="icon" className="rounded-none border-r" onClick={handlePrev}>
              ‹
            </Button>
            <Button variant="ghost" size="icon" className="rounded-none" onClick={handleNext}>
              ›
            </Button>
          </div>
          <div className="inline-flex rounded-full border bg-white shadow-sm overflow-hidden">
            {viewOptions.map((option) => (
              <button
                key={option.key}
                className={`px-3 py-1.5 text-sm font-medium transition-colors ${
                  activeView === option.key
                    ? "bg-slate-900 text-white"
                    : "text-slate-500 hover:bg-slate-100"
                }`}
                onClick={() => changeView(option.key)}
                type="button"
              >
                {option.label}
              </button>
            ))}
          </div>
          <Button className="bg-indigo-600 hover:bg-indigo-700" onClick={() => navigate("/?reserve=1")}>예약 등록</Button>
        </div>
      </div>

      <Card>
        <CardContent className="flex flex-col gap-6 p-4 xl:flex-row">
          <div className="w-full space-y-4 xl:w-72">
            <Card className="shadow-sm">
              <CardHeader className="pb-3">
                <CardTitle className="text-base font-semibold text-gray-800">달력</CardTitle>
                <span className="text-xs text-muted-foreground">{currentTitle}</span>
              </CardHeader>
              <CardContent className="pt-0">
                <MiniCalendar
                  mode="single"
                  selected={selectedDate}
                  onSelect={onMiniSelect}
                  className="rounded-2xl bg-slate-50 p-3"
                  classNames={{
                    months: "flex flex-col gap-2",
                    month: "space-y-2",
                    caption: "flex justify-between items-center text-sm font-semibold text-slate-700",
                    caption_label: "text-sm font-semibold",
                    nav: "flex items-center gap-2",
                    nav_button: "h-7 w-7 rounded-full border border-slate-200 bg-white text-slate-500 hover:bg-slate-100 transition shadow-sm",
                    nav_button_previous: "",
                    nav_button_next: "",
                    table: "w-full border-collapse text-center text-sm text-slate-600",
                    head_row: "",
                    head_cell: "pb-1 font-semibold text-slate-400",
                    row: "",
                    cell: "h-8 w-8",
                    day: "h-8 w-8 flex items-center justify-center rounded-full text-sm font-medium hover:bg-blue-50",
                    day_selected: "bg-blue-600 text-white shadow-sm hover:bg-blue-600",
                    day_today: "border border-blue-500 text-blue-600",
                    day_outside: "text-slate-300",
                    day_disabled: "opacity-50",
                    day_hidden: "invisible",
                  }}
                />
              </CardContent>
            </Card>

            <Card className="shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-base font-semibold text-gray-800">예약 범례</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 pt-0">
                {colorLegend.map((item) => (
                  <div key={item.label} className="flex items-center gap-2 text-sm text-gray-600">
                    <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                    {item.label}
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card className="shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-base font-semibold text-gray-800">다가오는 예약</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 pt-0">
                {upcomingReservations.length === 0 ? (
                  <p className="text-sm text-muted-foreground">예정된 예약이 없습니다.</p>
                ) : (
                  upcomingReservations.map((event) => {
                    const palette = colorByType[event.type || "예약"] || colorByType["예약"];
                    return (
                      <div key={event.id} className="rounded-lg border bg-white p-3 shadow-xs">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-gray-900">{event.title}</span>
                          <Badge
                            variant="outline"
                            style={{
                              borderColor: palette.border,
                              color: palette.text,
                              backgroundColor: palette.bg,
                            }}
                            className="border-0"
                          >
                            {event.type || "예약"}
                          </Badge>
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          {format(new Date(event.start), "yyyy년 MM월 dd일 HH:mm")}
                        </div>
                        {event.patientName && (
                          <div className="mt-1 text-xs text-gray-500">
                            환자: {event.patientName}
                            {event.patientId ? ` (${event.patientId})` : ""}
                          </div>
                        )}
                      </div>
                    );
                  })
                )}
              </CardContent>
            </Card>
          </div>

          <div className="flex-1 min-w-0">
            <Card className="shadow-sm h-full">
              <CardContent className="p-2 md:p-4">
                <style>{`
                  .reservation-calendar .fc .fc-toolbar { margin-bottom: 0; }
                  .reservation-calendar .fc { font-family: 'Pretendard', 'Inter', sans-serif; }
                  .reservation-calendar .fc .fc-scrollgrid { border-radius: 12px; overflow: hidden; }
                  .reservation-calendar .fc table { width: 100%; table-layout: fixed; }
                  .reservation-calendar .fc .fc-col-header-cell-cushion { padding: 10px 8px; font-size: 13px; color: #475569; }
                  .reservation-calendar .fc .fc-daygrid-day-top { display: flex; align-items: center; justify-content: flex-end; padding: 8px; }
                  .reservation-calendar .fc .fc-daygrid-day-number { font-size: 13px; color: #475569; }
                  .reservation-calendar .fc .fc-daygrid-day-frame { padding: 6px 8px 12px; }
                  .reservation-calendar .fc .fc-daygrid-day.fc-day-today { background: #eef2ff; }
                  .reservation-calendar .fc .fc-daygrid-day.fc-day-today .fc-daygrid-day-number { color: #4338ca; font-weight: 700; }
                  .reservation-calendar .fc .fc-daygrid-event { border-radius: 10px; padding: 4px 8px; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12); white-space: normal; overflow: hidden; }
                  .reservation-calendar .fc .fc-daygrid-event:hover { transform: translateY(-1px); }
                  .reservation-calendar .fc .fc-daygrid-day.fc-day-sat,
                  .reservation-calendar .fc .fc-daygrid-day.fc-day-sun { background: #fafafa; }
                  .reservation-calendar .fc .fc-timegrid-slot { height: 32px; }
                  .reservation-calendar .fc .fc-timegrid-axis-cushion { font-size: 11px; color: #94a3b8; }
                  .reservation-calendar .fc .fc-timegrid-event { border-radius: 10px; padding: 6px 8px; box-shadow: 0 4px 10px rgba(15, 23, 42, 0.14); }
                `}</style>
                <div
                  className="reservation-calendar"
                  style={{
                    ["--fc-border-color" as any]: "#e5e7eb",
                    ["--fc-page-bg-color" as any]: "transparent",
                    ["--fc-neutral-bg-color" as any]: "transparent",
                    ["--fc-today-bg-color" as any]: "transparent",
                  }}
                >
                  <FullCalendar
                    ref={calendarRef as any}
                    plugins={[dayGridPlugin, timeGridPlugin, interactionPlugin]}
                    initialView={toFcView(activeView)}
                    headerToolbar={false}
                    height="auto"
                    events={fcEvents}
                    dayMaxEventRows={3}
                    eventDisplay="block"
                    locale="ko"
                    firstDay={0}
                    dateClick={onDateClick}
                    eventClick={(info) => {
                      const event = info.event;
                      setDetailEvent({
                        id: event.id,
                        title: event.title,
                        start: event.startStr,
                        end: event.endStr || undefined,
                        type: (event.extendedProps as any)?.type,
                        patientId: (event.extendedProps as any)?.patientId,
                        patientName: (event.extendedProps as any)?.patientName,
                        patientGender: (event.extendedProps as any)?.patientGender,
                        patientAge: (event.extendedProps as any)?.patientAge,
                      });
                      setOpenDetail(true);
                    }}
                    eventDidMount={(info) => {
                      const event = info.event;
                      const type = (event.extendedProps as any)?.type || "";
                      const patient = (event.extendedProps as any)?.patientName || "";
                      const tooltipLines = [
                        event.title,
                        patient ? `환자: ${patient}` : "",
                        type ? `유형: ${type}` : "",
                        event.start ? `시작: ${new Date(event.start).toLocaleString("ko-KR")}` : "",
                        event.end ? `종료: ${new Date(event.end).toLocaleString("ko-KR")}` : "",
                      ].filter(Boolean);
                      info.el.setAttribute("title", tooltipLines.join("\n"));
                    }}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>

      <Dialog open={openDetail} onOpenChange={setOpenDetail}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle className="text-xl font-semibold text-gray-900">{detailEvent?.title || "예약 상세"}</DialogTitle>
            <DialogDescription className="text-sm text-muted-foreground">
              예약에 대한 세부 정보를 확인하세요.
            </DialogDescription>
          </DialogHeader>
          {detailEvent && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-600">유형</span>
                <Badge variant="outline" className="text-xs">
                  {detailEvent.type || "예약"}
                </Badge>
              </div>
              <Separator />
              <div className="space-y-2 text-sm text-gray-700">
                <div className="flex justify-between">
                  <span className="text-gray-500">시작</span>
                  <span>{new Date(detailEvent.start).toLocaleString("ko-KR")}</span>
                </div>
                {detailEvent.end && (
                  <div className="flex justify-between">
                    <span className="text-gray-500">종료</span>
                    <span>{new Date(detailEvent.end).toLocaleString("ko-KR")}</span>
                  </div>
                )}
                {detailEvent.patientName && (
                  <div className="flex justify-between">
                    <span className="text-gray-500">환자</span>
                    <span>
                      {detailEvent.patientName}
                      {detailEvent.patientId ? ` (${detailEvent.patientId})` : ""}
                    </span>
                  </div>
                )}
                {detailEvent.patientGender && (
                  <div className="flex justify-between">
                    <span className="text-gray-500">성별</span>
                    <span>{detailEvent.patientGender}</span>
                  </div>
                )}
                {typeof detailEvent.patientAge === "number" && (
                  <div className="flex justify-between">
                    <span className="text-gray-500">나이</span>
                    <span>{detailEvent.patientAge}세</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
