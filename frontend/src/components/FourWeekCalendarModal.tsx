import React, { useCallback, useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export interface Patient {
  id: string;
  name: string;
  birth_date: string;
  gender: string;
  phone?: string;
  address?: string;
  emergency_contact?: string;
  blood_type?: string;
  age: number;
  created_at: string;
  updated_at: string;
}

export type ReservationsDict = Record<string, { id: string; name: string; time: string; memo?: string }[]>;

interface FourWeekCalendarModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  reservations: ReservationsDict;
  onReserve: (payload: { date: Date; time: string; patient: Patient; memo?: string }) => void;
  searchPatients: (query: string) => Promise<Patient[]>;
  initialMonth?: Date;
  maxSameTimeReservations?: number; // default 2
}

const startOfDay = (d: Date) => {
  const x = new Date(d);
  x.setHours(0, 0, 0, 0);
  return x;
};

const addMonths = (date: Date, months: number) => {
  const d = new Date(date);
  d.setMonth(d.getMonth() + months, 1);
  d.setHours(0, 0, 0, 0);
  return d;
};

const getMonthGridDays = (month: Date): Date[] => {
  const firstOfMonth = new Date(month);
  firstOfMonth.setDate(1);
  firstOfMonth.setHours(0, 0, 0, 0);
  const weekDay = firstOfMonth.getDay(); // 0(일)~6(토)
  const diffToMonday = weekDay === 0 ? -6 : 1 - weekDay;
  const start = new Date(firstOfMonth);
  start.setDate(firstOfMonth.getDate() + diffToMonday);
  const days: Date[] = [];
  for (let i = 0; i < 42; i++) {
    const d = new Date(start);
    d.setDate(start.getDate() + i);
    days.push(d);
  }
  return days;
};

const initSuggestedTime = (date: Date) => {
  const now = new Date();
  const isToday = startOfDay(now).getTime() === startOfDay(date).getTime();
  if (!isToday) return "09:00";
  let hh = now.getHours();
  let mm = now.getMinutes();
  const rounded = Math.ceil(mm / 15) * 15;
  if (rounded === 60) { hh += 1; mm = 0; } else { mm = rounded; }
  if (hh < 9) hh = 9;
  if (hh > 17) hh = 17;
  if (hh === 17 && mm > 45) { hh = 17; mm = 45; }
  const h = String(hh).padStart(2, "0");
  const m = String(mm).padStart(2, "0");
  return `${h}:${m}`;
};

export default function FourWeekCalendarModal({
  open,
  onOpenChange,
  reservations,
  onReserve,
  searchPatients,
  initialMonth,
  maxSameTimeReservations = 2,
}: FourWeekCalendarModalProps) {
  const [currentMonth, setCurrentMonth] = useState<Date>(() => {
    const d = initialMonth ? new Date(initialMonth) : new Date();
    d.setDate(1);
    d.setHours(0, 0, 0, 0);
    return d;
  });
  const [selectedDate, setSelectedDate] = useState<Date | null>(null);

  const [rSearchQuery, setRSearchQuery] = useState("");
  const [rPatients, setRPatients] = useState<Patient[]>([]);
  const [rIsLoading, setRIsLoading] = useState(false);
  const [rShowSuggestions, setRShowSuggestions] = useState(false);
  const [rSelectedIndex, setRSelectedIndex] = useState(-1);
  const [rSelectedPatient, setRSelectedPatient] = useState<Patient | null>(null);

  const [selectedTime, setSelectedTime] = useState<string>("");
  const [selectedHour, setSelectedHour] = useState<string>("");
  const [selectedMinute, setSelectedMinute] = useState<string>("");
  const [reserveMemo, setReserveMemo] = useState<string>("");

  useEffect(() => {
    if (open) {
      const d = new Date();
      d.setDate(1);
      d.setHours(0, 0, 0, 0);
      setCurrentMonth(d);
      setSelectedDate(null);
      setRSearchQuery("");
      setRPatients([]);
      setRShowSuggestions(false);
      setRSelectedIndex(-1);
      setRSelectedPatient(null);
      setSelectedTime("");
      setSelectedHour("");
      setSelectedMinute("");
      setReserveMemo("");
    }
  }, [open]);

  const handleReserveSearch = useCallback(async (query: string) => {
    if (!query.trim()) {
      setRPatients([]);
      return;
    }
    setRIsLoading(true);
    try {
      const items = await searchPatients(query.trim());
      setRPatients(items || []);
    } catch {
      setRPatients([]);
    } finally {
      setRIsLoading(false);
    }
  }, [searchPatients]);

  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => {
      if (rSearchQuery.trim().length >= 1 && !rSelectedPatient) {
        handleReserveSearch(rSearchQuery);
        setRShowSuggestions(true);
      } else if (!rSearchQuery.trim()) {
        setRPatients([]);
        setRShowSuggestions(false);
      }
    }, 300);
    return () => clearTimeout(t);
  }, [open, rSearchQuery, rSelectedPatient, handleReserveSearch]);

  const handleReserveKeyDown = (e: React.KeyboardEvent) => {
    if (!rShowSuggestions || rPatients.length === 0) {
      if (e.key === "Enter") {
        handleReserveSearch(rSearchQuery);
      }
      return;
    }
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setRSelectedIndex((prev) => (prev < rPatients.length - 1 ? prev + 1 : 0));
        break;
      case "ArrowUp":
        e.preventDefault();
        setRSelectedIndex((prev) => (prev > 0 ? prev - 1 : rPatients.length - 1));
        break;
      case "Enter":
        e.preventDefault();
        if (rSelectedIndex >= 0 && rSelectedIndex < rPatients.length) {
          const p = rPatients[rSelectedIndex];
          setRSelectedPatient(p);
          setRSearchQuery(p.name);
          setRPatients([]);
          setRShowSuggestions(false);
          setRSelectedIndex(-1);
        }
        break;
      case "Escape":
        setRShowSuggestions(false);
        setRSelectedIndex(-1);
        break;
    }
  };

  const handleReservePatientSelect = (p: Patient) => {
    setRSelectedPatient(p);
    setRSearchQuery(p.name);
    setRPatients([]);
    setRShowSuggestions(false);
    setRSelectedIndex(-1);
  };

  const days = useMemo(() => getMonthGridDays(currentMonth), [currentMonth]);

  const submitReserve = () => {
    if (!selectedDate || !rSelectedPatient || !selectedTime) return;
    const key = selectedDate.toISOString().slice(0, 10);
    const existing = reservations[key] || [];
    const sameTimeCount = existing.filter((r) => r.time === selectedTime).length;
    if (sameTimeCount >= maxSameTimeReservations) {
      alert(`해당 시간대는 이미 ${maxSameTimeReservations}명 예약되어 있습니다. 다른 시간을 선택하세요.`);
      return;
    }
    onReserve({
      date: selectedDate,
      time: selectedTime,
      patient: rSelectedPatient,
      memo: reserveMemo || undefined,
    });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl">
        <DialogHeader>
          <DialogTitle>예약 날짜 선택</DialogTitle>
        </DialogHeader>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2">
            <div className="flex items-center justify-between mb-2">
              <Button variant="outline" size="icon" onClick={() => setCurrentMonth((prev) => addMonths(prev, -1))}>
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <div className="text-sm font-medium">
                {`${currentMonth.getFullYear()}년 ${String(currentMonth.getMonth() + 1).padStart(2, "0")}월`}
              </div>
              <Button variant="outline" size="icon" onClick={() => setCurrentMonth((prev) => addMonths(prev, 1))}>
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>

            <div className="grid grid-cols-7 gap-2">
              {["월", "화", "수", "목", "금", "토", "일"].map((d) => (
                <div key={d} className="text-center text-xs text-gray-500 py-1">
                  {d}
                </div>
              ))}

              {days.map((d) => {
                const isToday = new Date().toDateString() === d.toDateString();
                const isSelected = selectedDate && selectedDate.toDateString() === d.toDateString();
                const isOutside = d.getMonth() !== currentMonth.getMonth();
                const key = d.toISOString().slice(0, 10);
                const dayReservations = reservations[key] || [];
                return (
                  <button
                    key={d.toISOString()}
                    className={`border rounded-md p-3 text-sm transition text-left ${
                      isSelected
                        ? "border-blue-600 ring-2 ring-blue-300 bg-blue-50"
                        : dayReservations.length > 0
                          ? "border-blue-200 bg-blue-50 hover:bg-blue-100"
                          : isToday
                            ? "border-blue-500 ring-1 ring-blue-200 hover:bg-gray-50"
                            : "border-gray-200 hover:bg-gray-50"
                    } ${isOutside ? "opacity-40" : ""} disabled:opacity-50 disabled:cursor-not-allowed`}
                    disabled={startOfDay(d).getTime() < startOfDay(new Date()).getTime()}
                    onClick={() => {
                      if (startOfDay(d).getTime() < startOfDay(new Date()).getTime()) return;
                      setSelectedDate(d);
                      const t = initSuggestedTime(d);
                      setSelectedTime(t);
                      const [h, m] = t.split(":");
                      setSelectedHour(h);
                      setSelectedMinute(m);
                      setRSearchQuery("");
                      setRPatients([]);
                      setRSelectedIndex(-1);
                      setRSelectedPatient(null);
                      setRShowSuggestions(false);
                      setReserveMemo("");
                    }}
                  >
                    {(() => {
                      const todayOnly = startOfDay(new Date());
                      const dayOnly = startOfDay(d);
                      const isPast = dayOnly.getTime() < todayOnly.getTime();
                      return (
                        <div className={`flex items-start justify-between ${isPast ? "opacity-40" : ""}`}>
                          <span className="font-medium text-base">{d.getDate()}</span>
                          {isToday && <span className="text-xxs text-blue-600">오늘</span>}
                        </div>
                      );
                    })()}
                  </button>
                );
              })}
            </div>

            <div className="mt-4 border-t pt-3">
              <div className="text-sm font-medium text-gray-700 mb-2">해당 날짜 예약자</div>
              {(() => {
                if (!selectedDate)
                  return <div className="text-sm text-gray-500">날짜를 선택하세요</div>;
                const key = selectedDate.toISOString().slice(0, 10);
                const list = [...(reservations[key] || [])].sort((a, b) => a.time.localeCompare(b.time));
                if (list.length === 0)
                  return <div className="text-sm text-gray-500">예약자가 없습니다</div>;
                return (
                  <div className="space-y-2">
                    {list.map((p) => (
                      <div key={`${p.id}-${p.time}`} className="text-sm text-gray-800">
                        - {p.name} (<span className="text-xs text-gray-500">{p.id}</span>) · {p.time}
                      </div>
                    ))}
                  </div>
                );
              })()}
            </div>
          </div>

          <div>
            {!selectedDate ? (
              <div className="h-full flex items-center justify-center border rounded-md p-6 text-sm text-gray-500">
                날짜를 클릭하면 환자 검색창이 여기에 표시됩니다.
              </div>
            ) : (
              <div className="space-y-3">
                <div className="text-sm text-gray-600">
                  선택 날짜: {`${selectedDate.getFullYear()}.${String(selectedDate.getMonth() + 1).padStart(2, "0")}.${String(selectedDate.getDate()).padStart(2, "0")}`}
                </div>

                <div className="relative">
                  <Label htmlFor="reserve-search">환자 검색</Label>
                  <Input
                    id="reserve-search"
                    placeholder="환자 이름을 입력하세요"
                    value={rSearchQuery}
                    onChange={(e) => {
                      setRSearchQuery(e.target.value);
                      if (rSelectedPatient) setRSelectedPatient(null);
                      setRSelectedIndex(-1);
                    }}
                    onKeyDown={handleReserveKeyDown}
                    onFocus={() => {
                      if (rPatients.length > 0) setRShowSuggestions(true);
                    }}
                    onBlur={() => {
                      setTimeout(() => setRShowSuggestions(false), 200);
                    }}
                  />
                  {rShowSuggestions && rPatients.length > 0 && (
                    <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-y-auto">
                      {rPatients.map((p, idx) => (
                        <div
                          key={p.id}
                          className={`p-3 cursor-pointer border-b last:border-b-0 ${idx === rSelectedIndex ? 'bg-blue-50 border-blue-200' : 'hover:bg-gray-50'}`}
                          onClick={() => handleReservePatientSelect(p)}
                        >
                          <div className="font-medium">{p.name}</div>
                          <div className="text-sm text-gray-500">{p.gender} | {p.age}세 | {p.phone}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <Label htmlFor="reserve-time">시간</Label>
                    <div className="flex gap-2">
                      <Select
                        value={selectedHour}
                        onValueChange={(v) => {
                          setSelectedHour(v);
                          const mm = selectedMinute || '00';
                          setSelectedTime(`${v}:${mm}`);
                        }}
                      >
                        <SelectTrigger className="w-[100px]"><SelectValue placeholder="시" /></SelectTrigger>
                        <SelectContent>
                          {Array.from({ length: 9 }).map((_, i) => {
                            const h = String(9 + i).padStart(2, '0');
                            return <SelectItem key={h} value={h}>{h}</SelectItem>;
                          })}
                        </SelectContent>
                      </Select>
                      <Select
                        value={selectedMinute}
                        onValueChange={(v) => {
                          setSelectedMinute(v);
                          const hh = selectedHour || '09';
                          setSelectedTime(`${hh}:${v}`);
                        }}
                      >
                        <SelectTrigger className="w-[100px]"><SelectValue placeholder="분" /></SelectTrigger>
                        <SelectContent>
                          {['00','15','30','45'].map((m) => (
                            <SelectItem key={m} value={m}>{m}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div>
                    <Label htmlFor="reserve-memo">메모</Label>
                    <textarea
                      id="reserve-memo"
                      className="w-full border rounded-md p-2 h-[38px] text-sm"
                      placeholder="메모를 입력하세요"
                      value={reserveMemo}
                      onChange={(e) => setReserveMemo(e.target.value)}
                    />
                  </div>
                </div>

                {rSelectedPatient && (
                  <div className="bg-blue-50 p-3 rounded-md">
                    <div className="text-sm font-medium">선택 환자</div>
                    <div className="mt-1 text-sm">
                      {rSelectedPatient.name} · {rSelectedPatient.gender} · {rSelectedPatient.age}세
                    </div>
                    <div className="mt-1 text-xs text-gray-500">
                      ID: {rSelectedPatient.id} · {rSelectedPatient.phone}
                    </div>
                  </div>
                )}

                <div className="pt-2">
                  <Button
                    className="w-full"
                    disabled={!selectedDate || !rSelectedPatient || rIsLoading}
                    onClick={submitReserve}
                  >
                    {rIsLoading ? "처리 중..." : "이 날짜로 예약 등록"}
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
