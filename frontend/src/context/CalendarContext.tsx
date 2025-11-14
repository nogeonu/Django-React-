import {
  ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  createAppointmentApi,
  deleteAppointmentApi,
  getAppointmentsApi,
  updateAppointmentApi,
} from "@/lib/api";

export type CalendarEvent = {
  id: string;
  title: string;
  start: string;
  end?: string;
  type?: string;
  status?: string;
  memo?: string;
  patientId?: string;
  patientName?: string;
  patientGender?: string;
  patientAge?: number;
  patient?: number;
  doctorId?: string;
  doctorName?: string;
  doctorDepartment?: string;
};

export type CreateCalendarEventInput = {
  title: string;
  start: string;
  end?: string;
  type?: string;
  status?: string;
  memo?: string;
  doctor: string | number;
  patient?: number;
  patientId?: string;
  patientName?: string;
  patientGender?: string;
  patientAge?: number;
};

export type UpdateCalendarEventInput = Partial<CreateCalendarEventInput>;

type CalendarContextValue = {
  events: CalendarEvent[];
  isLoading: boolean;
  refresh: () => Promise<void>;
  addEvent: (input: CreateCalendarEventInput) => Promise<CalendarEvent>;
  updateEvent: (id: string, updates: UpdateCalendarEventInput) => Promise<CalendarEvent | null>;
  removeEvent: (id: string) => Promise<void>;
};

const CalendarContext = createContext<CalendarContextValue | undefined>(undefined);

const normalizeAppointmentList = (data: any) => {
  if (!data) return [];
  if (Array.isArray(data)) return data;
  if (Array.isArray(data?.results)) return data.results;
  return [];
};

const toCalendarEvent = (item: any): CalendarEvent => {
  const event = {
    id: String(item?.id || ''),
    title: item?.title || '제목 없음',
    start: item?.start_time || '',
    end: item?.end_time || undefined,
    type: item?.type || undefined,
    status: item?.status || undefined,
    memo: item?.memo || undefined,
    patientId: item?.patient_id || undefined,
    patientName: item?.patient_name || undefined,
    patientGender: item?.patient_gender || undefined,
    patientAge: typeof item?.patient_age === "number" ? item.patient_age : undefined,
    patient: typeof item?.patient === "number" ? item.patient : undefined,
    doctorId: item?.doctor_id || undefined,
    doctorName: item?.doctor_name || item?.doctor_username || undefined,
    doctorDepartment: item?.doctor_department || undefined,
  };
  
  console.log("🔄 toCalendarEvent 변환:", {
    원본: item,
    변환결과: event
  });
  
  return event;
};

const toCreatePayload = (input: CreateCalendarEventInput) => {
  const payload: Record<string, unknown> = {
    title: input.title,
    start_time: input.start,
    type: input.type ?? "예약",
    status: input.status ?? "scheduled",
    doctor: typeof input.doctor === "string" ? Number(input.doctor) : input.doctor,
  };
  if (input.end) payload.end_time = input.end;
  if (input.memo !== undefined) payload.memo = input.memo;
  if (input.patient !== undefined) payload.patient = input.patient;
  if (input.patientId !== undefined) payload.patient_id = input.patientId;
  if (input.patientName !== undefined) payload.patient_name = input.patientName;
  if (input.patientGender !== undefined) payload.patient_gender = input.patientGender;
  if (input.patientAge !== undefined) payload.patient_age = input.patientAge;
  
  console.log("예약 등록 payload:", payload);
  return payload;
};

const toUpdatePayload = (input: UpdateCalendarEventInput) => {
  const payload: Record<string, unknown> = {};
  if (input.title !== undefined) payload.title = input.title;
  if (input.start !== undefined) payload.start_time = input.start;
  if (input.end !== undefined) payload.end_time = input.end;
  if (input.type !== undefined) payload.type = input.type;
  if (input.status !== undefined) payload.status = input.status;
  if (input.memo !== undefined) payload.memo = input.memo;
  if (input.doctor !== undefined) payload.doctor = input.doctor;
  if (input.patient !== undefined) payload.patient = input.patient;
  if (input.patientId !== undefined) payload.patient_id = input.patientId;
  if (input.patientName !== undefined) payload.patient_name = input.patientName;
  if (input.patientGender !== undefined) payload.patient_gender = input.patientGender;
  if (input.patientAge !== undefined) payload.patient_age = input.patientAge;
  return payload;
};

export function CalendarProvider({ children }: { children: ReactNode }) {
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const refresh = useCallback(async () => {
    console.log("🔄 refresh() 호출 - 예약 데이터 새로고침 시작");
    setIsLoading(true);
    try {
      const data = await getAppointmentsApi({ page_size: 500 });
      console.log("📥 API 응답 받음:", data);
      
      const list = normalizeAppointmentList(data);
      console.log("📋 정규화된 리스트 (총 " + list.length + "건):", list);
      
      const events = list.map(toCalendarEvent);
      console.log("📅 변환된 이벤트 (총 " + events.length + "건):", events);
      
      setEvents(
        events.sort(
          (a: CalendarEvent, b: CalendarEvent) =>
            new Date(a.start).getTime() - new Date(b.start).getTime(),
        ),
      );
      console.log("✅ 이벤트 상태 업데이트 완료");
    } catch (error) {
      console.error("❌ 예약 데이터를 불러오지 못했습니다.", error);
      throw error;
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh().catch((error) => {
      console.error("초기 예약 데이터 로딩 실패", error);
    });
  }, [refresh]);

  const addEvent = useCallback(async (input: CreateCalendarEventInput) => {
    console.log("addEvent 호출됨, input:", input);
    try {
      const payload = toCreatePayload(input);
      console.log("API 호출 직전, payload:", payload);
      const created = await createAppointmentApi(payload);
      console.log("API 응답 받음:", created);
      const event = toCalendarEvent(created);
      
      // 즉시 UI에 반영
      setEvents((prev) => {
        const newEvents = [...prev, event].sort(
          (a: CalendarEvent, b: CalendarEvent) =>
            new Date(a.start).getTime() - new Date(b.start).getTime(),
        );
        console.log("새로운 이벤트 목록:", newEvents);
        return newEvents;
      });
      
      return event;
    } catch (error: any) {
      console.error("예약 생성에 실패했습니다.", error);
      console.error("에러 상세:", error?.response?.data || error?.message);
      throw error;
    }
  }, []);

  const updateEvent = useCallback(async (id: string, updates: UpdateCalendarEventInput) => {
    try {
      const payload = toUpdatePayload(updates);
      const updated = await updateAppointmentApi(id, payload);
      const event = toCalendarEvent(updated);
      setEvents((prev) =>
        prev
          .map((item) => (item.id === id ? event : item))
          .sort(
            (a: CalendarEvent, b: CalendarEvent) =>
              new Date(a.start).getTime() - new Date(b.start).getTime(),
          ),
      );
      return event;
    } catch (error) {
      console.error("예약 수정에 실패했습니다.", error);
      throw error;
    }
  }, []);

  const removeEvent = useCallback(async (id: string) => {
    try {
      await deleteAppointmentApi(id);
      setEvents((prev) => prev.filter((item) => item.id !== id));
    } catch (error) {
      console.error("예약 삭제에 실패했습니다.", error);
      throw error;
    }
  }, []);

  const value = useMemo(
    () => ({
      events,
      isLoading,
      refresh,
      addEvent,
      updateEvent,
      removeEvent,
    }),
    [events, isLoading, refresh, addEvent, updateEvent, removeEvent],
  );

  return <CalendarContext.Provider value={value}>{children}</CalendarContext.Provider>;
}

export function useCalendar() {
  const context = useContext(CalendarContext);
  if (!context) {
    throw new Error("useCalendar는 CalendarProvider 내부에서만 사용할 수 있습니다.");
  }
  return context;
}
