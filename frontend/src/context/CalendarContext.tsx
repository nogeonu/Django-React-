import {
  ReactNode,
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";

export type CalendarEvent = {
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

type CalendarContextValue = {
  events: CalendarEvent[];
  addEvent: (event: Omit<CalendarEvent, "id">) => void;
  updateEvent: (id: string, updates: Partial<CalendarEvent>) => void;
  removeEvent: (id: string) => void;
};

const CalendarContext = createContext<CalendarContextValue | undefined>(undefined);

const createId = () => Math.random().toString(36).slice(2, 10);

const initialEvents: CalendarEvent[] = [
  {
    id: createId(),
    title: "진료 – 호흡기 내과",
    start: new Date().toISOString(),
    type: "검진",
    patientId: "P-2025-001",
    patientName: "김건강",
    patientGender: "남",
    patientAge: 42,
  },
  {
    id: createId(),
    title: "의국 회의",
    start: new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString(),
    end: new Date(Date.now() + 3 * 60 * 60 * 1000).toISOString(),
    type: "회의",
  },
];

export function CalendarProvider({ children }: { children: ReactNode }) {
  const [events, setEvents] = useState<CalendarEvent[]>(initialEvents);

  const addEvent = useCallback((event: Omit<CalendarEvent, "id">) => {
    setEvents((prev) => [...prev, { ...event, id: createId() }]);
  }, []);

  const updateEvent = useCallback((id: string, updates: Partial<CalendarEvent>) => {
    setEvents((prev) =>
      prev.map((event) => (event.id === id ? { ...event, ...updates } : event)),
    );
  }, []);

  const removeEvent = useCallback((id: string) => {
    setEvents((prev) => prev.filter((event) => event.id !== id));
  }, []);

  const value = useMemo(
    () => ({
      events,
      addEvent,
      updateEvent,
      removeEvent,
    }),
    [events, addEvent, updateEvent, removeEvent],
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
