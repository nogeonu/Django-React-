import { createContext, useContext, useMemo, useState, ReactNode, useEffect } from 'react';

export type CalendarEvent = {
  id: string;
  title: string;
  baseTitle?: string;
  start: string; // ISO string
  end?: string;  // ISO string
  type?: '일반검진' | '정기검진' | '추가검사';
  patientId?: string;
  patientName?: string;
  patientGender?: string;
  patientAge?: number;
};

type CalendarEventInput = Omit<CalendarEvent, 'id'> & { id?: string };

type CalendarContextType = {
  events: CalendarEvent[];
  addEvent: (e: CalendarEventInput) => CalendarEvent;
  updateEvent: (id: string, patch: Partial<Omit<CalendarEvent, 'id'>>) => void;
  removeEvent: (id: string) => void;
  clearEvents: () => void;
};

const CalendarContext = createContext<CalendarContextType | undefined>(undefined);

const STORAGE_KEY = 'app.calendar.events';

export function CalendarProvider({ children }: { children: ReactNode }) {
  const [events, setEvents] = useState<CalendarEvent[]>([]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed: CalendarEvent[] = JSON.parse(raw);
        setEvents(parsed.map((ev) => ({
          ...ev,
          baseTitle: ev.baseTitle ?? ev.title,
        })));
      }
    } catch {}
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(events));
    } catch {}
  }, [events]);

  const api = useMemo<CalendarContextType>(() => ({
    events,
    addEvent: (e) => {
      const baseTitle = e.baseTitle ?? e.title;
      const id = e.id ?? Math.random().toString(36).slice(2);
      const withId: CalendarEvent = { id, ...e, baseTitle };
      setEvents((prev) => [...prev, withId]);
      return withId;
    },
    updateEvent: (id, patch) => {
      setEvents((prev) => prev.map((ev) => (
        ev.id === id
          ? { ...ev, ...patch, baseTitle: patch.baseTitle ?? ev.baseTitle ?? patch.title ?? ev.title }
          : ev
      )));
    },
    removeEvent: (id) => setEvents((prev) => prev.filter((e) => e.id !== id)),
    clearEvents: () => setEvents([]),
  }), [events]);

  return <CalendarContext.Provider value={api}>{children}</CalendarContext.Provider>;
}

export function useCalendar() {
  const ctx = useContext(CalendarContext);
  if (!ctx) throw new Error('useCalendar must be used within CalendarProvider');
  return ctx;
}
