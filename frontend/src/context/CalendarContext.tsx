import { createContext, useContext, useMemo, useState, ReactNode, useEffect } from 'react';

export type CalendarEvent = {
  id: string;
  title: string;
  start: string; // ISO string
  end?: string;  // ISO string
  type?: '검진' | '회의' | '내근' | '외근';
  patientId?: string;
  patientName?: string;
  patientGender?: string;
  patientAge?: number;
};

type CalendarContextType = {
  events: CalendarEvent[];
  addEvent: (e: Omit<CalendarEvent, 'id'>) => CalendarEvent;
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
      if (raw) setEvents(JSON.parse(raw));
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
      const withId: CalendarEvent = { id: Math.random().toString(36).slice(2), ...e };
      setEvents((prev) => [...prev, withId]);
      return withId;
    },
    updateEvent: (id, patch) => {
      setEvents((prev) => prev.map((ev) => (ev.id === id ? { ...ev, ...patch } : ev)));
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
