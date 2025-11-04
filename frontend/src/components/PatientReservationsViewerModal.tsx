import { Button } from "@/components/ui/button";
import { X } from "lucide-react";

export interface PatientReservationsViewerModalProps {
  open: boolean;
  onClose: () => void;
  patient: { id: string; name: string } | null;
  reservations: Record<string, { id: string; name: string; time: string; memo?: string }[]>;
}

export default function PatientReservationsViewerModal({ open, onClose, patient, reservations }: PatientReservationsViewerModalProps) {
  if (!open || !patient) return null;

  const entries = Object.entries(reservations || {});
  const items: { date: string; time: string; memo?: string }[] = [];
  for (const [dateKey, list] of entries) {
    for (const r of list || []) {
      // 기본적으로 id로 매칭, 필요 시 이름으로 보조 매칭
      if (r.id === patient.id || r.name === patient.name) {
        items.push({ date: dateKey, time: r.time, memo: r.memo });
      }
    }
  }
  items.sort((a, b) => (a.date + " " + a.time).localeCompare(b.date + " " + b.time));

  const fmt = (dateKey: string) => {
    const parts = dateKey.split('-').map(Number);
    const m = parts[1];
    const d = parts[2];
    return `${m}/${d}`;
  };

  return (
    <div 
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
      data-testid="modal-patient-reservations"
    >
      <div 
        className="bg-white rounded-lg w-full max-w-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-5 py-4 border-b flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">예약 검사 일정</h3>
          <Button size="icon" variant="ghost" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </div>
        <div className="p-5 space-y-4">
          {items.length === 0 ? (
            <div className="h-24 flex items-center justify-center text-sm text-gray-500">
              예약 내역이 없습니다
            </div>
          ) : (
            <div className="space-y-4">
              {items.map((it, idx) => (
                <div key={`${it.date}-${it.time}-${idx}`} className="flex items-center justify-between border rounded-md px-4 py-3">
                  <div className="text-gray-900 text-sm font-medium">{fmt(it.date)}&nbsp;&nbsp;{it.time} 예약</div>
                  <div className="w-48 min-h-[40px] bg-gray-100 rounded flex items-center px-3 text-sm text-gray-700">
                    {it.memo || ''}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
