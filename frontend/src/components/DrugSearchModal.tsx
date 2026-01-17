import { useState, FormEvent } from "react";
import { Search, X } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { searchDrugsApi, type Drug } from "@/lib/api";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (drug: Drug) => void;
}

export default function DrugSearchModal({ isOpen, onClose, onSelect }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Drug[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    
    setLoading(true);
    setResults([]);
    
    try {
      const drugs = await searchDrugsApi(query.trim(), 15);
      setResults(drugs);
    } catch (err) {
      console.error("약물 검색 오류:", err);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>약품 검색/추가</DialogTitle>
        </DialogHeader>
        
        <div className="flex-1 overflow-y-auto space-y-4">
          <form onSubmit={handleSearch} className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="약품명, 성분명, 증상 또는 진단명 입력 (예: 타이레놀, 두통, 유방암)"
                className="pl-8"
                autoFocus
              />
            </div>
            <Button type="submit" disabled={loading}>
              검색
            </Button>
          </form>

          {loading && (
            <div className="text-center py-8 text-muted-foreground">
              검색 중...
            </div>
          )}

          {!loading && results.length === 0 && query && (
            <div className="text-center py-8 text-muted-foreground">
              검색 결과가 없습니다.
            </div>
          )}

          {!loading && query.length === 0 && (
            <div className="text-center py-8 text-muted-foreground">
              검색어를 입력하세요.
            </div>
          )}

          <div className="space-y-2">
            {results.map((drug) => (
              <button
                key={drug.item_seq}
                type="button"
                onClick={() => {
                  onSelect(drug);
                  onClose();
                }}
                className="w-full text-left p-3 border rounded-lg hover:bg-accent transition-colors"
              >
                <div className="font-semibold">{drug.name_kor}</div>
                <div className="text-sm text-muted-foreground mt-1">
                  {drug.company_name} | EDI: {drug.edi_code}
                </div>
              </button>
            ))}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
