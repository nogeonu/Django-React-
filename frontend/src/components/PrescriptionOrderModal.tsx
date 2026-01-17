import { useState, useEffect } from "react";
import type { Drug, DrugInteractionResult } from "../types/drug";
import { searchDrugsApi, checkDrugInteractionsApi } from "../lib/api";

interface Props {
    isOpen: boolean;
    onClose: () => void;
}

export default function PrescriptionOrderModal({ isOpen, onClose }: Props) {
    // Form States
    const [patientQuery, setPatientQuery] = useState("");
    const [orderType, setOrderType] = useState("처방전");
    const [priority, setPriority] = useState("routine");
    const [memo, setMemo] = useState("");

    // Drug Entry States
    const [drugQuery, setDrugQuery] = useState("");
    const [showDrugResults, setShowDrugResults] = useState(false);
    const [searchResults, setSearchResults] = useState<Drug[]>([]);
    const [isSearching, setIsSearching] = useState(false);

    // Cart
    const [selectedDrugs, setSelectedDrugs] = useState<Drug[]>([]);

    // Interaction Analysis
    const [interactionResult, setInteractionResult] = useState<DrugInteractionResult | null>(null);
    const [isChecking, setIsChecking] = useState(false);

    // Debounced Search for Drugs/Symptoms...

    async function handleSearch(e: React.FormEvent) {
        e.preventDefault();
        if (!drugQuery.trim()) return;

        setIsSearching(true);
        setSearchResults([]);
        setShowDrugResults(true);

        try {
            // 1. Try Drug Name Search
            const drugs = await searchDrugsApi(drugQuery, 10);
            setSearchResults(drugs);
        } catch (err) {
            console.error(err);
        } finally {
            setIsSearching(false);
        }
    }

    function addDrug(drug: Drug) {
        if (!selectedDrugs.find(d => d.item_seq === drug.item_seq)) {
            setSelectedDrugs([...selectedDrugs, drug]);
        }
        setDrugQuery(""); // Clear input
        setShowDrugResults(false);
    }

    function removeDrug(seq: string) {
        setSelectedDrugs(prev => prev.filter(d => d.item_seq !== seq));
    }

    // Auto-Check Interactions
    useEffect(() => {
        if (selectedDrugs.length >= 2) {
            checkInteractionsNow();
        } else {
            setInteractionResult(null);
        }
    }, [selectedDrugs]);

    async function checkInteractionsNow() {
        setIsChecking(true);
        try {
            const res = await checkDrugInteractionsApi(selectedDrugs.map(d => d.item_seq));
            setInteractionResult(res);
        } catch (err) {
            console.error(err);
        } finally {
            setIsChecking(false);
        }
    }

    if (!isOpen) return null;

    return (
        <div style={{
            position: 'fixed', inset: 0, backgroundColor: 'rgba(0,0,0,0.4)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50
        }}>
            <div style={{
                backgroundColor: 'white', width: '700px', maxHeight: '90vh',
                borderRadius: '12px', display: 'flex', flexDirection: 'column',
                boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)'
            }}>
                {/* Header */}
                <div style={{ padding: '1.25rem 1.5rem', borderBottom: '1px solid #e5e7eb', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                        <h2 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0, color: '#111827' }}>새 주문 생성</h2>
                        <p style={{ margin: '0.2rem 0 0', fontSize: '0.85rem', color: '#6b7280' }}>처방전, 검사, 또는 영상촬영 주문을 생성합니다.</p>
                    </div>
                    <button onClick={onClose} style={{ border: 'none', background: 'transparent', fontSize: '1.5rem', cursor: 'pointer', color: '#9ca3af' }}>&times;</button>
                </div>

                {/* Body */}
                <div style={{ padding: '1.5rem', overflowY: 'auto', flex: 1 }}>

                    {/* Patient Search */}
                    <div style={{ marginBottom: '1.25rem' }}>
                        <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#374151', marginBottom: '0.4rem' }}>환자 검색</label>
                        <input
                            type="text"
                            placeholder="환자명 또는 환자번호로 검색..."
                            value={patientQuery}
                            onChange={e => setPatientQuery(e.target.value)}
                            style={{ width: '100%', padding: '0.6rem', borderRadius: '6px', border: '2px solid #3b82f6', outline: 'none', fontSize: '0.9rem' }}
                        />
                    </div>

                    {/* Type & Priority Row */}
                    <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.25rem' }}>
                        <div style={{ flex: 1 }}>
                            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#374151', marginBottom: '0.4rem' }}>주문 유형</label>
                            <select
                                value={orderType}
                                onChange={e => setOrderType(e.target.value)}
                                style={{ width: '100%', padding: '0.6rem', borderRadius: '6px', border: '1px solid #d1d5db', fontSize: '0.9rem', backgroundColor: 'white' }}
                            >
                                <option>처방전</option>
                                <option>진단검사</option>
                                <option>영상의학</option>
                            </select>
                        </div>
                        <div style={{ flex: 1 }}>
                            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#374151', marginBottom: '0.4rem' }}>우선순위</label>
                            <select
                                value={priority}
                                onChange={e => setPriority(e.target.value)}
                                style={{ width: '100%', padding: '0.6rem', borderRadius: '6px', border: '1px solid #d1d5db', fontSize: '0.9rem', backgroundColor: 'white' }}
                            >
                                <option value="routine">일반</option>
                                <option value="urgent">응급</option>
                                <option value="stat">즉시(Stat)</option>
                            </select>
                        </div>
                    </div>

                    {/* Drug Info Section */}
                    <div style={{ marginBottom: '1.25rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'end', marginBottom: '0.4rem' }}>
                            <label style={{ fontSize: '0.85rem', fontWeight: 600, color: '#374151' }}>약물 정보</label>
                            {selectedDrugs.length > 0 && <span style={{ fontSize: '0.8rem', color: '#2563eb' }}>{selectedDrugs.length}개 선택됨</span>}
                        </div>

                        <form onSubmit={handleSearch} style={{ position: 'relative' }}>
                            <div style={{ display: 'flex', gap: '0.5rem' }}>
                                <input
                                    type="text"
                                    placeholder="약물명 / 성분명 검색 (Enter)..."
                                    value={drugQuery}
                                    onChange={e => setDrugQuery(e.target.value)}
                                    style={{ flex: 1, padding: '0.6rem', borderRadius: '6px', border: '1px solid #d1d5db', fontSize: '0.9rem' }}
                                />
                                <button type="submit" style={{ padding: '0 1rem', borderRadius: '6px', border: '1px solid #d1d5db', backgroundColor: '#f3f4f6', cursor: 'pointer', fontWeight: 600 }}>검색</button>
                            </div>

                            {/* Search Results Dropdown */}
                            {showDrugResults && (
                                <div style={{
                                    position: 'absolute', top: '100%', left: 0, right: 0, zIndex: 20,
                                    backgroundColor: 'white', border: '1px solid #e5e7eb', borderRadius: '6px',
                                    boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)', marginTop: '4px',
                                    maxHeight: '250px', overflowY: 'auto'
                                }}>
                                    {isSearching && <div style={{ padding: '1rem', textAlign: 'center', color: '#6b7280' }}>검색 중...</div>}
                                    {!isSearching && searchResults.length === 0 && <div style={{ padding: '1rem', textAlign: 'center', color: '#6b7280' }}>검색 결과가 없습니다.</div>}
                                    {searchResults.map(drug => (
                                        <div
                                            key={drug.item_seq}
                                            onClick={() => addDrug(drug)}
                                            style={{
                                                padding: '0.6rem 0.8rem', borderBottom: '1px solid #f3f4f6', cursor: 'pointer',
                                                transition: 'background-color 0.1s'
                                            }}
                                            onMouseEnter={e => e.currentTarget.style.backgroundColor = '#f9fafb'}
                                            onMouseLeave={e => e.currentTarget.style.backgroundColor = 'white'}
                                        >
                                            <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{drug.name_kor}</div>
                                            <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>{drug.company_name} | EDI: {drug.edi_code}</div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </form>

                        {/* Selected Drugs List */}
                        <div style={{ marginTop: '0.8rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                            {selectedDrugs.map(drug => (
                                <div key={drug.item_seq} style={{
                                    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                                    padding: '0.6rem', backgroundColor: '#f9fafb', borderRadius: '6px', border: '1px solid #e5e7eb'
                                }}>
                                    <div style={{ flex: 1 }}>
                                        <div style={{ fontWeight: 600, fontSize: '0.9rem', color: '#1f2937' }}>{drug.name_kor}</div>
                                        <div style={{ display: 'flex', gap: '1rem', marginTop: '0.2rem' }}>
                                            <input type="text" placeholder="용량" style={{ width: '60px', padding: '2px 4px', fontSize: '0.8rem', border: '1px solid #d1d5db', borderRadius: '4px' }} />
                                            <input type="text" placeholder="용법" style={{ width: '60px', padding: '2px 4px', fontSize: '0.8rem', border: '1px solid #d1d5db', borderRadius: '4px' }} />
                                            <input type="text" placeholder="기간" style={{ width: '60px', padding: '2px 4px', fontSize: '0.8rem', border: '1px solid #d1d5db', borderRadius: '4px' }} />
                                        </div>
                                    </div>
                                    <button onClick={() => removeDrug(drug.item_seq)} style={{ color: '#ef4444', background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.2rem', padding: '0 0.5rem' }}>&times;</button>
                                </div>
                            ))}
                        </div>

                        {/* Interaction Alert Area */}
                        {selectedDrugs.length >= 2 && (
                            <div style={{ marginTop: '1rem' }}>
                                {isChecking ? (
                                    <div style={{ padding: '0.8rem', backgroundColor: '#f3f4f6', borderRadius: '6px', color: '#6b7280', fontSize: '0.9rem', textAlign: 'center' }}>
                                        💊 약물 상호작용 분석 중...
                                    </div>
                                ) : interactionResult ? (
                                    <div style={{
                                        border: `1px solid ${interactionResult.has_critical ? '#fecaca' : interactionResult.has_warnings ? '#fde68a' : '#a7f3d0'}`,
                                        backgroundColor: interactionResult.has_critical ? '#fef2f2' : interactionResult.has_warnings ? '#fffbeb' : '#ecfdf5',
                                        borderRadius: '8px', padding: '0.8rem'
                                    }}>
                                        <div style={{ fontWeight: 700, fontSize: '0.95rem', marginBottom: '0.5rem', color: interactionResult.has_critical ? '#b91c1c' : interactionResult.has_warnings ? '#92400e' : '#047857' }}>
                                            {interactionResult.summary}
                                        </div>
                                        {interactionResult.interactions.map((inter, idx) => (
                                            <div key={idx} style={{ fontSize: '0.85rem', marginBottom: '0.6rem', paddingLeft: '0.5rem', borderLeft: '3px solid', borderColor: inter.severity === 'CRITICAL' ? '#ef4444' : '#f59e0b' }}>
                                                <div style={{ fontWeight: 700, marginBottom: '0.2rem' }}>{inter.drug_name_a} + {inter.drug_name_b}:</div>
                                                <div style={{ color: '#374151', lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>
                                                    {inter.warning_message}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : null}
                            </div>
                        )}
                    </div>

                    <div style={{ marginBottom: '1.25rem' }}>
                        <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#374151', marginBottom: '0.4rem' }}>메모</label>
                        <textarea
                            placeholder="추가 메모를 입력하세요..."
                            value={memo}
                            onChange={e => setMemo(e.target.value)}
                            style={{ width: '100%', padding: '0.6rem', borderRadius: '6px', border: '1px solid #d1d5db', minHeight: '80px', fontSize: '0.9rem', resize: 'vertical' }}
                        />
                    </div>

                </div>

                {/* Footer */}
                <div style={{ padding: '1.25rem 1.5rem', borderTop: '1px solid #e5e7eb', display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
                    <button onClick={onClose} style={{ padding: '0.6rem 1.2rem', borderRadius: '6px', border: '1px solid #d1d5db', backgroundColor: 'white', color: '#374151', fontWeight: 600, cursor: 'pointer' }}>취소</button>
                    <button style={{ padding: '0.6rem 1.2rem', borderRadius: '6px', border: 'none', backgroundColor: '#5ea2a8', color: 'white', fontWeight: 600, cursor: 'pointer' }}>주문 생성</button>
                </div>
            </div>
        </div>
    );
}
