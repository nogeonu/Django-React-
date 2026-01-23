// frontend/src/types.ts

// 약 기본 정보
export interface Drug {
  item_seq: string;
  name_kor: string;
  company_name?: string | null;
  rx_otc?: string | null;
  edi_code?: string | null;
  atc_code?: string | null;
  is_anticancer?: boolean | null;
}

// DUR 경고
export interface DurWarning {
  item_seq: string;
  drug_name: string;
  rule_type: string;
  raw_type_name?: string | null;
  note?: string | null;
  severity: "HIGH" | "MEDIUM" | "INFO" | string;
}

// 성분 / 약효
export interface IngredientEffect {
  gnl_nm_cd?: string | null;
  gnl_nm?: string | null;
  meft_div_no?: string | null;
  div_nm?: string | null;
  fomn_tp_nm?: string | null;
  injc_pth_nm?: string | null;
}

// MFDS DUR ingredient search
export interface MfdsDurIngredientSummary {
  dur_serial_no: number;
  dur_type?: string | null;
  dur_ing_code?: string | null;
  dur_ing_eng_name?: string | null;
  dur_ing_kor_name?: string | null;
  notice_date?: string | null;
  grade?: string | null;
  status?: string | null;
  class_name?: string | null;
  source_file?: string | null;
}

export interface MfdsDurIngredientDetail {
  dur_serial_no: number;
  dur_type?: string | null;
  single_combo_code?: string | null;
  dur_ing_code?: string | null;
  dur_ing_eng_name?: string | null;
  dur_ing_kor_name?: string | null;
  combo_info?: string | null;
  related_ingredient?: string | null;
  efficacy_class_code?: string | null;
  efficacy_group?: string | null;
  notice_date?: string | null;
  contraindication?: string | null;
  dosage_form?: string | null;
  age_standard?: string | null;
  max_duration?: string | null;
  max_daily_dose?: string | null;
  grade?: string | null;
  coadmin_single_combo_code?: string | null;
  coadmin_dur_ing_code?: string | null;
  coadmin_dur_ing_eng_name?: string | null;
  coadmin_dur_ing_kor_name?: string | null;
  coadmin_combo_info?: string | null;
  coadmin_related_ingredient?: string | null;
  coadmin_efficacy_class?: string | null;
  note?: string | null;
  status?: string | null;
  class_name?: string | null;
  source_file?: string | null;
  created_at?: string | null;
}

export interface MfdsDurIngredientSearchResult {
  total: number;
  limit: number;
  offset: number;
  items: MfdsDurIngredientSummary[];
}

// /drugs/{item_seq}/safety-summary 응답
export interface DrugSafetySummary {
  item_seq: string;
  name_kor: string;
  company_name?: string | null;
  edi_code?: string | null;
  dur_warnings: DurWarning[];
  ingredient_effects: IngredientEffect[];
}

// 처방 아이템(오더셋 내부 약 구성도 이 타입 사용)
export interface PrescriptionItem {
  item_seq: string;
  dose_amount?: string | null;
  dose_frequency?: string | null;
  dose_duration?: number | null;
  memo?: string | null;
}

// 추천 오더셋
export interface RecommendedOrderSet {
  order_set_id: number;
  name: string;
  description?: string | null;
  items: PrescriptionItem[];
  warnings: DurWarning[];
  match_score: number;
  match_reason?: string | null;
}

// 자동 처방 요청 페이로드
export interface AutoPrescribePayload {
  diagnosis_code: string;
  patient_id?: number | null;
  patient_age?: number | null;
  patient_sex?: "M" | "F" | "" | null;
  is_pregnant?: boolean | null;
}

// 자동 처방 응답
export interface AutoPrescribeResult {
  prescription_id: number;
  order_set_id: number;
  order_set_name: string;
  item_count: number;
  match_score: number;
  risk_score: number;
  warnings: DurWarning[];
  alternatives: RecommendedOrderSet[];
}

export interface SymptomInput {
  name: string;
  severity?: number | null;
  days?: number | null;
}

export interface DiagnosisCandidate {
  code: string;
  name: string;
  score: number;
}

export interface RecommendOrderSetsPayload {
  diagnosis_code: string;
  patient_id?: number | null;
  patient_age?: number | null;
  patient_sex?: "M" | "F" | "" | null;
  is_pregnant?: boolean | null;
}

export interface RecommendBySymptomsPayload {
  symptoms: SymptomInput[];
  patient_id?: number | null;
  patient_age?: number | null;
  patient_sex?: "M" | "F" | "" | null;
  is_pregnant?: boolean | null;
  topk_diagnosis?: number | null;
  topk_order_sets_per_diag?: number | null;
  ranking_scope?: "global" | "top_diagnosis_only";
}

export interface OrderSetCandidate {
  order_set_id: number;
  name: string;
  description?: string | null;
  items: PrescriptionItem[];
  warnings: DurWarning[];
  match_score: number;
  match_reason?: string | null;
  diagnosis_code?: string | null;
  diagnosis_name?: string | null;
  recommendation_rank?: number | null;
}

export interface RecommendBySymptomsResult {
  symptoms_normalized: string[];
  diagnosis_candidates: DiagnosisCandidate[];
  order_set_candidates: OrderSetCandidate[];
  recommendation_session_id?: string | null;
}

export interface AutoBySymptomsPayload {
  symptoms: SymptomInput[];
  patient_id?: number | null;
  patient_age?: number | null;
  patient_sex?: "M" | "F" | "" | null;
  is_pregnant?: boolean | null;
  topk_diagnosis?: number | null;
}

export interface AutoBySymptomsResult {
  result: AutoPrescribeResult;
  symptoms_normalized: string[];
  diagnosis_candidates: DiagnosisCandidate[];
  selected_diagnosis_code?: string | null;
  selected_diagnosis_name?: string | null;
}

// Drug Interaction Check
export interface DrugInteractionWarning {
  item_seq_a: string;
  drug_name_a: string;
  item_seq_b: string;
  drug_name_b: string;
  interaction_type: string;
  severity: string;
  warning_message: string;
  prohbt_content?: string | null;
  ai_analysis?: {
    confidence: number;
    summary: string;
    mechanism: string;
    recommendation: string;
  } | null;
}

export interface DrugInteractionResult {
  checked_drugs: Drug[];
  interactions: DrugInteractionWarning[];
  has_critical: boolean;
  has_warnings: boolean;
  total_interactions: number;
  summary: string;
}

