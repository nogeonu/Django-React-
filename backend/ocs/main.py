# main.py
import os
import json
import uuid
import hmac
import hashlib
import queue
from datetime import date, datetime
from typing import List, Optional, Literal

import pymysql
from dotenv import load_dotenv
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from drug_dictionary import get_english_name # 사전 기반 성분 변환

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "drug")

DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
_CONN_POOL: queue.LifoQueue[pymysql.connections.Connection] = queue.LifoQueue(maxsize=max(DB_POOL_SIZE, 1))

# 가상 성분 매핑 (DUR DB에 없는 성분을 위한 인메모리 저장소 + 파일 지속성)
# Key: fake_id (20억 이상), Value: 성분명 (영문 우선)
VIRTUAL_INGREDIENT_MAP = {}
VIRTUAL_FILE = "virtual_ingredients.json"

def get_virtual_map():
    global VIRTUAL_INGREDIENT_MAP
    if not VIRTUAL_INGREDIENT_MAP and os.path.exists(VIRTUAL_FILE):
        try:
            with open(VIRTUAL_FILE, "r", encoding="utf-8") as f:
                VIRTUAL_INGREDIENT_MAP = {int(k): v for k, v in json.load(f).items()}
        except:
            pass
    return VIRTUAL_INGREDIENT_MAP

def add_virtual_ingredients(items: dict):
    m = get_virtual_map()
    m.update(items)
    try:
        with open(VIRTUAL_FILE, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False)
    except:
        pass


def _new_conn() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


class _PooledConn:
    def __init__(self, raw: pymysql.connections.Connection):
        self._raw = raw
        self._returned = False

    def __getattr__(self, name):
        return getattr(self._raw, name)

    def close(self):
        if self._returned:
            return
        self._returned = True
        raw, self._raw = self._raw, None

        try:
            raw.ping(reconnect=True)
        except Exception:
            try:
                raw.close()
            except Exception:
                pass
            return

        try:
            _CONN_POOL.put_nowait(raw)
        except queue.Full:
            try:
                raw.close()
            except Exception:
                pass


def get_conn():
    try:
        raw = _CONN_POOL.get_nowait()
    except queue.Empty:
        raw = _new_conn()
    else:
        try:
            raw.ping(reconnect=True)
        except Exception:
            try:
                raw.close()
            except Exception:
                pass
            raw = _new_conn()

    return _PooledConn(raw)


# =========================
# FastAPI App
# =========================
app = FastAPI(title="CDSS Drug API")

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Must be False when allow_origins is ["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_http_detail(detail, status_code: int) -> dict:
    if isinstance(detail, dict):
        out = dict(detail)
        if "message" not in out:
            out = {"message": "Request failed", "data": detail}
    else:
        out = {"message": str(detail)}

    out.setdefault("status_code", status_code)
    return out


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": _normalize_http_detail(exc.detail, exc.status_code)},
    )


@app.exception_handler(RequestValidationError)
async def _request_validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": {
                "message": "Validation error",
                "status_code": 422,
                "errors": exc.errors(),
            }
        },
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    debug = os.getenv("DEBUG_ERRORS", "0").lower() in ("1", "true", "yes")
    detail = {"message": "Internal server error", "status_code": 500}
    if debug:
        detail["error"] = repr(exc)
    return JSONResponse(status_code=500, content={"detail": detail})


# =========================
# Pydantic Models
# =========================
class DrugOut(BaseModel):
    item_seq: str
    name_kor: str
    company_name: str
    rx_otc: Optional[str] = None
    edi_code: Optional[str] = None
    atc_code: Optional[str] = None
    is_anticancer: Optional[bool] = None


class DurWarningOut(BaseModel):
    item_seq: str
    drug_name: str
    rule_type: str
    raw_type_name: str | None = None
    note: str | None = None
    severity: str


class InvolvedDrugOut(BaseModel):
    item_seq: str
    drug_name: str | None = None


class InteractionWarningOut(BaseModel):
    warning_type: Literal["DUPLICATE_INGREDIENT"]
    severity: Literal["HIGH", "MEDIUM", "INFO"] = "MEDIUM"
    title: str
    message: str
    ingredient_code: str | None = None
    ingredient_name: str | None = None
    drugs: list[InvolvedDrugOut] = []


class PrescriptionItemIn(BaseModel):
    item_seq: str
    dose_amount: str | None = None
    dose_frequency: str | None = None
    dose_duration: int | None = None
    memo: str | None = None


class PrescriptionCreateIn(BaseModel):
    patient_id: int | None = None
    patient_age: int | None = None
    patient_sex: str | None = None  # "M" / "F"
    is_pregnant: bool | None = None
    items: list[PrescriptionItemIn]


class PrescriptionWithDurOut(BaseModel):
    prescription_id: int
    item_count: int
    warnings: List[DurWarningOut]


class ApplyOrderSetIn(BaseModel):
    order_set_id: int
    patient_id: int | None = None
    patient_age: int | None = None
    patient_sex: str | None = None
    is_pregnant: bool | None = None


class RecommendRequest(BaseModel):
    diagnosis_code: str
    patient_age: int | None = None
    patient_sex: str | None = None
    is_pregnant: bool | None = None


class RecommendedOrderSet(BaseModel):
    order_set_id: int
    name: str
    description: str | None = None
    items: list[PrescriptionItemIn]
    warnings: list[DurWarningOut]
    match_score: int
    match_reason: str | None = None


class AutoPrescribeRequest(BaseModel):
    diagnosis_code: str
    patient_id: int | None = None
    patient_age: int | None = None
    patient_sex: str | None = None
    is_pregnant: bool | None = None


class SymptomIn(BaseModel):
    name: str
    severity: int | None = None
    days: int | None = None


class AutoBySymptomsRequest(BaseModel):
    symptoms: list[SymptomIn]
    patient_id: int | None = None
    patient_age: int | None = None
    patient_sex: str | None = None
    is_pregnant: bool | None = None
    topk_diagnosis: int = 5


class AutoPrescribeOut(BaseModel):
    prescription_id: int
    order_set_id: int
    order_set_name: str
    item_count: int
    match_score: int
    risk_score: int
    warnings: list[DurWarningOut]
    alternatives: list[RecommendedOrderSet]


class DiagnosisCandidateOut(BaseModel):
    code: str
    name: str
    score: float


class AutoBySymptomsOut(BaseModel):
    result: AutoPrescribeOut
    symptoms_normalized: list[str]
    diagnosis_candidates: list[DiagnosisCandidateOut]
    selected_diagnosis_code: str | None = None
    selected_diagnosis_name: str | None = None


class IngredientEffectOut(BaseModel):
    gnl_nm_cd: str | None = None
    gnl_nm: str | None = None
    meft_div_no: str | None = None
    div_nm: str | None = None
    fomn_tp_nm: str | None = None
    injc_pth_nm: str | None = None


class MfdsDurIngredientSummaryOut(BaseModel):
    dur_serial_no: int
    dur_type: str | None = None
    dur_ing_code: str | None = None
    dur_ing_eng_name: str | None = None
    dur_ing_kor_name: str | None = None
    notice_date: date | None = None
    grade: str | None = None
    status: str | None = None
    class_name: str | None = None
    source_file: str | None = None


class MfdsDurIngredientDetailOut(BaseModel):
    dur_serial_no: int
    dur_type: str | None = None
    single_combo_code: str | None = None
    dur_ing_code: str | None = None
    dur_ing_eng_name: str | None = None
    dur_ing_kor_name: str | None = None
    combo_info: str | None = None
    related_ingredient: str | None = None
    efficacy_class_code: str | None = None
    efficacy_group: str | None = None
    notice_date: date | None = None
    contraindication: str | None = None
    dosage_form: str | None = None
    age_standard: str | None = None
    max_duration: str | None = None
    max_daily_dose: str | None = None
    grade: str | None = None
    coadmin_single_combo_code: str | None = None
    coadmin_dur_ing_code: str | None = None
    coadmin_dur_ing_eng_name: str | None = None
    coadmin_dur_ing_kor_name: str | None = None
    coadmin_combo_info: str | None = None
    coadmin_related_ingredient: str | None = None
    coadmin_efficacy_class: str | None = None
    note: str | None = None
    status: str | None = None
    class_name: str | None = None
    source_file: str | None = None
    created_at: datetime | None = None


class MfdsDurIngredientSearchOut(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[MfdsDurIngredientSummaryOut]


class DrugSafetySummaryOut(BaseModel):
    item_seq: str
    name_kor: str
    company_name: str | None = None
    edi_code: str | None = None
    atc_code: str | None = None
    is_anticancer: bool | None = None
    dur_warnings: list[DurWarningOut]
    ingredient_effects: list[IngredientEffectOut]


class OrderSetCandidateOut(BaseModel):
    candidate_id: str | None = None
    diagnosis_code: str
    diagnosis_name: str
    order_set_id: int
    name: str
    description: str | None = None
    items: list[PrescriptionItemIn]
    warnings: list[DurWarningOut]
    match_score: int
    match_reason: str | None = None
    risk_score: int
    composite_score: int
    recommendation_rank: int | None = None
    recommendation_label: str | None = None

    # 카드필드
    item_names_preview: list[str] = []
    high_warning_count: int = 0
    medium_warning_count: int = 0
    info_warning_count: int = 0
    worst_severity: str = "NONE"
    card_summary: str | None = None

    # UX 카드용
    primary_symptom_tags: list[str] = []
    drug_count: int = 0
    contains_high_risk_drug: bool = False
    top_warning_reasons: list[str] = []
    estimated_risk_level: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
    duplicate_ingredient_count: int = 0
    interaction_warnings: list[InteractionWarningOut] = []

    # score/risk 설명용
    risk_breakdown: dict[str, int] = {}
    score_breakdown: dict[str, int] = {}


class RecommendBySymptomsRequest(BaseModel):
    symptoms: list[SymptomIn]
    patient_age: int | None = None
    patient_sex: str | None = None
    is_pregnant: bool | None = None
    topk_diagnosis: int = 5
    topk_order_sets_per_diag: int = 3
    ranking_scope: Literal["global", "top_diagnosis_only"] = "global"


class RecommendBySymptomsOut(BaseModel):
    symptoms_normalized: list[str]
    diagnosis_candidates: list[DiagnosisCandidateOut]
    order_set_candidates: list[OrderSetCandidateOut]
    recommendation_session_id: str | None = None


class PrescriptionItemOut(BaseModel):
    item_seq: str
    drug_name: str | None = None
    dose_amount: str | None = None
    dose_frequency: str | None = None
    dose_duration: int | None = None
    memo: str | None = None


class PrescriptionDetailOut(BaseModel):
    prescription_id: int
    patient_id: int | None = None
    patient_age: int | None = None
    patient_sex: str | None = None
    is_pregnant: bool | None = None
    item_count: int
    items: list[PrescriptionItemOut]
    warnings: list[DurWarningOut]


class ApplyOrderSetOut(BaseModel):
    prescription_id: int
    order_set_id: int
    order_set_name: str
    diagnosis_code: str | None = None
    diagnosis_name: str | None = None
    item_count: int
    warnings: list[DurWarningOut]


class AutoFromRecommendationRequest(BaseModel):
    symptoms: list[SymptomIn] = []
    patient_id: int | None = None
    patient_age: int | None = None
    patient_sex: str | None = None
    is_pregnant: bool | None = None

    recommendation_session_id: str | None = None
    selected_candidate_id: str | None = None

    topk_diagnosis: int = 5
    topk_order_sets_per_diag: int = 5
    ranking_scope: Literal["global", "top_diagnosis_only"] = "top_diagnosis_only"

    selected_rank: int = 1


class AutoFromRecommendationOut(BaseModel):
    prescription_id: int
    selected_rank: int
    selected_candidate_id: str | None = None
    recommendation_session_id: str | None = None
    order_set_id: int
    order_set_name: str
    diagnosis_code: str
    diagnosis_name: str
    item_count: int
    warnings: list[DurWarningOut]


class PrescriptionValidateIn(BaseModel):
    patient_id: int | None = None
    patient_age: int | None = None
    patient_sex: str | None = None
    is_pregnant: bool | None = None
    recommendation_session_id: str | None = None
    selected_candidate_id: str | None = None
    items: list[PrescriptionItemIn]


class PrescriptionValidateOut(BaseModel):
    is_valid: bool
    contraindications: list[DurWarningOut] = []
    dur_warnings: list[DurWarningOut]
    interaction_warnings: list[InteractionWarningOut]
    risk_score: int
    estimated_risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    risk_breakdown: dict[str, int]

    # ✅ NEW: 프론트 토스트용
    resolve_meta: dict | None = None
    resolved_item_seqs: list[str] = []


class PrescriptionCommitOut(BaseModel):
    prescription_id: int
    item_count: int
    dur_warnings: list[DurWarningOut]
    interaction_warnings: list[InteractionWarningOut]
    risk_score: int
    estimated_risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    risk_breakdown: dict[str, int]

    # ✅ NEW
    resolve_meta: dict | None = None
    resolved_item_seqs: list[str] = []


# =========================
# Symptom Helpers
# =========================
def normalize_symptoms(symptoms: list[SymptomIn]) -> list[str]:
    alias = {
        "열": "발열",
        "목아픔": "인후통",
        "목통증": "인후통",
        "코막": "코막힘",
        "콧물남": "콧물",
        "배아픔": "복통",
        "속메스꺼움": "오심",
        "어지러움": "어지럼",
        "가려워": "가려움",
    }
    out: list[str] = []
    for s in symptoms:
        name = (s.name or "").strip()
        if not name:
            continue
        out.append(alias.get(name, name))
    return list(dict.fromkeys(out))  # 중복 제거(순서 유지)


def recommend_diagnoses_by_symptoms(symptom_names: list[str], topk: int = 5):
    """
    symptoms.name 매칭 → symptom_diagnosis_weights 합산 → diagnoses 후보 TOPK 반환
    ※ order_sets가 존재하는 진단만 (EXISTS)
    """
    if not symptom_names:
        return []

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            in_clause = ", ".join(["%s"] * len(symptom_names))
            cur.execute(
                f"SELECT id, name FROM symptoms WHERE name IN ({in_clause})",
                symptom_names,
            )
            sym_rows = cur.fetchall()
            if not sym_rows:
                return []

            sym_ids = [r["id"] for r in sym_rows]
            sym_in = ", ".join(["%s"] * len(sym_ids))

            cur.execute(
                f"""
                SELECT d.id, d.code, d.name, SUM(w.weight) AS score
                FROM symptom_diagnosis_weights w
                JOIN diagnoses d ON d.id = w.diagnosis_id
                WHERE w.symptom_id IN ({sym_in})
                  AND EXISTS (SELECT 1 FROM order_sets os WHERE os.diagnosis_id = d.id)
                GROUP BY d.id, d.code, d.name
                ORDER BY score DESC
                LIMIT %s
                """,
                sym_ids + [topk],
            )
            return cur.fetchall()
    finally:
        conn.close()


# =========================
# DUR Helpers
# =========================
def dedup_sort_dur_warnings(warnings: list[DurWarningOut]) -> list[DurWarningOut]:
    """
    - dedup key: (item_seq, rule_type, raw_type_name)
    - sort: HIGH -> MEDIUM -> INFO
    """
    if not warnings:
        return []

    severity_order = {"HIGH": 0, "MEDIUM": 1, "INFO": 2, "NONE": 3}
    best_by_key: dict[tuple[str, str, str | None], DurWarningOut] = {}

    for w in warnings:
        key = (w.item_seq, w.rule_type, w.raw_type_name)
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = w
            continue

        existing_note = (existing.note or "").strip()
        new_note = (w.note or "").strip()
        if (not existing_note and new_note) or (len(new_note) > len(existing_note)):
            best_by_key[key] = w

    out = list(best_by_key.values())
    out.sort(
        key=lambda w: (
            severity_order.get((w.severity or "INFO").upper(), 99),
            w.item_seq or "",
            w.rule_type or "",
            w.raw_type_name or "",
        )
    )
    return out


def get_dur_warnings_for_items(item_seqs: list[str], is_pregnant: bool | None = None) -> list[DurWarningOut]:
    if not item_seqs:
        return []

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            in_clause = ", ".join(["%s"] * len(item_seqs))

            cur.execute(
                f"""
                SELECT item_seq, name_kor
                FROM drugs
                WHERE item_seq IN ({in_clause})
                """,
                item_seqs,
            )
            name_rows = cur.fetchall()
            name_by_item = {r["item_seq"]: r["name_kor"] for r in name_rows}

            cur.execute(
                f"""
                SELECT DISTINCT item_seq, rule_type, raw_type_name, note
                FROM dur_product_rules
                WHERE item_seq IN ({in_clause})
                """,
                item_seqs,
            )
            rules = cur.fetchall()
    finally:
        conn.close()

    if not rules:
        return []

    preg = bool(is_pregnant) if is_pregnant is not None else False

    warnings: list[DurWarningOut] = []
    for rule in rules:
        item_seq = rule["item_seq"]
        rule_type = rule.get("rule_type") or "OTHER"
        raw_type_name = rule.get("raw_type_name")
        note = rule.get("note")

        if rule_type == "PREGNANCY":
            severity = "HIGH" if preg else "INFO"
        elif rule_type == "AGE":
            severity = "HIGH"
        elif rule_type == "ELDERLY":
            severity = "MEDIUM"
        else:
            severity = "INFO"

        warnings.append(
            DurWarningOut(
                item_seq=item_seq,
                drug_name=name_by_item.get(item_seq, ""),
                rule_type=rule_type,
                raw_type_name=raw_type_name,
                note=note,
                severity=severity,
            )
        )

    return dedup_sort_dur_warnings(warnings)


def get_dur_warnings_for_prescription(prescription_id: int) -> list[DurWarningOut]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, patient_age, patient_sex, is_pregnant
                FROM prescriptions
                WHERE id = %s
                """,
                (prescription_id,),
            )
            presc = cur.fetchone()
            if not presc:
                return []

            raw_is_pregnant = presc.get("is_pregnant")
            is_pregnant = bool(raw_is_pregnant) if raw_is_pregnant is not None else False

            cur.execute(
                """
                SELECT pi.item_seq, d.name_kor
                FROM prescription_items pi
                JOIN drugs d
                  ON d.item_seq COLLATE utf8mb4_general_ci
                   = pi.item_seq COLLATE utf8mb4_general_ci
                WHERE pi.prescription_id = %s
                """,
                (prescription_id,),
            )
            items = cur.fetchall()
            if not items:
                return []

            item_seqs = [row["item_seq"] for row in items]
            in_clause = ", ".join(["%s"] * len(item_seqs))

            cur.execute(
                f"""
                SELECT DISTINCT item_seq, rule_type, raw_type_name, note
                FROM dur_product_rules
                WHERE item_seq IN ({in_clause})
                """,
                item_seqs,
            )
            rules = cur.fetchall()
    finally:
        conn.close()

    if not rules:
        return []

    name_by_item = {row["item_seq"]: row["name_kor"] for row in items}
    warnings: list[DurWarningOut] = []

    for rule in rules:
        item_seq = rule["item_seq"]
        rule_type = rule.get("rule_type") or "OTHER"
        raw_type_name = rule.get("raw_type_name")
        note = rule.get("note")

        if rule_type == "PREGNANCY":
            severity = "HIGH" if is_pregnant else "INFO"
        elif rule_type == "AGE":
            severity = "HIGH"
        elif rule_type == "ELDERLY":
            severity = "MEDIUM"
        else:
            severity = "INFO"

        warnings.append(
            DurWarningOut(
                item_seq=item_seq,
                drug_name=name_by_item.get(item_seq, ""),
                rule_type=rule_type,
                raw_type_name=raw_type_name,
                note=note,
                severity=severity,
            )
        )

    return dedup_sort_dur_warnings(warnings)


# =========================
# Ingredient Risk Helpers
# =========================
def compute_ingredient_risk_for_items(item_seqs: list[str]) -> dict:
    if not item_seqs:
        return {
            "risk_score": 0,
            "ingredient_count": 0,
            "high_risk_count": 0,
            "duplicate_ingredient_count": 0,
            "duplicate_ingredients": [],
        }

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            in_clause = ", ".join(["%s"] * len(item_seqs))
            sql = f"""
                SELECT DISTINCT
                    d.item_seq,
                    d.edi_code,
                    m.gnl_nm_cd,
                    m.gnl_nm,
                    e.meft_div_no,
                    e.div_nm,
                    e.fomn_tp_nm,
                    e.injc_pth_nm
                FROM drugs d
                JOIN hira_drug_ingredient_map m
                    ON BINARY m.edi_code = BINARY d.edi_code
                LEFT JOIN hira_ingredient_effects e
                    ON BINARY e.gnl_nm_cd = BINARY m.gnl_nm_cd
                WHERE d.item_seq IN ({in_clause})
            """
            cur.execute(sql, item_seqs)
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return {
            "risk_score": 0,
            "ingredient_count": 0,
            "high_risk_count": 0,
            "duplicate_ingredient_count": 0,
            "duplicate_ingredients": [],
        }

    seen_gnl: set[str] = set()
    gnl_to_items: dict[str, set[str]] = {}
    gnl_to_name: dict[str, str] = {}
    high_risk_count = 0
    risk_score = 0

    for r in rows:
        gnl_cd = r.get("gnl_nm_cd")
        if gnl_cd:
            seen_gnl.add(gnl_cd)
            item_seq = r.get("item_seq")
            if item_seq:
                gnl_to_items.setdefault(gnl_cd, set()).add(str(item_seq))
            if gnl_cd not in gnl_to_name and r.get("gnl_nm"):
                gnl_to_name[gnl_cd] = str(r.get("gnl_nm"))

        div_nm = (r.get("div_nm") or "").lower()
        inj_route = (r.get("injc_pth_nm") or "")

        high_keywords = [
            "항암", "antineoplastic",
            "항응고", "anticoagulant",
            "면역억제", "immunosuppress",
            "steroid", "스테로이드",
        ]
        if any(kw in div_nm for kw in high_keywords):
            high_risk_count += 1
            risk_score += 5

        inj_keywords = ["정맥", "근육", "주사", "intravenous", "intramuscular", "injection"]
        if any(kw in inj_route for kw in inj_keywords):
            risk_score += 1

    ingredient_count = len(seen_gnl)
    if ingredient_count >= 5:
        risk_score += 3
    if ingredient_count >= 10:
        risk_score += 5

    duplicate_ingredients = []
    for gnl_cd, item_set in gnl_to_items.items():
        item_list = sorted([s for s in item_set if s])
        if len(item_list) < 2:
            continue
        duplicate_ingredients.append(
            {
                "gnl_nm_cd": gnl_cd,
                "gnl_nm": gnl_to_name.get(gnl_cd),
                "item_seqs": item_list,
            }
        )

    duplicate_ingredients.sort(
        key=lambda x: (
            -len(x.get("item_seqs") or []),
            (x.get("gnl_nm") or ""),
            x.get("gnl_nm_cd") or "",
        )
    )

    return {
        "risk_score": risk_score,
        "ingredient_count": ingredient_count,
        "high_risk_count": high_risk_count,
        "duplicate_ingredient_count": len(duplicate_ingredients),
        "duplicate_ingredients": duplicate_ingredients,
    }


def get_ingredient_effects_for_item(item_seq: str) -> list[IngredientEffectOut]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            sql = """
                SELECT DISTINCT
                    m.gnl_nm_cd,
                    m.gnl_nm,
                    e.meft_div_no,
                    e.div_nm,
                    e.fomn_tp_nm,
                    e.injc_pth_nm
                FROM drugs d
                JOIN hira_drug_ingredient_map m
                    ON BINARY m.edi_code = BINARY d.edi_code
                LEFT JOIN hira_ingredient_effects e
                    ON BINARY e.gnl_nm_cd = BINARY m.gnl_nm_cd
                WHERE d.item_seq = %s
            """
            cur.execute(sql, (item_seq,))
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        IngredientEffectOut(
            gnl_nm_cd=r.get("gnl_nm_cd"),
            gnl_nm=r.get("gnl_nm"),
            meft_div_no=r.get("meft_div_no"),
            div_nm=r.get("div_nm"),
            fomn_tp_nm=r.get("fomn_tp_nm"),
            injc_pth_nm=r.get("injc_pth_nm"),
        )
        for r in rows
    ]


def get_drug_name_map(item_seqs: list[str]) -> dict[str, str]:
    if not item_seqs:
        return {}

    uniq = list(dict.fromkeys([s for s in item_seqs if s]))
    if not uniq:
        return {}

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            in_clause = ", ".join(["%s"] * len(uniq))
            cur.execute(
                f"""
                SELECT item_seq, name_kor
                FROM drugs
                WHERE item_seq IN ({in_clause})
                """,
                uniq,
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return {r["item_seq"]: r["name_kor"] for r in rows if r.get("item_seq") and r.get("name_kor")}


def build_duplicate_ingredient_warnings(
    ing_profile: dict,
    name_by_item: dict[str, str],
) -> list[InteractionWarningOut]:
    warnings: list[InteractionWarningOut] = []
    for dup in ing_profile.get("duplicate_ingredients") or []:
        ingredient_code = dup.get("gnl_nm_cd")
        ingredient_name = dup.get("gnl_nm") or ingredient_code or ""
        item_list = dup.get("item_seqs") or []
        drugs = [InvolvedDrugOut(item_seq=str(seq), drug_name=name_by_item.get(str(seq))) for seq in item_list]
        drug_labels = [d.drug_name or d.item_seq for d in drugs]
        warnings.append(
            InteractionWarningOut(
                warning_type="DUPLICATE_INGREDIENT",
                severity="MEDIUM",
                title="중복 성분 발견",
                message=f"중복 성분({ingredient_name}) 포함: {', '.join(drug_labels)}",
                ingredient_code=ingredient_code,
                ingredient_name=dup.get("gnl_nm"),
                drugs=drugs,
            )
        )
    return warnings


def validate_prescription_items(
    items: list[PrescriptionItemIn],
    patient_age: int | None,
    patient_sex: str | None,
    is_pregnant: bool | None,
) -> dict:
    if not items:
        raise HTTPException(status_code=400, detail="items가 비어있습니다.")

    item_seqs = [it.item_seq for it in items if (it.item_seq or "").strip()]
    item_seqs = list(dict.fromkeys(item_seqs))
    if not item_seqs:
        raise HTTPException(status_code=400, detail="item_seq가 비어있습니다.")

    name_by_item = get_drug_name_map(item_seqs)
    missing = [seq for seq in item_seqs if seq not in name_by_item]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={"message": "알 수 없는 item_seq가 포함되어 있습니다.", "missing_item_seqs": missing},
        )

    preg = bool(is_pregnant) if is_pregnant is not None else False
    dur_warnings = get_dur_warnings_for_items(item_seqs, preg)

    ing_profile = compute_ingredient_risk_for_items(item_seqs)
    interaction_warnings = build_duplicate_ingredient_warnings(ing_profile, name_by_item)

    dur_risk = 0
    for w in dur_warnings:
        if w.severity == "HIGH":
            dur_risk += 10
        elif w.severity == "MEDIUM":
            dur_risk += 5
        else:
            dur_risk += 1

    duplicate_risk = len(interaction_warnings) * 5
    ingredient_risk = int(ing_profile.get("risk_score", 0))
    risk_score = dur_risk + ingredient_risk + duplicate_risk

    worst = compute_worst_severity([w.severity for w in dur_warnings] + [w.severity for w in interaction_warnings])
    estimated_risk_level = estimate_risk_level(risk_score, worst)

    contraindications = [w for w in dur_warnings if w.severity == "HIGH"]

    return {
        "dur_warnings": dur_warnings,
        "interaction_warnings": interaction_warnings,
        "contraindications": contraindications,
        "risk_score": risk_score,
        "estimated_risk_level": estimated_risk_level,
        "risk_breakdown": {
            "dur": dur_risk,
            "ingredient_profile": ingredient_risk,
            "duplicate_ingredient": duplicate_risk,
        },
    }


# =========================
# Helper: order_set -> prescriptions 저장
# =========================
def apply_order_set_to_prescription(
    order_set_id: int,
    patient_id: int | None,
    patient_age: int | None,
    patient_sex: str | None,
    is_pregnant: bool | None,
) -> int:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prescriptions (patient_id, patient_age, patient_sex, is_pregnant)
                VALUES (%s, %s, %s, %s)
                """,
                (patient_id, patient_age, patient_sex, is_pregnant),
            )
            presc_id = cur.lastrowid

            cur.execute(
                """
                SELECT item_seq, default_dose, default_freq, default_days
                FROM order_set_items
                WHERE order_set_id = %s
                """,
                (order_set_id,),
            )
            items = cur.fetchall()
            if not items:
                raise HTTPException(status_code=400, detail="오더셋에 등록된 약이 없습니다.")

            for it in items:
                cur.execute(
                    """
                    INSERT INTO prescription_items
                        (prescription_id, item_seq, dose_amount, dose_frequency, dose_duration, memo)
                    VALUES
                        (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        presc_id,
                        it["item_seq"],
                        it["default_dose"],
                        it["default_freq"],
                        it["default_days"],
                        None,
                    ),
                )

            return presc_id
    finally:
        conn.close()


# =========================
# Card/Scoring Helpers
# =========================
_SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "INFO": 2, "NONE": 3}


def estimate_risk_level(total_risk: int, worst_severity: str) -> Literal["LOW", "MEDIUM", "HIGH"]:
    if worst_severity == "HIGH" or total_risk >= 25:
        return "HIGH"
    if worst_severity == "MEDIUM" or total_risk >= 12:
        return "MEDIUM"
    return "LOW"


def summarize_top_warning_reasons(warnings: list[DurWarningOut], limit: int = 2) -> list[str]:
    if not warnings or limit <= 0:
        return []

    def _key(w: DurWarningOut):
        return (
            _SEVERITY_ORDER.get(w.severity or "INFO", 99),
            w.rule_type or "",
            w.raw_type_name or "",
            w.item_seq or "",
        )

    ordered = sorted(warnings, key=_key)

    def _pick(severity: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for w in ordered:
            if w.severity != severity:
                continue
            reason = (w.raw_type_name or w.rule_type or "").strip() or "DUR"
            label = f"{w.drug_name}: {reason}" if (w.drug_name or "").strip() else reason
            if label in seen:
                continue
            seen.add(label)
            out.append(label)
            if len(out) >= limit:
                break
        return out

    top = _pick("HIGH")
    if top:
        return top
    top = _pick("MEDIUM")
    if top:
        return top
    return _pick("INFO")


def compute_worst_severity(severities: list[str]) -> str:
    if not severities:
        return "NONE"

    def _key(sev: str) -> int:
        return _SEVERITY_ORDER.get((sev or "INFO").upper(), 99)

    return min(severities, key=_key).upper()


# =========================
# Helper: 진단 코드의 order_set 후보(저장 X)
# =========================
def build_order_set_candidates_for_diagnosis(
    diagnosis_code: str,
    patient_age: int | None,
    is_preg: bool,
    topk: int = 3,
) -> list[dict]:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, code, name FROM diagnoses WHERE code=%s", (diagnosis_code,))
            diag = cur.fetchone()
            if not diag:
                return []

            cur.execute(
                """
                SELECT id, name, description, min_age, max_age, allow_pregnant
                FROM order_sets
                WHERE diagnosis_id = %s
                """,
                (diag["id"],),
            )
            os_rows = cur.fetchall()
            if not os_rows:
                return []

            out = []
            for os_row in os_rows:
                os_id = os_row["id"]

                cur.execute(
                    """
                    SELECT item_seq, default_dose, default_freq, default_days
                    FROM order_set_items
                    WHERE order_set_id = %s
                    """,
                    (os_id,),
                )
                items = cur.fetchall()
                if not items:
                    continue

                # --- match score ---
                score = 0
                reasons: list[str] = []
                score_breakdown: dict[str, int] = {}

                min_age = os_row.get("min_age")
                max_age = os_row.get("max_age")
                allow_pregnant = os_row.get("allow_pregnant")

                if patient_age is not None:
                    if min_age is not None:
                        if patient_age >= min_age:
                            score += 1
                            reasons.append(f"{min_age}세 이상 조건 충족")
                            score_breakdown["age_min"] = 1
                        else:
                            continue
                    if max_age is not None:
                        if patient_age <= max_age:
                            score += 1
                            reasons.append(f"{max_age}세 이하 조건 충족")
                            score_breakdown["age_max"] = 1
                        else:
                            continue

                if allow_pregnant is not None:
                    if is_preg and allow_pregnant == 0:
                        continue
                    if not is_preg and allow_pregnant == 0:
                        score += 1
                        reasons.append("비임부 권장")
                        score_breakdown["non_pregnant_preferred"] = 1
                    if allow_pregnant == 1:
                        score += 1
                        reasons.append("임부 사용 가능")
                        score_breakdown["pregnancy_allowed"] = 1

                item_seqs = [it["item_seq"] for it in items]
                warnings = get_dur_warnings_for_items(item_seqs, is_preg)

                # 약 이름 미리보기용 조회
                name_by_item = {}
                if item_seqs:
                    in_clause = ", ".join(["%s"] * len(item_seqs))
                    cur.execute(
                        f"""
                        SELECT item_seq, name_kor
                        FROM drugs
                        WHERE item_seq IN ({in_clause})
                        """,
                        item_seqs,
                    )
                    name_rows = cur.fetchall()
                    name_by_item = {r["item_seq"]: r["name_kor"] for r in name_rows}

                # 경고 집계(카드용)
                high_cnt = sum(1 for w in warnings if w.severity == "HIGH")
                med_cnt = sum(1 for w in warnings if w.severity == "MEDIUM")
                info_cnt = sum(1 for w in warnings if w.severity == "INFO")

                if high_cnt > 0:
                    worst = "HIGH"
                elif med_cnt > 0:
                    worst = "MEDIUM"
                elif info_cnt > 0:
                    worst = "INFO"
                else:
                    worst = "NONE"

                # 약 이름 미리보기(최대 3개)
                preview = []
                for seq in item_seqs:
                    nm = name_by_item.get(seq)
                    if nm:
                        preview.append(nm)
                    if len(preview) >= 3:
                        break

                # --- risk scoring ---
                dur_risk = 0
                for w in warnings:
                    if w.severity == "HIGH":
                        dur_risk += 10
                    elif w.severity == "MEDIUM":
                        dur_risk += 5
                    else:
                        dur_risk += 1

                ing_profile = compute_ingredient_risk_for_items(item_seqs)
                interaction_warnings = build_duplicate_ingredient_warnings(ing_profile, name_by_item)
                duplicate_ingredient_count = len(interaction_warnings)
                duplicate_risk = duplicate_ingredient_count * 5

                total_risk = dur_risk + int(ing_profile.get("risk_score", 0)) + duplicate_risk

                composite = score * 10 - total_risk
                risk_breakdown = {
                    "dur": dur_risk,
                    "ingredient_profile": int(ing_profile.get("risk_score", 0)),
                    "duplicate_ingredient": duplicate_risk,
                }

                drug_count = len(item_seqs)
                contains_high_risk_drug = bool(ing_profile.get("high_risk_count", 0) > 0)
                estimated_risk_level = estimate_risk_level(total_risk, worst)
                top_warning_reasons = summarize_top_warning_reasons(warnings, limit=2)

                card_summary = (
                    f"{drug_count}개 약 · 위험 {estimated_risk_level} · 경고 {worst} "
                    f"(H{high_cnt}/M{med_cnt}/I{info_cnt})"
                )

                rec_items = [
                    PrescriptionItemIn(
                        item_seq=it["item_seq"],
                        dose_amount=it["default_dose"],
                        dose_frequency=it["default_freq"],
                        dose_duration=it["default_days"],
                        memo=None,
                    )
                    for it in items
                ]

                out.append(
                    {
                        "diagnosis_code": diag["code"],
                        "diagnosis_name": diag["name"],
                        "order_set_id": os_id,
                        "name": os_row["name"],
                        "description": os_row.get("description"),
                        "items": rec_items,
                        "warnings": warnings,
                        "match_score": score,
                        "match_reason": "; ".join(reasons) if reasons else None,
                        "risk_score": total_risk,
                        "risk_breakdown": risk_breakdown,
                        "score_breakdown": score_breakdown,
                        "composite_score": composite,

                        # 카드필드
                        "item_names_preview": preview,
                        "high_warning_count": high_cnt,
                        "medium_warning_count": med_cnt,
                        "info_warning_count": info_cnt,
                        "worst_severity": worst,
                        "card_summary": card_summary,

                        # UX 카드용
                        "drug_count": drug_count,
                        "contains_high_risk_drug": contains_high_risk_drug,
                        "top_warning_reasons": top_warning_reasons,
                        "estimated_risk_level": estimated_risk_level,
                        "duplicate_ingredient_count": duplicate_ingredient_count,
                        "interaction_warnings": interaction_warnings,
                    }
                )

            out.sort(
                key=lambda x: (
                    -x["composite_score"],
                    x["risk_score"],
                    -x["match_score"],
                    x["order_set_id"],
                )
            )

            return out[:topk]
    finally:
        conn.close()


# =========================
# Recommendation Session Helpers + Audit
# =========================
_RECOMMENDATION_TOKEN_SECRET = os.getenv("RECOMMENDATION_TOKEN_SECRET") or (DB_PASSWORD or "")
_AUTO_CREATE_TABLES = os.getenv("AUTO_CREATE_TABLES", "0").lower() in ("1", "true", "yes")


def _jsonable(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def make_candidate_token(order_set_id: int, diagnosis_code: str) -> str:
    base = f"os:{int(order_set_id)}:{diagnosis_code}"
    if not _RECOMMENDATION_TOKEN_SECRET:
        return base
    sig = hmac.new(
        _RECOMMENDATION_TOKEN_SECRET.encode("utf-8"),
        base.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:16]
    return f"{base}:{sig}"


def parse_candidate_token(candidate_id: str) -> tuple[int, str] | None:
    if not candidate_id or not candidate_id.startswith("os:"):
        return None

    parts = candidate_id.split(":")
    if len(parts) not in (3, 4):
        return None

    try:
        order_set_id = int(parts[1])
    except ValueError:
        return None

    diagnosis_code = parts[2]
    if len(parts) == 4 and _RECOMMENDATION_TOKEN_SECRET:
        base = ":".join(parts[:3])
        expected = hmac.new(
            _RECOMMENDATION_TOKEN_SECRET.encode("utf-8"),
            base.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:16]
        if not hmac.compare_digest(parts[3], expected):
            return None

    return order_set_id, diagnosis_code


def _parse_json_field(val):
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, (bytes, bytearray)):
        val = val.decode("utf-8", errors="ignore")
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return None
    return None


def ensure_audit_tables() -> None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_sessions (
                    id VARCHAR(36) PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    request_payload LONGTEXT NOT NULL,
                    candidates_payload LONGTEXT NOT NULL,
                    KEY idx_created_at (created_at)
                ) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prescription_audit_logs (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    event_type VARCHAR(64) NOT NULL,
                    patient_id INT NULL,
                    prescription_id INT NULL,
                    recommendation_session_id VARCHAR(36) NULL,
                    selected_candidate_id VARCHAR(128) NULL,
                    payload LONGTEXT NOT NULL,
                    KEY idx_event_type_created_at (event_type, created_at),
                    KEY idx_prescription_id (prescription_id),
                    KEY idx_recommendation_session_id (recommendation_session_id)
                ) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci
                """
            )
    finally:
        conn.close()


def log_audit_event(
    event_type: str,
    payload: dict,
    patient_id: int | None = None,
    prescription_id: int | None = None,
    recommendation_session_id: str | None = None,
    selected_candidate_id: str | None = None,
) -> None:
    if not event_type:
        return

    data = json.dumps(_jsonable(payload), ensure_ascii=False)

    def _insert() -> None:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prescription_audit_logs
                        (event_type, patient_id, prescription_id, recommendation_session_id, selected_candidate_id, payload)
                    VALUES
                        (%s, %s, %s, %s, %s, %s)
                    """,
                    (event_type, patient_id, prescription_id, recommendation_session_id, selected_candidate_id, data),
                )
        finally:
            conn.close()

    try:
        _insert()
    except Exception:
        if _AUTO_CREATE_TABLES:
            try:
                ensure_audit_tables()
                _insert()
            except Exception:
                return


def create_recommendation_session(request_payload: dict, candidates_payload: list[dict]) -> str | None:
    session_id = str(uuid.uuid4())
    request_json = json.dumps(_jsonable(request_payload), ensure_ascii=False)
    candidates_json = json.dumps(_jsonable(candidates_payload), ensure_ascii=False)

    def _insert() -> None:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO recommendation_sessions (id, request_payload, candidates_payload)
                    VALUES (%s, %s, %s)
                    """,
                    (session_id, request_json, candidates_json),
                )
        finally:
            conn.close()

    try:
        _insert()
        return session_id
    except Exception:
        if _AUTO_CREATE_TABLES:
            try:
                ensure_audit_tables()
                _insert()
                return session_id
            except Exception:
                return None
        return None


def get_recommendation_session(session_id: str) -> dict | None:
    if not session_id:
        return None

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, request_payload, candidates_payload, created_at
                FROM recommendation_sessions
                WHERE id = %s
                """,
                (session_id,),
            )
            row = cur.fetchone()
    except Exception:
        return None
    finally:
        conn.close()

    if not row:
        return None

    return {
        "id": row.get("id"),
        "created_at": row.get("created_at"),
        "request": _parse_json_field(row.get("request_payload")),
        "candidates": _parse_json_field(row.get("candidates_payload")) or [],
    }


@app.on_event("startup")
def _startup_init_tables():
    if _AUTO_CREATE_TABLES:
        try:
            ensure_audit_tables()
        except Exception:
            pass


def _to_prescription_items(items_any) -> list[PrescriptionItemIn]:
    out: list[PrescriptionItemIn] = []
    for it in (items_any or []):
        if isinstance(it, PrescriptionItemIn):
            out.append(it)
        elif isinstance(it, dict):
            out.append(
                PrescriptionItemIn(
                    item_seq=str(it.get("item_seq") or "").strip(),
                    dose_amount=it.get("dose_amount"),
                    dose_frequency=it.get("dose_frequency"),
                    dose_duration=it.get("dose_duration"),
                    memo=it.get("memo"),
                )
            )
    return [x for x in out if (x.item_seq or "").strip()]


def get_candidate_items_from_session(session_id: str, candidate_id: str) -> list[PrescriptionItemIn] | None:
    session = get_recommendation_session(session_id)
    if not session:
        return None
    for cand in session.get("candidates") or []:
        if cand.get("candidate_id") == candidate_id:
            return _to_prescription_items(cand.get("items"))
    return None


# =========================
# ✅ Overwrite selection: validate/commit items resolver
# =========================
def _item_seqs(items: list[PrescriptionItemIn]) -> list[str]:
    return sorted({(x.item_seq or "").strip() for x in items if (x.item_seq or "").strip()})


def resolve_items_for_validation(body: PrescriptionValidateIn) -> tuple[list[PrescriptionItemIn], dict]:
    """
    ✅ Overwrite 모드:
    - recommendation_session_id + selected_candidate_id가 있으면 스냅샷 items를 무조건 사용
    - body.items가 있고 스냅샷과 다르면: 거절 X, 스냅샷으로 덮어쓰기 + meta로 기록
    """
    meta = {
        "used_snapshot": False,
        "items_overwritten": False,
        "client_item_seqs": [],
        "snapshot_item_seqs": [],
    }

    if body.recommendation_session_id and body.selected_candidate_id:
        snap_items = get_candidate_items_from_session(body.recommendation_session_id, body.selected_candidate_id)
        if not snap_items:
            raise HTTPException(status_code=400, detail="selected_candidate_id not in recommendation_session.")

        meta["used_snapshot"] = True
        meta["snapshot_item_seqs"] = _item_seqs(snap_items)

        if body.items:
            meta["client_item_seqs"] = _item_seqs(body.items)
            if meta["client_item_seqs"] != meta["snapshot_item_seqs"]:
                meta["items_overwritten"] = True

        return snap_items, meta

    meta["client_item_seqs"] = _item_seqs(body.items)
    return body.items, meta


# =========================
# Endpoints
# =========================
@app.get("/drugs/search", response_model=List[DrugOut])
def search_drugs(
    q: str = Query(..., description="검색어 (제품명 일부 or 증상)"),
    limit: int = Query(20, ge=1, le=100),
):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 1. Drug Name search
            sql_name = """
            SELECT item_seq,
                   name_kor,
                   company_name,
                   rx_otc,
                   edi_code,
                   atc_code,
                   is_anticancer
            FROM drugs
            WHERE name_kor LIKE %s
            ORDER BY name_kor
            LIMIT %s
            """
            cur.execute(sql_name, (f"%{q}%", limit))
            name_rows = cur.fetchall()

            # 2. Symptom search (Optional, handle if table exists)
            symptom_rows = []
            try:
                cur.execute("SHOW TABLES LIKE 'symptoms'")
                if cur.fetchone(): # symptoms 테이블이 존재하는 경우에만 쿼리 실행
                    sql_symptom = """
                    SELECT DISTINCT d.item_seq,
                           d.name_kor,
                           d.company_name,
                           d.rx_otc,
                           d.edi_code,
                           d.atc_code,
                           d.is_anticancer
                    FROM symptoms s
                    JOIN symptom_diagnosis_weights sdw ON sdw.symptom_id = s.id
                    JOIN order_sets os ON os.diagnosis_id = sdw.diagnosis_id
                    JOIN order_set_items osi ON osi.order_set_id = os.id
                    JOIN drugs d ON d.item_seq = osi.item_seq
                    WHERE s.name LIKE %s
                    LIMIT %s
                    """
                    cur.execute(sql_symptom, (f"%{q}%", limit))
                    symptom_rows = cur.fetchall()
                else:
                    print("⚠️ 'symptoms' 테이블이 존재하지 않아 증상 검색을 건너뜁니다.")
            except Exception as e:
                if hasattr(e, 'args') and len(e.args) > 0 and e.args[0] == 1146: # Table doesn't exist
                    print(f"⚠️ 'symptoms' 테이블이 존재하지 않아 증상 검색을 건너뜁니다: {e}")
                else:
                    raise # 다른 오류는 다시 발생

            # 3. Diagnosis Name search (e.g., "유방암") (Optional, handle if table exists)
            diagnosis_rows = []
            try:
                cur.execute("SHOW TABLES LIKE 'diagnoses'")
                if cur.fetchone(): # diagnoses 테이블이 존재하는 경우에만 쿼리 실행
                    sql_diagnosis = """
                    SELECT DISTINCT d.item_seq,
                           d.name_kor,
                           d.company_name,
                           d.rx_otc,
                           d.edi_code,
                           d.atc_code,
                           d.is_anticancer
                    FROM diagnoses diag
                    JOIN order_sets os ON os.diagnosis_id = diag.id
                    JOIN order_set_items osi ON osi.order_set_id = os.id
                    JOIN drugs d ON d.item_seq = osi.item_seq
                    WHERE diag.name LIKE %s
                    LIMIT %s
                    """
                    cur.execute(sql_diagnosis, (f"%{q}%", limit))
                    diagnosis_rows = cur.fetchall()
                else:
                    print("⚠️ 'diagnoses' 테이블이 존재하지 않아 진단명 검색을 건너뜁니다.")
            except Exception as e:
                if hasattr(e, 'args') and len(e.args) > 0 and e.args[0] == 1146: # Table doesn't exist
                    print(f"⚠️ 'diagnoses' 테이블이 존재하지 않아 진단명 검색을 건너뜁니다: {e}")
                else:
                    raise # 다른 오류는 다시 발생

            # 4. Order Set Name search (e.g., "맘모그래피" associated order sets) (Optional, handle if table exists)
            orderset_rows = []
            try:
                cur.execute("SHOW TABLES LIKE 'order_sets'")
                if cur.fetchone(): # order_sets 테이블이 존재하는 경우에만 쿼리 실행
                    sql_orderset = """
                    SELECT DISTINCT d.item_seq,
                           d.name_kor,
                           d.company_name,
                           d.rx_otc,
                           d.edi_code,
                           d.atc_code,
                           d.is_anticancer
                    FROM order_sets os
                    JOIN order_set_items osi ON osi.order_set_id = os.id
                    JOIN drugs d ON d.item_seq = osi.item_seq
                    WHERE os.name LIKE %s
                    LIMIT %s
                    """
                    cur.execute(sql_orderset, (f"%{q}%", limit))
                    orderset_rows = cur.fetchall()
                else:
                    print("⚠️ 'order_sets' 테이블이 존재하지 않아 오더셋 검색을 건너뜁니다.")
            except Exception as e:
                if hasattr(e, 'args') and len(e.args) > 0 and e.args[0] == 1146: # Table doesn't exist
                    print(f"⚠️ 'order_sets' 테이블이 존재하지 않아 오더셋 검색을 건너뜁니다: {e}")
                else:
                    raise # 다른 오류는 다시 발생

            # Merge results (Dedup by item_seq)
            # Prioritize: Name > Symptom > Diagnosis > OrderSet
            combined = {row['item_seq']: row for row in name_rows}
            
            for row in symptom_rows:
                if row['item_seq'] not in combined:
                    combined[row['item_seq']] = row
            
            for row in diagnosis_rows:
                if row['item_seq'] not in combined:
                    combined[row['item_seq']] = row

            for row in orderset_rows:
                if row['item_seq'] not in combined:
                    combined[row['item_seq']] = row
            
            # Sort by name for consistency
            results = sorted(combined.values(), key=lambda x: x['name_kor'])[:limit]

    finally:
        conn.close()

    return [DrugOut(**row) for row in results]


class MfdsDurIngredientSummaryOut(BaseModel):
    dur_serial_no: int
    dur_type: str | None = None
    dur_ing_code: str | None = None
    dur_ing_eng_name: str | None = None
    dur_ing_kor_name: str | None = None
    notice_date: str | date | None = None 
    grade: str | None = None
    status: str | None = None
    class_name: str | None = None
    source_file: str | None = None


class MfdsDurIngredientSearchOut(BaseModel):
    total: int
    limit: int
    offset: int
    items: List[MfdsDurIngredientSummaryOut]

@app.get("/mfds-dur-ingredients/search", response_model=MfdsDurIngredientSearchOut)
def search_mfds_dur_ingredients(
    q: str | None = Query(None, description="Search by ingredient name or code"),
    dur_type: str | None = Query(None, description="Exact DUR type"),
    dur_ing_code: str | None = Query(None, description="Exact DUR ingredient code"),
    source_file: str | None = Query(None, description="Exact source file name"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    conditions = []
    params: list = []

    terms: list[str] = []
    if q:
        q = q.strip()
        if q:
            raw = q.replace(",", " ").replace(";", " ").replace("|", " ")
            terms = [t for t in (s.strip() for s in raw.split()) if t]
            
            # 약품명 검색 지원 & 한글 성분명 영문 변환
            conn = get_conn()
            try:
                import re
                
                additional_terms = set()
                ingredient_to_drug = {} # 성분->약품명 매핑
                
                with conn.cursor() as cur:
                    for term in terms:
                        # 1. 약품명으로 검색 (약품이 검색된 경우)
                        cur.execute("""
                            SELECT name_kor, edi_code
                            FROM drugs
                            WHERE name_kor LIKE %s
                            LIMIT 5
                        """, (f"%{term}%",))
                        found_drugs = cur.fetchall()
                        
                        for d in found_drugs:
                            # A. HIRA 매핑 확인
                            if d['edi_code']:
                                cur.execute("""
                                    SELECT gnl_nm, gnl_nm_cd 
                                    FROM hira_drug_ingredient_map 
                                    WHERE edi_code = %s
                                """, (d['edi_code'],))
                                ings = cur.fetchall()
                                
                                has_mapping = False
                                for i in ings:
                                    if i['gnl_nm']: 
                                        additional_terms.add(i['gnl_nm'])
                                        ingredient_to_drug[i['gnl_nm']] = d['name_kor'] # 매핑
                                        has_mapping = True
                                    if i['gnl_nm_cd']: 
                                        additional_terms.add(i['gnl_nm_cd'])
                                        ingredient_to_drug[i['gnl_nm_cd']] = d['name_kor'] # 매핑
                                        has_mapping = True
                            else:
                                has_mapping = False
                                    
                            # B. 매핑 없으면 괄호 안 추출 (Fallback)
                            if not has_mapping:
                                match = re.search(r'\(([^)]+)\)', d['name_kor'])
                                if match:
                                    inner = match.group(1).strip()
                                    # 정규화
                                    suffixes = ['고체분산체', '염산염', '나트륨', '칼륨', '수화물', '무수물', '타르타르산염', '말레산염', '정', '캡슐', '주사액', '주']
                                    for s in suffixes: inner = inner.replace(s, '')
                                    inner = inner.strip()
                                    if len(inner) > 1: additional_terms.add(inner)

                    # 2. 한글 성분명을 영문으로 변환 (Simvastatin 매칭용)
                    # DB 조인 대신 2-step 조회로 Collation 에러 방지
                    for cand in (set(terms) | additional_terms):
                        if re.search('[가-힣]', cand): # 한글이면
                             try:
                                 # 2-1. 해당 성분명이 포함된 약품의 EDI 찾기 (실패 확률을 낮추기 위해 여러 개 조회)
                                 cur.execute("SELECT edi_code FROM drugs WHERE name_kor LIKE %s LIMIT 10", (f"%{cand}%",))
                                 drug_rows = cur.fetchall()
                                 
                                 for drug_res in drug_rows:
                                     if not drug_res['edi_code']: continue
                                     
                                     # 2-2. EDI로 영문 성분명 조회
                                     cur.execute("""
                                        SELECT gnl_nm 
                                        FROM hira_drug_ingredient_map 
                                        WHERE edi_code = %s AND gnl_nm IS NOT NULL AND gnl_nm != ''
                                        LIMIT 1
                                     """, (drug_res['edi_code'],))
                                     hira_res = cur.fetchone()
                                     if hira_res and hira_res['gnl_nm']:
                                         additional_terms.add(hira_res['gnl_nm'])
                                         # 매핑 정보에도 추가 (결과 화면 표시용)
                                         if hira_res['gnl_nm'] not in ingredient_to_drug:
                                             ingredient_to_drug[hira_res['gnl_nm']] = cand 
                                         break # 하나 찾으면 성공으로 간주
                                         
                             except Exception as e:
                                 # 변환 실패 시 무시
                                 pass
                                 
                    terms.extend(list(additional_terms))
                    
            except Exception as e:
                print(f"Error extending search terms: {e}")
            finally:
                conn.close()




    if terms:
        like_clauses = []
        for term in list(dict.fromkeys(terms)):
            like = f"%{term}%"
            like_clauses.append(
                "(dur_ing_kor_name LIKE %s OR dur_ing_eng_name LIKE %s OR dur_ing_code LIKE %s)"
            )
            params.extend([like, like, like])
        conditions.append("(" + " OR ".join(like_clauses) + ")")
    if dur_type:
        conditions.append("dur_type = %s")
        params.append(dur_type)
    if dur_ing_code:
        conditions.append("dur_ing_code = %s")
        params.append(dur_ing_code)
    if source_file:
        conditions.append("source_file = %s")
        params.append(source_file)

    where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 그룹화된 카운트 조회
            count_sql = f"""
            SELECT COUNT(DISTINCT dur_ing_kor_name) AS cnt 
            FROM mfds_dur_ingredient_raw 
            {where_sql}
            """
            cur.execute(count_sql, params)
            total = cur.fetchone()["cnt"]

            # 그룹화된 데이터 조회
            data_sql = f"""
            SELECT
                MAX(dur_serial_no) as dur_serial_no,
                GROUP_CONCAT(DISTINCT dur_type ORDER BY dur_type SEPARATOR ', ') as dur_type,
                MAX(dur_ing_code) as dur_ing_code,
                MAX(dur_ing_eng_name) as dur_ing_eng_name,
                dur_ing_kor_name,
                MAX(notice_date) as notice_date,
                MAX(grade) as grade,
                MAX(status) as status,
                MAX(class_name) as class_name,
                MAX(source_file) as source_file
            FROM mfds_dur_ingredient_raw
            {where_sql}
            GROUP BY dur_ing_kor_name
            ORDER BY dur_ing_kor_name
            LIMIT %s OFFSET %s
            """
            cur.execute(data_sql, params + [limit, offset])
            
            # DB 결과 안전하게 딕셔너리 리스트로 변환
            # (get_conn 설정에 따라 tuple일 수도, dict일 수도 있음)
            raw_rows = cur.fetchall()
            rows = []
            if raw_rows:
                if isinstance(raw_rows[0], dict):
                    rows = list(raw_rows)
                else:
                    # Tuple -> Dict 변환
                    cols = [d[0] for d in cur.description]
                    rows = [dict(zip(cols, r)) for r in raw_rows]

            # 이미 결과에 포함된 성분 추적
            found_ing_names = set()
            for r in rows:
                if r.get('dur_ing_kor_name'): found_ing_names.add(r['dur_ing_kor_name'])
                if r.get('dur_ing_eng_name'): found_ing_names.add(r['dur_ing_eng_name'])
            
            # 검색어가 약품명인 경우, 결과에 관련 약품명 표시 및 누락된 성분 추가
            for row in rows:
                k = row.get('dur_ing_kor_name')
                e = row.get('dur_ing_eng_name')
                d_name = ingredient_to_drug.get(k) or ingredient_to_drug.get(e)
                if d_name:
                    cls = row.get('class_name') or ''
                    row['class_name'] = f"관련약품: {d_name} | {cls}" if cls else f"관련약품: {d_name}"

            # MFDS DB에는 없지만 약품 검색으로 찾아낸 성분(예: 아세트아미노펜) 추가
            # 상호작용 체크를 위해 선택은 가능해야 함
            import zlib
            batch_virtual = {}
            for ing_name, drug_name in ingredient_to_drug.items():
                if ing_name not in found_ing_names:
                    # 가상 ID 생성 (해시 기반, 충돌 가능성 낮음)
                    # 32비트 정수 범위 내에서 양수화하여 20억 더함 (DB ID와 구분)
                    fake_id = 2000000000 + (zlib.crc32(ing_name.encode('utf-8')) & 0x7fffffff)
                    
                    batch_virtual[fake_id] = ing_name

                    # 가상 행 추가
                    rows.append({
                        'dur_serial_no': fake_id, 
                        'dur_type': '일반성분',
                        'dur_ing_code': '-',
                        'dur_ing_eng_name': ing_name,
                        'dur_ing_kor_name': ing_name, 
                        'notice_date': '-',
                        'grade': '-',
                        'status': '-',
                        'class_name': f"관련약품: {drug_name} (특이사항 없음)",
                        'source_file': '-'
                    })
                    found_ing_names.add(ing_name)
            
            if batch_virtual:
                add_virtual_ingredients(batch_virtual)
    
    except Exception as e:
        print(f"Error in search_mfds_dur_ingredients: {e}")
        import traceback
        traceback.print_exc()
        
        # 디버깅을 위해 파일로 에러 기록
        try:
            with open("error_log.txt", "w", encoding="utf-8") as f:
                f.write(f"Error: {str(e)}\n")
                f.write(traceback.format_exc())
        except:
            pass
            
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        conn.close()


    return MfdsDurIngredientSearchOut(
        total=total,
        limit=limit,
        offset=offset,
        items=[MfdsDurIngredientSummaryOut(**row) for row in rows],
    )


@app.get("/mfds-dur-ingredients/{dur_serial_no}", response_model=MfdsDurIngredientDetailOut)
@app.get("/mfds-dur-ingredients/{dur_serial_no}", response_model=MfdsDurIngredientDetailOut)
def get_mfds_dur_ingredient(dur_serial_no: int):
    try:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                # 1. 먼저 요청된 시리얼 번호로 성분명 조회
                cur.execute(
                    "SELECT dur_ing_kor_name FROM mfds_dur_ingredient_raw WHERE dur_serial_no = %s",
                    (dur_serial_no,),
                )
                initial_result = cur.fetchone()
                
                if not initial_result:
                    # 가상 ID 체크 (아세트아미노펜 등)
                    vmap = get_virtual_map()
                    if dur_serial_no >= 2000000000 and dur_serial_no in vmap:
                        ing_name = vmap[dur_serial_no]
                        
                        # 가상 상세 정보 리턴
                        # created_at 등 누락된 필드가 없도록 주의
                        # (임시로 현재 시간 또는 None 넣기)
                        from datetime import datetime
                        now = datetime.now()
                        
                        return MfdsDurIngredientDetailOut(
                            dur_serial_no=dur_serial_no,
                            dur_type="일반성분 (DUR 미등재)",
                            dur_ing_kor_name=ing_name,
                            dur_ing_eng_name=ing_name,
                            dur_ing_code="-",
                            notice_date=None, # Fix: '-' -> None (Date type expected)
                            grade="-",
                            status="-",
                            class_name="-",
                            source_file="-",
                            contraindication=None,
                            note="DUR 금기 정보가 없는 일반 성분입니다.",
                            created_at=now,
                            updated_at=now
                        )
                        
                    raise HTTPException(status_code=404, detail="Ingredinet not found")
                    
                dur_ing_kor_name = initial_result['dur_ing_kor_name']
                
                # 2. 해당 성분명의 모든 정보 조회
                cur.execute(
                    """
                    SELECT *
                    FROM mfds_dur_ingredient_raw
                    WHERE dur_ing_kor_name = %s
                    ORDER BY dur_type
                    """,
                    (dur_ing_kor_name,),
                )
                rows = cur.fetchall()
                
                if not rows:
                    raise HTTPException(status_code=404, detail="Detail info not found")

                # 3. 데이터 병합 (가장 최근/대표 정보 및 리스트 정보)
                # 기본 정보는 요청된 dur_serial_no에 해당하는 행 또는 첫 번째 행 사용
                base_row = next((r for r in rows if r['dur_serial_no'] == dur_serial_no), rows[0])
                
                # 여러 유형 합치기
                dur_types = sorted(list(set(r['dur_type'] for r in rows if r['dur_type'])))
                dur_type_str = ", ".join(dur_types)
                
                # 금기 내용 등 상세 정보 합치기 (중복 제거)
                contraindications = [r['contraindication'] for r in rows if r['contraindication']]
                notes = [r['note'] for r in rows if r['note']]
                
                # 대표 필드 업데이트
                base_row['dur_type'] = dur_type_str
                base_row['contraindication'] = " | ".join(sorted(list(set(contraindications)))) if contraindications else None
                base_row['note'] = " | ".join(sorted(list(set(notes)))) if notes else None
                
                row = base_row
                
        finally:
            conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="dur_serial_no not found.")

        return MfdsDurIngredientDetailOut(**row)

    except Exception as e:
        import traceback
        # 디버깅을 위해 파일로 에러 기록
        try:
            with open("error_log_detail.txt", "w", encoding="utf-8") as f:
                f.write(f"Error in detail API: {str(e)}\n")
                f.write(traceback.format_exc())
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mfds-dur-ingredients/{dur_serial_no}/drugs", response_model=List[DrugOut])
def get_drugs_by_ingredient(dur_serial_no: int):
    """
    DUR 성분으로 해당 성분을 포함하는 약품 조회
    안정성을 위해 쿼리 분리 및 명시적 타입 변환 적용
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 1. 성분명 조회
            # 1. 성분명 조회
            cur.execute(
                "SELECT dur_ing_kor_name FROM mfds_dur_ingredient_raw WHERE dur_serial_no = %s",
                (dur_serial_no,)
            )
            ingredient = cur.fetchone()
            
            ing_name = None
            if ingredient:
                ing_name = ingredient.get('dur_ing_kor_name')
            else:
                # 가상 ID 체크 (아세트아미노펜 등)
                vmap = get_virtual_map()
                if dur_serial_no >= 2000000000 and dur_serial_no in vmap:
                    ing_name = vmap[dur_serial_no]

            if not ing_name:
                return []

            # 2. 검색 대상 EDI 코드 수집
            target_edi_codes = set()
            import re
            
            # 영문 성분명인 경우 HIRA 매핑에서 EDI 코드 조회
            if re.search('[a-zA-Z]', ing_name):
                cur.execute(
                    "SELECT edi_code FROM hira_drug_ingredient_map WHERE gnl_nm = %s OR gnl_nm LIKE %s",
                    (ing_name, f"{ing_name}%")
                )
                for row in cur.fetchall():
                    if row['edi_code']:
                        target_edi_codes.add(row['edi_code'])

            # 3. 약품 검색 쿼리 구성
            # 조건 1: 약품명에 성분명 포함
            where_clauses = ["name_kor LIKE %s"]
            params = [f"%{ing_name}%"]
            
            # 조건 2: EDI 코드가 일치하는 경우 (있을 때만)
            if target_edi_codes:
                placeholders = ', '.join(['%s'] * len(target_edi_codes))
                where_clauses.append(f"edi_code IN ({placeholders})")
                params.extend(list(target_edi_codes))
            
            # OR 로 연결
            query = f"""
                SELECT item_seq, name_kor, company_name, rx_otc, edi_code, atc_code, is_anticancer
                FROM drugs
                WHERE {' OR '.join(where_clauses)}
                ORDER BY name_kor
                LIMIT 200
            """
            
            cur.execute(query, params)
            rows = cur.fetchall()
            
            # 4. 안전한 데이터 변환 (DrugOut 모델 매핑)
            results = []
            for r in rows:
                try:
                    # null safe handling
                    is_anti = r.get('is_anticancer')
                    is_anti_bool = False
                    if is_anti in [1, '1', True, 'true', 'True']:
                        is_anti_bool = True
                        
                    dto = DrugOut(
                        item_seq=str(r['item_seq']), # 문자열 강제 변환
                        name_kor=str(r['name_kor'] or ''),
                        company_name=str(r['company_name'] or ''), # NULL 방지
                        rx_otc=str(r['rx_otc']) if r.get('rx_otc') else None,
                        edi_code=str(r['edi_code']) if r.get('edi_code') else None,
                        atc_code=str(r['atc_code']) if r.get('atc_code') else None,
                        is_anticancer=is_anti_bool
                    )
                    results.append(dto)
                except Exception as e:
                    print(f"Skipping row due to error: {e}, Row: {r}")
                    continue
                    
            return results

    except Exception as e:
        print(f"Error in get_drugs_by_ingredient: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        conn.close()




class DrugInteractionCheck(BaseModel):
    item_seqs: List[str]


class DrugInteractionWarning(BaseModel):
    item_seq_a: str
    drug_name_a: str
    item_seq_b: str
    drug_name_b: str
    interaction_type: str
    severity: str
    warning_message: str
    prohbt_content: Optional[str] = None
    ai_analysis: Optional[dict] = None


class DrugInteractionResult(BaseModel):
    checked_drugs: List[DrugOut]
    interactions: List[DrugInteractionWarning]
    has_critical: bool
    has_warnings: bool
    total_interactions: int
    total_interactions: int
    summary: str


class DrugsBySeqsRequest(BaseModel):
    item_seqs: List[str]


@app.post("/drugs/by-seqs", response_model=List[DrugOut])
def get_drugs_by_seqs(body: DrugsBySeqsRequest):
    """
    여러 약품 코드로 약품 정보 일괄 조회 (Internal/Integration용)
    """
    if not body.item_seqs:
        return []

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 1. DB 조회
            placeholders = ','.join(['%s'] * len(body.item_seqs))
            cur.execute(
                f"""
                SELECT item_seq, name_kor, company_name, 
                       rx_otc, edi_code, atc_code, is_anticancer
                FROM drugs
                WHERE item_seq IN ({placeholders})
                """,
                tuple(body.item_seqs)
            )
            rows = cur.fetchall()
            
            # 2. 결과 매핑
            results = []
            for r in rows:
                try:
                    is_anti = r.get('is_anticancer')
                    is_anti_bool = False
                    if is_anti in [1, '1', True, 'true', 'True']:
                        is_anti_bool = True
                        
                    dto = DrugOut(
                        item_seq=str(r['item_seq']),
                        name_kor=str(r['name_kor'] or ''),
                        company_name=str(r['company_name'] or ''),
                        rx_otc=str(r['rx_otc']) if r.get('rx_otc') else None,
                        edi_code=str(r['edi_code']) if r.get('edi_code') else None,
                        atc_code=str(r['atc_code']) if r.get('atc_code') else None,
                        is_anticancer=is_anti_bool
                    )
                    results.append(dto)
                except Exception:
                    continue
            return results
    finally:
        conn.close()


@app.post("/drugs/check-interactions", response_model=DrugInteractionResult)
def check_drug_interactions(body: DrugInteractionCheck):
    """
    여러 약물 간 상호작용 체크 (성분 기반)
    """
    if len(body.item_seqs) < 2:
        raise HTTPException(status_code=400, detail="최소 2개 이상의 약품이 필요합니다.")
    
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # item_seqs 분리 (실제 DB ID vs 가상 ID)
            real_seqs = []
            virtual_drugs = []
            
            for seq in body.item_seqs:
                if str(seq).startswith("VIRTUAL_") or str(seq).startswith("V_"):
                    # 가상 약물 (포맷: VIRTUAL_약물명)
                    drug_name = str(seq).replace("VIRTUAL_", "").replace("V_", "")
                    # 괄호 안 영문명 등 제거하고 순수 이름 추출 (선택사항)
                    virtual_drugs.append({
                        'item_seq': seq,
                        'name_kor': drug_name,
                        'company_name': 'Virtual',
                        'rx_otc': None,
                        'edi_code': None,
                        'atc_code': None,
                        'is_anticancer': False
                    })
                else:
                    real_seqs.append(seq)

            # 1. 실제 db 약품 정보 조회
            drugs = []
            if real_seqs:
                placeholders = ','.join(['%s'] * len(real_seqs))
                cur.execute(
                    f"""
                    SELECT item_seq, name_kor, company_name, 
                           rx_otc, edi_code, atc_code, is_anticancer
                    FROM drugs
                    WHERE item_seq IN ({placeholders})
                    """,
                    tuple(real_seqs)
                )
                drugs = cur.fetchall()
            
            # 가상 약물 합치기
            drugs.extend(virtual_drugs)
            
            drugs_dict = {str(d['item_seq']): d for d in drugs}
            
            # 2. 각 약품의 성분 조회 (HIRA 데이터 활용)
            drug_ingredients = {}
            for drug in drugs:
                ingredients = []
                
                # hira_drug_ingredient_map 테이블 존재 여부 확인
                try:
                    cur.execute("SHOW TABLES LIKE 'hira_drug_ingredient_map'")
                    has_hira_table = cur.fetchone() is not None
                except:
                    has_hira_table = False
                
                if drug['edi_code'] and has_hira_table:
                    try:
                    cur.execute(
                        """
                        SELECT DISTINCT gnl_nm_cd, gnl_nm
                        FROM hira_drug_ingredient_map
                        WHERE edi_code = %s
                        """,
                        (drug['edi_code'],)
                    )
                    ingredients = cur.fetchall()
                    except Exception as e:
                        print(f"⚠️ hira_drug_ingredient_map 조회 실패: {e}")
                        ingredients = []
                
                # HIRA 매핑이 없으면 약품명에서 성분 추출 시도
                if not ingredients:
                    drug_name = drug['name_kor'].lower()
                    # 약품명에서 일반적인 성분명 키워드 찾기
                    ingredient_name = None
                    if 'aspirin' in drug_name or '아스피린' in drug_name:
                        ingredient_name = 'aspirin'
                    elif 'ibuprofen' in drug_name or '이부프로펜' in drug_name:
                        ingredient_name = 'ibuprofen'
                    elif 'acetaminophen' in drug_name or '아세트아미노펜' in drug_name or '타이레놀' in drug_name:
                        ingredient_name = 'acetaminophen'
                    
                    if ingredient_name:
                        ingredients = [{'gnl_nm_cd': ingredient_name, 'gnl_nm': ingredient_name}]
                
                drug_ingredients[drug['item_seq']] = ingredients
            

            # 3. 성분 후보 추출 및 정규화 함수
            import re
            def get_ingredient_candidates(name: str) -> set:
                if not name: return set()
                candidates = set()
                
                # 0. 사전 조회 (정확 일치 & 부분 일치)
                # 1) 정확 일치
                dict_name = get_english_name(name)
                if dict_name: 
                    candidates.add(dict_name)
                
                # 2) 부분 일치 (사전의 키가 약물명에 포함된 경우)
                from drug_dictionary import KOREAN_TO_ENGLISH
                for k, v in KOREAN_TO_ENGLISH.items():
                    if len(k) >= 2 and k in name: # 2글자 이상만
                        candidates.add(v)
                
                # 1. 기본 정규화 (염/제형 제거)
                def clean_utils(n):
                    suffixes = [
                        '고체분산체', '염산염', '나트륨', '칼륨', '수화물', '무수물', 
                        '타르타르산염', '말레산염', '장용정', '연질캡슐', '정', '캡슐', 
                        '주사액', '주', '서방정', '이초산염', '질산염', '황산염', 
                        '붕산염', '인산염', '구연산염', '아세트산염', '염화물',
                        '베실산염', '토실산염', '메실산염', '서방캡슐', '액', '과립',
                        '시럽', '산', '염', '정제', '필름코팅정', '이수화물', '일수화물'
                    ]
                    n = n.strip()
                    # 긴 것부터 제거 (예: '고체분산체'를 '체'보다 먼저)
                    for suffix in sorted(suffixes, key=len, reverse=True):
                        n = n.replace(suffix, '')
                    return n.strip()

                # 2. 괄호 안의 내용 추출 (약품명인 경우 성분명이 괄호 안에 있을 수 있음)
                # 예: "뉴심바드정(심바스타틴)" -> "심바스타틴"
                match = re.search(r'\(([^)]+)\)', name)
                if match:
                    inner = match.group(1)
                    candidates.add(clean_utils(inner))
                    
                # 3. 괄호 제거한 이름 (성분명 뒤 옵션인 경우)
                # 예: "Aspirin (enteric)" -> "Aspirin"
                no_paren = re.sub(r'\([^)]*\)', '', name)
                candidates.add(clean_utils(no_paren))
                
                # 원본도 정규화해서 추가
                candidates.add(clean_utils(name))
                
                # 4. 한글 성분명인 경우 영문명도 찾아 추가 (DB 활용 - Simvastatin 등 영문 전용 데이터 매칭용)
                try:
                    # hira_drug_ingredient_map 테이블 존재 여부 확인
                    cur.execute("SHOW TABLES LIKE 'hira_drug_ingredient_map'")
                    has_hira_table = cur.fetchone() is not None
                    
                    if has_hira_table:
                    for cand in list(candidates):
                        if re.search('[가-힣]', cand): # 한글이 포함된 경우
                            # 해당 성분명이 포함된 약품을 찾아 영문 성분명(gnl_nm) 조회
                            # 주의: 성분명만으로 정확히 찾기 위해 LIKE 패턴 조심
                                try:
                            cur.execute("""
                                SELECT h.gnl_nm 
                                FROM hira_drug_ingredient_map h
                                JOIN drugs d ON h.edi_code = d.edi_code
                                WHERE d.name_kor LIKE %s 
                                  AND h.gnl_nm IS NOT NULL 
                                  AND h.gnl_nm != ''
                                LIMIT 1
                            """, (f"%{cand}%",))
                            row = cur.fetchone()
                            if row and row['gnl_nm']:
                                eng_name = clean_utils(row['gnl_nm'])
                                if eng_name: candidates.add(eng_name)
                                except Exception as e:
                                    # 개별 쿼리 실패해도 계속 진행
                                    pass
                except Exception as e:
                    # 영문 변환 실패해도 기존 후보로 계속 진행
                    pass

                # 5. 영문 성분명인 경우 한글 성분명도 찾아 추가 (MFDS DB 활용)
                try:
                    for cand in list(candidates):
                         # 영문만 있는 경우 (한글 미포함)
                         if not re.search('[가-힣]', cand): 
                             # MFDS RAW 테이블에서 영문명에 대응하는 한글명 조회
                             cur.execute("""
                                 SELECT dur_ing_kor_name 
                                 FROM mfds_dur_ingredient_raw
                                 WHERE dur_ing_eng_name LIKE %s 
                                   AND dur_ing_kor_name IS NOT NULL
                                 LIMIT 1
                             """, (f"%{cand}%",))
                             row = cur.fetchone()
                             if row and row['dur_ing_kor_name']:
                                 kor_name = clean_utils(row['dur_ing_kor_name'])
                                 if kor_name: candidates.add(kor_name)
                except Exception as e:
                    pass
                
                return [c for c in candidates if len(c) >= 2] # 2글자 이상만


            # 4. 약물 조합 체크 (정확도 우선 루프 방식)
            def clean_drug_name(name):
                if not name: return ""
                return re.sub(r'\((수출명|수입명):[^)]+\)', '', name).strip()

            interactions = []
            interaction_map = {} # For deduplication
            checked_pairs = set()
            
            # 모든 성분 쌍에 대해 검사
            for i, item_a in enumerate(body.item_seqs):
                for item_b in body.item_seqs[i+1:]:
                    drug_a = drugs_dict.get(item_a)
                    drug_b = drugs_dict.get(item_b)
                    
                    if not drug_a or not drug_b:
                        continue
                        
                    ings_a = drug_ingredients.get(item_a, [])
                    ings_b = drug_ingredients.get(item_b, [])
                    
                    # 성분 정보 확보 (없으면 약품명 사용)
                    raw_names_a = [i.get('gnl_nm') or i.get('gnl_nm_cd') for i in ings_a]
                    raw_names_b = [i.get('gnl_nm') or i.get('gnl_nm_cd') for i in ings_b]
                    
                    # 약품명도 항상 추가 (EDI 매핑이 불완전하거나 한글명이 누락될 수 있음)
                    raw_names_a.append(drug_a['name_kor'])
                    raw_names_b.append(drug_b['name_kor'])

                    candidates_a = set()
                    for name in raw_names_a:
                        for cand in get_ingredient_candidates(name):
                            candidates_a.add(cand.lower())
                        
                    candidates_b = set()
                    for name in raw_names_b:
                        for cand in get_ingredient_candidates(name):
                            candidates_b.add(cand.lower())

                    # 4-1. [추가] 성분 중복 처방(Duplication) 체크
                    # HIRA 성분코드 또는 정규화된 성분명 일치 여부 확인
                    codes_a = {i.get('gnl_nm_cd') for i in ings_a if i.get('gnl_nm_cd')}
                    codes_b = {i.get('gnl_nm_cd') for i in ings_b if i.get('gnl_nm_cd')}
                    
                    common_codes = codes_a.intersection(codes_b)
                    common_names = candidates_a.intersection(candidates_b)
                    
                    # 7자리 성분코드가 같으면 확실한 중복 (앞 4자리가 주성분)
                    is_duplicate = False
                    dup_cause = ""
                    
                    if common_codes:
                        is_duplicate = True
                        dup_cause = list(common_codes)[0]
                    elif common_names:
                        # 이름이 같아도 중복 가능성 높음
                        is_duplicate = True
                        dup_cause = list(common_names)[0]
                    
                    if is_duplicate:
                        interactions.append(DrugInteractionWarning(
                            item_seq_a=item_a,
                            drug_name_a=clean_drug_name(drug_a['name_kor']),
                            item_seq_b=item_b,
                            drug_name_b=clean_drug_name(drug_b['name_kor']),
                            interaction_type="Duplication",
                            severity="HIGH", # 중복 처방은 주의 필요
                            warning_message=f"동일/유사 성분 중복 처방 의심: {dup_cause}\n└ AI 분석: 두 약물 모두 '{dup_cause}' 성분을 포함하고 있거나 유사한 계열입니다.",
                            prohbt_content="동일하거나 유사한 효능을 가진 성분이 중복 처방되었습니다. 과다 복용의 위험이 있으므로 주의가 필요합니다.",
                            ai_analysis={
                                "confidence": 90,
                                "summary": f"두 약물 모두 '{dup_cause}' 성분을 포함하고 있거나 유사한 계열입니다. 중복 복용 시 부작용 위험이 증가할 수 있습니다.",
                                "mechanism": "Duplicate Ingredient",
                                "recommendation": "중복 처방 여부를 확인하고, 필요 시 하나만 복용하도록 지도하십시오."
                            }
                        ).model_dump())
                    
                    # 디버그 로그
                    print(f"\n[DEBUG] Checking pair:")
                    print(f"  Drug A: {drug_a['name_kor']}")
                    print(f"  Candidates A: {candidates_a}")
                    print(f"  Drug B: {drug_b['name_kor']}")
                    print(f"  Candidates B: {candidates_b}")
                    
                    for norm_a in candidates_a:
                        for norm_b in candidates_b:

                            
                            if len(norm_a) < 2 or len(norm_b) < 2: continue
                            
                            # 중복 검사
                            pair_key = tuple(sorted([norm_a, norm_b]))
                            if pair_key in checked_pairs: continue
                            checked_pairs.add(pair_key)
                            
                            try:
                                # LIKE 검색으로 유연하게 매칭
                                cur.execute(
                                    """
                                    SELECT 
                                        dur_ing_kor_name,
                                        coadmin_dur_ing_kor_name,
                                        dur_type,
                                        contraindication,
                                        note
                                    FROM mfds_dur_ingredient_raw
                                    WHERE coadmin_dur_ing_kor_name IS NOT NULL
                                      AND (
                                          (dur_ing_kor_name LIKE %s AND coadmin_dur_ing_kor_name LIKE %s)
                                          OR
                                          (dur_ing_kor_name LIKE %s AND coadmin_dur_ing_kor_name LIKE %s)
                                          OR
                                          (dur_ing_eng_name LIKE %s AND coadmin_dur_ing_kor_name LIKE %s)
                                      )
                                    ORDER BY 
                                        CASE dur_type
                                            WHEN '병용금기' THEN 1
                                            WHEN '용량주의' THEN 2
                                            WHEN '효능군중복' THEN 3
                                            ELSE 4
                                        END
                                    LIMIT 1
                                    """,
                                    (f"%{norm_a}%", f"%{norm_b}%", 
                                     f"%{norm_b}%", f"%{norm_a}%",
                                     f"%{norm_a}%", f"%{norm_b}%") # 영문 이름 매칭 시도
                                )
                                result = cur.fetchone()
                                
                                if result:
                                    dur_type = result.get('dur_type') or '상호작용'
                                    contraindication = result.get('contraindication') or result.get('note') or dur_type
                                    
                                    if dur_type == '병용금기': severity = "CRITICAL"
                                    elif dur_type in ['용량주의', '효능군중복']: severity = "HIGH"
                                    else: severity = "MEDIUM"
                                    
                                    # AI 분석 자동 호출
                                    try:
                                        ai_result = ai_service.analyze_interaction(
                                            drug_a_name=drug_a['name_kor'],
                                            drug_a_ingr=result['dur_ing_kor_name'],
                                            drug_b_name=drug_b['name_kor'],
                                            drug_b_ingr=result['coadmin_dur_ing_kor_name'],
                                            reason_from_db=contraindication
                                        )
                                        # 정확도 퍼센트 (severity 기반)
                                        confidence = {
                                            "CRITICAL": 95,
                                            "HIGH": 85,
                                            "MEDIUM": 75
                                        }.get(severity, 70)
                                        
                                        ai_analysis = {
                                            "confidence": confidence,
                                            "summary": ai_result.get("summary", "분석 불가"),
                                            "mechanism": ai_result.get("mechanism", ""),
                                            "recommendation": ai_result.get("clinical_recommendation", "")
                                        }
                                    except Exception as e:
                                        print(f"AI analysis failed: {e}")
                                        ai_analysis = None
                                    
                                    
                                    # [DEDUPLICATION START]
                                    # Use dictionary to store interactions by pair key
                                    # Key: frozenset({item_a, item_b})
                                    pair_key = frozenset([str(item_a), str(item_b)])
                                    
                                    existing = interaction_map.get(pair_key)
                                    if existing:
                                        # Already found (likely by another ingredient pair of same drugs)
                                        # But here we stick with the first one or logic to prioritize
                                        pass
                                    else:
                                        # Clean and Format - 조원 형식: 약물명 + AI 분석
                                        # 약물명 형식: "코다론정(아미오다론염산염) + 삼진드론정(드로네다론염산염):"
                                        drug_a_display = clean_drug_name(drug_a['name_kor'])
                                        drug_b_display = clean_drug_name(drug_b['name_kor'])
                                        
                                        # 성분명이 있으면 괄호에 표시
                                        ing_a = result.get('dur_ing_kor_name', '')
                                        ing_b = result.get('coadmin_dur_ing_kor_name', '')
                                        
                                        if ing_a and ing_a not in drug_a_display:
                                            drug_a_display = f"{drug_a_display}({ing_a})"
                                        if ing_b and ing_b not in drug_b_display:
                                            drug_b_display = f"{drug_b_display}({ing_b})"
                                        
                                        # 기본 메시지: 약물명 + 상세 설명
                                        base_msg = f"{drug_a_display} + {drug_b_display}:"
                                        
                                        # AI 분석이 있으면 AI 분석 메시지 추가
                                        if ai_analysis and ai_analysis.get('summary'):
                                            base_msg = f"{base_msg}\nAI 분석: {ai_analysis['summary']}"
                                        else:
                                            # AI 분석이 없으면 DB의 contraindication 사용
                                            if contraindication:
                                                # contraindication에서 핵심 정보만 추출
                                                contra_clean = contraindication.split('|')[0].strip() if '|' in contraindication else contraindication[:200]
                                                base_msg = f"{base_msg} {contra_clean}"
                                            else:
                                                base_msg = f"{base_msg} {dur_type} (DUR 경고)"

                                        interaction_map[pair_key] = {
                                            "item_seq_a": item_a,
                                            "drug_name_a": clean_drug_name(drug_a['name_kor']),
                                            "item_seq_b": item_b,
                                            "drug_name_b": clean_drug_name(drug_b['name_kor']),
                                            "interaction_type": dur_type,
                                            "severity": severity,
                                            "warning_message": base_msg,
                                            "prohbt_content": contraindication[:500] if contraindication else None,
                                            "ai_analysis": ai_analysis
                                        }
                                        interactions.append(interaction_map[pair_key]) # Keep ref in list order

                            except Exception as e:
                                print(f"Interaction check error: {e}")
                                continue


            # 4-2. DDInter 국제 DB 추가 검색 (174K+ interactions)
            print(f"\n[DEBUG] Searching DDInter international database...")
            for i, item_a in enumerate(body.item_seqs):
                for item_b in body.item_seqs[i+1:]:
                    drug_a = drugs_dict.get(item_a)
                    drug_b = drugs_dict.get(item_b)
                    
                    if not drug_a or not drug_b:
                        continue
                    
                    ddinter_result = check_ddinter_interactions(
                        cursor=cur,
                        drug_a=drug_a,
                        drug_b=drug_b, 
                        item_a=item_a,
                        item_b=item_b,
                        ai_service=ai_service
                    )
                    
                    if ddinter_result:
                        # Clean names
                        ddinter_result['drug_name_a'] = clean_drug_name(ddinter_result['drug_name_a'])
                        ddinter_result['drug_name_b'] = clean_drug_name(ddinter_result['drug_name_b'])
                        
                        # AI Summary extraction
                        ai_sum = ddinter_result.get('ai_analysis', {}).get('summary') if ddinter_result.get('ai_analysis') else None
                        
                        # Get RAW DB message (strip existing AI text if any, just in case)
                        raw_dd_msg = ddinter_result['warning_message']
                        if "└ AI 분석" in raw_dd_msg:
                            raw_dd_msg = raw_dd_msg.split("└ AI 분석")[0].strip()

                        # [DEDUPLICATION MERGE]
                        # Check if this pair already exists (from Korean DB)
                        pair_key = frozenset([str(item_a), str(item_b)])
                        existing = interaction_map.get(pair_key)
                        
                        if existing:
                            # 1. Clean existing message (separate DB vs AI)
                            existing_raw = existing['warning_message']
                            existing_ai = None
                            if "└ AI 분석" in existing_raw:
                                parts = existing_raw.split("└ AI 분석")
                                existing_raw = parts[0].strip()
                                existing_ai = parts[1].strip().lstrip(":").strip() # Handle ": " 
                            
                            # 2. Combine DB messages
                            # 2. Combine DB messages (Default)
                            new_msg = f"{existing_raw}\n[국제 DB 확인] {raw_dd_msg}"
                            
                            # 3. Use AI Message if available (Overwrite)
                            final_ai = existing_ai if existing_ai else ai_sum
                            if existing.get('ai_analysis'):
                                final_ai = existing['ai_analysis'].get('summary') or final_ai
                            
                            if final_ai:
                                new_msg = f"└ AI 분석: {final_ai}"
                            
                            existing['warning_message'] = new_msg
                            # Update ai_analysis object if needed
                            if not existing.get('ai_analysis') and ddinter_result.get('ai_analysis'):
                                existing['ai_analysis'] = ddinter_result['ai_analysis']

                        else:
                            # New interaction found only in DDInter
                            # Reconstruct clean message
                            clean_msg = raw_dd_msg
                            if ai_sum:
                                clean_msg = f"└ AI 분석: {ai_sum}"
                            
                            ddinter_result['warning_message'] = clean_msg
                            interaction_map[pair_key] = ddinter_result
                            interactions.append(ddinter_result)

            # 4-3. AI Safety Check (DB에서 상호작용 미발견 시 최후의 보루 - 딥러닝 분석)
            if not interactions and len(drugs) >= 2:
                print("ℹ️ DB Check SAFE -> Running AI Deep Analysis...")
                
                # AI 분석 요청
                ai_findings = ai_service.detect_interactions(drugs) # drugs: list of dict
                
                for finding in ai_findings:
                    # AI가 찾은 'drug_a', 'drug_b' 이름을 실제 seq와 매칭
                    item_a = "UNKNOWN"
                    item_b = "UNKNOWN"
                    name_a = finding.get('drug_a', 'Unknown')
                    name_b = finding.get('drug_b', 'Unknown')
                    
                    # 가장 유사한 약물 찾기 (단순 문자열 포함 여부)
                    for seq, d in drugs_dict.items():
                        d_name = d['name_kor']
                        # AI가 반환한 이름이 우리 DB 이름의 일부이거나 그 반대인 경우
                        if (name_a in d_name) or (d_name in name_a):
                            item_a = seq
                            name_a = d_name # 정확한 이름으로 교체
                        elif (name_b in d_name) or (d_name in name_b):
                            item_b = seq
                            name_b = d_name

                    # Warning 객체 생성
                    sev = finding.get('severity', 'HIGH')
                    if sev == 'SAFE': continue

                    interactions.append(DrugInteractionWarning(
                        item_seq_a=str(item_a),
                        drug_name_a=str(name_a),
                        item_seq_b=str(item_b),
                        drug_name_b=str(name_b),
                        interaction_type=f"AI-{finding.get('type', 'Risk')}",
                        severity=sev,
                        warning_message=finding.get('summary', 'Detected by AI analysis'),
                        prohbt_content="AI 분석 결과 잠재적인 상호작용 위험이 감지되었습니다.",
                        ai_analysis={
                            "confidence": 75, # DB 근거가 없으므로 신뢰도는 약간 낮게 설정
                            "summary": finding.get('summary'),
                            "mechanism": "AI-detected Interaction (No DB Record)",
                            "recommendation": finding.get('recommendation', '전문가 상의 필요')
                        }
                    ).model_dump())

            # 5. 결과 중복 제거 및 요약
            unique_interactions = []
            seen_keys = set()
            
            for i in interactions:
                # Key: 약품A-약품B-유형-내용 (A,B 순서 무관하게 정렬)
                ab_seqs = sorted([i['item_seq_a'], i['item_seq_b']])
                key = (ab_seqs[0], ab_seqs[1], i['interaction_type'], i['warning_message'])
                
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_interactions.append(i)
            
            interactions = unique_interactions

            has_critical = any(i['severity'] == 'CRITICAL' for i in interactions)
            has_warnings = len(interactions) > 0
            
            if has_critical:
                summary = f"⚠️ 치명적인 병용금기 {len(interactions)}건 발견!"
            elif has_warnings:
                summary = f"⚠️ 약물 상호작용 {len(interactions)}건 발견"
            else:
                summary = "✅ 검사한 약물 간 상호작용이 발견되지 않았습니다."
    
    finally:
        conn.close()
    
    return DrugInteractionResult(
        checked_drugs=[DrugOut(**d) for d in drugs],
        interactions=[DrugInteractionWarning(**i) for i in interactions],
        has_critical=has_critical,
        has_warnings=has_warnings,
        total_interactions=len(interactions),
        summary=summary
    )






@app.post("/prescriptions")
def create_prescription(body: PrescriptionCreateIn):
    if not body.items:
        raise HTTPException(status_code=400, detail="items가 비어있습니다.")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prescriptions (patient_id, patient_age, patient_sex, is_pregnant)
                VALUES (%s, %s, %s, %s)
                """,
                (body.patient_id, body.patient_age, body.patient_sex, body.is_pregnant),
            )
            presc_id = cur.lastrowid

            for item in body.items:
                cur.execute(
                    """
                    INSERT INTO prescription_items
                        (prescription_id, item_seq, dose_amount, dose_frequency, dose_duration, memo)
                    VALUES
                        (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        presc_id,
                        item.item_seq,
                        item.dose_amount,
                        item.dose_frequency,
                        item.dose_duration,
                        item.memo,
                    ),
                )

        return {"prescription_id": presc_id, "item_count": len(body.items)}
    finally:
        conn.close()


@app.post("/prescriptions/validate", response_model=PrescriptionValidateOut)
def validate_prescription(body: PrescriptionValidateIn):
    resolved_items, resolve_meta = resolve_items_for_validation(body)

    result = validate_prescription_items(
        items=resolved_items,
        patient_age=body.patient_age,
        patient_sex=body.patient_sex,
        is_pregnant=body.is_pregnant,
    )

    log_audit_event(
        event_type="prescription_validated",
        payload={
            "request": body.model_dump(),
            "resolve_meta": resolve_meta,
            "resolved_items": _jsonable(resolved_items),
            "result": _jsonable(result),
        },
        patient_id=body.patient_id,
        recommendation_session_id=body.recommendation_session_id,
        selected_candidate_id=body.selected_candidate_id,
    )

    if resolve_meta.get("items_overwritten"):
        log_audit_event(
            event_type="prescription_items_overwritten",
            payload={
                "request": body.model_dump(),
                "resolve_meta": resolve_meta,
            },
            patient_id=body.patient_id,
            recommendation_session_id=body.recommendation_session_id,
            selected_candidate_id=body.selected_candidate_id,
        )

    if result["contraindications"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "금기(Contraindication) 경고가 포함되어 있습니다.",
                "is_valid": False,
                "resolve_meta": resolve_meta,
                "resolved_item_seqs": _item_seqs(resolved_items),
                **_jsonable(result),
            },
        )

    return PrescriptionValidateOut(
        is_valid=True,
        **result,
        resolve_meta=resolve_meta,
        resolved_item_seqs=_item_seqs(resolved_items),
    )


@app.post("/prescriptions/commit", response_model=PrescriptionCommitOut)
def commit_prescription(body: PrescriptionValidateIn):
    resolved_items, resolve_meta = resolve_items_for_validation(body)

    result = validate_prescription_items(
        items=resolved_items,
        patient_age=body.patient_age,
        patient_sex=body.patient_sex,
        is_pregnant=body.is_pregnant,
    )

    if result["contraindications"]:
        log_audit_event(
            event_type="prescription_commit_rejected",
            payload={
                "request": body.model_dump(),
                "resolve_meta": resolve_meta,
                "resolved_items": _jsonable(resolved_items),
                "result": _jsonable(result),
            },
            patient_id=body.patient_id,
            recommendation_session_id=body.recommendation_session_id,
            selected_candidate_id=body.selected_candidate_id,
        )
        raise HTTPException(
            status_code=400,
            detail={
                "message": "금기(Contraindication) 경고가 포함되어 저장할 수 없습니다.",
                "is_valid": False,
                "resolve_meta": resolve_meta,
                "resolved_item_seqs": _item_seqs(resolved_items),
                **_jsonable(result),
            },
        )

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prescriptions (patient_id, patient_age, patient_sex, is_pregnant)
                VALUES (%s, %s, %s, %s)
                """,
                (body.patient_id, body.patient_age, body.patient_sex, body.is_pregnant),
            )
            presc_id = cur.lastrowid

            for item in resolved_items:
                cur.execute(
                    """
                    INSERT INTO prescription_items
                        (prescription_id, item_seq, dose_amount, dose_frequency, dose_duration, memo)
                    VALUES
                        (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        presc_id,
                        item.item_seq,
                        item.dose_amount,
                        item.dose_frequency,
                        item.dose_duration,
                        item.memo,
                    ),
                )
    finally:
        conn.close()

    log_audit_event(
        event_type="prescription_committed",
        payload={
            "request": body.model_dump(),
            "resolve_meta": resolve_meta,
            "resolved_items": _jsonable(resolved_items),
            "result": _jsonable(result),
        },
        patient_id=body.patient_id,
        prescription_id=presc_id,
        recommendation_session_id=body.recommendation_session_id,
        selected_candidate_id=body.selected_candidate_id,
    )

    if resolve_meta.get("items_overwritten"):
        log_audit_event(
            event_type="prescription_items_overwritten_on_commit",
            payload={
                "prescription_id": presc_id,
                "resolve_meta": resolve_meta,
            },
            patient_id=body.patient_id,
            prescription_id=presc_id,
            recommendation_session_id=body.recommendation_session_id,
            selected_candidate_id=body.selected_candidate_id,
        )

    return PrescriptionCommitOut(
        prescription_id=presc_id,
        item_count=len(resolved_items),
        dur_warnings=result["dur_warnings"],
        interaction_warnings=result["interaction_warnings"],
        risk_score=result["risk_score"],
        estimated_risk_level=result["estimated_risk_level"],
        risk_breakdown=result["risk_breakdown"],
        resolve_meta=resolve_meta,
        resolved_item_seqs=_item_seqs(resolved_items),
    )


@app.get("/dur/check", response_model=list[DurWarningOut])
def dur_check(prescription_id: int):
    return get_dur_warnings_for_prescription(prescription_id)


@app.post("/prescriptions/with-dur", response_model=PrescriptionWithDurOut)
def create_prescription_with_dur(body: PrescriptionCreateIn):
    if not body.items:
        raise HTTPException(status_code=400, detail="items가 비어있습니다.")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prescriptions (patient_id, patient_age, patient_sex, is_pregnant)
                VALUES (%s, %s, %s, %s)
                """,
                (body.patient_id, body.patient_age, body.patient_sex, body.is_pregnant),
            )
            presc_id = cur.lastrowid

            for item in body.items:
                cur.execute(
                    """
                    INSERT INTO prescription_items
                        (prescription_id, item_seq, dose_amount, dose_frequency, dose_duration, memo)
                    VALUES
                        (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        presc_id,
                        item.item_seq,
                        item.dose_amount,
                        item.dose_frequency,
                        item.dose_duration,
                        item.memo,
                    ),
                )
    finally:
        conn.close()

    warnings = get_dur_warnings_for_prescription(presc_id)
    return PrescriptionWithDurOut(prescription_id=presc_id, item_count=len(body.items), warnings=warnings)


@app.post("/prescriptions/from-order-set", response_model=PrescriptionWithDurOut)
def create_prescription_from_order_set(body: ApplyOrderSetIn):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM order_sets WHERE id=%s", (body.order_set_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail="해당 order_set_id를 찾을 수 없습니다.")
    finally:
        conn.close()

    presc_id = apply_order_set_to_prescription(
        order_set_id=body.order_set_id,
        patient_id=body.patient_id,
        patient_age=body.patient_age,
        patient_sex=body.patient_sex,
        is_pregnant=body.is_pregnant,
    )
    warnings = get_dur_warnings_for_prescription(presc_id)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM prescription_items WHERE prescription_id=%s", (presc_id,))
            cnt = cur.fetchone()["cnt"]
    finally:
        conn.close()

    return PrescriptionWithDurOut(prescription_id=presc_id, item_count=cnt, warnings=warnings)


@app.post("/prescriptions/apply-order-set", response_model=ApplyOrderSetOut)
def apply_order_set(body: ApplyOrderSetIn):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    os.id AS order_set_id,
                    os.name AS order_set_name,
                    d.code AS diagnosis_code,
                    d.name AS diagnosis_name
                FROM order_sets os
                JOIN diagnoses d ON d.id = os.diagnosis_id
                WHERE os.id = %s
                """,
                (body.order_set_id,),
            )
            os_info = cur.fetchone()
            if not os_info:
                raise HTTPException(status_code=404, detail="해당 order_set_id를 찾을 수 없습니다.")
    finally:
        conn.close()

    presc_id = apply_order_set_to_prescription(
        order_set_id=body.order_set_id,
        patient_id=body.patient_id,
        patient_age=body.patient_age,
        patient_sex=body.patient_sex,
        is_pregnant=body.is_pregnant,
    )

    warnings = get_dur_warnings_for_prescription(presc_id)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM prescription_items WHERE prescription_id=%s", (presc_id,))
            item_count = int(cur.fetchone()["cnt"])
    finally:
        conn.close()

    return ApplyOrderSetOut(
        prescription_id=presc_id,
        order_set_id=int(os_info["order_set_id"]),
        order_set_name=os_info["order_set_name"],
        diagnosis_code=os_info.get("diagnosis_code"),
        diagnosis_name=os_info.get("diagnosis_name"),
        item_count=item_count,
        warnings=warnings,
    )


@app.post("/prescriptions/auto", response_model=AutoPrescribeOut)
def auto_prescribe(body: AutoPrescribeRequest):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, code, name FROM diagnoses WHERE code=%s", (body.diagnosis_code,))
            diag = cur.fetchone()
            if not diag:
                raise HTTPException(status_code=404, detail="해당 진단 코드가 없습니다.")

            cur.execute(
                """
                SELECT id, name, description, min_age, max_age, allow_pregnant
                FROM order_sets
                WHERE diagnosis_id=%s
                """,
                (diag["id"],),
            )
            os_rows = cur.fetchall()
            if not os_rows:
                raise HTTPException(status_code=400, detail="이 진단에 대한 오더셋이 없습니다.")

            patient_age = body.patient_age
            is_preg = bool(body.is_pregnant) if body.is_pregnant is not None else False

            candidates = []
            for os_row in os_rows:
                os_id = os_row["id"]

                cur.execute(
                    """
                    SELECT item_seq, default_dose, default_freq, default_days
                    FROM order_set_items
                    WHERE order_set_id=%s
                    """,
                    (os_id,),
                )
                items = cur.fetchall()
                if not items:
                    continue

                score = 0
                reasons: list[str] = []

                min_age = os_row.get("min_age")
                max_age = os_row.get("max_age")
                allow_pregnant = os_row.get("allow_pregnant")

                if patient_age is not None:
                    if min_age is not None and patient_age < min_age:
                        continue
                    if max_age is not None and patient_age > max_age:
                        continue
                    if min_age is not None and patient_age >= min_age:
                        score += 1
                        reasons.append(f"{min_age}세 이상 조건 충족")
                    if max_age is not None and patient_age <= max_age:
                        score += 1
                        reasons.append(f"{max_age}세 이하 조건 충족")

                if allow_pregnant is not None:
                    if is_preg and allow_pregnant == 0:
                        continue
                    if not is_preg and allow_pregnant == 0:
                        score += 1
                        reasons.append("비임부 권장")
                    if allow_pregnant == 1:
                        score += 1
                        reasons.append("임부 사용 가능")

                item_seqs = [it["item_seq"] for it in items]
                warnings = get_dur_warnings_for_items(item_seqs, is_preg)

                dur_risk = 0
                for w in warnings:
                    if w.severity == "HIGH":
                        dur_risk += 10
                    elif w.severity == "MEDIUM":
                        dur_risk += 5
                    else:
                        dur_risk += 1

                ing_profile = compute_ingredient_risk_for_items(item_seqs)
                dup_risk = int(ing_profile.get("duplicate_ingredient_count", 0)) * 5
                total_risk = dur_risk + int(ing_profile.get("risk_score", 0)) + dup_risk

                candidates.append(
                    {
                        "os_row": os_row,
                        "items": items,
                        "warnings": warnings,
                        "match_score": score,
                        "match_reason": "; ".join(reasons) if reasons else None,
                        "risk_score": total_risk,
                    }
                )
    finally:
        conn.close()

    if not candidates:
        raise HTTPException(status_code=400, detail="환자 조건을 만족하는 오더셋 후보가 없습니다.")

    candidates.sort(key=lambda c: (c["match_score"] * 10 - c["risk_score"]), reverse=True)

    best = candidates[0]
    best_os = best["os_row"]
    best_items = best["items"]

    alternatives: list[RecommendedOrderSet] = []
    for alt in candidates[1:4]:
        alt_os = alt["os_row"]
        alt_items = [
            PrescriptionItemIn(
                item_seq=it["item_seq"],
                dose_amount=it["default_dose"],
                dose_frequency=it["default_freq"],
                dose_duration=it["default_days"],
                memo=None,
            )
            for it in alt["items"]
        ]
        alternatives.append(
            RecommendedOrderSet(
                order_set_id=alt_os["id"],
                name=alt_os["name"],
                description=alt_os.get("description"),
                items=alt_items,
                warnings=alt["warnings"],
                match_score=alt["match_score"],
                match_reason=alt["match_reason"],
            )
        )

    presc_id = apply_order_set_to_prescription(
        order_set_id=best_os["id"],
        patient_id=body.patient_id,
        patient_age=body.patient_age,
        patient_sex=body.patient_sex,
        is_pregnant=body.is_pregnant,
    )
    final_warnings = get_dur_warnings_for_prescription(presc_id)

    return AutoPrescribeOut(
        prescription_id=presc_id,
        order_set_id=best_os["id"],
        order_set_name=best_os["name"],
        item_count=len(best_items),
        match_score=best["match_score"],
        risk_score=best["risk_score"],
        warnings=final_warnings,
        alternatives=alternatives,
    )


@app.post("/prescriptions/auto-by-symptoms", response_model=AutoBySymptomsOut)
def auto_prescribe_by_symptoms(body: AutoBySymptomsRequest):
    symptom_names = normalize_symptoms(body.symptoms)
    if not symptom_names:
        raise HTTPException(status_code=400, detail="symptoms가 비어있습니다.")

    diags = recommend_diagnoses_by_symptoms(symptom_names, topk=body.topk_diagnosis)
    if not diags:
        raise HTTPException(status_code=404, detail="증상과 매칭되는 진단 후보가 없습니다.")

    diag_out = [DiagnosisCandidateOut(code=d["code"], name=d["name"], score=float(d["score"])) for d in diags]

    last_err = None
    for d in diags:
        try:
            req = AutoPrescribeRequest(
                diagnosis_code=d["code"],
                patient_id=body.patient_id,
                patient_age=body.patient_age,
                patient_sex=body.patient_sex,
                is_pregnant=body.is_pregnant,
            )
            result = auto_prescribe(req)
            return AutoBySymptomsOut(
                result=result,
                symptoms_normalized=symptom_names,
                diagnosis_candidates=diag_out,
                selected_diagnosis_code=d["code"],
                selected_diagnosis_name=d["name"],
            )
        except HTTPException as e:
            last_err = e
            continue

    raise HTTPException(
        status_code=400,
        detail={
            "message": "TOPK 진단들로 자동처방을 시도했지만 가능한 오더셋이 없습니다.",
            "symptoms_normalized": symptom_names,
            "diagnosis_candidates": [dict(code=d["code"], name=d["name"], score=float(d["score"])) for d in diags],
            "last_error": getattr(last_err, "detail", None),
        },
    )


@app.post("/recommend/order-sets", response_model=list[RecommendedOrderSet])
def recommend_order_sets(body: RecommendRequest):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, code, name FROM diagnoses WHERE code=%s", (body.diagnosis_code,))
            diag = cur.fetchone()
            if not diag:
                return []

            cur.execute(
                """
                SELECT id, name, description, min_age, max_age, allow_pregnant
                FROM order_sets
                WHERE diagnosis_id=%s
                """,
                (diag["id"],),
            )
            os_rows = cur.fetchall()
            if not os_rows:
                return []

            is_preg = bool(body.is_pregnant) if body.is_pregnant is not None else False
            results: list[RecommendedOrderSet] = []

            for os_row in os_rows:
                os_id = os_row["id"]

                cur.execute(
                    """
                    SELECT item_seq, default_dose, default_freq, default_days
                    FROM order_set_items
                    WHERE order_set_id=%s
                    """,
                    (os_id,),
                )
                items = cur.fetchall()
                if not items:
                    continue

                score = 0
                reasons: list[str] = []

                min_age = os_row.get("min_age")
                max_age = os_row.get("max_age")
                allow_pregnant = os_row.get("allow_pregnant")
                patient_age = body.patient_age

                if patient_age is not None:
                    if min_age is not None and patient_age < min_age:
                        continue
                    if max_age is not None and patient_age > max_age:
                        continue
                    if min_age is not None and patient_age >= min_age:
                        score += 1
                        reasons.append(f"{min_age}세 이상 조건 충족")
                    if max_age is not None and patient_age <= max_age:
                        score += 1
                        reasons.append(f"{max_age}세 이하 조건 충족")

                if allow_pregnant is not None:
                    if is_preg and allow_pregnant == 0:
                        continue
                    if not is_preg and allow_pregnant == 0:
                        score += 1
                        reasons.append("비임부 권장")
                    if allow_pregnant == 1:
                        score += 1
                        reasons.append("임부 사용 가능")

                item_seqs = [it["item_seq"] for it in items]
                warnings = get_dur_warnings_for_items(item_seqs, is_preg)

                rec_items = [
                    PrescriptionItemIn(
                        item_seq=it["item_seq"],
                        dose_amount=it["default_dose"],
                        dose_frequency=it["default_freq"],
                        dose_duration=it["default_days"],
                        memo=None,
                    )
                    for it in items
                ]

                results.append(
                    RecommendedOrderSet(
                        order_set_id=os_id,
                        name=os_row["name"],
                        description=os_row.get("description"),
                        items=rec_items,
                        warnings=warnings,
                        match_score=score,
                        match_reason="; ".join(reasons) if reasons else None,
                    )
                )

        results.sort(key=lambda r: r.match_score, reverse=True)
        return results
    finally:
        conn.close()


@app.post("/recommend/order-sets-by-symptoms", response_model=RecommendBySymptomsOut)
def recommend_order_sets_by_symptoms(body: RecommendBySymptomsRequest):
    symptom_names = normalize_symptoms(body.symptoms)
    if not symptom_names:
        raise HTTPException(status_code=400, detail="symptoms가 비어있습니다.")

    diags = recommend_diagnoses_by_symptoms(symptom_names, topk=body.topk_diagnosis)
    if not diags:
        raise HTTPException(status_code=404, detail="증상과 매칭되는 진단 후보가 없습니다.")

    is_preg = bool(body.is_pregnant) if body.is_pregnant is not None else False

    diagnosis_candidates = [DiagnosisCandidateOut(code=d["code"], name=d["name"], score=float(d["score"])) for d in diags]

    if body.ranking_scope == "top_diagnosis_only":
        diag_list_for_order_sets = [diags[0]]
    else:
        diag_list_for_order_sets = diags

    pooled = []
    for d in diag_list_for_order_sets:
        pooled.extend(
            build_order_set_candidates_for_diagnosis(
                diagnosis_code=d["code"],
                patient_age=body.patient_age,
                is_preg=is_preg,
                topk=body.topk_order_sets_per_diag,
            )
        )

    if not pooled:
        raise HTTPException(status_code=400, detail="추천 가능한 오더셋 후보가 없습니다. (order_sets/items 확인)")

    pooled.sort(key=lambda x: (-x["composite_score"], x["risk_score"], -x["match_score"], x["order_set_id"]))

    for i, x in enumerate(pooled, start=1):
        x["recommendation_rank"] = i
        x["recommendation_label"] = "TOP1" if i == 1 else ("TOP2" if i == 2 else None)

    primary_tags = symptom_names[:3]
    for x in pooled:
        x["primary_symptom_tags"] = primary_tags

    for x in pooled:
        x["candidate_id"] = str(uuid.uuid4())

    session_id = create_recommendation_session(
        request_payload={
            "symptoms_normalized": symptom_names,
            "patient_age": body.patient_age,
            "patient_sex": body.patient_sex,
            "is_pregnant": body.is_pregnant,
            "topk_diagnosis": body.topk_diagnosis,
            "topk_order_sets_per_diag": body.topk_order_sets_per_diag,
            "ranking_scope": body.ranking_scope,
        },
        candidates_payload=pooled,
    )
    if not session_id:
        for x in pooled:
            x["candidate_id"] = make_candidate_token(x["order_set_id"], x["diagnosis_code"])

    log_audit_event(
        event_type="recommendation_candidates_generated",
        payload={
            "request": body.model_dump(),
            "symptoms_normalized": symptom_names,
            "candidate_count": len(pooled),
            "recommendation_session_id": session_id,
            "candidates_snapshot": pooled,
        },
        recommendation_session_id=session_id,
    )

    return RecommendBySymptomsOut(
        symptoms_normalized=symptom_names,
        diagnosis_candidates=diagnosis_candidates,
        order_set_candidates=[OrderSetCandidateOut(**x) for x in pooled],
        recommendation_session_id=session_id,
    )


@app.get("/drugs/{item_seq}/safety-summary", response_model=DrugSafetySummaryOut)
def get_drug_safety_summary(item_seq: str):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT item_seq, name_kor, company_name, edi_code, atc_code, is_anticancer
                FROM drugs
                WHERE item_seq=%s
                """,
                (item_seq,),
            )
            drug = cur.fetchone()
    finally:
        conn.close()

    if not drug:
        raise HTTPException(status_code=404, detail="해당 item_seq 약을 찾을 수 없습니다.")

    dur_warnings = get_dur_warnings_for_items([item_seq])
    ingredient_effects = get_ingredient_effects_for_item(item_seq)

    return DrugSafetySummaryOut(
        item_seq=drug["item_seq"],
        name_kor=drug["name_kor"],
        company_name=drug.get("company_name"),
        edi_code=drug.get("edi_code"),
        atc_code=drug.get("atc_code"),
        is_anticancer=bool(drug.get("is_anticancer")),
        dur_warnings=dur_warnings,
        ingredient_effects=ingredient_effects,
    )


@app.get("/prescriptions/{prescription_id}", response_model=PrescriptionDetailOut)
def get_prescription_detail(
    prescription_id: int,
    include_warnings: bool = Query(True, description="true면 DUR warnings 포함해서 반환"),
):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, patient_id, patient_age, patient_sex, is_pregnant
                FROM prescriptions
                WHERE id = %s
                """,
                (prescription_id,),
            )
            presc = cur.fetchone()
            if not presc:
                raise HTTPException(status_code=404, detail="해당 prescription_id를 찾을 수 없습니다.")

            cur.execute(
                """
                SELECT
                    pi.item_seq,
                    d.name_kor AS drug_name,
                    pi.dose_amount,
                    pi.dose_frequency,
                    pi.dose_duration,
                    pi.memo
                FROM prescription_items pi
                LEFT JOIN drugs d
                  ON d.item_seq COLLATE utf8mb4_0900_ai_ci
                   = pi.item_seq COLLATE utf8mb4_0900_ai_ci
                WHERE pi.prescription_id = %s
                ORDER BY pi.id ASC
                """,
                (prescription_id,),
            )
            items = cur.fetchall()
    finally:
        conn.close()

    warnings = get_dur_warnings_for_prescription(prescription_id) if include_warnings else []

    return PrescriptionDetailOut(
        prescription_id=presc["id"],
        patient_id=presc.get("patient_id"),
        patient_age=presc.get("patient_age"),
        patient_sex=presc.get("patient_sex"),
        is_pregnant=bool(presc.get("is_pregnant")) if presc.get("is_pregnant") is not None else None,
        item_count=len(items),
        items=[PrescriptionItemOut(**it) for it in items],
        warnings=warnings,
    )


@app.post("/prescriptions/auto-from-recommendation", response_model=AutoFromRecommendationOut)
def auto_from_recommendation(body: AutoFromRecommendationRequest):
    selected_rank = 0
    selected_candidate_id = body.selected_candidate_id
    recommendation_session_id = body.recommendation_session_id

    if recommendation_session_id and selected_candidate_id:
        session = get_recommendation_session(recommendation_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="recommendation_session_id not found.")

        selected = None
        for idx, cand in enumerate(session.get("candidates") or [], start=1):
            if cand.get("candidate_id") == selected_candidate_id:
                selected = cand
                selected_rank = idx
                break
        if not selected:
            raise HTTPException(status_code=400, detail="selected_candidate_id not in recommendation_session.")

        selected_order_set_id = int(selected["order_set_id"])
        order_set_name = selected.get("name") or ""
        diagnosis_code = selected.get("diagnosis_code") or ""
        diagnosis_name = selected.get("diagnosis_name") or ""

    elif selected_candidate_id:
        parsed = parse_candidate_token(selected_candidate_id)
        if not parsed:
            raise HTTPException(status_code=400, detail="selected_candidate_id is invalid.")

        token_order_set_id, token_diag_code = parsed
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        os.id AS order_set_id,
                        os.name AS order_set_name,
                        d.code AS diagnosis_code,
                        d.name AS diagnosis_name
                    FROM order_sets os
                    JOIN diagnoses d ON d.id = os.diagnosis_id
                    WHERE os.id = %s
                    """,
                    (token_order_set_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="order_set_id not found.")
        if token_diag_code and row.get("diagnosis_code") and token_diag_code != row.get("diagnosis_code"):
            raise HTTPException(status_code=400, detail="selected_candidate_id diagnosis mismatch.")

        selected_order_set_id = int(row["order_set_id"])
        order_set_name = row["order_set_name"]
        diagnosis_code = row["diagnosis_code"]
        diagnosis_name = row["diagnosis_name"]

    else:
        symptom_names = normalize_symptoms(body.symptoms)
        if not symptom_names:
            raise HTTPException(status_code=400, detail="symptoms가 비어있습니다.")

        diags = recommend_diagnoses_by_symptoms(symptom_names, topk=body.topk_diagnosis)
        if not diags:
            raise HTTPException(status_code=404, detail="증상과 매칭되는 진단 후보가 없습니다.")

        is_preg = bool(body.is_pregnant) if body.is_pregnant is not None else False
        if body.ranking_scope == "top_diagnosis_only":
            diag_list_for_order_sets = [diags[0]]
        else:
            diag_list_for_order_sets = diags

        pooled = []
        for d in diag_list_for_order_sets:
            pooled.extend(
                build_order_set_candidates_for_diagnosis(
                    diagnosis_code=d["code"],
                    patient_age=body.patient_age,
                    is_preg=is_preg,
                    topk=body.topk_order_sets_per_diag,
                )
            )

        if not pooled:
            raise HTTPException(status_code=400, detail="추천 가능한 오더셋 후보가 없습니다.")

        pooled.sort(key=lambda x: (-x["composite_score"], x["risk_score"], -x["match_score"], x["order_set_id"]))

        if body.selected_rank < 1 or body.selected_rank > len(pooled):
            raise HTTPException(status_code=400, detail=f"selected_rank 범위 오류: 1~{len(pooled)} 중 선택하세요.")

        selected = pooled[body.selected_rank - 1]
        selected_order_set_id = int(selected["order_set_id"])
        order_set_name = selected["name"]
        diagnosis_code = selected["diagnosis_code"]
        diagnosis_name = selected["diagnosis_name"]
        selected_rank = body.selected_rank
        selected_candidate_id = make_candidate_token(selected_order_set_id, diagnosis_code)

    presc_id = apply_order_set_to_prescription(
        order_set_id=selected_order_set_id,
        patient_id=body.patient_id,
        patient_age=body.patient_age,
        patient_sex=body.patient_sex,
        is_pregnant=body.is_pregnant,
    )

    warnings = get_dur_warnings_for_prescription(presc_id)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM prescription_items WHERE prescription_id=%s", (presc_id,))
            cnt = cur.fetchone()["cnt"]
    finally:
        conn.close()

    log_audit_event(
        event_type="recommendation_selected_and_applied",
        payload={
            "request": body.model_dump(),
            "selected_rank": selected_rank,
            "selected_candidate_id": selected_candidate_id,
            "recommendation_session_id": recommendation_session_id,
            "order_set_id": selected_order_set_id,
            "order_set_name": order_set_name,
            "diagnosis_code": diagnosis_code,
            "diagnosis_name": diagnosis_name,
            "prescription_id": presc_id,
            "item_count": cnt,
            "dur_warnings": warnings,
        },
        patient_id=body.patient_id,
        prescription_id=presc_id,
        recommendation_session_id=recommendation_session_id,
        selected_candidate_id=selected_candidate_id,
    )

    return AutoFromRecommendationOut(
        prescription_id=presc_id,
        selected_rank=selected_rank,
        selected_candidate_id=selected_candidate_id,
        recommendation_session_id=recommendation_session_id,
        order_set_id=selected_order_set_id,
        order_set_name=order_set_name,
        diagnosis_code=diagnosis_code,
        diagnosis_name=diagnosis_name,
        item_count=cnt,
        warnings=warnings,
    )
# OCS API 엔드포인트 모듈
# main.py에 추가할 OCS 관련 API 엔드포인트들

from typing import List, Optional
from datetime import datetime, date
from pydantic import BaseModel
from enum import Enum

# ============================================================
# Pydantic Models for OCS
# ============================================================

class AllergySeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class NotificationType(str, Enum):
    ALLERGY_WARNING = "ALLERGY_WARNING"
    DRUG_INTERACTION = "DRUG_INTERACTION"
    CRITICAL_IMAGING = "CRITICAL_IMAGING"
    PRESCRIPTION_APPROVED = "PRESCRIPTION_APPROVED"
    ORDER_STATUS_CHANGE = "ORDER_STATUS_CHANGE"
    SYSTEM_MESSAGE = "SYSTEM_MESSAGE"

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    APPROVED = "APPROVED"
    DISPENSING = "DISPENSING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    ON_HOLD = "ON_HOLD"

class OrderPriority(str, Enum):
    ROUTINE = "ROUTINE"
    URGENT = "URGENT"
    STAT = "STAT"

# --- Allergy Models ---
class AllergyCreate(BaseModel):
    patient_id: int
    item_seq: Optional[str] = None
    ingredient_code: Optional[str] = None
    ingredient_name: str
    allergy_type: str
    severity: AllergySeverity = AllergySeverity.MEDIUM
    reaction_description: Optional[str] = None
    onset_date: Optional[date] = None
    reported_by: Optional[str] = None

class AllergyOut(BaseModel):
    id: int
    patient_id: int
    item_seq: Optional[str]
    ingredient_code: Optional[str]
    ingredient_name: str
    allergy_type: str
    severity: str
    reaction_description: Optional[str]
    onset_date: Optional[date]
    reported_by: Optional[str]
    is_active: bool
    created_at: datetime

class AllergyCheckResult(BaseModel):
    has_allergy: bool
    warnings: List[AllergyOut]
    message: str

# --- Drug Interaction Models ---
class InteractionCheckResult(BaseModel):
    prescription_id: int
    has_interactions: bool
    interaction_count: int
    interactions: List[dict]

# --- Imaging Models ---
class ImagingCreate(BaseModel):
    patient_id: int
    imaging_type: str
    body_part: Optional[str] = None
    finding_summary: Optional[str] = None
    diagnosis_suggestion: Optional[str] = None
    ai_confidence_score: Optional[float] = None
    radiologist_name: Optional[str] = None
    is_critical: bool = False
    report_url: Optional[str] = None

class ImagingOut(BaseModel):
    id: int
    patient_id: int
    imaging_date: datetime
    imaging_type: str
    body_part: Optional[str]
    finding_summary: Optional[str]
    diagnosis_suggestion: Optional[str]
    ai_confidence_score: Optional[float]
    radiologist_name: Optional[str]
    is_critical: bool
    report_url: Optional[str]
    created_at: datetime

# --- Notification Models ---
class NotificationOut(BaseModel):
    id: int
    target_user_id: Optional[int]
    notification_type: str
    title: str
    message: Optional[str]
    severity: str
    related_prescription_id: Optional[int]
    is_read: bool
    read_at: Optional[datetime]
    created_at: datetime

# --- Order Models ---
class OrderCreate(BaseModel):
    prescription_id: int
    order_type: str = "NEW"
    ordered_by: Optional[int] = None
    priority: OrderPriority = OrderPriority.ROUTINE
    notes: Optional[str] = None

class OrderOut(BaseModel):
    id: int
    prescription_id: int
    order_number: str
    order_type: str
    order_status: str
    ordered_by: Optional[int]
    priority: str
    notes: Optional[str]
    ordered_at: datetime
    verified_at: Optional[datetime]
    approved_at: Optional[datetime]
    completed_at: Optional[datetime]

class OrderStatusUpdate(BaseModel):
    order_id: int
    to_status: OrderStatus
    changed_by: Optional[int] = None
    change_reason: Optional[str] = None
    notes: Optional[str] = None

# ============================================================
# API Endpoints
# ============================================================

# --- Allergy APIs ---
@app.post("/api/allergies/register", response_model=AllergyOut)
def register_allergy(body: AllergyCreate):
    """환자 알러지 등록"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ocs_allergycheck 
                (patient_id, item_seq, ingredient_code, ingredient_name, 
                 allergy_type, severity, reaction_description, onset_date, reported_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (body.patient_id, body.item_seq, body.ingredient_code, 
                 body.ingredient_name, body.allergy_type, body.severity.value,
                 body.reaction_description, body.onset_date, body.reported_by)
            )
            allergy_id = cur.lastrowid
            
            # 조회해서 반환
            cur.execute("SELECT * FROM ocs_allergycheck WHERE id = %s", (allergy_id,))
            result = cur.fetchone()
    finally:
        conn.close()
    
    return AllergyOut(**result)


@app.get("/api/allergies/patient/{patient_id}", response_model=List[AllergyOut])
def get_patient_allergies(patient_id: int, active_only: bool = True):
    """환자의 알러지 목록 조회"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if active_only:
                cur.execute(
                    "SELECT * FROM ocs_allergycheck WHERE patient_id = %s AND is_active = TRUE ORDER BY severity DESC",
                    (patient_id,)
                )
            else:
                cur.execute(
                    "SELECT * FROM ocs_allergycheck WHERE patient_id = %s ORDER BY created_at DESC",
                    (patient_id,)
                )
            results = cur.fetchall()
    finally:
        conn.close()
    
    return [AllergyOut(**row) for row in results]


@app.post("/api/allergies/check", response_model=AllergyCheckResult)
def check_allergy(prescription_id: int):
    """처방전의 약물에 대한 알러지 체크"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 처방 정보 조회
            cur.execute(
                "SELECT patient_id FROM prescriptions WHERE id = %s",
                (prescription_id,)
            )
            presc = cur.fetchone()
            if not presc:
                raise HTTPException(status_code=404, detail="처방전을 찾을 수 없습니다.")
            
            patient_id = presc['patient_id']
            
            # 처방 약물 조회
            cur.execute(
                """
                SELECT DISTINCT pi.item_seq, d.name_kor
                FROM prescription_items pi
                LEFT JOIN drugs d ON d.item_seq = pi.item_seq
                WHERE pi.prescription_id = %s
                """,
                (prescription_id,)
            )
            items = cur.fetchall()
            
            if not items:
                return AllergyCheckResult(
                    has_allergy=False,
                    warnings=[],
                    message="처방 약물이 없습니다."
                )
            
            # 환자 알러지 조회
            cur.execute(
                "SELECT * FROM ocs_allergycheck WHERE patient_id = %s AND is_active = TRUE",
                (patient_id,)
            )
            allergies = cur.fetchall()
            
            if not allergies:
                return AllergyCheckResult(
                    has_allergy=False,
                    warnings=[],
                    message="등록된 알러지가 없습니다."
                )
            
            # 매칭 확인
            warnings = []
            for allergy in allergies:
                for item in items:
                    # item_seq 매칭
                    if allergy.get('item_seq') == item['item_seq']:
                        warnings.append(AllergyOut(**allergy))
                        break
            
            return AllergyCheckResult(
                has_allergy=len(warnings) > 0,
                warnings=warnings,
                message=f"{len(warnings)}개의 알러지 경고가 있습니다." if warnings else "알러지 경고 없음"
            )
    finally:
        conn.close()


# --- Notification APIs ---
@app.get("/api/notifications/unread", response_model=List[NotificationOut])
def get_unread_notifications(user_id: int, limit: int = 50):
    """읽지 않은 알림 조회"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM ocs_notification 
                WHERE target_user_id = %s AND is_read = FALSE
                ORDER BY severity DESC, created_at DESC
                LIMIT %s
                """,
                (user_id, limit)
            )
            results = cur.fetchall()
    finally:
        conn.close()
    
    return [NotificationOut(**row) for row in results]


@app.patch("/api/notifications/{notification_id}/read")
def mark_notification_read(notification_id: int):
    """알림 읽음 처리"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ocs_notification 
                SET is_read = TRUE, read_at = NOW()
                WHERE id = %s
                """,
                (notification_id,)
            )
    finally:
        conn.close()
    
    return {"message": "알림을 읽음 처리했습니다."}


# --- Order APIs ---
@app.post("/api/orders/create", response_model=OrderOut)
def create_order(body: OrderCreate):
    """처방 주문 생성"""
    import uuid
    
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 주문 번호 생성
            order_number = f"ORD-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
            
            cur.execute(
                """
                INSERT INTO ocs_order 
                (prescription_id, order_number, order_type, ordered_by, priority, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (body.prescription_id, order_number, body.order_type, 
                 body.ordered_by, body.priority.value, body.notes)
            )
            order_id = cur.lastrowid
            
            # 상태 이력 기록
            cur.execute(
                """
                INSERT INTO ocs_orderstatushistory 
                (order_id, from_status, to_status, changed_by, notes)
                VALUES (%s, NULL, 'PENDING', %s, 'Order created')
                """,
                (order_id, body.ordered_by)
            )
            
            # 조회
            cur.execute("SELECT * FROM ocs_order WHERE id = %s", (order_id,))
            result = cur.fetchone()
    finally:
        conn.close()
    
    return OrderOut(**result)


@app.patch("/api/orders/{order_id}/status")
def update_order_status(order_id: int, body: OrderStatusUpdate):
    """주문 상태 변경"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 기존 상태 조회
            cur.execute("SELECT order_status FROM ocs_order WHERE id = %s", (order_id,))
            order = cur.fetchone()
            if not order:
                raise HTTPException(status_code=404, detail="주문을 찾을 수 없습니다.")
            
            from_status = order['order_status']
            
            # 상태 업데이트
            timestamp_field = {
                'VERIFIED': 'verified_at',
                'APPROVED': 'approved_at',
                'COMPLETED': 'completed_at'
            }.get(body.to_status.value)
            
            if timestamp_field:
                cur.execute(
                    f"UPDATE ocs_order SET order_status = %s, {timestamp_field} = NOW() WHERE id = %s",
                    (body.to_status.value, order_id)
                )
            else:
                cur.execute(
                    "UPDATE ocs_order SET order_status = %s WHERE id = %s",
                    (body.to_status.value, order_id)
                )
            
            # 이력 기록
            cur.execute(
                """
                INSERT INTO ocs_orderstatushistory 
                (order_id, from_status, to_status, changed_by, change_reason, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (order_id, from_status, body.to_status.value, 
                 body.changed_by, body.change_reason, body.notes)
            )
    finally:
        conn.close()
    
    return {"message": f"주문 상태를 {body.to_status.value}로 변경했습니다."}


@app.get("/api/orders/pending", response_model=List[OrderOut])
def get_pending_orders(priority: Optional[OrderPriority] = None, limit: int = 100):
    """대기 중인 주문 목록"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if priority:
                cur.execute(
                    """
                    SELECT * FROM ocs_order 
                    WHERE order_status IN ('PENDING', 'VERIFIED')
                      AND priority = %s
                    ORDER BY 
                        CASE priority 
                            WHEN 'STAT' THEN 1 
                            WHEN 'URGENT' THEN 2 
                            ELSE 3 
                        END,
                        ordered_at ASC
                    LIMIT %s
                    """,
                    (priority.value, limit)
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM ocs_order 
                    WHERE order_status IN ('PENDING', 'VERIFIED')
                    ORDER BY 
                        CASE priority 
                            WHEN 'STAT' THEN 1 
                            WHEN 'URGENT' THEN 2 
                            ELSE 3 
                        END,
                        ordered_at ASC
                    LIMIT %s
                    """,
                    (limit,)
                )
            results = cur.fetchall()
    finally:
        conn.close()
    
    return [OrderOut(**row) for row in results]


# --- Imaging APIs ---
@app.post("/api/imaging/upload", response_model=ImagingOut)
def upload_imaging_result(body: ImagingCreate):
    """영상 분석 결과 업로드"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ocs_imaginganalysisresult
                (patient_id, imaging_type, body_part, finding_summary, 
                 diagnosis_suggestion, ai_confidence_score, radiologist_name, 
                 is_critical, report_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (body.patient_id, body.imaging_type, body.body_part, 
                 body.finding_summary, body.diagnosis_suggestion, 
                 body.ai_confidence_score, body.radiologist_name,
                 body.is_critical, body.report_url)
            )
            imaging_id = cur.lastrowid
            
            # 긴급인 경우 알림 생성
            if body.is_critical:
                cur.execute(
                    """
                    INSERT INTO ocs_notification
                    (notification_type, title, message, severity, related_entity_type, related_entity_id)
                    VALUES ('CRITICAL_IMAGING', %s, %s, 'CRITICAL', 'imaging', %s)
                    """,
                    (
                        f"긴급 영상 판독: {body.imaging_type}",
                        f"환자 ID {body.patient_id}의 {body.imaging_type} 검사에서 긴급 소견이 발견되었습니다.",
                        imaging_id
                    )
                )
            
            # 조회
            cur.execute("SELECT * FROM ocs_imaginganalysisresult WHERE id = %s", (imaging_id,))
            result = cur.fetchone()
    finally:
        conn.close()
    
    return ImagingOut(**result)


@app.get("/api/imaging/patient/{patient_id}", response_model=List[ImagingOut])
def get_patient_imaging(patient_id: int, limit: int = 50):
    """환자의 영상 검사 이력"""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM ocs_imaginganalysisresult 
                WHERE patient_id = %s 
                ORDER BY imaging_date DESC 
                LIMIT %s
                """,
                (patient_id, limit)
            )
            results = cur.fetchall()
    finally:
        conn.close()
    
    return [ImagingOut(**row) for row in results]


# =========================
# AI Analysis Endpoints
# =========================
from ai_service import ai_service
from ddinter_helper import check_ddinter_interactions

class AnalyzeInteractionRequest(BaseModel):
    drug_a_name: str
    drug_a_code: str
    drug_b_name: str
    drug_b_code: str
    db_reason: str

@app.post("/drugs/interaction/analyze")
async def analyze_drug_interaction(req: AnalyzeInteractionRequest):
    """
    두 약물 간의 상호작용 원인과 임상적 권고안을 AI(또는 DB)를 통해 분석합니다.
    """
    result = ai_service.analyze_interaction(
        drug_a_name=req.drug_a_name,
        drug_a_ingr=req.drug_a_code,
        drug_b_name=req.drug_b_name,
        drug_b_ingr=req.drug_b_code,
        reason_from_db=req.db_reason
    )
    return {"result": result}
