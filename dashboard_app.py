import json
import os
from pathlib import Path
import re
from urllib import error, request

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError as exc:
    raise ImportError(
        "plotly가 필요합니다. `pip install plotly streamlit` 후 다시 실행해주세요."
    ) from exc


st.set_page_config(
    page_title="신상품 센터 대시보드",
    page_icon="📦",
    layout="wide",
)


APP_DIR = Path(__file__).resolve().parent
LOCAL_DATA_DIR = APP_DIR / "데이터"
SHARED_DATA_DIR = APP_DIR.parent / "데이터"
DATA_DIR = LOCAL_DATA_DIR if LOCAL_DATA_DIR.exists() else SHARED_DATA_DIR
PREORDER_PATH = APP_DIR / "final_preorder.csv"
SALES_PATH = APP_DIR / "center_sales_final.csv"
STOCK_PATH = APP_DIR / "A4_final_CENTER_STK.csv"
CENTER_ORDER_PATH = APP_DIR / "A1_final_center_order.csv"
PREDICTIONS_PATH = APP_DIR / "predictions.parquet"
MASTER_ITEM_PATH = DATA_DIR / "A7_신상품_상품마스터.csv"
CENTER_MAP_PATH = DATA_DIR / "target_centers_for_map.csv"

MASTER_ACCOUNT_ID = "master"
MASTER_ACCOUNT_PASSWORD = "master123!"
APP_SESSION_VERSION = "md-login-v2"

PREORDER_DAY_COLUMNS = [
    "D-11",
    "D-10",
    "D-9",
    "D-8",
    "D-7",
    "D-6",
    "D-5",
    "D-4",
    "D-3",
    "D-2",
    "D-1",
    "D-0",
]

PREORDER_DAY_OFFSETS = {column: int(column.replace("D", "")) for column in PREORDER_DAY_COLUMNS}

TABLEAU_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]

GEMINI_MODEL = "gemini-2.0-flash"
INITIAL_ORDER_MULTIPLIER = 2.5
OUTFLOW_STATUS_BOUNDS = {
    "slow_stock": 0.333,
    "over_order_risk": 0.5,
    "normal": 0.778,
    "shortage_risk": 1.0,
}
OUTFLOW_STATUS_ORDER = ["부진재고", "과발주 위험", "정상", "결품 위험", "결품"]
OUTFLOW_STATUS_COLORS = {
    "부진재고": "#E68163",
    "과발주 위험": "#F7D79A",
    "정상": "#A9CF7A",
    "결품 위험": "#F3C96C",
    "결품": "#EF6F6C",
}

CENTER_WEIGHT_CONFIG = {
    "20079": {"weight": 0.8, "store_count": 672, "ldu": "광주"},
    "20081": {"weight": 0.9, "store_count": 800, "ldu": "양산"},
    "20080": {"weight": 0.9, "store_count": 721, "ldu": "의왕"},
    "20083": {"weight": 0.75, "store_count": 579, "ldu": "김제"},
    "20007": {"weight": 1.0, "store_count": 903, "ldu": "대구"},
    "20034": {"weight": 1.25, "store_count": 937, "ldu": "구성(용인)"},
    "20006": {"weight": 1.3, "store_count": 717, "ldu": "성남"},
    "20010": {"weight": 1.18, "store_count": 948, "ldu": "양주"},
    "20050": {"weight": 0.9, "store_count": 907, "ldu": "세종"},
    "20017": {"weight": 0.8, "store_count": 660, "ldu": "울산"},
    "20065": {"weight": 0.9, "store_count": 755, "ldu": "원주"},
    "20075": {"weight": 0.95, "store_count": 892, "ldu": "인천"},
    "20085": {"weight": 1.0, "store_count": 860, "ldu": "인천B"},
    "20084": {"weight": 0.8, "store_count": 653, "ldu": "천안"},
    "20033": {"weight": 0.75, "store_count": 282, "ldu": "제주"},
}


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def format_int(value: float) -> str:
    return f"{value:,.0f}"


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def format_won(value: float) -> str:
    if pd.isna(value):
        return "-"
    if value >= 100_000_000:
        return f"{value / 100_000_000:,.1f}억"
    if value >= 10_000:
        return f"{value / 10_000:,.0f}만"
    return f"{value:,.0f}"


def normalize_center_code(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def format_md_weekday(value) -> str:
    date_value = pd.to_datetime(value, errors="coerce")
    if pd.isna(date_value):
        return "-"
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    return f"{date_value.month:02d}/{date_value.day:02d}({weekdays[date_value.weekday()]})"


def classify_outflow_status(outflow_ratio: pd.Series) -> pd.Series:
    ratio = pd.to_numeric(outflow_ratio, errors="coerce")
    return pd.Series(
        np.select(
            [
                ratio < OUTFLOW_STATUS_BOUNDS["slow_stock"],
                ratio < OUTFLOW_STATUS_BOUNDS["over_order_risk"],
                ratio < OUTFLOW_STATUS_BOUNDS["normal"],
                ratio < OUTFLOW_STATUS_BOUNDS["shortage_risk"],
                ratio >= OUTFLOW_STATUS_BOUNDS["shortage_risk"],
            ],
            OUTFLOW_STATUS_ORDER,
            default="판정 제외",
        ),
        index=ratio.index,
    )


def clean_item_description(text: str) -> list[str]:
    if pd.isna(text):
        return []
    raw_lines = str(text).replace("\r", "\n").split("\n")
    cleaned_lines: list[str] = []
    for line in raw_lines:
        line = re.sub(r"^\s*[0-9]+\.\s*", "", line)
        line = re.sub(r"^\s*[-*]+\s*", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line or line.startswith("※"):
            continue
        cleaned_lines.append(line)
    return cleaned_lines


def summarize_item_description(text: str, max_points: int = 2, max_length: int = 120) -> str:
    lines = clean_item_description(text)
    if not lines:
        return ""

    summary = " ".join(lines[:max_points])
    summary = re.sub(r"([가-힣A-Za-z0-9%])\1{3,}", r"\1", summary)
    summary = re.sub(r"\s+", " ", summary).strip()

    if len(summary) <= max_length:
        return summary
    truncated = summary[:max_length].rsplit(" ", 1)[0].strip()
    return f"{truncated}..."


def get_gemini_api_key() -> str:
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""
    return str(secret_key or os.getenv("GEMINI_API_KEY", "")).strip()


def get_master_account_password() -> str:
    try:
        secret_password = st.secrets.get("MASTER_ACCOUNT_PASSWORD", "")
    except Exception:
        secret_password = ""
    return str(secret_password or os.getenv("MASTER_ACCOUNT_PASSWORD", "") or MASTER_ACCOUNT_PASSWORD).strip()


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def generate_item_description_summary(
    item_name: str,
    brand: str,
    middle_category: str,
    small_category: str,
    description: str,
    fallback_summary: str,
) -> str:
    description = str(description or "").strip()
    fallback_summary = str(fallback_summary or "").strip()
    api_key = get_gemini_api_key()

    if not description:
        return fallback_summary
    if not api_key:
        return fallback_summary

    prompt = f"""
다음은 편의점 신상품 소개 원문이다.
상품명: {item_name}
브랜드: {brand}
중분류: {middle_category}
소분류: {small_category}
원문:
{description}

요구사항:
- 한국어로 1~2문장만 작성
- 운영 화면에 들어갈 짧은 상품 소개 문구처럼 작성
- 불필요한 과장 표현, 번호 목록, 따옴표는 제외
- 핵심 특징과 맛/포인트만 자연스럽게 정리
""".strip()

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 120},
    }
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    req = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=8) as response:
            result = json.loads(response.read().decode("utf-8"))
        text = (
            result.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        text = re.sub(r"\s+", " ", str(text)).strip()
        text = text.strip("\"' ")
        return text or fallback_summary
    except (error.URLError, error.HTTPError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
        return fallback_summary


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #f6f7fb;
            --panel: #ffffff;
            --ink: #16202a;
            --muted: #5f6b7a;
            --line: #d9dde7;
            --blue: #4f6df5;
            --orange: #ffb32c;
            --green: #2fd3b1;
            --red: #e15759;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 122, 75, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(255, 111, 31, 0.10), transparent 22%),
                linear-gradient(180deg, #fbfcff 0%, var(--bg) 100%);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(135deg, rgba(15,122,75,0.98), rgba(255,111,31,0.96));
            color: white;
            padding: 1.35rem 1.5rem;
            border-radius: 18px;
            box-shadow: 0 18px 40px rgba(34, 52, 89, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            font-size: 1.85rem;
            line-height: 1.2;
            margin: 0 0 0.45rem 0;
        }
        .hero p {
            margin: 0;
            color: rgba(255,255,255,0.86);
            font-size: 0.98rem;
        }
        .section-label {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            color: var(--blue);
            text-transform: uppercase;
            margin-top: 0.35rem;
            margin-bottom: 0.2rem;
        }
        .kpi-card {
            background: var(--panel);
            border: 1px solid rgba(217, 221, 231, 0.9);
            border-radius: 16px;
            padding: 1rem 1.05rem 0.95rem 1.05rem;
            box-shadow: 0 8px 26px rgba(25, 40, 67, 0.06);
            min-height: 118px;
        }
        .kpi-label {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 0.55rem;
        }
        .kpi-value {
            color: var(--ink);
            font-size: 1.7rem;
            line-height: 1.05;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .kpi-sub {
            color: var(--muted);
            font-size: 0.82rem;
        }
        .insight-card {
            background: rgba(255,255,255,0.9);
            border-left: 4px solid var(--orange);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 8px 22px rgba(25, 40, 67, 0.05);
        }
        .weekly-shell {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(217,221,231,0.92);
            border-radius: 20px;
            padding: 1.1rem;
            box-shadow: 0 14px 34px rgba(25, 40, 67, 0.06);
            margin-bottom: 1rem;
        }
        .weekly-header {
            display: grid;
            grid-template-columns: 1.7fr 0.8fr 0.8fr 0.9fr 0.7fr;
            gap: 0.8rem;
            padding: 0.2rem 1rem 0.65rem 1rem;
            color: #6b7280;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .weekly-row {
            display: grid;
            grid-template-columns: 1.7fr 0.8fr 0.8fr 0.9fr 0.7fr;
            gap: 0.8rem;
            align-items: center;
            background: #ffffff;
            border: 1px solid rgba(223,227,236,0.95);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 10px 24px rgba(25,40,67,0.04);
        }
        .weekly-product {
            display: flex;
            align-items: center;
            gap: 0.85rem;
            min-width: 0;
        }
        .weekly-avatar {
            width: 42px;
            height: 42px;
            border-radius: 50%;
            background: linear-gradient(135deg, #4f6df5, #35c8e8);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            font-weight: 800;
            flex-shrink: 0;
        }
        .weekly-name {
            color: #16202a;
            font-size: 0.98rem;
            font-weight: 800;
            line-height: 1.3;
        }
        .weekly-meta {
            color: #6b7280;
            font-size: 0.78rem;
            margin-top: 0.16rem;
        }
        .weekly-metric {
            color: #16202a;
            font-size: 0.96rem;
            font-weight: 700;
            text-align: left;
        }
        .weekly-submetric {
            color: #6b7280;
            font-size: 0.76rem;
            margin-top: 0.15rem;
        }
        .weekly-status {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            font-size: 0.77rem;
            font-weight: 800;
            width: fit-content;
        }
        .status-high {
            background: rgba(15,122,75,0.12);
            color: #4f6df5;
        }
        .status-mid {
            background: rgba(255,111,31,0.12);
            color: #e56a20;
        }
        .status-low {
            background: rgba(156,117,95,0.12);
            color: #8b6b54;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.72);
            border-radius: 999px;
            padding: 0.4rem 0.95rem;
            border: 1px solid rgba(217, 221, 231, 0.9);
        }
        .stTabs [aria-selected="true"] {
            background: #ffffff;
            box-shadow: 0 6px 18px rgba(25, 40, 67, 0.07);
        }
        .login-panel {
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(217, 221, 231, 0.95);
            border-radius: 26px;
            padding: 1.7rem 1.5rem 1.35rem 1.5rem;
            box-shadow: 0 20px 42px rgba(25, 40, 67, 0.09);
        }
        .login-title {
            font-size: 2.05rem;
            font-weight: 800;
            color: #16202a;
            margin-bottom: 0.25rem;
            text-align: center;
        }
        .login-sub {
            color: #5f6b7a;
            font-size: 0.88rem;
            margin-bottom: 1.2rem;
            text-align: center;
            line-height: 1.5;
        }
        .login-chip {
            display: inline-block;
            padding: 0.28rem 0.68rem;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(15,122,75,0.12), rgba(255,111,31,0.12));
            border: 1px solid rgba(15,122,75,0.18);
            color: #4f6df5;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            margin-bottom: 0.95rem;
        }
        div[data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
            background: transparent !important;
        }
        div[data-testid="stTextInput"] input {
            border-radius: 999px !important;
            border: 1.5px solid #d6dbe4 !important;
            padding-top: 0.95rem !important;
            padding-bottom: 0.95rem !important;
            background: #ffffff !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #4f6df5 !important;
            box-shadow: 0 0 0 1px #4f6df5 !important;
        }
        div[data-testid="stTextInput"] label p {
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            color: #5f6b7a !important;
        }
        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="stButton"] button {
            border-radius: 999px !important;
            min-height: 3.15rem !important;
            font-weight: 800 !important;
        }
        div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(90deg, #4f6df5, #35c8e8) !important;
            color: white !important;
            border: none !important;
        }
        :root {
            --bg: #eef3fb;
            --panel: #ffffff;
            --ink: #161b2d;
            --muted: #9aa4b8;
            --line: #e9edf6;
            --blue: #4f6df5;
            --blue-deep: #2637c9;
            --cyan: #35c8e8;
            --mint: #2fd3b1;
            --violet: #7b4cf3;
            --pink: #f455a7;
            --amber: #ffb32c;
            --red: #ff4d6d;
            --shadow: 0 18px 45px rgba(40, 51, 86, 0.10);
            --soft-shadow: 0 8px 22px rgba(40, 51, 86, 0.08);
        }
        .stApp {
            background:
                radial-gradient(circle at 4% 0%, rgba(79, 109, 245, 0.16), transparent 25%),
                radial-gradient(circle at 100% 5%, rgba(53, 200, 232, 0.13), transparent 22%),
                linear-gradient(135deg, #f7f9fd 0%, #eef3fb 100%) !important;
            color: var(--ink);
        }
        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.86) !important;
            border-right: 1px solid rgba(226, 232, 245, 0.9);
            box-shadow: 16px 0 42px rgba(41, 56, 93, 0.06);
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 1.35rem;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #24304f !important;
            font-weight: 800 !important;
        }
        .block-container {
            max-width: 1480px;
            padding-top: 1.25rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        .hero {
            position: relative;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.92) !important;
            color: var(--ink) !important;
            border: 1px solid rgba(231, 235, 247, 0.95);
            border-radius: 24px !important;
            padding: 1.45rem 1.65rem 1.35rem 1.65rem !important;
            box-shadow: var(--shadow) !important;
        }
        .hero::before {
            content: "";
            position: absolute;
            inset: auto -60px -85px auto;
            width: 250px;
            height: 250px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(79, 109, 245, 0.24), rgba(53, 200, 232, 0.08) 58%, transparent 70%);
            pointer-events: none;
        }
        .hero::after {
            content: "Dashboard";
            position: absolute;
            right: 1.45rem;
            top: 1.15rem;
            padding: 0.45rem 0.78rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(79,109,245,0.12), rgba(47,211,177,0.12));
            color: #4f6df5;
            font-size: 0.74rem;
            font-weight: 800;
        }
        .hero h1 {
            color: #151a2e !important;
            font-size: 2rem !important;
            letter-spacing: 0 !important;
        }
        .hero p {
            max-width: 760px;
            color: #8b95aa !important;
            font-size: 0.95rem !important;
        }
        .kpi-card {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(231, 235, 247, 0.95) !important;
            border-radius: 18px !important;
            background: rgba(255, 255, 255, 0.94) !important;
            box-shadow: var(--soft-shadow) !important;
            min-height: 112px !important;
            padding: 1.08rem 1.1rem !important;
        }
        .kpi-card::before {
            content: "";
            display: block;
            width: 38px;
            height: 38px;
            border-radius: 13px;
            margin-bottom: 0.65rem;
            background: linear-gradient(135deg, #4f6df5, #35c8e8);
            box-shadow: 0 10px 22px rgba(79, 109, 245, 0.22);
        }
        .kpi-card::after {
            content: "...";
            position: absolute;
            right: 1rem;
            top: 0.75rem;
            color: #b8bfd0;
            font-weight: 800;
            letter-spacing: 0.08em;
        }
        .kpi-label {
            color: #a0a9bb !important;
            font-size: 0.76rem !important;
            font-weight: 700 !important;
        }
        .kpi-value {
            color: #161b2d !important;
            font-size: 1.58rem !important;
            font-weight: 850 !important;
        }
        .kpi-sub {
            color: #2fc39e !important;
            font-size: 0.78rem !important;
            font-weight: 750;
        }
        .insight-card,
        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"],
        div[data-testid="stPlotlyChart"] {
            border-radius: 20px !important;
        }
        .insight-card {
            background: rgba(255,255,255,0.94) !important;
            border: 1px solid rgba(231, 235, 247, 0.95) !important;
            border-left: 0 !important;
            box-shadow: var(--soft-shadow) !important;
            color: #29324a;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(231, 235, 247, 0.95);
            box-shadow: var(--soft-shadow);
            padding: 1rem 1.05rem;
        }
        div[data-testid="stMetricLabel"] p {
            color: #9aa4b8 !important;
            font-size: 0.78rem !important;
            font-weight: 750 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #161b2d !important;
            font-weight: 850 !important;
        }
        div[data-testid="stDataFrame"],
        div[data-testid="stPlotlyChart"] {
            background: rgba(255,255,255,0.94);
            border: 1px solid rgba(231, 235, 247, 0.95);
            box-shadow: var(--soft-shadow);
            padding: 0.55rem;
        }
        h1, h2, h3 {
            color: #161b2d !important;
            letter-spacing: 0 !important;
        }
        div[data-testid="stCaptionContainer"],
        p, label {
            color: #8f9aae;
        }
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255,255,255,0.72);
            border: 1px solid rgba(231, 235, 247, 0.92);
            border-radius: 18px;
            padding: 0.35rem;
            box-shadow: var(--soft-shadow);
        }
        .stTabs [data-baseweb="tab"] {
            border: 0 !important;
            border-radius: 14px !important;
            color: #9aa4b8;
            background: transparent !important;
            min-height: 2.45rem;
        }
        .stTabs [aria-selected="true"] {
            color: #4f6df5 !important;
            background: #ffffff !important;
            box-shadow: 0 9px 22px rgba(79, 109, 245, 0.12) !important;
        }
        div[data-testid="stTextInput"] input,
        div[data-testid="stDateInput"] input,
        div[data-baseweb="select"] > div,
        div[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
            border-radius: 15px !important;
            border: 1px solid rgba(226, 232, 245, 0.95) !important;
            background: rgba(255,255,255,0.92) !important;
            box-shadow: 0 5px 14px rgba(40, 51, 86, 0.04);
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #4f6df5 !important;
            box-shadow: 0 0 0 3px rgba(79, 109, 245, 0.12) !important;
        }
        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            border-radius: 15px !important;
            border: 1px solid rgba(226, 232, 245, 0.95) !important;
            background: #ffffff !important;
            color: #4f6df5 !important;
            box-shadow: var(--soft-shadow);
            min-height: 2.75rem !important;
        }
        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg, #4f6df5, #35c8e8) !important;
            color: #ffffff !important;
            border: 0 !important;
        }
        div[role="radiogroup"] {
            gap: 0.35rem;
        }
        div[role="radiogroup"] label {
            background: rgba(255,255,255,0.76);
            border: 1px solid rgba(231, 235, 247, 0.95);
            border-radius: 14px;
            padding: 0.45rem 0.65rem;
            margin-bottom: 0.25rem;
        }
        .js-plotly-plot .plotly .gridlayer path {
            stroke: rgba(218, 225, 240, 0.72) !important;
        }
        @media (max-width: 900px) {
            .login-title {
                font-size: 1.8rem;
            }
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, subtext: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=56, b=12),
        legend_title_text="",
        font=dict(size=13, color="#2d3448", family="Inter, Pretendard, Arial, sans-serif"),
        title=dict(font=dict(size=18, color="#161b2d")),
        colorway=["#4f6df5", "#35c8e8", "#2fd3b1", "#7b4cf3", "#ffb32c", "#f455a7"],
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(218,225,240,0.75)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(218,225,240,0.75)", zeroline=False)
    return fig


def get_latest_week_range(item_master: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    latest_date = item_master["NP_RLSE_DATE"].max()
    week_start = latest_date - pd.Timedelta(days=latest_date.weekday())
    week_end = week_start + pd.Timedelta(days=6)
    return week_start, week_end


def build_weekly_item_list(item_master: pd.DataFrame, preorder_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    week_start, week_end = get_latest_week_range(item_master)
    weekly_items = item_master[item_master["NP_RLSE_DATE"].between(week_start, week_end)].copy()

    if weekly_items.empty:
        return weekly_items

    preorder_summary = (
        preorder_df[preorder_df["ITEM_CODE"].isin(weekly_items["ITEM_CODE"])]
        .groupby("ITEM_CODE", as_index=False)
        .agg(
            WEEK_INITIAL_ORD_QTY=("INITIAL_ORD_QTY", "sum"),
            WEEK_RESERVATION_QTY=("reservation_qty_total", "sum"),
            WEEK_ORDERING_STORE_CNT=("ordering_store_cnt", "sum"),
            WEEK_TOTAL_STORE_CNT=("total_store_cnt", "sum"),
        )
    )
    sales_summary = (
        sales_df[sales_df["ITEM_CODE"].isin(weekly_items["ITEM_CODE"])]
        .groupby("ITEM_CODE", as_index=False)
        .agg(
            WEEK_SALE_QTY=("CENTER_SALE_QTY", "sum"),
            WEEK_SALE_AMT=("CENTER_SALE_AMT_VAT", "sum"),
        )
    )
    out = weekly_items.merge(preorder_summary, on="ITEM_CODE", how="left").merge(
        sales_summary, on="ITEM_CODE", how="left"
    )
    fill_cols = [
        "WEEK_INITIAL_ORD_QTY",
        "WEEK_RESERVATION_QTY",
        "WEEK_ORDERING_STORE_CNT",
        "WEEK_TOTAL_STORE_CNT",
        "WEEK_SALE_QTY",
        "WEEK_SALE_AMT",
    ]
    out[fill_cols] = out[fill_cols].fillna(0)
    out["WEEK_RESERVATION_RATE"] = (
        out["WEEK_RESERVATION_QTY"] / out["WEEK_INITIAL_ORD_QTY"].replace(0, pd.NA) * 100
    )
    out["WEEK_STORE_PARTICIPATION"] = (
        out["WEEK_ORDERING_STORE_CNT"] / out["WEEK_TOTAL_STORE_CNT"].replace(0, pd.NA) * 100
    )
    out["RESERVATION_BADGE"] = pd.cut(
        out["WEEK_RESERVATION_RATE"].fillna(0),
        bins=[-1, 10, 25, float("inf")],
        labels=["예약 낮음", "예약 보통", "예약 높음"],
    ).astype("string")
    return out.sort_values(["NP_RLSE_DATE", "WEEK_RESERVATION_QTY"], ascending=[False, False])


def build_item_center_preorder_detail(preorder_df: pd.DataFrame, item_code: str) -> pd.DataFrame:
    detail = preorder_df[preorder_df["ITEM_CODE"] == item_code].copy()
    if detail.empty:
        return detail
    center_detail = (
        detail.groupby(["CENTER_CODE", "CENTER_NM"], as_index=False)
        .agg(
            INITIAL_ORD_QTY=("INITIAL_ORD_QTY", "sum"),
            RESERVATION_QTY=("reservation_qty_total", "sum"),
            ORDERING_STORE_CNT=("ordering_store_cnt", "sum"),
            TOTAL_STORE_CNT=("total_store_cnt", "sum"),
        )
    )
    center_detail["예약/초도 비율(%)"] = (
        center_detail["RESERVATION_QTY"] / center_detail["INITIAL_ORD_QTY"].replace(0, pd.NA) * 100
    )
    center_detail["예약 참여율(%)"] = (
        center_detail["ORDERING_STORE_CNT"] / center_detail["TOTAL_STORE_CNT"].replace(0, pd.NA) * 100
    )
    return center_detail.sort_values("RESERVATION_QTY", ascending=False)


def build_center_initial_order_plan(center_detail: pd.DataFrame) -> pd.DataFrame:
    if center_detail.empty:
        return center_detail

    plan = center_detail.copy()
    plan["CENTER_CODE"] = plan["CENTER_CODE"].map(normalize_center_code)
    plan["센터 가중치"] = plan["CENTER_CODE"].map(
        lambda code: CENTER_WEIGHT_CONFIG.get(code, {}).get("weight", 1.0)
    )
    plan["기준 점포수"] = plan["CENTER_CODE"].map(
        lambda code: CENTER_WEIGHT_CONFIG.get(code, {}).get("store_count")
    )
    center_name_by_code = plan.drop_duplicates("CENTER_CODE").set_index("CENTER_CODE")["CENTER_NM"].to_dict()
    plan["LDU"] = plan["CENTER_CODE"].map(
        lambda code: CENTER_WEIGHT_CONFIG.get(code, {}).get("ldu") or center_name_by_code.get(code, code)
    )
    plan["산식 초도예측량"] = (
        plan["RESERVATION_QTY"] * INITIAL_ORDER_MULTIPLIER * plan["센터 가중치"]
    ).round()
    plan["초도 차이"] = plan["INITIAL_ORD_QTY"] - plan["산식 초도예측량"]
    plan["예약 충족 배수"] = np.where(
        plan["RESERVATION_QTY"] > 0,
        plan["INITIAL_ORD_QTY"] / plan["RESERVATION_QTY"],
        np.nan,
    )
    return plan.sort_values("RESERVATION_QTY", ascending=False)


def build_center_map_view(center_plan: pd.DataFrame, center_locations: pd.DataFrame) -> pd.DataFrame:
    if center_plan.empty or center_locations.empty:
        return pd.DataFrame()

    mapped = center_plan.merge(center_locations, on=["CENTER_CODE", "CENTER_NM"], how="left")
    mapped["상태 판정"] = np.select(
        [
            mapped["초도 차이"] > mapped["산식 초도예측량"].fillna(0) * 0.15,
            mapped["초도 차이"] < -mapped["산식 초도예측량"].fillna(0) * 0.15,
        ],
        ["과다 가능", "부족 가능"],
        default="적정",
    )
    mapped["지도 라벨"] = mapped["CENTER_LABEL"].fillna(mapped["CENTER_NM"])
    mapped["마커 크기"] = mapped["INITIAL_ORD_QTY"].clip(lower=1)
    return mapped.dropna(subset=["LAT", "LON"]).copy()


@st.cache_data(show_spinner=False)
def build_preorder_sales_analysis(preorder_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    preorder = preorder_df.copy()
    sales = sales_df.copy()

    preorder["NP_RLSE_DATE"] = pd.to_datetime(preorder["NP_RLSE_DATE"], errors="coerce")
    sales["SALE_DATE"] = pd.to_datetime(sales["SALE_DATE"], errors="coerce")

    merged = preorder.merge(
        sales,
        left_on=["ITEM_CODE", "CENTER_NM"],
        right_on=["ITEM_CODE", "CENTER_NM"],
        how="left",
    )

    in_window = merged[
        (merged["SALE_DATE"] >= merged["NP_RLSE_DATE"])
        & (merged["SALE_DATE"] < merged["NP_RLSE_DATE"] + pd.Timedelta(days=7))
    ].copy()

    actual_sales = (
        in_window.groupby(["ITEM_CODE", "CENTER_NM"], as_index=False)["CENTER_SALE_QTY"]
        .sum()
        .rename(columns={"CENTER_SALE_QTY": "actual_sales_qty_7d"})
    )

    base = preorder.merge(actual_sales, on=["ITEM_CODE", "CENTER_NM"], how="left")
    base["actual_sales_qty_7d"] = base["actual_sales_qty_7d"].fillna(0)
    base["preorder_qty"] = pd.to_numeric(base["reservation_qty_total"], errors="coerce").fillna(0)
    base["initial_order_qty"] = pd.to_numeric(base["INITIAL_ORD_QTY"], errors="coerce").fillna(0)
    base["over_order_gap"] = base["initial_order_qty"] - base["actual_sales_qty_7d"]
    return base


def build_item_detail_analysis(analysis_df: pd.DataFrame, item_code: str) -> tuple[pd.DataFrame, pd.Series | None]:
    scoped = analysis_df[analysis_df["ITEM_CODE"] == item_code].copy()
    if scoped.empty:
        return pd.DataFrame(), None

    center_view = (
        scoped.groupby(["CENTER_CODE", "CENTER_NM"], as_index=False)[
            ["preorder_qty", "initial_order_qty", "actual_sales_qty_7d", "over_order_gap"]
        ]
        .sum()
        .sort_values(["initial_order_qty", "actual_sales_qty_7d"], ascending=False)
    )
    summary = center_view[["preorder_qty", "initial_order_qty", "actual_sales_qty_7d", "over_order_gap"]].sum()
    return center_view, summary


def build_item_preorder_profile(preorder_df: pd.DataFrame, item_code: str) -> pd.DataFrame:
    detail = preorder_df[preorder_df["ITEM_CODE"] == item_code].copy()
    if detail.empty:
        return pd.DataFrame(columns=["예약일자", "예약 수량"])
    melted = detail.melt(
        id_vars=["ITEM_CODE", "CENTER_NM", "NP_RLSE_DATE"],
        value_vars=PREORDER_DAY_COLUMNS,
        var_name="예약 시점",
        value_name="예약 수량",
    )
    melted["예약일자"] = melted["NP_RLSE_DATE"] + pd.to_timedelta(
        melted["예약 시점"].map(PREORDER_DAY_OFFSETS), unit="D"
    )
    return (
        melted.groupby("예약일자", as_index=False)["예약 수량"]
        .sum()
        .sort_values("예약일자")
    )


def build_item_center_preorder_profile(preorder_df: pd.DataFrame, item_code: str) -> pd.DataFrame:
    detail = preorder_df[preorder_df["ITEM_CODE"] == item_code].copy()
    if detail.empty:
        return pd.DataFrame(columns=["CENTER_NM", "예약일자", "예약 수량"])
    melted = detail.melt(
        id_vars=["ITEM_CODE", "CENTER_NM", "NP_RLSE_DATE"],
        value_vars=PREORDER_DAY_COLUMNS,
        var_name="예약 시점",
        value_name="예약 수량",
    )
    melted["예약일자"] = melted["NP_RLSE_DATE"] + pd.to_timedelta(
        melted["예약 시점"].map(PREORDER_DAY_OFFSETS), unit="D"
    )
    return (
        melted.groupby(["CENTER_NM", "예약일자"], as_index=False)["예약 수량"]
        .sum()
        .sort_values(["CENTER_NM", "예약일자"])
    )


def render_product_card(item_row: pd.Series) -> None:
    badge_color = {
        "예약 높음": "#4e79a7",
        "예약 보통": "#f28e2b",
        "예약 낮음": "#9c755f",
    }.get(item_row["RESERVATION_BADGE"], "#4e79a7")
    st.markdown(
        f"""
        <div style="background:#fff;border:1px solid rgba(217,221,231,0.9);border-radius:18px;padding:1rem 1rem 0.9rem 1rem;box-shadow:0 10px 28px rgba(25,40,67,0.06);min-height:230px;">
            <div style="display:flex;justify-content:space-between;gap:0.8rem;align-items:flex-start;">
                <div>
                    <div style="font-size:1.02rem;font-weight:800;color:#16202a;line-height:1.4;">{item_row['ITEM_NM']}</div>
                    <div style="font-size:0.82rem;color:#5f6b7a;margin-top:0.25rem;">{item_row['BRAND']} · {item_row['ITEM_MDDV_NM']} · {item_row['ITEM_SMDV_NM']}</div>
                </div>
                <div style="background:{badge_color};color:#fff;padding:0.28rem 0.55rem;border-radius:999px;font-size:0.75rem;font-weight:700;white-space:nowrap;">{item_row['RESERVATION_BADGE']}</div>
            </div>
            <div style="margin-top:0.9rem;display:grid;grid-template-columns:repeat(3, minmax(0,1fr));gap:0.55rem;">
                <div style="background:#f7f9fc;border-radius:12px;padding:0.6rem 0.7rem;"><div style="font-size:0.72rem;color:#6b7280;">출시일</div><div style="font-size:0.92rem;font-weight:700;color:#16202a;">{item_row['NP_RLSE_DATE'].date()}</div></div>
                <div style="background:#f7f9fc;border-radius:12px;padding:0.6rem 0.7rem;"><div style="font-size:0.72rem;color:#6b7280;">판매가</div><div style="font-size:0.92rem;font-weight:700;color:#16202a;">{format_won(item_row['ST_SLEM_AMT'])}원</div></div>
                <div style="background:#f7f9fc;border-radius:12px;padding:0.6rem 0.7rem;"><div style="font-size:0.72rem;color:#6b7280;">최소주문</div><div style="font-size:0.92rem;font-weight:700;color:#16202a;">{format_int(item_row['MIN_ORD_QTY'])}</div></div>
            </div>
            <div style="margin-top:0.8rem;display:grid;grid-template-columns:repeat(3, minmax(0,1fr));gap:0.55rem;">
                <div><div style="font-size:0.72rem;color:#6b7280;">예약수량</div><div style="font-size:1.05rem;font-weight:800;color:#16202a;">{format_int(item_row['WEEK_RESERVATION_QTY'])}</div></div>
                <div><div style="font-size:0.72rem;color:#6b7280;">예약점포</div><div style="font-size:1.05rem;font-weight:800;color:#16202a;">{format_int(item_row['WEEK_ORDERING_STORE_CNT'])}</div></div>
                <div><div style="font-size:0.72rem;color:#6b7280;">예약/초도</div><div style="font-size:1.05rem;font-weight:800;color:#16202a;">{format_pct(item_row['WEEK_RESERVATION_RATE'] if pd.notna(item_row['WEEK_RESERVATION_RATE']) else 0)}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_weekly_product_row(item_row: pd.Series) -> None:
    badge = str(item_row.get("RESERVATION_BADGE", "예약 보통"))
    status_class = {
        "예약 높음": "status-high",
        "예약 보통": "status-mid",
        "예약 낮음": "status-low",
    }.get(badge, "status-mid")
    initials = str(item_row["BRAND"])[:2].upper()
    st.markdown(
        f"""
        <div class="weekly-row">
            <div class="weekly-product">
                <div class="weekly-avatar">{initials}</div>
                <div>
                    <div class="weekly-name">{item_row['ITEM_NM']}</div>
                    <div class="weekly-meta">{item_row['BRAND']} · {item_row['ITEM_MDDV_NM']} · 출시일 {item_row['NP_RLSE_DATE'].date()}</div>
                </div>
            </div>
            <div>
                <div class="weekly-metric">{format_won(item_row['ST_SLEM_AMT'])}원</div>
                <div class="weekly-submetric">최소주문 {format_int(item_row['MIN_ORD_QTY'])}</div>
            </div>
            <div>
                <div class="weekly-metric">{format_int(item_row['WEEK_RESERVATION_QTY'])}</div>
                <div class="weekly-submetric">예약수량</div>
            </div>
            <div>
                <div class="weekly-metric">{format_int(item_row['WEEK_ORDERING_STORE_CNT'])}</div>
                <div class="weekly-submetric">예약점포수</div>
            </div>
            <div>
                <div class="weekly-status {status_class}">{badge}</div>
                <div class="weekly-submetric">{format_pct(item_row['WEEK_RESERVATION_RATE'] if pd.notna(item_row['WEEK_RESERVATION_RATE']) else 0)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_login_screen() -> None:
    spacer_left, center_col, spacer_right = st.columns([1.45, 1.1, 1.45])
    with center_col:
        st.markdown('<div style="height:11vh"></div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="login-panel">
                <div style="text-align:center;">
                    <div class="login-chip">7-ELEVEN INTERNAL</div>
                </div>
                <div class="login-title">로그인</div>
                <div class="login-sub">담당 MD는 본인 등록 상품만 조회합니다.<br>권한이 있는 계정으로 접속해주세요.</div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            user_id = st.text_input("ID", placeholder="예: hhj30540")
            password = st.text_input("PASSWORD", type="password", placeholder="비밀번호")
            submit = st.form_submit_button("대시보드 입장", width="stretch")

        if submit:
            user_id = user_id.strip().lower()
            password = password.strip()
            valid_ids = st.session_state.get("valid_md_ids", set())
            master_account_password = get_master_account_password()

            if user_id == MASTER_ACCOUNT_ID and password == master_account_password:
                st.session_state["is_logged_in"] = True
                st.session_state["login_user"] = user_id
                st.session_state["is_master_user"] = True
                st.rerun()
            elif user_id in valid_ids and password:
                st.session_state["is_logged_in"] = True
                st.session_state["login_user"] = user_id
                st.session_state["is_master_user"] = False
                st.rerun()
            else:
                st.error("등록된 MD 계정이 아니거나 비밀번호가 입력되지 않았습니다.")

        st.markdown(
            """
            </div>
            """,
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def load_item_md_mapping() -> pd.DataFrame:
    df = pd.read_csv(
        MASTER_ITEM_PATH,
        usecols=["ITEM_CD", "REG_USER_ID", "ITEM_CRTR_CN"],
        dtype="string",
    )
    df["ITEM_CODE"] = df["ITEM_CD"].astype("string").str.strip()
    df["REG_USER_ID"] = df["REG_USER_ID"].astype("string").str.strip().str.lower()
    df["ITEM_CRTR_CN"] = df["ITEM_CRTR_CN"].astype("string").fillna("").str.strip()
    df["ITEM_CRTR_SUMMARY"] = df["ITEM_CRTR_CN"].apply(summarize_item_description)
    df = df.dropna(subset=["ITEM_CODE", "REG_USER_ID"])
    return df[["ITEM_CODE", "REG_USER_ID", "ITEM_CRTR_CN", "ITEM_CRTR_SUMMARY"]].drop_duplicates()


@st.cache_data(show_spinner=False)
def load_preorder() -> pd.DataFrame:
    df = pd.read_csv(PREORDER_PATH)
    numeric_cols = [
        "GOAL_INTRO_RT",
        "MIN_ORD_QTY",
        "INITIAL_ORD_QTY",
        "PRE_D11",
        "total_pre_order_qty(D-11~D-8)",
        "ordering_store_cnt",
        "total_store_cnt",
        "ST_CPM_AMT",
        "ST_SLEM_AMT",
    ] + PREORDER_DAY_COLUMNS
    for col in numeric_cols:
        df[col] = clean_numeric(df[col]).fillna(0)

    df["ITEM_CODE"] = df["ITEM_CODE"].astype(str).str.strip()
    df["CENTER_CODE"] = df["CENTER_CODE"].map(normalize_center_code)
    df["NP_RLSE_DATE"] = pd.to_datetime(df["NP_RLSE_YMD"].astype(str), format="%Y%m%d", errors="coerce")
    df["intro_rate_pct"] = df["GOAL_INTRO_RT"]
    df["reservation_qty_total"] = df["total_pre_order_qty(D-11~D-8)"]
    df["reservation_to_initial_ratio"] = (
        df["reservation_qty_total"] / df["INITIAL_ORD_QTY"].replace(0, pd.NA) * 100
    )
    df["store_participation_pct"] = (
        df["ordering_store_cnt"] / df["total_store_cnt"].replace(0, pd.NA) * 100
    )
    df["gross_margin_rate_pct"] = (
        (df["ST_SLEM_AMT"] - df["ST_CPM_AMT"]) / df["ST_SLEM_AMT"].replace(0, pd.NA) * 100
    )
    df["price_band"] = pd.cut(
        df["ST_SLEM_AMT"],
        bins=[-1, 1000, 1500, 2000, 3000, 5000, float("inf")],
        labels=["~1000", "1001~1500", "1501~2000", "2001~3000", "3001~5000", "5001+"],
    )
    return df


@st.cache_data(show_spinner=False)
def load_sales() -> pd.DataFrame:
    df = pd.read_csv(SALES_PATH)
    df["ITEM_CODE"] = df["ITEM_CD"].astype(str).str.strip()
    df["CENTER_NM"] = df["CENT_NM"].astype(str).str.strip()
    df["SALE_DATE"] = pd.to_datetime(df["판매일자"], errors="coerce")
    df["CENTER_SALE_QTY"] = clean_numeric(df["CENTER_SALE_QTY"]).fillna(0)
    df["CENTER_SALE_AMT_VAT"] = clean_numeric(df["CENTER_SALE_AMT_VAT"]).fillna(0)
    df["Ratio"] = clean_numeric(df["Ratio"]).fillna(0)
    return df


@st.cache_data(show_spinner=False)
def load_stock() -> pd.DataFrame:
    df = pd.read_csv(STOCK_PATH)
    df["ITEM_CODE"] = df["ITEM_CODE"].astype(str).str.strip()
    df["CENTER_CODE"] = df["CENTER_CODE"].map(normalize_center_code)
    df["BIZ_DT"] = pd.to_datetime(df["BIZ_DATE"].astype(str), format="%Y%m%d", errors="coerce")
    df["BOOK_END_QTY"] = clean_numeric(df["BOOK_END_QTY"]).fillna(0)
    return df


@st.cache_data(show_spinner=False)
def load_center_order() -> pd.DataFrame:
    if not CENTER_ORDER_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(CENTER_ORDER_PATH)
    if "ITEM_CD" in df.columns:
        df["ITEM_CODE"] = df["ITEM_CD"].astype(str).str.strip()
    if "CENT_CD" in df.columns:
        df["CENTER_CODE"] = df["CENT_CD"].map(normalize_center_code)
    if "SUM(A.CONV_QTY)" in df.columns:
        df["CONV_QTY"] = clean_numeric(df["SUM(A.CONV_QTY)"]).fillna(0)
    else:
        df["CONV_QTY"] = 0
    if "ORD_YMD" in df.columns:
        df["ORD_DATE"] = pd.to_datetime(df["ORD_YMD"].astype(str), format="%Y%m%d", errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    predictions_path = PREDICTIONS_PATH if PREDICTIONS_PATH.exists() else DATA_DIR / "predictions.parquet"
    if not predictions_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(predictions_path)
    rename_map = {
        "ITEM_CD": "ITEM_CODE",
        "item_cd": "ITEM_CODE",
        "item_code": "ITEM_CODE",
        "CENT_CD": "CENTER_CODE",
        "cent_cd": "CENTER_CODE",
        "CENTER_CD": "CENTER_CODE",
        "center_cd": "CENTER_CODE",
        "center_code": "CENTER_CODE",
        "OUTFLOW_7D": "OUTFLOW_7D",
        "outflow_7d": "OUTFLOW_7D",
    }
    df = df.rename(columns={col: rename_map.get(col, col) for col in df.columns})
    if "OUTFLOW_7D" not in df.columns:
        return pd.DataFrame()
    if "ITEM_CODE" in df.columns:
        df["ITEM_CODE"] = df["ITEM_CODE"].map(normalize_center_code)
    if "CENTER_CODE" in df.columns:
        df["CENTER_CODE"] = df["CENTER_CODE"].map(normalize_center_code)
    df["OUTFLOW_7D"] = clean_numeric(df["OUTFLOW_7D"]).fillna(0)
    return df


def build_outflow_7d_summary(
    predictions_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    if predictions_df.empty or "OUTFLOW_7D" not in predictions_df.columns:
        return pd.DataFrame(columns=group_cols + ["실출고량"])
    missing_cols = [col for col in group_cols if col not in predictions_df.columns]
    if missing_cols:
        return pd.DataFrame(columns=group_cols + ["실출고량"])

    scoped = predictions_df[group_cols + ["OUTFLOW_7D"]].copy()
    if scoped.empty:
        return pd.DataFrame(columns=group_cols + ["실출고량"])

    return (
        scoped.groupby(group_cols, as_index=False)["OUTFLOW_7D"]
        .sum()
        .rename(columns={"OUTFLOW_7D": "실출고량"})
    )


@st.cache_data(show_spinner=False)
def load_center_locations() -> pd.DataFrame:
    if not CENTER_MAP_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(CENTER_MAP_PATH, low_memory=False)
    rename_map = {
        "CENT_CD": "CENTER_CODE",
        "CENT_NM": "CENTER_NM",
        "CENTER_LABEL": "CENTER_LABEL",
        "LAT": "LAT",
        "LON": "LON",
        "STORE_COUNT": "STORE_COUNT",
    }
    df = df.rename(columns=rename_map)
    for col in ["CENTER_CODE", "CENTER_NM", "CENTER_LABEL"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "CENTER_CODE" in df.columns:
        df["CENTER_CODE"] = df["CENTER_CODE"].map(normalize_center_code)
    for col in ["LAT", "LON", "STORE_COUNT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep_cols = [col for col in ["CENTER_CODE", "CENTER_NM", "CENTER_LABEL", "LAT", "LON", "STORE_COUNT"] if col in df.columns]
    return df[keep_cols].drop_duplicates()


@st.cache_data(show_spinner=False)
def build_item_master(preorder_df: pd.DataFrame) -> pd.DataFrame:
    md_map = load_item_md_mapping()
    item_master = (
        preorder_df.sort_values(["ITEM_CODE", "NP_RLSE_DATE"])
        .groupby("ITEM_CODE", as_index=False)
        .agg(
            ITEM_NM=("ITEM_NM", "first"),
            BRAND=("BRAND", "first"),
            ITEM_MDDV_NM=("ITEM_MDDV_NM", "first"),
            ITEM_SMDV_NM=("ITEM_SMDV_NM", "first"),
            NP_RLSE_DATE=("NP_RLSE_DATE", "first"),
            ST_SLEM_AMT=("ST_SLEM_AMT", "first"),
            ST_CPM_AMT=("ST_CPM_AMT", "first"),
            MIN_ORD_QTY=("MIN_ORD_QTY", "first"),
            GOAL_INTRO_RT=("GOAL_INTRO_RT", "first"),
            CENTER_COUNT=("CENTER_CODE", "nunique"),
            TOTAL_INITIAL_ORD_QTY=("INITIAL_ORD_QTY", "sum"),
            TOTAL_RESERVATION_QTY=("reservation_qty_total", "sum"),
            TOTAL_ORDERING_STORE_CNT=("ordering_store_cnt", "sum"),
            TOTAL_STORE_CNT=("total_store_cnt", "sum"),
        )
    )
    item_master["LABEL"] = item_master["ITEM_CODE"] + " | " + item_master["ITEM_NM"]
    item_master["reservation_to_initial_ratio"] = (
        item_master["TOTAL_RESERVATION_QTY"]
        / item_master["TOTAL_INITIAL_ORD_QTY"].replace(0, pd.NA)
        * 100
    )
    item_master = item_master.merge(md_map, on="ITEM_CODE", how="left")
    item_master["REG_USER_ID"] = item_master["REG_USER_ID"].fillna("unassigned")
    return item_master.sort_values(["NP_RLSE_DATE", "ITEM_CODE"], ascending=[False, True])


@st.cache_data(show_spinner=False)
def build_center_master(preorder_df: pd.DataFrame) -> pd.DataFrame:
    return (
        preorder_df[["CENTER_CODE", "CENTER_NM"]]
        .drop_duplicates()
        .sort_values("CENTER_NM")
        .reset_index(drop=True)
    )


def filter_preorder(
    preorder_df: pd.DataFrame,
    selected_items: list[str],
    selected_centers: list[str],
    selected_brands: list[str],
    date_range: tuple,
) -> pd.DataFrame:
    filtered = preorder_df.copy()
    if selected_items:
        filtered = filtered[filtered["ITEM_CODE"].isin(selected_items)]
    if selected_centers:
        filtered = filtered[filtered["CENTER_NM"].isin(selected_centers)]
    if selected_brands:
        filtered = filtered[filtered["BRAND"].isin(selected_brands)]
    start_date, end_date = date_range
    return filtered[
        filtered["NP_RLSE_DATE"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ].copy()


def summarize_kpis(filtered_preorder: pd.DataFrame, filtered_sales: pd.DataFrame, filtered_stock: pd.DataFrame) -> dict:
    total_initial = filtered_preorder["INITIAL_ORD_QTY"].sum()
    total_reservation = filtered_preorder["reservation_qty_total"].sum()
    total_sales_qty = filtered_sales["CENTER_SALE_QTY"].sum()
    total_sales_amt = filtered_sales["CENTER_SALE_AMT_VAT"].sum()
    avg_stock = filtered_stock["BOOK_END_QTY"].mean() if not filtered_stock.empty else 0
    return {
        "상품 수": filtered_preorder["ITEM_CODE"].nunique(),
        "센터 수": filtered_preorder["CENTER_NM"].nunique(),
        "초도 발주량": total_initial,
        "사전 예약량": total_reservation,
        "예약/초도 비율": (total_reservation / total_initial * 100) if total_initial else 0,
        "누적 판매수량": total_sales_qty,
        "누적 판매금액": total_sales_amt,
        "평균 재고": avg_stock,
    }


def build_daily_sales_chart(filtered_sales: pd.DataFrame) -> pd.DataFrame:
    if filtered_sales.empty:
        return pd.DataFrame(columns=["SALE_DATE", "CENTER_NM", "CENTER_SALE_QTY", "CENTER_SALE_AMT_VAT"])
    return (
        filtered_sales.groupby(["SALE_DATE", "CENTER_NM"], as_index=False)
        .agg(
            CENTER_SALE_QTY=("CENTER_SALE_QTY", "sum"),
            CENTER_SALE_AMT_VAT=("CENTER_SALE_AMT_VAT", "sum"),
        )
        .sort_values(["SALE_DATE", "CENTER_NM"])
    )


def build_stock_chart(filtered_stock: pd.DataFrame, center_lookup: pd.DataFrame) -> pd.DataFrame:
    if filtered_stock.empty:
        return pd.DataFrame(columns=["BIZ_DT", "CENTER_NM", "BOOK_END_QTY"])
    chart_df = filtered_stock.merge(center_lookup, on="CENTER_CODE", how="left")
    chart_df["CENTER_NM"] = chart_df["CENTER_NM"].fillna(chart_df["CENTER_CODE"])
    return (
        chart_df.groupby(["BIZ_DT", "CENTER_NM"], as_index=False)["BOOK_END_QTY"]
        .sum()
        .sort_values(["BIZ_DT", "CENTER_NM"])
    )


def build_center_summary(filtered_preorder: pd.DataFrame, filtered_sales: pd.DataFrame, filtered_stock: pd.DataFrame) -> pd.DataFrame:
    preorder_summary = (
        filtered_preorder.groupby(["CENTER_CODE", "CENTER_NM"], as_index=False)
        .agg(
            ITEM_CNT=("ITEM_CODE", "nunique"),
            INITIAL_ORD_QTY=("INITIAL_ORD_QTY", "sum"),
            RESERVATION_QTY=("reservation_qty_total", "sum"),
            ORDERING_STORE_CNT=("ordering_store_cnt", "sum"),
            TOTAL_STORE_CNT=("total_store_cnt", "sum"),
        )
    )
    sales_summary = (
        filtered_sales.groupby("CENTER_NM", as_index=False)
        .agg(
            CENTER_SALE_QTY=("CENTER_SALE_QTY", "sum"),
            CENTER_SALE_AMT_VAT=("CENTER_SALE_AMT_VAT", "sum"),
        )
    )
    stock_summary = (
        filtered_stock.groupby("CENTER_CODE", as_index=False)
        .agg(
            AVG_STOCK_QTY=("BOOK_END_QTY", "mean"),
            LAST_STOCK_QTY=("BOOK_END_QTY", "last"),
        )
    )
    summary = preorder_summary.merge(sales_summary, on="CENTER_NM", how="left").merge(
        stock_summary, on="CENTER_CODE", how="left"
    )
    fill_cols = [
        "CENTER_SALE_QTY",
        "CENTER_SALE_AMT_VAT",
        "AVG_STOCK_QTY",
        "LAST_STOCK_QTY",
    ]
    summary[fill_cols] = summary[fill_cols].fillna(0)
    summary["예약/초도 비율(%)"] = (
        summary["RESERVATION_QTY"] / summary["INITIAL_ORD_QTY"].replace(0, pd.NA) * 100
    )
    summary["예약 참여율(%)"] = (
        summary["ORDERING_STORE_CNT"] / summary["TOTAL_STORE_CNT"].replace(0, pd.NA) * 100
    )
    return summary.sort_values("INITIAL_ORD_QTY", ascending=False)


def build_item_summary(filtered_preorder: pd.DataFrame, filtered_sales: pd.DataFrame) -> pd.DataFrame:
    preorder_summary = (
        filtered_preorder.groupby(["ITEM_CODE", "ITEM_NM", "BRAND", "ITEM_MDDV_NM", "ITEM_SMDV_NM"], as_index=False)
        .agg(
            RELEASE_DATE=("NP_RLSE_DATE", "first"),
            CENTER_CNT=("CENTER_NM", "nunique"),
            INITIAL_ORD_QTY=("INITIAL_ORD_QTY", "sum"),
            RESERVATION_QTY=("reservation_qty_total", "sum"),
            ORDERING_STORE_CNT=("ordering_store_cnt", "sum"),
            TOTAL_STORE_CNT=("total_store_cnt", "sum"),
            AVG_SELL_PRICE=("ST_SLEM_AMT", "mean"),
        )
    )
    sales_summary = (
        filtered_sales.groupby("ITEM_CODE", as_index=False)
        .agg(
            CENTER_SALE_QTY=("CENTER_SALE_QTY", "sum"),
            CENTER_SALE_AMT_VAT=("CENTER_SALE_AMT_VAT", "sum"),
        )
    )
    summary = preorder_summary.merge(sales_summary, on="ITEM_CODE", how="left")
    summary[["CENTER_SALE_QTY", "CENTER_SALE_AMT_VAT"]] = summary[
        ["CENTER_SALE_QTY", "CENTER_SALE_AMT_VAT"]
    ].fillna(0)
    summary["예약/초도 비율(%)"] = (
        summary["RESERVATION_QTY"] / summary["INITIAL_ORD_QTY"].replace(0, pd.NA) * 100
    )
    summary["판매/초도 비율(%)"] = (
        summary["CENTER_SALE_QTY"] / summary["INITIAL_ORD_QTY"].replace(0, pd.NA) * 100
    )
    return summary.sort_values("INITIAL_ORD_QTY", ascending=False)


@st.cache_data(show_spinner=False)
def build_past_reference_item_analysis(
    preorder_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    if preorder_df.empty:
        return pd.DataFrame()

    item_df = (
        preorder_df.groupby("ITEM_CODE", as_index=False)
        .agg(
            ITEM_NM=("ITEM_NM", "first"),
            NP_RLSE_DATE=("NP_RLSE_DATE", "first"),
            BRAND=("BRAND", "first"),
            ITEM_MDDV_NM=("ITEM_MDDV_NM", "first"),
            ITEM_SMDV_NM=("ITEM_SMDV_NM", "first"),
            GOAL_INTRO_RT=("GOAL_INTRO_RT", "first"),
            MIN_ORD_QTY=("MIN_ORD_QTY", "first"),
            ST_SLEM_AMT=("ST_SLEM_AMT", "first"),
            초도발주량=("INITIAL_ORD_QTY", "sum"),
            초기예약발주=("reservation_qty_total", "sum"),
            참여점포수=("ordering_store_cnt", "sum"),
            전체점포수=("total_store_cnt", "sum"),
        )
    )

    if not sales_df.empty:
        sales_summary = (
            sales_df.groupby("ITEM_CODE", as_index=False)["CENTER_SALE_QTY"]
            .sum()
            .rename(columns={"CENTER_SALE_QTY": "실수요량"})
        )
        item_df = item_df.merge(sales_summary, on="ITEM_CODE", how="left")
    item_df["실수요량"] = item_df.get("실수요량", 0).fillna(0)

    if not predictions_df.empty:
        shipped = build_outflow_7d_summary(predictions_df, ["ITEM_CODE"])
        item_df = item_df.merge(shipped, on="ITEM_CODE", how="left")
    item_df["실출고량"] = item_df.get("실출고량", 0).fillna(0)

    safe_initial = item_df["초도발주량"].replace(0, pd.NA)
    item_df["실제출고율(%)"] = (item_df["실출고량"] / safe_initial * 100).round(1)
    item_df["출고율"] = item_df["실출고량"] / safe_initial
    item_df["상태"] = classify_outflow_status(item_df["출고율"])
    item_df["결품여부"] = item_df["상태"].isin(["결품 위험", "결품"])
    item_df["부진여부"] = item_df["상태"].isin(["부진재고", "과발주 위험"])
    return item_df.sort_values(["NP_RLSE_DATE", "초도발주량"], ascending=[False, False])


def build_past_product_dashboard_table(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame()
    return (
        analysis_df.groupby(
            ["ITEM_MDDV_NM", "ITEM_SMDV_NM", "ITEM_CODE", "ITEM_NM"],
            as_index=False,
        )[["preorder_qty", "initial_order_qty", "actual_sales_qty_7d"]]
        .sum()
        .rename(
            columns={
                "ITEM_MDDV_NM": "중분류",
                "ITEM_SMDV_NM": "소분류",
                "ITEM_CODE": "제품코드",
                "ITEM_NM": "제품명",
                "preorder_qty": "예약주문 수",
                "initial_order_qty": "초도발주량",
                "actual_sales_qty_7d": "실수요",
            }
        )
        .sort_values(["중분류", "소분류", "제품명", "제품코드"])
    )


def build_past_center_dashboard_table(analysis_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df.empty:
        return pd.DataFrame()
    return (
        analysis_df.groupby(
            ["CENTER_CODE", "CENTER_NM", "ITEM_MDDV_NM", "ITEM_SMDV_NM", "ITEM_CODE", "ITEM_NM"],
            as_index=False,
        )[["preorder_qty", "initial_order_qty", "actual_sales_qty_7d"]]
        .sum()
        .rename(
            columns={
                "CENTER_CODE": "센터코드",
                "CENTER_NM": "센터",
                "ITEM_MDDV_NM": "중분류",
                "ITEM_SMDV_NM": "소분류",
                "ITEM_CODE": "제품코드",
                "ITEM_NM": "제품명",
                "preorder_qty": "예약주문 수",
                "initial_order_qty": "초도발주량",
                "actual_sales_qty_7d": "실수요",
            }
        )
        .sort_values(["센터코드", "센터", "중분류", "소분류", "제품명", "제품코드"])
    )


def render_past_product_data_tab(
    preorder_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    base_date: pd.Timestamp,
) -> None:
    analysis = build_preorder_sales_analysis(preorder_df, sales_df)
    if analysis.empty:
        st.info("제품별 데이터를 만들기 위한 데이터가 부족합니다.")
        return

    analysis = analysis[analysis["NP_RLSE_DATE"].le(base_date)].copy()
    if analysis.empty:
        st.info("기준일 이전 데이터가 없습니다.")
        return

    filters = st.columns([1, 1, 1, 1, 1.4, 1])
    view_mode = filters[0].selectbox("보기 기준", ["제품별", "센터별"], key="past_product_data_view_mode")
    filters[1].selectbox("대분류", ["과자"], key="past_product_data_top_category")

    selected_center = filters[2].selectbox(
        "센터",
        ["전체"] + sorted(analysis["CENTER_NM"].dropna().astype(str).unique().tolist()),
        key="past_product_data_center",
    )
    if selected_center != "전체":
        analysis = analysis[analysis["CENTER_NM"].astype(str) == selected_center].copy()

    table = (
        build_past_center_dashboard_table(analysis)
        if view_mode == "센터별"
        else build_past_product_dashboard_table(analysis)
    )
    if table.empty:
        st.info("선택한 조건에 맞는 데이터가 없습니다.")
        return

    mddv_options = ["전체"] + sorted(table["중분류"].dropna().astype(str).unique().tolist())
    selected_mddv = filters[3].selectbox("중분류", mddv_options, key="past_product_data_mddv")
    filtered = table.copy()
    if selected_mddv != "전체":
        filtered = filtered[filtered["중분류"].astype(str) == selected_mddv]

    smdv_options = ["전체"] + sorted(filtered["소분류"].dropna().astype(str).unique().tolist())
    selected_smdv = filters[4].selectbox("소분류", smdv_options, key="past_product_data_smdv")
    if selected_smdv != "전체":
        filtered = filtered[filtered["소분류"].astype(str) == selected_smdv]

    keyword = filters[5].text_input(
        "제품 검색",
        placeholder="제품코드 또는 제품명",
        key="past_product_data_keyword",
    )
    if keyword.strip():
        filtered = filtered[
            filtered["제품코드"].astype(str).str.contains(keyword, case=False, na=False)
            | filtered["제품명"].astype(str).str.contains(keyword, case=False, na=False)
        ]

    sort_choice = st.selectbox("정렬 기준", ["예약주문 수", "초도발주량", "실수요"], key="past_product_data_sort")
    filtered = filtered.sort_values(sort_choice, ascending=False)

    summary_cols = st.columns(4)
    summary_cols[0].metric("제품 수", f"{len(filtered):,}")
    summary_cols[1].metric("예약주문 합계", f"{filtered['예약주문 수'].sum():,.0f}")
    summary_cols[2].metric("초도발주 합계", f"{filtered['초도발주량'].sum():,.0f}")
    summary_cols[3].metric("실수요 합계", f"{filtered['실수요'].sum():,.0f}")

    display_columns = ["중분류", "소분류", "제품코드", "제품명", "예약주문 수", "초도발주량", "실수요"]
    if view_mode == "센터별":
        display_columns = ["센터코드", "센터"] + display_columns
    st.dataframe(filtered[display_columns], width="stretch", height=520, hide_index=True)


def render_past_product_lookup(
    preorder_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    center_order_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    base_date: pd.Timestamp,
) -> None:
    st.subheader("과거 신상품 조회")

    item_df = build_past_reference_item_analysis(preorder_df, sales_df, predictions_df)
    if item_df.empty:
        st.info("과거 신상품 조회를 위한 데이터가 부족합니다.")
        return

    item_df = item_df[item_df["NP_RLSE_DATE"].le(base_date)].copy()
    if item_df.empty:
        st.info("기준일 이전 상품이 없습니다.")
        return

    filtered = item_df.copy()
    keyword = st.text_input(
        "키워드 검색",
        key="past_lookup_search",
        placeholder="예: 초코, 감자, 젤리",
    )
    if keyword.strip():
        keyword_mask = (
            filtered["ITEM_NM"].astype(str).str.contains(keyword.strip(), na=False)
            | filtered["ITEM_CODE"].astype(str).str.contains(keyword.strip(), na=False)
        )
        filtered = filtered[keyword_mask]

    item_opts = filtered[["ITEM_CODE", "ITEM_NM"]].drop_duplicates().reset_index(drop=True)
    if item_opts.empty:
        st.info("필터 조건에 해당하는 과거 상품이 없습니다.")
        return

    st.markdown("##### 상품 선택")
    labels = (item_opts["ITEM_NM"] + " [" + item_opts["ITEM_CODE"].astype(str) + "]").tolist()
    sel_idx = st.selectbox(
        "상품 선택",
        options=range(len(labels)),
        format_func=lambda i: labels[i],
        key="past_lookup_item_select",
        label_visibility="collapsed",
    )
    selected_item_code = str(item_opts.loc[sel_idx, "ITEM_CODE"])

    item_pre = preorder_df[
        (preorder_df["ITEM_CODE"].astype(str) == selected_item_code)
        & (preorder_df["NP_RLSE_DATE"].le(base_date))
    ].copy()
    if item_pre.empty:
        st.info("선택한 상품 상세 데이터를 불러오지 못했습니다.")
        return

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("##### 신상품 정보")
        first_row = item_pre.iloc[0]
        info_rows = [
            ("상품코드", str(first_row["ITEM_CODE"])),
            ("상품명", first_row["ITEM_NM"]),
            ("출시일자", first_row["NP_RLSE_DATE"].strftime("%Y-%m-%d") if pd.notna(first_row["NP_RLSE_DATE"]) else "-"),
            ("브랜드", first_row["BRAND"]),
            ("중분류", first_row["ITEM_MDDV_NM"]),
            ("소분류", first_row["ITEM_SMDV_NM"]),
            ("목표도입율", f"{first_row['GOAL_INTRO_RT']:.0f}%"),
            ("최소발주수량", f"{first_row['MIN_ORD_QTY']:.0f}"),
            ("총 초도발주량", f"{item_pre['INITIAL_ORD_QTY'].sum():,.0f}"),
            ("총 사전예약발주", f"{item_pre['reservation_qty_total'].sum():,.0f}"),
        ]
        st.dataframe(
            pd.DataFrame(info_rows, columns=["항목", "내용"]).set_index("항목"),
            width="stretch",
            height=300,
        )

        d_df = pd.DataFrame(
            [
                {"D-day": day_col, "발주수량": float(pd.to_numeric(item_pre[day_col], errors="coerce").sum())}
                for day_col in PREORDER_DAY_COLUMNS
                if day_col in item_pre.columns
            ]
        )
        if not d_df.empty and d_df["발주수량"].sum() > 0:
            fig_d = px.bar(
                d_df,
                x="D-day",
                y="발주수량",
                color_discrete_sequence=["#4f6df5"],
                title="",
            )
            style_figure(fig_d)
            fig_d.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=260)
            st.plotly_chart(fig_d, use_container_width=True, config={})

    with col_r:
        st.markdown("##### 센터별 예약주문 · 초도발주량 · 실수요량")
        center_pre = item_pre[
            ["CENTER_CODE", "CENTER_NM", "INITIAL_ORD_QTY", "reservation_qty_total", "ordering_store_cnt", "total_store_cnt"]
        ].copy()
        center_pre.columns = ["센터코드", "센터명", "초도발주량", "사전예약발주", "참여점포", "전체점포"]
        center_pre["센터코드"] = pd.to_numeric(center_pre["센터코드"], errors="coerce")

        center_ship = pd.DataFrame(columns=["센터코드", "실출고량"])
        if not predictions_df.empty:
            center_ship = build_outflow_7d_summary(predictions_df, ["ITEM_CODE", "CENTER_CODE"])
            center_ship = center_ship[center_ship["ITEM_CODE"].astype(str) == selected_item_code]
            center_ship = center_ship.rename(columns={"CENTER_CODE": "센터코드"})[["센터코드", "실출고량"]]
            center_ship["센터코드"] = pd.to_numeric(center_ship["센터코드"], errors="coerce")

        center_sales = pd.DataFrame(columns=["센터명", "실수요량"])
        if not sales_df.empty:
            center_sales = (
                sales_df[sales_df["ITEM_CODE"].astype(str) == selected_item_code]
                .groupby("CENTER_NM", as_index=False)["CENTER_SALE_QTY"]
                .sum()
                .rename(columns={"CENTER_NM": "센터명", "CENTER_SALE_QTY": "실수요량"})
            )

        center_merged = center_pre.merge(center_ship, on="센터코드", how="left").merge(center_sales, on="센터명", how="left")
        center_merged["실출고량"] = center_merged["실출고량"].fillna(0)
        center_merged["실수요량"] = center_merged["실수요량"].fillna(0)

        st.dataframe(
            center_merged.set_index("센터명")[["사전예약발주", "초도발주량", "실수요량", "참여점포", "전체점포"]],
            width="stretch",
            height=300,
        )

        fig_c = go.Figure()
        fig_c.add_trace(
            go.Bar(name="사전예약발주", x=center_merged["센터명"], y=center_merged["사전예약발주"], marker_color="#4f6df5")
        )
        fig_c.add_trace(
            go.Bar(name="실수요량", x=center_merged["센터명"], y=center_merged["실수요량"], marker_color="#35c8e8")
        )
        fig_c.add_trace(
            go.Bar(name="초도발주량", x=center_merged["센터명"], y=center_merged["초도발주량"], marker_color="#ff4d4f")
        )
        style_figure(fig_c)
        fig_c.update_layout(
            barmode="group",
            margin=dict(l=0, r=0, t=20, b=80),
            showlegend=True,
            height=320,
        )
        st.plotly_chart(fig_c, use_container_width=True, config={})


def render_past_raw_data_tab(
    preorder_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    center_order_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    base_date: pd.Timestamp,
) -> None:
    st.subheader("과거 Raw Data")
    raw_keyword = st.text_input(
        "코드명/제품명 검색",
        key="past_raw_data_search",
        placeholder="제품코드, 코드명 또는 제품명을 입력하세요",
    ).strip()
    datasets = [
        ("센터 발주 Raw", center_order_df, "center_order_filtered.csv"),
        ("센터 재고 Raw", stock_df, "center_stock_filtered.csv"),
        ("매출/수요 Raw", sales_df, "sales_filtered.csv"),
        ("예약주문 Raw", preorder_df, "preorder_filtered.csv"),
    ]
    for title, frame, filename in datasets:
        st.markdown(f"##### {title}")
        if frame.empty:
            st.info(f"{title} 데이터가 없습니다.")
            continue
        filtered = frame.copy()
        if "ORD_YMD" in filtered.columns:
            dates = pd.to_datetime(filtered["ORD_YMD"], errors="coerce", format="%Y%m%d")
            filtered = filtered[dates.isna() | (dates <= base_date)]
        elif "BIZ_DT" in filtered.columns:
            filtered = filtered[filtered["BIZ_DT"].isna() | (filtered["BIZ_DT"] <= base_date)]
        elif "SALE_DATE" in filtered.columns:
            filtered = filtered[filtered["SALE_DATE"].isna() | (filtered["SALE_DATE"] <= base_date)]
        elif "NP_RLSE_DATE" in filtered.columns:
            filtered = filtered[filtered["NP_RLSE_DATE"].isna() | (filtered["NP_RLSE_DATE"] <= base_date)]

        if raw_keyword:
            searchable_columns = [
                col
                for col in ["ITEM_CODE", "ITEM_CD", "ITEM_NM", "상품명", "제품명", "CENTER_CODE", "CENT_CD"]
                if col in filtered.columns
            ]
            if searchable_columns:
                search_mask = pd.Series(False, index=filtered.index)
                for col in searchable_columns:
                    search_mask = search_mask | filtered[col].astype(str).str.contains(
                        raw_keyword, case=False, na=False
                    )
                filtered = filtered[search_mask]

        left, right = st.columns([2.4, 1])
        left.dataframe(filtered.head(200), width="stretch", height=300)
        right.download_button(
            label=f"{filename} 다운로드",
            data=filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name=filename,
            mime="text/csv",
            width="stretch",
            key=f"download_{filename}",
        )
        right.write(f"조회 건수: {len(filtered):,}")
        right.write(f"표시 컬럼 수: {len(filtered.columns):,}")


def render_past_simple_lookup(
    preorder_df: pd.DataFrame,
    center_order_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> None:
    if preorder_df.empty:
        st.info("조회할 과거 상품 데이터가 없습니다.")
        return

    # ── 필터 행 ──────────────────────────────────────────────────────────────
    filters = st.columns([1, 1, 1, 1, 1.4, 1])
    filters[0].selectbox("대분류", ["과자"], key="psl_top_category")

    all_centers = sorted(preorder_df["CENTER_NM"].dropna().astype(str).unique().tolist())
    selected_center = filters[1].selectbox("센터", ["전체"] + all_centers, key="psl_center")

    all_mddv = sorted(preorder_df["ITEM_MDDV_NM"].dropna().unique().tolist())
    selected_mddv = filters[2].selectbox("중분류", ["전체"] + all_mddv, key="psl_mddv")

    mddv_pool = preorder_df if selected_mddv == "전체" else preorder_df[preorder_df["ITEM_MDDV_NM"] == selected_mddv]
    all_smdv = sorted(mddv_pool["ITEM_SMDV_NM"].dropna().unique().tolist())
    selected_smdv = filters[3].selectbox("소분류", ["전체"] + all_smdv, key="psl_smdv")

    keyword = filters[4].text_input("제품 검색", placeholder="제품코드 또는 제품명", key="psl_keyword")

    sort_choice = st.selectbox(
        "정렬 기준",
        ["예약주문 수", "초도발주량", "실수요"],
        key="psl_sort",
    )

    # ── item_df 빌드 ──────────────────────────────────────────────────────────
    item_info = preorder_df.groupby("ITEM_CODE").agg(
        ITEM_NM=("ITEM_NM", "first"),
        NP_RLSE_YMD=("NP_RLSE_YMD", "first"),
        BRAND=("BRAND", "first"),
        ITEM_MDDV_NM=("ITEM_MDDV_NM", "first"),
        ITEM_SMDV_NM=("ITEM_SMDV_NM", "first"),
        GOAL_INTRO_RT=("GOAL_INTRO_RT", "first"),
        MIN_ORD_QTY=("MIN_ORD_QTY", "first"),
        ST_CPM_AMT=("ST_CPM_AMT", "first"),
        ST_SLEM_AMT=("ST_SLEM_AMT", "first"),
        초도발주량=("INITIAL_ORD_QTY", "sum"),
        초기예약발주=("total_pre_order_qty(D-11~D-8)", "sum"),
        참여점포수=("ordering_store_cnt", "sum"),
        전체점포수=("total_store_cnt", "sum"),
    ).reset_index()

    if not predictions_df.empty:
        item_orders = build_outflow_7d_summary(predictions_df, ["ITEM_CODE"])
    else:
        item_orders = pd.DataFrame(columns=["ITEM_CODE", "실출고량"])

    if not sales_df.empty:
        item_sales = (
            sales_df.groupby("ITEM_CODE")[["CENTER_SALE_QTY", "CENTER_SALE_AMT_VAT"]].sum()
            .reset_index().rename(columns={"CENTER_SALE_QTY": "실수요", "CENTER_SALE_AMT_VAT": "실수요금액"})
        )
    else:
        item_sales = pd.DataFrame(columns=["ITEM_CODE", "실수요", "실수요금액"])

    item_df = item_info.merge(item_orders, on="ITEM_CODE", how="left").merge(item_sales, on="ITEM_CODE", how="left")
    for col in ["실출고량", "실수요", "실수요금액"]:
        item_df[col] = item_df[col].fillna(0)

    safe_initial = item_df["초도발주량"].replace(0, pd.NA)
    item_df["실출고율(%)"] = (item_df["실출고량"] / safe_initial * 100).round(1)
    item_df["출고율"] = item_df["실출고량"] / safe_initial
    item_df["실수요비율(%)"] = (item_df["실수요"] / safe_initial * 100).round(1)
    item_df["상태"] = classify_outflow_status(item_df["출고율"])
    item_df["결품여부"] = item_df["상태"].isin(["결품 위험", "결품"])
    item_df["부진여부"] = item_df["상태"].isin(["부진재고", "과발주 위험"])
    item_df["출시일자"] = pd.to_datetime(item_df["NP_RLSE_YMD"].astype(str), format="%Y%m%d", errors="coerce")

    # ── 필터 적용 ─────────────────────────────────────────────────────────────
    filtered = item_df.copy()
    if selected_center != "전체":
        items_in_center = preorder_df[preorder_df["CENTER_NM"].astype(str) == selected_center]["ITEM_CODE"].unique()
        filtered = filtered[filtered["ITEM_CODE"].isin(items_in_center)]
    if selected_mddv != "전체":
        filtered = filtered[filtered["ITEM_MDDV_NM"] == selected_mddv]
    if selected_smdv != "전체":
        filtered = filtered[filtered["ITEM_SMDV_NM"] == selected_smdv]
    if keyword.strip():
        kw = keyword.strip()
        filtered = filtered[
            filtered["ITEM_CODE"].astype(str).str.contains(kw, case=False, na=False)
            | filtered["ITEM_NM"].astype(str).str.contains(kw, case=False, na=False)
        ]

    sort_col_map = {"예약주문 수": "초기예약발주", "초도발주량": "초도발주량", "실수요": "실수요"}

    # ══ SECTION 1: Raw Data Explorer ══════════════════════════════════════════
    st.markdown("---")
    st.markdown("#### Raw Data Explorer")
    st.markdown("##### 중/소분류별 초도발주량 · 실출고량(7일치) · 실수요")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("제품 수", f"{len(filtered):,} 종")
    k2.metric("총 초도발주량", f"{filtered['초도발주량'].sum():,.0f}")
    k3.metric("총 실출고량(7일치)", f"{filtered['실출고량'].sum():,.0f}")
    k4.metric("총 실수요", f"{filtered['실수요'].sum():,.0f}")

    tbl = filtered[[
        "ITEM_CODE", "ITEM_NM", "출시일자", "BRAND", "ITEM_MDDV_NM", "ITEM_SMDV_NM",
        "초도발주량", "초기예약발주", "실출고량", "실수요", "실출고율(%)",
    ]].copy()
    tbl = tbl.sort_values(sort_col_map[sort_choice], ascending=False)
    tbl["출시일자"] = tbl["출시일자"].dt.strftime("%Y-%m-%d")
    tbl = tbl.rename(columns={
        "ITEM_CODE": "제품코드", "ITEM_NM": "제품명", "BRAND": "브랜드",
        "ITEM_MDDV_NM": "중분류", "ITEM_SMDV_NM": "소분류",
        "실출고량": "실출고량(7일치)", "실출고율(%)": "실출고율(7일치)(%)",
    })
    st.dataframe(
        tbl.set_index("제품코드"),
        use_container_width=True,
        height=360,
        column_config={
            "초도발주량": st.column_config.NumberColumn(format="%,.0f"),
            "초기예약발주": st.column_config.NumberColumn(format="%,.0f"),
            "실출고량(7일치)": st.column_config.NumberColumn(format="%,.0f"),
            "실수요": st.column_config.NumberColumn(format="%,.0f"),
            "실출고율(7일치)(%)": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    csv_bytes = tbl.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("CSV 다운로드", data=csv_bytes, file_name="신상품_이동분석.csv", mime="text/csv")

    # ══ SECTION 2: 신상품 정보표 및 센터별 출고 현황 ══════════════════════════
    st.markdown("---")
    st.markdown("#### 신상품 정보표 및 센터별 출고 현황")

    item_opts = filtered[["ITEM_CODE", "ITEM_NM"]].drop_duplicates().reset_index(drop=True)
    if item_opts.empty:
        st.info("필터 조건에 해당하는 상품이 없습니다.")
    else:
        item_labels = (item_opts["ITEM_NM"] + "  [" + item_opts["ITEM_CODE"] + "]").tolist()
        sel_idx = st.selectbox(
            "상품 선택",
            options=range(len(item_labels)),
            format_func=lambda i: item_labels[i],
            key="psl_item_select",
        )
        sel_code = item_opts.loc[sel_idx, "ITEM_CODE"]

        col_l, col_r = st.columns(2, gap="large")

        with col_l:
            st.markdown("##### 신상품 정보표")
            item_pre = preorder_df[preorder_df["ITEM_CODE"] == sel_code].copy()
            if not item_pre.empty:
                b = item_pre.iloc[0]
                launch_dt = pd.to_datetime(str(int(float(b["NP_RLSE_YMD"]))), format="%Y%m%d", errors="coerce")
                info_rows = [
                    ("상품코드", str(b["ITEM_CODE"])),
                    ("상품명", b["ITEM_NM"]),
                    ("출시일자", launch_dt.strftime("%Y-%m-%d") if pd.notna(launch_dt) else "-"),
                    ("브랜드", b["BRAND"]),
                    ("중분류", b["ITEM_MDDV_NM"]),
                    ("소분류", b["ITEM_SMDV_NM"]),
                    ("목표도입율", f"{b['GOAL_INTRO_RT']:.0f}%"),
                    ("최소발주수량", f"{b['MIN_ORD_QTY']:.0f}"),
                    ("점포 원가", f"{b['ST_CPM_AMT']:,.0f} 원"),
                    ("점포 매가", f"{b['ST_SLEM_AMT']:,.0f} 원"),
                    ("총 초도발주량", f"{item_pre['INITIAL_ORD_QTY'].sum():,.0f}"),
                    ("총 사전예약발주", f"{item_pre['total_pre_order_qty(D-11~D-8)'].sum():,.0f}"),
                ]
                info_df = pd.DataFrame(info_rows, columns=["항목", "내용"])
                st.dataframe(info_df.set_index("항목"), use_container_width=True, height=410)

                d_vals = [
                    {"D-day": dc, "발주수량": float(pd.to_numeric(item_pre[dc], errors="coerce").sum())}
                    for dc in PREORDER_DAY_COLUMNS
                    if dc in item_pre.columns
                ]
                d_df = pd.DataFrame(d_vals)
                if not d_df.empty and d_df["발주수량"].sum() > 0:
                    st.markdown("**D-day별 사전예약 발주 수량 (전체 센터 합계)**")
                    fig_d = px.bar(
                        d_df, x="D-day", y="발주수량",
                        color_discrete_sequence=["#4f6df5"], height=220,
                        labels={"발주수량": "수량"},
                    )
                    fig_d.update_layout(
                        margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        font_size=11, showlegend=False,
                    )
                    fig_d.update_xaxes(showgrid=False)
                    fig_d.update_yaxes(gridcolor="#F1F5F9", tickformat=",")
                    st.plotly_chart(fig_d, use_container_width=True)

        with col_r:
            st.markdown("##### 센터별 초도발주량 및 실 출고량")
            center_pre = item_pre[[
                "CENTER_CODE", "CENTER_NM", "INITIAL_ORD_QTY",
                "total_pre_order_qty(D-11~D-8)", "ordering_store_cnt", "total_store_cnt",
                *[col for col in PREORDER_DAY_COLUMNS if col in item_pre.columns],
            ]].copy()
            center_pre = center_pre.rename(
                columns={
                    "CENTER_CODE": "센터코드",
                    "CENTER_NM": "센터명",
                    "INITIAL_ORD_QTY": "초도발주량",
                    "total_pre_order_qty(D-11~D-8)": "4일치 예약주문",
                    "ordering_store_cnt": "참여점포",
                    "total_store_cnt": "전체점포",
                }
            )
            ten_day_cols = [col for col in PREORDER_DAY_COLUMNS[:10] if col in center_pre.columns]
            center_pre["10일치 예약주문"] = center_pre[ten_day_cols].sum(axis=1) if ten_day_cols else center_pre["4일치 예약주문"]
            center_pre = center_pre[["센터코드", "센터명", "초도발주량", "4일치 예약주문", "10일치 예약주문", "참여점포", "전체점포"]]
            center_pre["센터코드"] = center_pre["센터코드"].map(normalize_center_code)

            if not predictions_df.empty:
                c_orders = build_outflow_7d_summary(predictions_df, ["ITEM_CODE", "CENTER_CODE"])
                c_orders = c_orders[c_orders["ITEM_CODE"].astype(str) == str(sel_code)]
                c_orders = c_orders.rename(columns={"CENTER_CODE": "센터코드"})[["센터코드", "실출고량"]]
                c_orders["센터코드"] = c_orders["센터코드"].map(normalize_center_code)
            else:
                c_orders = pd.DataFrame(columns=["센터코드", "실출고량"])

            if not sales_df.empty:
                c_sales = (
                    sales_df[sales_df["ITEM_CODE"] == sel_code]
                    .groupby("CENTER_NM")["CENTER_SALE_QTY"].sum()
                    .reset_index().rename(columns={"CENTER_NM": "센터명", "CENTER_SALE_QTY": "실수요"})
                )
            else:
                c_sales = pd.DataFrame(columns=["센터명", "실수요"])

            c_merged = center_pre.merge(c_orders, on="센터코드", how="left").merge(c_sales, on="센터명", how="left")
            c_merged["실출고량"] = c_merged["실출고량"].fillna(0)
            c_merged["실수요"] = c_merged["실수요"].fillna(0)
            c_merged["실수요량(또는 실출고량(7일치))"] = np.where(
                c_merged["실수요"] > 0,
                c_merged["실수요"],
                c_merged["실출고량"],
            )
            safe_init = c_merged["초도발주량"].replace(0, pd.NA)
            c_merged["출고율(7일치)(%)"] = (c_merged["실출고량"] / safe_init * 100).round(1)

            st.dataframe(
                c_merged.set_index("센터명")[["4일치 예약주문", "10일치 예약주문", "초도발주량", "실출고량", "출고율(7일치)(%)", "실수요", "참여점포", "전체점포"]],
                use_container_width=True,
                height=300,
                column_config={
                    "초도발주량": st.column_config.NumberColumn(format="%,.0f"),
                    "4일치 예약주문": st.column_config.NumberColumn(format="%,.0f"),
                    "10일치 예약주문": st.column_config.NumberColumn(format="%,.0f"),
                    "실출고량": st.column_config.NumberColumn("실출고량(7일치)", format="%,.0f"),
                    "실수요": st.column_config.NumberColumn(format="%,.0f"),
                    "출고율(7일치)(%)": st.column_config.NumberColumn(format="%.1f"),
                    "참여점포": st.column_config.NumberColumn(format="%,.0f"),
                    "전체점포": st.column_config.NumberColumn(format="%,.0f"),
                },
            )

            if not c_merged.empty:
                fig_c = go.Figure()
                fig_c.add_trace(go.Bar(name="4일치 예약주문", x=c_merged["센터명"], y=c_merged["4일치 예약주문"], marker_color="#A7F3D0"))
                fig_c.add_trace(go.Bar(name="10일치 예약주문", x=c_merged["센터명"], y=c_merged["10일치 예약주문"], marker_color="#34D399"))
                fig_c.add_trace(go.Bar(name="초도발주량", x=c_merged["센터명"], y=c_merged["초도발주량"], marker_color="#7b4cf3"))
                fig_c.add_trace(go.Bar(name="실수요량(또는 실출고량(7일치))", x=c_merged["센터명"], y=c_merged["실수요량(또는 실출고량(7일치))"], marker_color="#35c8e8"))
                fig_c.update_layout(
                    barmode="group", height=290,
                    margin=dict(l=0, r=0, t=20, b=80),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    font_size=10,
                )
                fig_c.update_xaxes(tickangle=-40, showgrid=False)
                fig_c.update_yaxes(gridcolor="#F1F5F9", tickformat=",")
                st.plotly_chart(fig_c, use_container_width=True)

    # ══ SECTION 3: 분석 자료 ══════════════════════════════════════════
    st.markdown("---")
    st.markdown("#### 분석 자료")

    sub1, sub2 = st.tabs(["상품별 분석", "중/소분류 집계"])

    with sub1:
        st.markdown("#### 상품별 상태 분석")

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("전체 상품", f"{len(filtered):,} 종")
        r2.metric("정상", f"{(filtered['상태'] == '정상').sum():,} 종")
        r3.metric("결품/위험", f"{filtered['상태'].isin(['결품 위험', '결품']).sum():,} 종")
        r4.metric("과발주/부진", f"{filtered['상태'].isin(['부진재고', '과발주 위험']).sum():,} 종")
        avg_exit_rt = filtered["실출고율(%)"].mean()
        r5.metric("평균 출고율(7일치)", f"{avg_exit_rt:.1f} %" if pd.notna(avg_exit_rt) else "-")

        status_filter = st.radio("상태 필터", ["전체", *OUTFLOW_STATUS_ORDER], horizontal=True, key="psl_status_filter")
        res_df = filtered.copy()
        if status_filter != "전체":
            res_df = res_df[res_df["상태"] == status_filter]

        res_tbl = res_df[[
            "ITEM_CODE", "ITEM_NM", "출시일자", "BRAND", "ITEM_MDDV_NM", "ITEM_SMDV_NM",
            "초도발주량", "초기예약발주", "실출고량", "실출고율(%)", "실수요",
            "결품여부", "부진여부", "상태",
        ]].rename(columns={
            "ITEM_CODE": "제품코드", "ITEM_NM": "제품명", "BRAND": "브랜드",
            "ITEM_MDDV_NM": "중분류", "ITEM_SMDV_NM": "소분류",
            "결품여부": "결품/위험", "부진여부": "과발주/부진",
            "실출고량": "실출고량(7일치)", "실출고율(%)": "실출고율(7일치)(%)",
        }).copy()
        res_tbl["출시일자"] = res_tbl["출시일자"].dt.strftime("%Y-%m-%d")
        res_tbl["결품/위험"] = res_tbl["결품/위험"].map({True: "Y", False: "-"})
        res_tbl["과발주/부진"] = res_tbl["과발주/부진"].map({True: "Y", False: "-"})

        st.dataframe(
            res_tbl.set_index("제품코드"),
            use_container_width=True,
            height=420,
            column_config={
                "초도발주량": st.column_config.NumberColumn(format="%,.0f"),
                "초기예약발주": st.column_config.NumberColumn(format="%,.0f"),
                "실출고량(7일치)": st.column_config.NumberColumn(format="%,.0f"),
                "실수요": st.column_config.NumberColumn(format="%,.0f"),
                "실출고율(7일치)(%)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        res_csv = res_tbl.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("상품별 결과 CSV 다운로드", data=res_csv, file_name="상품별_이동결과.csv", mime="text/csv")

    with sub2:
        st.markdown("#### 중/소분류 단위 집계")

        agg_level = st.radio("집계 기준", ["중분류", "소분류"], horizontal=True, key="psl_agg_level")
        agg_col = "ITEM_MDDV_NM" if agg_level == "중분류" else "ITEM_SMDV_NM"

        cat_agg = filtered.groupby(agg_col).agg(
            상품수=("ITEM_CODE", "count"),
            총초도발주량=("초도발주량", "sum"),
            총초기예약발주=("초기예약발주", "sum"),
            총실출고량=("실출고량", "sum"),
            총실수요=("실수요", "sum"),
            결품상품수=("결품여부", "sum"),
            부진상품수=("부진여부", "sum"),
        ).reset_index()
        cat_agg.columns = [agg_level, "상품수", "총 초도발주량", "총 초기예약발주", "총 실출고량(7일치)", "총 실수요", "결품/위험 상품수", "과발주/부진 상품수"]
        safe_tot = cat_agg["총 초도발주량"].replace(0, pd.NA)
        cat_agg["평균 출고율(7일치)(%)"] = (cat_agg["총 실출고량(7일치)"] / safe_tot * 100).round(1)

        st.dataframe(
            cat_agg.set_index(agg_level),
            use_container_width=True,
            height=320,
            column_config={
                "총 초도발주량": st.column_config.NumberColumn(format="%,.0f"),
                "총 초기예약발주": st.column_config.NumberColumn(format="%,.0f"),
                "총 실출고량(7일치)": st.column_config.NumberColumn(format="%,.0f"),
                "총 실수요": st.column_config.NumberColumn(format="%,.0f"),
                "평균 출고율(7일치)(%)": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        cat_csv = cat_agg.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("집계 결과 CSV 다운로드", data=cat_csv, file_name=f"{agg_level}_이동결과집계.csv", mime="text/csv")

        ch1, ch2 = st.columns(2, gap="large")

        with ch1:
            st.markdown(f"**{agg_level}별 초도발주량 / 실출고량(7일치) / 실수요**")
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(name="초도발주량", x=cat_agg[agg_level], y=cat_agg["총 초도발주량"], marker_color="#BFDBFE"))
            fig_bar.add_trace(go.Bar(name="실출고량(7일치)", x=cat_agg[agg_level], y=cat_agg["총 실출고량(7일치)"], marker_color="#2563EB"))
            fig_bar.add_trace(go.Bar(name="실수요", x=cat_agg[agg_level], y=cat_agg["총 실수요"], marker_color="#35c8e8"))
            fig_bar.update_layout(
                barmode="group", height=340,
                margin=dict(l=0, r=0, t=20, b=60),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                font_size=11,
            )
            fig_bar.update_xaxes(showgrid=False)
            fig_bar.update_yaxes(gridcolor="#F1F5F9", tickformat=",")
            st.plotly_chart(fig_bar, use_container_width=True)

        with ch2:
            st.markdown(f"**{agg_level}별 출고율 상태 현황**")

            def _cnt(grp, state):
                return grp.apply(lambda x: (x["상태"] == state).sum()).reset_index()

            norm_c = _cnt(filtered.groupby(agg_col), "정상"); norm_c.columns = [agg_level, "정상"]
            slow_stock_c = _cnt(filtered.groupby(agg_col), "부진재고"); slow_stock_c.columns = [agg_level, "부진재고"]
            over_c = _cnt(filtered.groupby(agg_col), "과발주 위험"); over_c.columns = [agg_level, "과발주 위험"]
            risk_c = _cnt(filtered.groupby(agg_col), "결품 위험"); risk_c.columns = [agg_level, "결품 위험"]
            shortage_c = _cnt(filtered.groupby(agg_col), "결품"); shortage_c.columns = [agg_level, "결품"]
            status_df_chart = (
                slow_stock_c.merge(over_c, on=agg_level)
                .merge(norm_c, on=agg_level)
                .merge(risk_c, on=agg_level)
                .merge(shortage_c, on=agg_level)
            )

            fig_s = go.Figure()
            for status in OUTFLOW_STATUS_ORDER:
                fig_s.add_trace(
                    go.Bar(
                        name=status,
                        x=status_df_chart[agg_level],
                        y=status_df_chart[status],
                        marker_color=OUTFLOW_STATUS_COLORS[status],
                    )
                )
            fig_s.update_layout(
                barmode="stack", height=340,
                margin=dict(l=0, r=0, t=20, b=60),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                font_size=11,
            )
            fig_s.update_xaxes(showgrid=False)
            fig_s.update_yaxes(gridcolor="#F1F5F9")
            st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("**상품별 OPTIMAL(OUTFLOW_7D) vs MD 실제 초도발주량 (로그 스케일)**")
        scatter_df = filtered.copy()
        scatter_df["출시일자_str"] = scatter_df["출시일자"].dt.strftime("%Y-%m-%d")
        scatter_df = scatter_df[(scatter_df["초도발주량"] > 0) & (scatter_df["실출고량"] > 0)].copy()
        if scatter_df.empty:
            st.info("로그 스케일 그래프를 그릴 양수 데이터가 없습니다.")
        else:
            scatter_df["MD/OPTIMAL 배수"] = scatter_df["초도발주량"] / scatter_df["실출고량"]
            fig_sc = px.scatter(
                scatter_df,
                x="실출고량", y="초도발주량",
                color="상태",
                category_orders={"상태": OUTFLOW_STATUS_ORDER},
                color_discrete_map=OUTFLOW_STATUS_COLORS,
                hover_data={
                    "ITEM_NM": True, "출시일자_str": True,
                    "ITEM_MDDV_NM": True, "ITEM_SMDV_NM": True,
                    "실출고량": ":,.0f", "초도발주량": ":,.0f", "MD/OPTIMAL 배수": ":.2f", "실출고율(%)": ":.1f", "상태": True,
                },
                labels={
                    "ITEM_NM": "제품명", "출시일자_str": "출시일자",
                    "ITEM_MDDV_NM": "중분류", "ITEM_SMDV_NM": "소분류",
                    "실출고량": "OPTIMAL = OUTFLOW_7D (log)",
                    "초도발주량": "MD 실제 초도발주량 (log)",
                },
                height=420,
            )
            min_axis = max(1, min(scatter_df["실출고량"].min(), scatter_df["초도발주량"].min()))
            max_axis = max(scatter_df["실출고량"].max(), scatter_df["초도발주량"].max())
            line_x = np.geomspace(min_axis, max_axis, 80)
            for label, multiplier, dash, color in [
                ("y=x (이상)", 1.0, "dash", "#2D3748"),
                ("MD = 2 x OPTIMAL (과발주)", 2.0, "dot", "#9BA6BD"),
                ("MD = 0.5 x OPTIMAL (결품)", 0.5, "dot", "#9BA6BD"),
            ]:
                fig_sc.add_trace(
                    go.Scatter(
                        x=line_x,
                        y=line_x * multiplier,
                        mode="lines",
                        line=dict(color=color, dash=dash, width=1.4),
                        name=label,
                        hoverinfo="skip",
                    )
                )
            fig_sc.update_layout(
                margin=dict(l=0, r=0, t=20, b=20),
                plot_bgcolor="white", paper_bgcolor="white",
                font_size=11,
                annotations=[
                    dict(
                        x=0.08,
                        y=0.92,
                        xref="paper",
                        yref="paper",
                        text="과발주 영역<br>(MD > OPTIMAL)",
                        showarrow=False,
                        align="left",
                        bgcolor="rgba(255,255,255,0.78)",
                        bordercolor="#94A3B8",
                        borderwidth=1,
                        font=dict(color="#475569", size=11),
                    ),
                    dict(
                        x=0.92,
                        y=0.08,
                        xref="paper",
                        yref="paper",
                        text="결품 영역<br>(MD < OPTIMAL)",
                        showarrow=False,
                        align="right",
                        bgcolor="rgba(255,255,255,0.78)",
                        bordercolor="#EF4444",
                        borderwidth=1,
                        font=dict(color="#DC2626", size=11),
                    ),
                ],
            )
            fig_sc.update_xaxes(type="log", gridcolor="#F1F5F9", tickformat=",")
            fig_sc.update_yaxes(type="log", gridcolor="#F1F5F9", tickformat=",")
            st.plotly_chart(fig_sc, use_container_width=True)
            st.markdown(
                "<span style='font-size:0.72rem;color:#94A3B8;'>"
                "X축은 OPTIMAL = OUTFLOW_7D, Y축은 MD 실제 초도발주량입니다. "
                "y=x는 이상선, y=2x는 과발주 기준, y=0.5x는 결품 기준입니다."
                "</span>",
                unsafe_allow_html=True,
            )


def render_past_category_compare(preorder_df: pd.DataFrame, sales_df: pd.DataFrame, base_date: pd.Timestamp) -> None:
    st.subheader("카테고리")
    st.caption("중/소분류별 예약주문, 초도발주, 실수요 흐름을 요약합니다.")

    analysis = build_preorder_sales_analysis(preorder_df, sales_df)
    if analysis.empty:
        st.info("카테고리 데이터를 만들기 위한 데이터가 부족합니다.")
        return
    analysis = analysis[analysis["NP_RLSE_DATE"].le(base_date)].copy()
    if analysis.empty:
        st.info("기준일 이전 데이터가 없습니다.")
        return

    category_level = st.radio("집계 기준", ["중분류", "소분류"], horizontal=True, key="past_category_level")
    group_cols = ["ITEM_MDDV_NM"] if category_level == "중분류" else ["ITEM_MDDV_NM", "ITEM_SMDV_NM"]
    category_summary = (
        analysis.groupby(group_cols, as_index=False)
        .agg(
            상품수=("ITEM_CODE", "nunique"),
            예약주문량=("preorder_qty", "sum"),
            초도발주량=("initial_order_qty", "sum"),
            실수요량=("actual_sales_qty_7d", "sum"),
        )
        .rename(columns={"ITEM_MDDV_NM": "중분류", "ITEM_SMDV_NM": "소분류"})
    )
    category_summary["과발주 갭"] = category_summary["초도발주량"] - category_summary["실수요량"]

    metrics = category_summary[["예약주문량", "초도발주량", "실수요량", "과발주 갭"]].sum()
    top_cols = st.columns(4)
    top_cols[0].metric("예약주문량", f"{metrics['예약주문량']:,.0f}")
    top_cols[1].metric("초도발주량", f"{metrics['초도발주량']:,.0f}")
    top_cols[2].metric("매출 기반 실수요량", f"{metrics['실수요량']:,.0f}")
    top_cols[3].metric("과발주 갭", f"{metrics['과발주 갭']:,.0f}")

    category_label_col = "중분류" if category_level == "중분류" else "소분류"
    chart_source = category_summary.sort_values("초도발주량", ascending=False).head(20)
    fig = go.Figure()
    fig.add_trace(go.Bar(name="예약주문량", x=chart_source[category_label_col], y=chart_source["예약주문량"], marker_color="#34D399"))
    fig.add_trace(go.Bar(name="초도발주량", x=chart_source[category_label_col], y=chart_source["초도발주량"], marker_color="#7b4cf3"))
    fig.add_trace(go.Bar(name="매출 기반 실수요량", x=chart_source[category_label_col], y=chart_source["실수요량"], marker_color="#35c8e8"))
    fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=0, r=0, t=20, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font_size=11,
    )
    fig.update_xaxes(tickangle=-35, showgrid=False)
    fig.update_yaxes(gridcolor="#F1F5F9", tickformat=",")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        category_summary.sort_values("초도발주량", ascending=False),
        width="stretch",
        height=360,
        hide_index=True,
        column_config={
            "상품수": st.column_config.NumberColumn(format="%,.0f"),
            "예약주문량": st.column_config.NumberColumn(format="%,.0f"),
            "초도발주량": st.column_config.NumberColumn(format="%,.0f"),
            "실수요량": st.column_config.NumberColumn(format="%,.0f"),
            "과발주 갭": st.column_config.NumberColumn(format="%,.0f"),
        },
    )


def render_past_current_release_focus(preorder_df: pd.DataFrame, base_date: pd.Timestamp) -> None:
    st.subheader("기준일 시점 신상품 관점")
    release_window = preorder_df[
        preorder_df["NP_RLSE_DATE"].between(base_date - pd.Timedelta(days=31), base_date, inclusive="both")
    ].copy()
    release_window = release_window.sort_values("NP_RLSE_DATE", ascending=False)
    if release_window.empty:
        st.info("기준일 시점에 해당하는 신상품이 없습니다.")
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("기준월 신상품 행 수", f"{len(release_window):,}")
    summary_cols[1].metric("기준월 상품 수", f"{release_window['ITEM_CODE'].nunique():,}")
    summary_cols[2].metric("센터 수", f"{release_window['CENTER_CODE'].nunique():,}")
    summary_cols[3].metric("초도발주 합계", f"{release_window['INITIAL_ORD_QTY'].fillna(0).sum():,.0f}")

    show_columns = [
        "NP_RLSE_DATE",
        "ITEM_CODE",
        "ITEM_NM",
        "CENTER_NM",
        "BRAND",
        "ITEM_MDDV_NM",
        "ITEM_SMDV_NM",
        "reservation_qty_total",
        "INITIAL_ORD_QTY",
    ]
    show_columns = [column for column in show_columns if column in release_window.columns]
    st.dataframe(release_window[show_columns], width="stretch", height=360, hide_index=True)


def render_past_dashboard_page(
    preorder_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    center_order_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    base_date: pd.Timestamp,
) -> None:
    st.markdown("## 과거 신상품 조회")
    tabs = st.tabs(
        ["제품별 데이터", "과거 Raw Data", "과거 신상품 조회", "카테고리", "기준일 신상품"]
    )
    with tabs[0]:
        render_past_product_data_tab(preorder_df, sales_df, base_date)
    with tabs[1]:
        render_past_raw_data_tab(preorder_df, sales_df, center_order_df, stock_df, base_date)
    with tabs[2]:
        render_past_simple_lookup(preorder_df, center_order_df, sales_df, predictions_df)
    with tabs[3]:
        render_past_category_compare(preorder_df, sales_df, base_date)
    with tabs[4]:
        render_past_current_release_focus(preorder_df, base_date)


preorder_df = load_preorder()
sales_df = load_sales()
stock_df = load_stock()
center_order_df = load_center_order()
predictions_df = load_predictions()
item_master = build_item_master(preorder_df)
center_master = build_center_master(preorder_df)

full_preorder_df = preorder_df.copy()
full_sales_df = sales_df.copy()
full_stock_df = stock_df.copy()
full_center_order_df = center_order_df.copy()
full_predictions_df = predictions_df.copy()
full_item_master = item_master.copy()

inject_theme()

if "is_logged_in" not in st.session_state:
    st.session_state["is_logged_in"] = False
if "is_master_user" not in st.session_state:
    st.session_state["is_master_user"] = False
if st.session_state.get("app_session_version") != APP_SESSION_VERSION:
    st.session_state["app_session_version"] = APP_SESSION_VERSION
    st.session_state["is_logged_in"] = False
    st.session_state["is_master_user"] = False
    st.session_state.pop("login_user", None)
    st.session_state.pop("weekly_selected_item", None)

st.session_state["valid_md_ids"] = set(
    item_master.loc[item_master["REG_USER_ID"].ne("unassigned"), "REG_USER_ID"].dropna().tolist()
)

if not st.session_state["is_logged_in"]:
    render_login_screen()
    st.stop()

logged_user = st.session_state.get("login_user", "").strip().lower()
is_master_user = st.session_state.get("is_master_user", False)

if not is_master_user:
    allowed_items = item_master[item_master["REG_USER_ID"] == logged_user]["ITEM_CODE"].unique().tolist()
    item_master = item_master[item_master["ITEM_CODE"].isin(allowed_items)].copy()
    preorder_df = preorder_df[preorder_df["ITEM_CODE"].isin(allowed_items)].copy()
    sales_df = sales_df[sales_df["ITEM_CODE"].isin(allowed_items)].copy()
    stock_df = stock_df[stock_df["ITEM_CODE"].isin(allowed_items)].copy()
    center_master = build_center_master(preorder_df)

if item_master.empty:
    st.error("현재 로그인한 ID에 연결된 상품이 없습니다. REG_USER_ID 기준 매핑을 확인해주세요.")
    if st.button("로그아웃", width="stretch"):
        st.session_state["is_logged_in"] = False
        st.session_state["is_master_user"] = False
        st.session_state.pop("login_user", None)
        st.rerun()
    st.stop()

topbar_left, topbar_right = st.columns([1, 0.35])
with topbar_left:
    role_label = "MASTER" if is_master_user else "MD"
    st.caption(f"접속 사용자: `{st.session_state.get('login_user', 'unknown')}` | 권한: `{role_label}`")
with topbar_right:
    if st.button("로그아웃", width="stretch"):
        st.session_state["is_logged_in"] = False
        st.session_state["is_master_user"] = False
        st.session_state.pop("login_user", None)
        st.rerun()

page_options = ["금주 신상품", "과거 신상품 조회"]
with st.sidebar:
    st.header("메뉴")
    selected_page = st.radio(
        "최상위 탭",
        options=page_options,
        key="app_page_selector",
        label_visibility="collapsed",
    )
    st.divider()

min_release = item_master["NP_RLSE_DATE"].min().date()
max_release = item_master["NP_RLSE_DATE"].max().date()

if selected_page == "과거 신상품 조회":
    with st.sidebar:
        st.header("조회 기준")
        past_base_date = pd.Timestamp(
            st.date_input(
                "기준일",
                value=max_release,
                min_value=min_release,
                max_value=max_release,
                key="past_dashboard_base_date",
            )
        )
        st.caption("기준일 이후 데이터는 자동으로 제외됩니다.")
        st.divider()
        st.write("파일 연결 상태")
        status_items = [
            ("센터 발주 Raw", not full_center_order_df.empty),
            ("센터 재고 Raw", not full_stock_df.empty),
            ("매출/수요 Raw", not full_sales_df.empty),
            ("예약주문 Raw", not full_preorder_df.empty),
            ("OUTFLOW_7D 예측", not full_predictions_df.empty),
        ]
        for label, connected in status_items:
            if connected:
                st.success(f"{label}: 연결됨")
            else:
                st.warning(f"{label}: 없음")

    render_past_dashboard_page(
        full_preorder_df,
        full_sales_df,
        full_center_order_df,
        full_stock_df,
        full_predictions_df,
        past_base_date,
    )
    st.stop()

st.markdown(
    """
    <div class="hero">
        <h1>신상품 센터 운영 대시보드</h1>
        <p>예약 반응, 초도 발주, 실제 판매, 잔여 재고를 카드형 분석 화면에서 빠르게 비교합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("필터")

    brand_options = sorted(item_master["BRAND"].dropna().unique().tolist())
    selected_brands = st.multiselect(
        "브랜드",
        options=brand_options,
        default=[],
        help="비워두면 전체 브랜드를 봅니다.",
    )

    item_candidates = item_master.copy()
    if selected_brands:
        item_candidates = item_candidates[item_candidates["BRAND"].isin(selected_brands)]

    selected_labels = st.multiselect(
        "상품",
        options=item_candidates["LABEL"].tolist(),
        default=item_candidates["LABEL"].head(5).tolist(),
    )
    selected_items = [label.split(" | ", 1)[0] for label in selected_labels]

    center_options = center_master["CENTER_NM"].tolist()
    selected_centers = st.multiselect(
        "센터",
        options=center_options,
        default=[],
        help="비워두면 전체 센터를 봅니다.",
    )

    date_range = st.date_input(
        "출시일 기준 기간",
        value=(min_release, max_release),
        min_value=min_release,
        max_value=max_release,
    )

    sales_metric = st.radio("판매 추이 지표", ["판매수량", "판매금액"], horizontal=True)
    top_n = st.slider("랭킹 표시 개수", min_value=5, max_value=30, value=15, step=1)

if isinstance(date_range, tuple) and len(date_range) == 2:
    selected_date_range = date_range
else:
    selected_date_range = (min_release, max_release)

filtered_preorder = filter_preorder(
    preorder_df,
    selected_items=selected_items,
    selected_centers=selected_centers,
    selected_brands=selected_brands,
    date_range=selected_date_range,
)

available_item_codes = filtered_preorder["ITEM_CODE"].unique().tolist()
available_center_names = filtered_preorder["CENTER_NM"].unique().tolist()
available_center_codes = filtered_preorder["CENTER_CODE"].unique().tolist()

filtered_sales = sales_df[
    sales_df["ITEM_CODE"].isin(available_item_codes) & sales_df["CENTER_NM"].isin(available_center_names)
].copy()
filtered_stock = stock_df[
    stock_df["ITEM_CODE"].isin(available_item_codes) & stock_df["CENTER_CODE"].isin(available_center_codes)
].copy()

if filtered_preorder.empty:
    st.warning("선택한 조건에 맞는 대시보드 데이터가 없습니다. 필터를 조금 넓혀보세요.")
    st.stop()

kpis = summarize_kpis(filtered_preorder, filtered_sales, filtered_stock)

kpi_cols = st.columns(4)
with kpi_cols[0]:
    render_kpi_card("Product Revenue", format_won(kpis["누적 판매금액"]), "누적 판매금액")
with kpi_cols[1]:
    render_kpi_card("Initial Orders", format_int(kpis["초도 발주량"]), f"{format_int(kpis['상품 수'])}개 상품")
with kpi_cols[2]:
    render_kpi_card("Preorder Response", format_int(kpis["사전 예약량"]), f"{format_pct(kpis['예약/초도 비율'])}")
with kpi_cols[3]:
    render_kpi_card("Center Coverage", format_int(kpis["센터 수"]), f"평균 재고 {format_int(kpis['평균 재고'])}")

item_summary = build_item_summary(filtered_preorder, filtered_sales)
center_summary = build_center_summary(filtered_preorder, filtered_sales, filtered_stock)
weekly_items = build_weekly_item_list(item_master, preorder_df, sales_df)
week_start, week_end = get_latest_week_range(item_master)
analysis_df = build_preorder_sales_analysis(preorder_df, sales_df)

st.subheader("금주 신상품 리스트")
st.caption(f"최신 출시 주차 기준 {week_start.date()} ~ {week_end.date()} 상품만 표시합니다.")

if weekly_items.empty:
    st.info("최신 주차에 해당하는 출시 상품이 없습니다.")
    st.stop()

weekly_top = weekly_items.copy()
weekly_top["LABEL"] = weekly_top["ITEM_CODE"] + " | " + weekly_top["ITEM_NM"]
weekly_top = weekly_top.sort_values(["WEEK_RESERVATION_QTY", "NP_RLSE_DATE"], ascending=[False, False])

if "weekly_view_mode" not in st.session_state:
    st.session_state["weekly_view_mode"] = "list"
if "weekly_selected_item" not in st.session_state:
    st.session_state["weekly_selected_item"] = weekly_top.iloc[0]["ITEM_CODE"]

if st.session_state["weekly_view_mode"] == "list":
    selection_table = weekly_top[
        [
            "ITEM_CODE",
            "ITEM_NM",
            "BRAND",
            "ITEM_MDDV_NM",
            "NP_RLSE_DATE",
            "ST_SLEM_AMT",
            "MIN_ORD_QTY",
            "WEEK_RESERVATION_QTY",
            "WEEK_ORDERING_STORE_CNT",
            "WEEK_RESERVATION_RATE",
        ]
    ].rename(
        columns={
            "ITEM_CODE": "상품코드",
            "ITEM_NM": "상품명",
            "BRAND": "브랜드",
            "ITEM_MDDV_NM": "중분류",
            "NP_RLSE_DATE": "출시일",
            "ST_SLEM_AMT": "판매가",
            "MIN_ORD_QTY": "최소주문",
            "WEEK_RESERVATION_QTY": "예약수량",
            "WEEK_ORDERING_STORE_CNT": "예약점포수",
            "WEEK_RESERVATION_RATE": "예약/초도 비율(%)",
        }
    )

    selected_event = st.dataframe(
        selection_table,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "출시일": st.column_config.DateColumn("출시일"),
            "판매가": st.column_config.NumberColumn("판매가", format="%,.0f"),
            "최소주문": st.column_config.NumberColumn("최소주문", format="%,.0f"),
            "예약수량": st.column_config.NumberColumn("예약수량", format="%,.0f"),
            "예약점포수": st.column_config.NumberColumn("예약점포수", format="%,.0f"),
            "예약/초도 비율(%)": st.column_config.NumberColumn("예약/초도 비율(%)", format="%.1f"),
        },
    )

    selected_rows = selected_event.selection.rows
    if selected_rows:
        selected_idx = selected_rows[0]
        st.session_state["weekly_selected_item"] = weekly_top.iloc[selected_idx]["ITEM_CODE"]
        st.session_state["weekly_view_mode"] = "detail"
        st.rerun()

    st.caption("행 전체를 클릭하면 해당 상품 상세 화면으로 이동합니다.")

else:
    selected_weekly_item = st.session_state["weekly_selected_item"]
    selected_weekly_row = weekly_top[weekly_top["ITEM_CODE"] == selected_weekly_item]
    if selected_weekly_row.empty:
        st.session_state["weekly_view_mode"] = "list"
        st.rerun()
    selected_weekly_row = selected_weekly_row.iloc[0]
    category_parts = [
        str(selected_weekly_row.get("BRAND", "")).strip(),
        str(selected_weekly_row.get("ITEM_MDDV_NM", "")).strip(),
        str(selected_weekly_row.get("ITEM_SMDV_NM", "")).strip(),
    ]
    category_text = " · ".join([part for part in category_parts if part and part != "nan"])
    item_summary_text = generate_item_description_summary(
        item_name=str(selected_weekly_row.get("ITEM_NM", "") or "").strip(),
        brand=str(selected_weekly_row.get("BRAND", "") or "").strip(),
        middle_category=str(selected_weekly_row.get("ITEM_MDDV_NM", "") or "").strip(),
        small_category=str(selected_weekly_row.get("ITEM_SMDV_NM", "") or "").strip(),
        description=str(selected_weekly_row.get("ITEM_CRTR_CN", "") or "").strip(),
        fallback_summary=str(selected_weekly_row.get("ITEM_CRTR_SUMMARY", "") or "").strip(),
    )

    st.subheader("상품 소개 + 기본 정보")
    top_left, top_right = st.columns([0.8, 5.2])
    with top_left:
        if st.button("목록으로", width="stretch"):
            st.session_state["weekly_view_mode"] = "list"
            st.rerun()
    with top_right:
        st.markdown(
            f"""
            <div class="insight-card" style="border-left-color:#4e79a7;">
                <strong>{selected_weekly_row['ITEM_NM']}</strong><br>
                {category_text}<br>
                출시일 {selected_weekly_row['NP_RLSE_DATE'].date()} / 판매가 {format_won(selected_weekly_row['ST_SLEM_AMT'])}원 / 최소주문 {format_int(selected_weekly_row['MIN_ORD_QTY'])}
                {f"<br><span style='color:#5f6b7a;'>{item_summary_text}</span>" if item_summary_text else ""}
            </div>
            """,
            unsafe_allow_html=True,
        )

    selected_center_detail = build_item_center_preorder_detail(preorder_df, selected_weekly_item)
    selected_center_plan = build_center_initial_order_plan(selected_center_detail)
    selected_center_map = build_center_map_view(selected_center_plan, load_center_locations())
    selected_profile = build_item_preorder_profile(preorder_df, selected_weekly_item)
    selected_center_profile = build_item_center_preorder_profile(preorder_df, selected_weekly_item)
    detail_analysis, detail_summary = build_item_detail_analysis(analysis_df, selected_weekly_item)
    if not selected_profile.empty:
        selected_profile["예약일자 표시"] = selected_profile["예약일자"].apply(format_md_weekday)
    if not selected_center_profile.empty:
        selected_center_profile["예약일자 표시"] = selected_center_profile["예약일자"].apply(format_md_weekday)

    center_share = selected_center_plan.copy()
    total_reservation_qty = center_share["RESERVATION_QTY"].sum()
    center_share["센터 비중(%)"] = np.where(
        total_reservation_qty > 0,
        center_share["RESERVATION_QTY"] / total_reservation_qty * 100,
        0,
    )
    center_share = center_share.sort_values("RESERVATION_QTY", ascending=False).reset_index(drop=True)

    center_options = center_share["CENTER_NM"].dropna().astype(str).tolist()
    default_center = center_options[0] if center_options else None
    center_key = f"detail_center_{selected_weekly_item}"
    if center_key not in st.session_state and default_center:
        st.session_state[center_key] = default_center

    left, right = st.columns([1.15, 1])
    with left:
        fig = px.line(
            selected_profile,
            x="예약일자 표시",
            y="예약 수량",
            markers=True,
            title="전체 예약주문 추이",
            labels={"예약일자 표시": "예약일자", "예약 수량": "예약 수량"},
            color_discrete_sequence=[TABLEAU_COLORS[0]],
            category_orders={"예약일자 표시": selected_profile["예약일자 표시"].tolist()},
        )
        style_figure(fig)
        st.plotly_chart(fig, use_container_width=True, config={})

    with right:
        if center_options:
            fig = px.bar(
                center_share,
                x="CENTER_NM",
                y="센터 비중(%)",
                custom_data=["CENTER_NM", "RESERVATION_QTY", "ORDERING_STORE_CNT"],
                title="센터별 예약주문 비중",
                labels={"CENTER_NM": "센터", "센터 비중(%)": "예약 비중(%)"},
                color="센터 비중(%)",
                color_continuous_scale=["#edf2ff", "#4f6df5"],
            )
            style_figure(fig)
            fig.update_traces(
                hovertemplate=(
                    "센터=%{customdata[0]}<br>"
                    "예약 비중=%{y:.1f}%<br>"
                    "예약 수량=%{customdata[1]:,.0f}<br>"
                    "예약 점포 수=%{customdata[2]:,.0f}<extra></extra>"
                )
            )
            fig.update_layout(coloraxis_showscale=False)
            center_select_event = st.plotly_chart(
                fig,
                use_container_width=True,
                config={},
                on_select="rerun",
                key=f"center_share_chart_{selected_weekly_item}",
            )

            selected_points = center_select_event.selection.points if center_select_event else []
            if selected_points:
                clicked_center_name = selected_points[0].get("x")
                if clicked_center_name in center_options:
                    st.session_state[center_key] = clicked_center_name
        else:
            st.info("센터 비중을 보여줄 데이터가 없습니다.")

    selected_center_name = (
        st.session_state.get(center_key, default_center) if center_options else None
    )

    selected_center_row = (
        selected_center_plan[selected_center_plan["CENTER_NM"] == selected_center_name].iloc[0]
        if selected_center_name and not selected_center_plan.empty
        else None
    )

    if selected_center_row is not None:
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("4일 예약수량", format_int(selected_center_row["RESERVATION_QTY"]))
        metric2.metric("현재 초도발주량", format_int(selected_center_row["INITIAL_ORD_QTY"]))
        metric3.metric(
            f"산식 초도예측량 ({INITIAL_ORDER_MULTIPLIER}x가중치)",
            format_int(selected_center_row["산식 초도예측량"]),
        )
        metric4.metric("현재 초도 - 산식 차이", format_int(selected_center_row["초도 차이"]))
        st.caption(
            f"선택 센터: {selected_center_name} | 가중치 {selected_center_row['센터 가중치']:.2f} | "
            f"산식 기준: 4일 예약수량 x {INITIAL_ORDER_MULTIPLIER} x 센터 가중치"
        )
    elif not selected_center_plan.empty:
        st.info("센터를 클릭하면 해당 센터 기준으로 초도 산식이 계산됩니다.")

    map_col, trend_col = st.columns([1.15, 0.85], gap="large")
    with map_col:
        if not selected_center_map.empty:
            st.markdown(
                """
                <div class="insight-card" style="border-left-color:#4e79a7;">
                    <strong>센터 지도 뷰</strong><br>
                    점 크기는 초도발주량, 색상은 현재 초도와 산식 초도예측량 차이를 기준으로 표시합니다.
                </div>
                """,
                unsafe_allow_html=True,
            )
            map_fig = px.scatter_geo(
                selected_center_map,
                lat="LAT",
                lon="LON",
                color="상태 판정",
                size="마커 크기",
                size_max=26,
                hover_name="CENTER_NM",
                hover_data={
                    "지도 라벨": True,
                    "RESERVATION_QTY": ":,.0f",
                    "INITIAL_ORD_QTY": ":,.0f",
                    "산식 초도예측량": ":,.0f",
                    "초도 차이": ":,.0f",
                    "LAT": False,
                    "LON": False,
                    "마커 크기": False,
                },
                color_discrete_map={
                    "과다 가능": "#ff6b6b",
                    "적정": "#2fd3b1",
                    "부족 가능": "#4e79a7",
                },
                custom_data=["CENTER_NM", "RESERVATION_QTY", "INITIAL_ORD_QTY", "산식 초도예측량", "초도 차이"],
            )
            map_fig.update_geos(
                projection_type="mercator",
                showland=True,
                landcolor="#eef4ef",
                showcountries=True,
                countrycolor="#c9d4cf",
                coastlinecolor="#b9c9c1",
                showocean=True,
                oceancolor="#f4f8fb",
                lataxis_range=[33, 39],
                lonaxis_range=[124.5, 131],
                bgcolor="rgba(0,0,0,0)",
            )
            map_fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                legend_title_text="센터 상태",
                height=420,
                paper_bgcolor="white",
                plot_bgcolor="white",
            )
            map_fig.update_traces(
                marker=dict(line=dict(width=1, color="rgba(255,255,255,0.95)"), opacity=0.88),
                hovertemplate=(
                    "센터=%{hovertext}<br>"
                    "예약수량=%{customdata[1]:,.0f}<br>"
                    "초도발주량=%{customdata[2]:,.0f}<br>"
                    "산식 초도예측량=%{customdata[3]:,.0f}<br>"
                    "현재 초도-산식 차이=%{customdata[4]:,.0f}<extra></extra>"
                ),
            )
            map_event = st.plotly_chart(
                map_fig,
                use_container_width=True,
                config={},
                on_select="rerun",
                key=f"center_map_chart_{selected_weekly_item}",
            )
            selected_points = map_event.selection.points if map_event else []
            if selected_points:
                clicked_center_name = selected_points[0].get("customdata", [None])[0]
                if clicked_center_name in center_options:
                    st.session_state[center_key] = clicked_center_name
        else:
            st.info("지도에 표시할 센터 좌표 데이터가 없습니다.")

    with trend_col:
        if selected_center_name:
            scoped_center_profile = selected_center_profile[
                selected_center_profile["CENTER_NM"] == selected_center_name
            ].copy()
            st.markdown(
                f"""
                <div class="insight-card" style="border-left-color:#f28e2b;">
                    <strong>{selected_center_name}</strong> 센터 예약주문 추이
                </div>
                """,
                unsafe_allow_html=True,
            )
            fig = px.line(
                scoped_center_profile,
                x="예약일자 표시",
                y="예약 수량",
                markers=True,
                title="",
                labels={"예약일자 표시": "예약일자", "예약 수량": "예약 수량"},
                color_discrete_sequence=[TABLEAU_COLORS[1]],
                category_orders={"예약일자 표시": scoped_center_profile["예약일자 표시"].tolist()},
            )
            style_figure(fig)
            fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True, config={})
        else:
            st.info("센터를 클릭하면 해당 센터 예약주문 추이를 볼 수 있습니다.")

    detail_table = selected_center_plan.copy()
    st.dataframe(
        detail_table,
        width="stretch",
        hide_index=True,
        column_config={
            "LDU": st.column_config.TextColumn("LDU"),
            "CENTER_CODE": st.column_config.TextColumn("센터코드"),
            "INITIAL_ORD_QTY": st.column_config.NumberColumn("초도 발주량", format="%,.0f"),
            "RESERVATION_QTY": st.column_config.NumberColumn("예약 수량", format="%,.0f"),
            "센터 가중치": st.column_config.NumberColumn("센터 가중치", format="%.2f"),
            "기준 점포수": st.column_config.NumberColumn("점포수", format="%,.0f"),
            "산식 초도예측량": st.column_config.NumberColumn("산식 초도예측량", format="%,.0f"),
            "초도 차이": st.column_config.NumberColumn("현재 초도-산식 차이", format="%,.0f"),
            "예약 충족 배수": st.column_config.NumberColumn("초도/예약 배수", format="%.2f"),
            "ORDERING_STORE_CNT": st.column_config.NumberColumn("예약 점포 수", format="%,.0f"),
            "TOTAL_STORE_CNT": st.column_config.NumberColumn("전체 점포 수", format="%,.0f"),
            "예약/초도 비율(%)": st.column_config.NumberColumn("예약/초도 비율(%)", format="%.1f"),
            "예약 참여율(%)": st.column_config.NumberColumn("예약 참여율(%)", format="%.1f"),
        },
    )
