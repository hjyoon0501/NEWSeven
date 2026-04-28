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
W_RECOMMEND_CANDIDATES = [
    APP_DIR / "asymmetric_recommended_W.csv",
    DATA_DIR / "asymmetric_recommended_W.csv",
    APP_DIR / "W_RECOMMEND.parquet",
    APP_DIR / "W_RECOMMEND.csv",
    APP_DIR / "W_RECOMMEND.xlsx",
    DATA_DIR / "W_RECOMMEND.parquet",
    DATA_DIR / "W_RECOMMEND.csv",
    DATA_DIR / "W_RECOMMEND.xlsx",
]

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


def extract_capacity_from_name(value) -> str:
    text = "" if pd.isna(value) else str(value)
    match = re.search(r"(\d+(?:\.\d+)?)\s*(g|kg|ml|l|G|KG|ML|L)\b", text)
    return match.group(0) if match else "-"


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
            max-width: 1380px;
            padding-top: 0.85rem !important;
            padding-left: 1.35rem !important;
            padding-right: 1.35rem !important;
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
            border-radius: 14px !important;
            background: rgba(255, 255, 255, 0.94) !important;
            box-shadow: var(--soft-shadow) !important;
            min-height: 82px !important;
            padding: 0.72rem 0.85rem !important;
        }
        .kpi-card::before {
            content: "";
            display: block;
            width: 26px;
            height: 26px;
            border-radius: 9px;
            margin-bottom: 0.42rem;
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
            font-size: 0.7rem !important;
            font-weight: 700 !important;
        }
        .kpi-value {
            color: #161b2d !important;
            font-size: 1.25rem !important;
            font-weight: 850 !important;
        }
        .kpi-sub {
            color: #2fc39e !important;
            font-size: 0.68rem !important;
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
        .stTabs [aria-selected="true"]::after,
        .stTabs [data-baseweb="tab-highlight"] {
            background: linear-gradient(90deg, #7da7ff, #9a86ff) !important;
            border-radius: 999px !important;
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
        div[data-baseweb="tag"] {
            background: linear-gradient(135deg, rgba(222, 244, 255, 0.98), rgba(235, 225, 255, 0.98)) !important;
            border: 1px solid rgba(147, 173, 255, 0.62) !important;
            border-radius: 11px !important;
            color: #5169d7 !important;
            box-shadow: 0 5px 13px rgba(121, 145, 240, 0.12) !important;
        }
        div[data-baseweb="tag"] span,
        div[data-baseweb="tag"] div,
        div[data-baseweb="tag"] svg,
        div[data-baseweb="tag"] path {
            color: #5169d7 !important;
            fill: #5169d7 !important;
        }
        div[data-testid="stMultiSelect"] div[data-baseweb="tag"],
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
            background: linear-gradient(135deg, #def4ff, #ebe1ff) !important;
            border-color: rgba(147, 173, 255, 0.62) !important;
        }
        div[data-testid="stMultiSelect"] div[data-baseweb="tag"] *,
        div[data-testid="stMultiSelect"] span[data-baseweb="tag"] * {
            color: #5169d7 !important;
            fill: #5169d7 !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
            background-color: #dce3f1 !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
            background: linear-gradient(135deg, #7da7ff, #9a86ff) !important;
            border-color: #ffffff !important;
            box-shadow: 0 6px 16px rgba(121, 145, 240, 0.22) !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] [aria-valuenow] ~ div,
        div[data-testid="stSlider"] [data-baseweb="slider"] div[style*="background"] {
            background-color: #8f92ff !important;
        }
        div[role="radiogroup"] label p,
        div[role="radiogroup"] label span {
            background: transparent !important;
        }
        div[role="radiogroup"] label:has(input:checked) svg,
        div[role="radiogroup"] label:has(input:checked) path {
            fill: #8f92ff !important;
            color: #8f92ff !important;
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
        section[data-testid="stSidebar"] div[data-testid="stButton"] button {
            min-height: 1.9rem !important;
            padding: 0.2rem 0.85rem !important;
            border-radius: 999px !important;
            font-size: 0.74rem !important;
            font-weight: 800 !important;
            box-shadow: 0 6px 14px rgba(40, 51, 86, 0.07) !important;
            white-space: nowrap !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 0.55rem;
            width: 100%;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background: rgba(255,255,255,0.76);
            border: 1px solid rgba(231, 235, 247, 0.95);
            border-radius: 16px;
            box-shadow: 0 7px 18px rgba(40, 51, 86, 0.04);
            display: flex !important;
            align-items: center !important;
            min-height: 2.8rem;
            margin-bottom: 0 !important;
            padding: 0.48rem 0.95rem !important;
            width: 100% !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label p {
            color: #8995aa !important;
            font-size: 0.96rem !important;
            font-weight: 800 !important;
            line-height: 1.15 !important;
            white-space: nowrap !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, rgba(222, 244, 255, 0.98), rgba(235, 225, 255, 0.98)) !important;
            border-color: rgba(147, 173, 255, 0.72) !important;
            box-shadow: 0 10px 24px rgba(121, 145, 240, 0.16) !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
            color: #5169d7 !important;
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
        "INITIAL_ORD_QTY": "INITIAL_ORD_QTY",
        "Initial_ord_qty": "INITIAL_ORD_QTY",
        "initial_ord_qty": "INITIAL_ORD_QTY",
        "initial_order_qty": "INITIAL_ORD_QTY",
        "초도발주량": "INITIAL_ORD_QTY",
    }
    df = df.rename(columns={col: rename_map.get(col, col) for col in df.columns})
    if "OUTFLOW_7D" not in df.columns:
        return pd.DataFrame()
    if "ITEM_CODE" in df.columns:
        df["ITEM_CODE"] = df["ITEM_CODE"].map(normalize_center_code)
    if "CENTER_CODE" in df.columns:
        df["CENTER_CODE"] = df["CENTER_CODE"].map(normalize_center_code)
    df["OUTFLOW_7D"] = clean_numeric(df["OUTFLOW_7D"]).fillna(0)
    if "INITIAL_ORD_QTY" in df.columns:
        df["INITIAL_ORD_QTY"] = clean_numeric(df["INITIAL_ORD_QTY"]).fillna(0)
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


def build_prediction_initial_outflow_scatter(
    predictions_df: pd.DataFrame,
    preorder_df: pd.DataFrame,
    item_codes: list[str],
    selected_center: str,
) -> pd.DataFrame:
    required_cols = {"ITEM_CODE", "OUTFLOW_7D", "INITIAL_ORD_QTY"}
    if predictions_df.empty or not required_cols.issubset(predictions_df.columns):
        return pd.DataFrame()

    scoped = predictions_df.copy()
    if item_codes:
        scoped = scoped[scoped["ITEM_CODE"].astype(str).isin([str(code) for code in item_codes])]

    if selected_center != "전체" and "CENTER_CODE" in scoped.columns:
        center_codes = (
            preorder_df[preorder_df["CENTER_NM"].astype(str) == selected_center]["CENTER_CODE"]
            .map(normalize_center_code)
            .dropna()
            .unique()
            .tolist()
        )
        if center_codes:
            scoped = scoped[scoped["CENTER_CODE"].isin(center_codes)]

    if scoped.empty:
        return pd.DataFrame()

    scatter_df = (
        scoped.groupby("ITEM_CODE", as_index=False)
        .agg(
            OUTFLOW_7D=("OUTFLOW_7D", "sum"),
            INITIAL_ORD_QTY=("INITIAL_ORD_QTY", "sum"),
        )
    )
    item_meta = (
        preorder_df[
            ["ITEM_CODE", "ITEM_NM", "BRAND", "ITEM_MDDV_NM", "ITEM_SMDV_NM", "NP_RLSE_DATE"]
        ]
        .drop_duplicates("ITEM_CODE")
    )
    scatter_df = scatter_df.merge(item_meta, on="ITEM_CODE", how="left")
    safe_initial = scatter_df["INITIAL_ORD_QTY"].replace(0, pd.NA)
    scatter_df["실출고율(%)"] = (scatter_df["OUTFLOW_7D"] / safe_initial * 100).round(1)
    scatter_df["출고율"] = scatter_df["OUTFLOW_7D"] / safe_initial
    scatter_df["상태"] = classify_outflow_status(scatter_df["출고율"])
    scatter_df["MD/OPTIMAL 배수"] = scatter_df["INITIAL_ORD_QTY"] / scatter_df["OUTFLOW_7D"].replace(0, pd.NA)
    return scatter_df


def build_prediction_simulation_base(
    predictions_df: pd.DataFrame,
    preorder_df: pd.DataFrame,
) -> pd.DataFrame:
    required_cols = {"ITEM_CODE", "OUTFLOW_7D", "INITIAL_ORD_QTY"}
    if predictions_df.empty or not required_cols.issubset(predictions_df.columns):
        return pd.DataFrame()

    base = predictions_df.copy()
    if "NP_RLSE_YMD" in base.columns:
        base["NP_RLSE_DATE"] = pd.to_datetime(base["NP_RLSE_YMD"], errors="coerce")
    elif "NP_RLSE_DATE" in base.columns:
        base["NP_RLSE_DATE"] = pd.to_datetime(base["NP_RLSE_DATE"], errors="coerce")

    item_meta = (
        preorder_df[
            ["ITEM_CODE", "ITEM_NM", "BRAND", "ITEM_MDDV_NM", "ITEM_SMDV_NM", "NP_RLSE_DATE"]
        ]
        .drop_duplicates("ITEM_CODE")
        .rename(columns={"NP_RLSE_DATE": "PREORDER_RLSE_DATE"})
    )
    base = base.merge(item_meta, on="ITEM_CODE", how="left")
    if "NP_RLSE_DATE" not in base.columns:
        base["NP_RLSE_DATE"] = pd.NaT
    base["NP_RLSE_DATE"] = base["NP_RLSE_DATE"].fillna(base["PREORDER_RLSE_DATE"])
    for col in ["ITEM_NM", "BRAND", "ITEM_MDDV_NM", "ITEM_SMDV_NM"]:
        if col not in base.columns:
            base[col] = ""
        base[col] = base[col].fillna("")
    return base


@st.cache_data(show_spinner=False)
def load_w_recommend() -> pd.DataFrame:
    recommend_path = next((path for path in W_RECOMMEND_CANDIDATES if path.exists()), None)
    if recommend_path is None:
        return pd.DataFrame()

    suffix = recommend_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(recommend_path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(recommend_path)
    else:
        df = pd.read_csv(recommend_path)

    rename_map = {
        "CENT_CD": "CENTER_CODE",
        "CENTER_CD": "CENTER_CODE",
        "CENT_CODE": "CENTER_CODE",
        "CENTER": "CENTER_CODE",
        "W": "W_RECOMMEND",
        "WEIGHT": "W_RECOMMEND",
        "RECOMMEND_WEIGHT": "W_RECOMMEND",
    }
    df = df.rename(columns={col: rename_map.get(str(col).strip().upper(), col) for col in df.columns})
    if "CENTER_CODE" not in df.columns or "W_RECOMMEND" not in df.columns:
        return pd.DataFrame()

    df = df[["CENTER_CODE", "W_RECOMMEND"]].copy()
    df["CENTER_CODE"] = df["CENTER_CODE"].map(normalize_center_code)
    df["W_RECOMMEND"] = clean_numeric(df["W_RECOMMEND"])
    return (
        df.dropna(subset=["CENTER_CODE", "W_RECOMMEND"])
        .groupby("CENTER_CODE", as_index=False)["W_RECOMMEND"]
        .mean()
    )


def build_center_weight_lookup(center_codes: list[str]) -> dict[str, float]:
    weights = {
        center_code: float(CENTER_WEIGHT_CONFIG.get(center_code, {}).get("weight", 1.0))
        for center_code in center_codes
    }
    recommend_df = load_w_recommend()
    if not recommend_df.empty:
        recommended = recommend_df.set_index("CENTER_CODE")["W_RECOMMEND"].to_dict()
        for center_code in center_codes:
            if center_code in recommended and pd.notna(recommended[center_code]):
                weights[center_code] = float(recommended[center_code])
    return weights


def parse_md_editor_numbers(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")
    cleaned = (
        values.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"": pd.NA, "None": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def render_md_order_simulation_tab(
    preorder_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> None:
    st.markdown("## 이달의 신상품 예측")
    st.caption("출시일별 신상품을 센터 전체 기준으로 보고, 모델 산출값을 참고해 MD 발주 수량을 한 번에 조정합니다.")

    base = build_prediction_simulation_base(predictions_df, preorder_df)
    if base.empty:
        st.info("시뮬레이션에 사용할 predictions.parquet 데이터가 없거나 OUTFLOW_7D, INITIAL_ORD_QTY 컬럼을 찾을 수 없습니다.")
        return

    base["ITEM_CODE"] = base["ITEM_CODE"].astype(str).str.strip()
    if "CENTER_CODE" in base.columns:
        base["CENTER_CODE"] = base["CENTER_CODE"].map(normalize_center_code)
    else:
        st.info("센터별 매트릭스를 만들기 위한 CENTER_CODE 컬럼이 predictions.parquet에 없습니다.")
        return

    center_lookup = (
        preorder_df[["CENTER_CODE", "CENTER_NM"]]
        .drop_duplicates()
        .assign(CENTER_CODE=lambda df: df["CENTER_CODE"].map(normalize_center_code))
    )
    if "CENTER_NM" in base.columns:
        base = base.drop(columns=["CENTER_NM"])
    base = base.merge(center_lookup, on="CENTER_CODE", how="left")
    base["CENTER_NM"] = base["CENTER_NM"].fillna(base["CENTER_CODE"])

    base["출시일"] = pd.to_datetime(base["NP_RLSE_DATE"], errors="coerce").dt.date
    available_release_dates = base["출시일"].dropna().unique().tolist()
    if not available_release_dates:
        st.info("출시일 정보가 없어 시뮬레이션 대상을 선택할 수 없습니다.")
        return
    latest_year = max(date_value.year for date_value in available_release_dates)
    max_release_date = pd.Timestamp(year=latest_year, month=12, day=31).date()
    release_dates = sorted(
        [
            date_value
            for date_value in available_release_dates
            if date_value <= max_release_date
        ],
        reverse=True,
    )
    if not release_dates:
        st.info("출시일 정보가 없어 시뮬레이션 대상을 선택할 수 없습니다.")
        return

    default_dates = release_dates[:1]
    selected_dates = st.multiselect(
        "출시일 선택",
        options=release_dates,
        default=default_dates,
        format_func=lambda value: f"{value} ({base.loc[base['출시일'] == value, 'ITEM_CODE'].nunique():,}개)",
        key="md_sim_release_dates",
    )
    if not selected_dates:
        st.info("출시일을 1개 이상 선택해주세요.")
        return

    unit_choice = st.radio("단위", ["EA", "BOX"], horizontal=True, key="md_sim_unit")
    unit_divisor = 1 if unit_choice == "EA" else 10

    scoped = base[base["출시일"].isin(selected_dates)].copy()

    keyword = st.text_input("상품 검색", placeholder="상품코드 또는 상품명을 입력하세요", key="md_sim_keyword_matrix")
    if keyword.strip():
        raw_keyword = keyword.strip()
        scoped = scoped[
            scoped["ITEM_CODE"].astype(str).str.contains(raw_keyword, case=False, na=False)
            | scoped["ITEM_NM"].astype(str).str.contains(raw_keyword, case=False, na=False)
        ]

    if scoped.empty:
        st.info("선택한 조건에 맞는 예측 상품이 없습니다.")
        return

    center_order = (
        scoped.groupby(["CENTER_CODE", "CENTER_NM"], as_index=False)["OUTFLOW_7D"]
        .sum()
        .sort_values("OUTFLOW_7D", ascending=False)
    )
    center_codes = center_order["CENTER_CODE"].tolist()
    center_names = dict(zip(center_order["CENTER_CODE"], center_order["CENTER_NM"]))
    center_weight_lookup = build_center_weight_lookup(center_codes)

    matrix_base = (
        scoped.groupby(["ITEM_CODE", "ITEM_NM", "CENTER_CODE"], as_index=False)
        .agg(
            OUTFLOW_7D=("OUTFLOW_7D", "sum"),
            INITIAL_ORD_QTY=("INITIAL_ORD_QTY", "sum"),
        )
    )

    ml_pivot = matrix_base.pivot_table(
        index=["ITEM_CODE", "ITEM_NM"],
        columns="CENTER_CODE",
        values="OUTFLOW_7D",
        aggfunc="sum",
        fill_value=0,
    )
    current_pivot = matrix_base.pivot_table(
        index=["ITEM_CODE", "ITEM_NM"],
        columns="CENTER_CODE",
        values="INITIAL_ORD_QTY",
        aggfunc="sum",
        fill_value=0,
    )
    ml_pivot = ml_pivot.reindex(columns=center_codes, fill_value=0)
    current_pivot = current_pivot.reindex(columns=center_codes, fill_value=0)

    state_key = "md_sim_matrix_values"
    signature_key = "md_sim_matrix_signature"
    edit_key = "md_sim_matrix_editing"
    signature = "|".join([str(date_value) for date_value in selected_dates]) + f"|{keyword}|{unit_choice}"
    default_md = ml_pivot.copy()
    if st.session_state.get(signature_key) != signature:
        st.session_state[signature_key] = signature
        st.session_state[state_key] = default_md.copy()
        st.session_state[edit_key] = False
    stored_md = st.session_state.get(state_key)
    if (
        not isinstance(stored_md, pd.DataFrame)
        or not stored_md.index.equals(current_pivot.index)
        or list(stored_md.columns) != list(current_pivot.columns)
    ):
        st.session_state[state_key] = default_md.copy()
    current_md = st.session_state[state_key].reindex_like(current_pivot).fillna(default_md)

    display_ml = (ml_pivot / unit_divisor).round(0).astype(int)
    display_md = (current_md / unit_divisor).round(0).astype(int)

    editor_df = pd.DataFrame()
    editor_df["상품"] = [
        f"{idx + 1}. {name} [{item_code}]"
        for idx, (item_code, name) in enumerate(display_ml.index)
    ]
    column_config = {"상품": st.column_config.TextColumn("상품", width="medium")}
    disabled_cols = ["상품"]
    md_column_by_center = {}
    for center_code in center_codes:
        center_name = center_names.get(center_code, center_code)
        ml_col = f"ML_{center_code}"
        md_col = f"MD_{center_code}"
        md_column_by_center[center_code] = md_col
        editor_df[ml_col] = display_ml[center_code].values
        editor_df[md_col] = display_md[center_code].values
        column_config[ml_col] = st.column_config.NumberColumn(f"{center_name} 모델", format="%,.0f")
        column_config[md_col] = st.column_config.NumberColumn(f"{center_name} MD", min_value=0, step=1, format="%,.0f")
        disabled_cols.append(ml_col)

    st.markdown("#### 센터 메타")
    meta_rows = []
    for center_code in center_codes:
        config = CENTER_WEIGHT_CONFIG.get(center_code, {})
        meta_rows.append(
            {
                "구분": center_names.get(center_code, center_code),
                "가중치 (W)": center_weight_lookup.get(center_code, 1.0),
                "점포수": config.get("store_count", "-"),
            }
        )
    meta_df = pd.DataFrame(meta_rows).set_index("구분").T
    st.dataframe(meta_df, use_container_width=True, height=130)

    matrix_title_col, matrix_action_col = st.columns([1, 0.48])
    with matrix_title_col:
        st.markdown(f"#### MD 입력 매트릭스 ({unit_choice})")
    with matrix_action_col:
        action_cols = st.columns(3)
        if action_cols[0].button("편집", width="stretch", key="md_sim_edit_button"):
            st.session_state[edit_key] = True
        if action_cols[1].button("초기화", width="stretch", key="md_sim_reset_button"):
            st.session_state[state_key] = default_md.copy()
            st.session_state[edit_key] = False
            st.rerun()
        apply_clicked = action_cols[2].button("적용", width="stretch", key="md_sim_apply_button")

    st.caption("센터마다 모델 산출값(ML)은 잠금 열로 두고, MD 열만 편집합니다. 편집 버튼을 누른 뒤 수정하고 적용하면 신호 매트릭스가 갱신됩니다.")
    is_editing = st.session_state.get(edit_key, False)
    edited_df = st.data_editor(
        editor_df,
        use_container_width=True,
        height=420,
        hide_index=True,
        disabled=disabled_cols if is_editing else list(editor_df.columns),
        column_config=column_config,
        key=f"md_sim_matrix_editor_{signature}_{'edit' if is_editing else 'view'}",
    )

    fallback_md = (current_md / unit_divisor).round(0)
    live_md_pivot = pd.DataFrame(index=current_pivot.index, columns=center_codes, dtype=float)
    for center_code in center_codes:
        md_col = md_column_by_center[center_code]
        edited_values = parse_md_editor_numbers(edited_df[md_col])
        edited_values = edited_values.where(edited_values.notna(), fallback_md[center_code].to_numpy())
        live_md_pivot[center_code] = edited_values * unit_divisor

    if apply_clicked:
        st.session_state[state_key] = live_md_pivot.copy()
        st.session_state[edit_key] = False
        st.rerun()

    md_pivot = live_md_pivot.reindex_like(current_pivot).fillna(0)
    ratio = md_pivot / ml_pivot.replace(0, pd.NA)
    signal = pd.DataFrame("-", index=ratio.index, columns=ratio.columns)
    signal[(ratio >= 0.5) & (ratio <= 2.0)] = "정상"
    signal[(ratio > 2.0) & (ratio <= 3.0)] = "과발주"
    signal[ratio > 3.0] = "과발주↑"
    signal[(ratio < 0.5) & ratio.notna()] = "결품↑"
    signal.columns = [center_names.get(center_code, center_code) for center_code in signal.columns]
    signal_df = signal.reset_index(drop=True)
    signal_df.insert(0, "상품", editor_df["상품"].values)

    md_total = md_pivot.to_numpy().sum()
    diff_total = md_total - ml_pivot.to_numpy().sum()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"MD 입력 합 ({unit_choice})", format_int(md_total / unit_divisor))
    m2.metric("ML 대비 차이", format_int(diff_total / unit_divisor))
    m3.metric("과발주 신호", f"{signal_df.astype(str).apply(lambda col: col.str.contains('과발주', regex=False)).sum().sum():,}")
    m4.metric("결품 신호", f"{signal_df.astype(str).apply(lambda col: col.str.contains('결품', regex=False)).sum().sum():,}")

    st.markdown("#### 신호 (MD 입력 / ML 비율)")

    def style_signal_cell(value):
        text = str(value)
        if text == "정상":
            return "background-color:#DFF5E6; color:#256D3B; font-weight:700;"
        if "결품" in text:
            return "background-color:#FFE4E8; color:#B4233A; font-weight:700;"
        if "과발주" in text:
            return "background-color:#FFF0C7; color:#8A5A00; font-weight:700;"
        return "background-color:#F3F6FB; color:#8A94A6;"

    styled_signal = signal_df.style.map(style_signal_cell, subset=signal_df.columns[1:])
    st.dataframe(styled_signal, use_container_width=True, height=360, hide_index=True)


_PLT_W_CM = 110.0
_PLT_D_CM = 110.0
_PLT_MAX_H_CM = 120.0
_MIN_DIM_CM = 3.0
DEFAULT_PALLET_DAILY_COST = 2035
DEFAULT_BOX_HANDLING_COST = 90
DEFAULT_ANNUAL_RATE = 0.05
INVENTORY_DETAIL_DISPLAY_ROWS = 5_000
INVENTORY_DETAIL_DOWNLOAD_ROWS = 50_000


@st.cache_data(show_spinner=False)
def load_item_dimension_master() -> pd.DataFrame:
    if not MASTER_ITEM_PATH.exists():
        return pd.DataFrame(columns=["ITEM_CODE", "CALC_EA_PER_PALLET", "CALC_OB_QTY"])

    requested_cols = [
        "ITEM_CD",
        "ITEM_WDTH_LENG",
        "ITEM_HGHT_LENG",
        "ITEM_HG",
        "OB_WDTH_LENG",
        "OB_HGHT_LENG",
        "OB_HG",
        "OB_OBT_QTY",
        "PLLT_OBT_QTY",
        "PLLT_TSNG_QTY",
    ]
    available_cols = pd.read_csv(MASTER_ITEM_PATH, nrows=0).columns.tolist()
    df = pd.read_csv(MASTER_ITEM_PATH, usecols=[col for col in requested_cols if col in available_cols])
    if "ITEM_CD" not in df.columns:
        return pd.DataFrame(columns=["ITEM_CODE", "CALC_EA_PER_PALLET", "CALC_OB_QTY"])

    for col in df.columns.difference(["ITEM_CD"]):
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce").fillna(0)
    df["ITEM_CODE"] = df["ITEM_CD"].astype(str).str.strip()

    def _ea_per_pallet(row: pd.Series) -> float:
        if row.get("PLLT_OBT_QTY", 0) > 0:
            return float(row["PLLT_OBT_QTY"])

        outer_w = row.get("OB_WDTH_LENG", 0)
        outer_d = row.get("OB_HGHT_LENG", 0)
        outer_h = row.get("OB_HG", 0)
        outer_qty = row.get("OB_OBT_QTY", 0)
        if outer_w >= _MIN_DIM_CM and outer_d >= _MIN_DIM_CM and outer_h >= _MIN_DIM_CM and outer_qty > 0:
            per_layer = np.floor(_PLT_W_CM / outer_w) * np.floor(_PLT_D_CM / outer_d)
            stack_layers = row.get("PLLT_TSNG_QTY", 0)
            layers = stack_layers if stack_layers > 0 else np.floor(_PLT_MAX_H_CM / outer_h)
            return max(float(per_layer), 1) * max(float(layers), 1) * outer_qty

        item_w = row.get("ITEM_WDTH_LENG", 0)
        item_d = row.get("ITEM_HGHT_LENG", 0)
        item_h = row.get("ITEM_HG", 0)
        if item_w >= _MIN_DIM_CM and item_d >= _MIN_DIM_CM and item_h >= _MIN_DIM_CM:
            per_layer = np.floor(_PLT_W_CM / item_w) * np.floor(_PLT_D_CM / item_d)
            layers = np.floor(_PLT_MAX_H_CM / item_h)
            return max(float(per_layer), 1) * max(float(layers), 1)
        return np.nan

    df["CALC_EA_PER_PALLET"] = df.apply(_ea_per_pallet, axis=1)
    median_ea = df["CALC_EA_PER_PALLET"].median()
    df["CALC_EA_PER_PALLET"] = df["CALC_EA_PER_PALLET"].fillna(median_ea).clip(lower=1.0)
    if "OB_OBT_QTY" in df.columns:
        df["CALC_OB_QTY"] = df["OB_OBT_QTY"].where(df["OB_OBT_QTY"] > 0, 1.0)
    else:
        df["CALC_OB_QTY"] = 1.0

    return df[["ITEM_CODE", "CALC_EA_PER_PALLET", "CALC_OB_QTY"]].drop_duplicates("ITEM_CODE")


@st.cache_data(show_spinner=False)
def build_inventory_cost_dataset(
    stock_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    center_order_df: pd.DataFrame,
    preorder_df: pd.DataFrame,
    pallet_daily_cost: float,
    box_handling_cost: float,
    annual_interest_rate: float,
) -> pd.DataFrame:
    if stock_df.empty:
        return pd.DataFrame()

    dim_master = load_item_dimension_master()
    item_meta = (
        preorder_df[
            ["ITEM_CODE", "ITEM_NM", "BRAND", "ITEM_MDDV_NM", "ITEM_SMDV_NM", "ST_CPM_AMT", "ST_SLEM_AMT"]
        ]
        .drop_duplicates("ITEM_CODE")
    )
    center_meta = (
        preorder_df[["CENTER_CODE", "CENTER_NM"]]
        .drop_duplicates("CENTER_CODE")
        .assign(CENTER_CODE=lambda df: df["CENTER_CODE"].map(normalize_center_code))
    )

    cost_df = stock_df.copy()
    cost_df["CENTER_CODE"] = cost_df["CENTER_CODE"].map(normalize_center_code)
    cost_df = (
        cost_df.merge(item_meta, on="ITEM_CODE", how="left")
        .merge(dim_master, on="ITEM_CODE", how="left")
        .merge(center_meta, on="CENTER_CODE", how="left")
    )
    cost_df["CENTER_NM"] = cost_df["CENTER_NM"].fillna(cost_df["CENTER_CODE"])
    cost_df["CALC_EA_PER_PALLET"] = cost_df["CALC_EA_PER_PALLET"].fillna(1.0).clip(lower=1.0)
    cost_df["CALC_OB_QTY"] = cost_df["CALC_OB_QTY"].fillna(1.0).clip(lower=1.0)
    cost_df["ST_CPM_AMT"] = pd.to_numeric(cost_df["ST_CPM_AMT"], errors="coerce").fillna(0)
    cost_df["ST_SLEM_AMT"] = pd.to_numeric(cost_df["ST_SLEM_AMT"], errors="coerce").fillna(0)

    if not sales_df.empty:
        daily_sales = (
            sales_df.groupby(["SALE_DATE", "ITEM_CODE", "CENTER_NM"], as_index=False)["CENTER_SALE_QTY"]
            .sum()
            .rename(columns={"SALE_DATE": "BIZ_DT", "CENTER_SALE_QTY": "DAILY_OUTBOUND_QTY"})
        )
        cost_df = cost_df.merge(daily_sales, on=["BIZ_DT", "ITEM_CODE", "CENTER_NM"], how="left")
    else:
        cost_df["DAILY_OUTBOUND_QTY"] = 0

    if not center_order_df.empty and {"ORD_DATE", "ITEM_CODE", "CENTER_CODE", "CONV_QTY"}.issubset(center_order_df.columns):
        daily_orders = (
            center_order_df.groupby(["ORD_DATE", "ITEM_CODE", "CENTER_CODE"], as_index=False)["CONV_QTY"]
            .sum()
            .rename(columns={"ORD_DATE": "BIZ_DT", "CONV_QTY": "DAILY_INBOUND_QTY"})
        )
        daily_orders["CENTER_CODE"] = daily_orders["CENTER_CODE"].map(normalize_center_code)
        cost_df = cost_df.merge(daily_orders, on=["BIZ_DT", "ITEM_CODE", "CENTER_CODE"], how="left")
    else:
        cost_df["DAILY_INBOUND_QTY"] = 0

    for col in ["DAILY_OUTBOUND_QTY", "DAILY_INBOUND_QTY", "BOOK_END_QTY"]:
        cost_df[col] = pd.to_numeric(cost_df[col], errors="coerce").fillna(0)

    cost_df = cost_df.sort_values(["ITEM_CODE", "CENTER_NM", "BIZ_DT"]).reset_index(drop=True)

    def _weighted_avg(series: pd.Series, window: int) -> pd.Series:
        weights = np.arange(1, window + 1, dtype=float)

        def _wa(values: np.ndarray) -> float:
            if len(values) == 0:
                return np.nan
            scoped_weights = weights[-len(values):]
            return float(np.dot(values, scoped_weights) / scoped_weights.sum())

        return series.rolling(window=window, min_periods=1).apply(_wa, raw=True)

    sale_for_velocity = cost_df["DAILY_OUTBOUND_QTY"].where(cost_df["BOOK_END_QTY"] > 0)
    sale_for_velocity = sale_for_velocity.groupby([cost_df["ITEM_CODE"], cost_df["CENTER_NM"]]).ffill()
    cost_df["SALE_VEL_3D"] = (
        sale_for_velocity.groupby([cost_df["ITEM_CODE"], cost_df["CENTER_NM"]])
        .transform(lambda series: _weighted_avg(series, window=3))
        .fillna(0)
    )
    cost_df["SALE_VEL_7D"] = (
        sale_for_velocity.groupby([cost_df["ITEM_CODE"], cost_df["CENTER_NM"]])
        .transform(lambda series: _weighted_avg(series, window=7))
        .fillna(0)
    )
    cost_df["SALE_ACCEL"] = cost_df["SALE_VEL_3D"] - cost_df["SALE_VEL_7D"]
    cost_df["EST_DAILY_DEMAND"] = np.where(
        cost_df["SALE_VEL_7D"] > 0,
        cost_df["SALE_VEL_7D"],
        cost_df["SALE_VEL_3D"],
    )

    daily_rate = float(annual_interest_rate) / 365

    cost_df["PALLET_COUNT"] = np.ceil(cost_df["BOOK_END_QTY"].clip(lower=0) / cost_df["CALC_EA_PER_PALLET"])
    cost_df["STORAGE_COST"] = cost_df["PALLET_COUNT"] * pallet_daily_cost
    cost_df["OUTBOUND_BOX_COUNT"] = np.ceil(cost_df["DAILY_OUTBOUND_QTY"].clip(lower=0) / cost_df["CALC_OB_QTY"])
    cost_df["INBOUND_BOX_COUNT"] = np.ceil(cost_df["DAILY_INBOUND_QTY"].clip(lower=0) / cost_df["CALC_OB_QTY"])
    cost_df["OUTBOUND_HANDLING_COST"] = cost_df["OUTBOUND_BOX_COUNT"] * box_handling_cost
    cost_df["INBOUND_HANDLING_COST"] = cost_df["INBOUND_BOX_COUNT"] * box_handling_cost
    cost_df["INVENTORY_VALUE"] = cost_df["BOOK_END_QTY"] * cost_df["ST_CPM_AMT"]
    cost_df["CAPITAL_COST"] = cost_df["INVENTORY_VALUE"] * daily_rate
    cost_df["TOTAL_HANDLING_COST"] = cost_df["OUTBOUND_HANDLING_COST"] + cost_df["INBOUND_HANDLING_COST"]
    cost_df["MARGIN_PER_EA"] = (cost_df["ST_SLEM_AMT"] - cost_df["ST_CPM_AMT"]).clip(lower=0)
    cost_df["STOCKOUT_OPP_COST"] = np.where(
        (cost_df["BOOK_END_QTY"] == 0) & (cost_df["EST_DAILY_DEMAND"] > 0),
        cost_df["EST_DAILY_DEMAND"] * cost_df["MARGIN_PER_EA"],
        0.0,
    )
    cost_df["DAILY_TOTAL_COST"] = (
        cost_df["STORAGE_COST"]
        + cost_df["TOTAL_HANDLING_COST"]
        + cost_df["CAPITAL_COST"]
        + cost_df["STOCKOUT_OPP_COST"]
    )
    cost_df["STORAGE_COST_PER_EA"] = pallet_daily_cost / cost_df["CALC_EA_PER_PALLET"]
    cost_df["HANDLING_COST_PER_EA"] = box_handling_cost / cost_df["CALC_OB_QTY"]
    cost_df["CAPITAL_COST_PER_EA"] = cost_df["ST_CPM_AMT"] * daily_rate
    cost_df["TOTAL_COST_PER_EA_PER_DAY"] = cost_df["STORAGE_COST_PER_EA"] + cost_df["CAPITAL_COST_PER_EA"]
    cost_df["LOGISTICS_TO_SLEM_PCT"] = np.where(
        cost_df["ST_SLEM_AMT"] > 0,
        cost_df["TOTAL_COST_PER_EA_PER_DAY"] / cost_df["ST_SLEM_AMT"] * 100,
        np.nan,
    )
    return cost_df


def render_inventory_cost_page(
    stock_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    center_order_df: pd.DataFrame,
    preorder_df: pd.DataFrame,
) -> None:
    st.markdown("## 재고비용 시뮬레이션")
    st.caption("보관비, 입출고 하역비, 자본비용 가정을 움직이면서 센터/상품별 재고비용 변화를 확인합니다.")

    if stock_df.empty:
        st.info("재고비용을 계산할 재고 데이터가 없습니다.")
        return

    min_date = stock_df["BIZ_DT"].min().date()
    max_date = stock_df["BIZ_DT"].max().date()

    with st.sidebar:
        st.header("재고비용 조건")
        date_range = st.date_input(
            "분석 기간",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="inventory_cost_date_range",
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date

        center_options = sorted(preorder_df["CENTER_NM"].dropna().astype(str).unique().tolist())
        selected_centers = st.multiselect(
            "센터",
            options=center_options,
            default=[],
            help="비워두면 전체 센터를 봅니다.",
            key="inventory_cost_centers",
        )
        mddv_options = sorted(preorder_df["ITEM_MDDV_NM"].dropna().astype(str).unique().tolist())
        selected_mddv = st.multiselect(
            "중분류",
            options=mddv_options,
            default=[],
            help="비워두면 전체 중분류를 봅니다.",
            key="inventory_cost_mddv",
        )
        st.divider()
        st.caption("파렛트·박스 입수량은 A7 상품마스터 치수 기반으로 상품별 자동 산출됩니다.")
        pallet_daily_cost = st.slider(
            "팔레트당 일 보관비 (원/PLT/일)",
            100,
            3000,
            DEFAULT_PALLET_DAILY_COST,
            50,
            key="inventory_cost_pallet_cost",
        )
        box_handling_cost = st.slider(
            "박스당 하역비 (원/박스, 편도)",
            50,
            500,
            DEFAULT_BOX_HANDLING_COST,
            10,
            key="inventory_cost_box_cost",
        )
        annual_interest_rate = (
            st.slider(
                "연 자본비용 이율 (%)",
                0.0,
                20.0,
                DEFAULT_ANNUAL_RATE * 100,
                0.5,
                key="inventory_cost_rate",
            )
            / 100
        )

    cost_df = build_inventory_cost_dataset(
        stock_df,
        sales_df,
        center_order_df,
        preorder_df,
        pallet_daily_cost,
        box_handling_cost,
        annual_interest_rate,
    )

    filtered = cost_df[
        (cost_df["BIZ_DT"].dt.date >= start_date)
        & (cost_df["BIZ_DT"].dt.date <= end_date)
    ].copy()
    if selected_centers:
        filtered = filtered[filtered["CENTER_NM"].isin(selected_centers)]
    if selected_mddv:
        filtered = filtered[filtered["ITEM_MDDV_NM"].isin(selected_mddv)]

    if filtered.empty:
        st.info("선택한 조건에 맞는 재고비용 데이터가 없습니다.")
        return

    total_storage = filtered["STORAGE_COST"].sum()
    total_handling = filtered["TOTAL_HANDLING_COST"].sum()
    total_capital = filtered["CAPITAL_COST"].sum()
    total_cost = filtered["DAILY_TOTAL_COST"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 보관비", format_won(total_storage))
    c2.metric("총 하역비", format_won(total_handling))
    c3.metric("총 자본비용", format_won(total_capital))
    c4.metric("총 재고비용", format_won(total_cost))

    tab1, tab2, tab3, tab4 = st.tabs(["비용 추이", "센터/상품별 비용", "부진재고", "상세 데이터"])

    with tab1:
        daily_cost = (
            filtered.groupby("BIZ_DT", as_index=False)[
                ["STORAGE_COST", "TOTAL_HANDLING_COST", "CAPITAL_COST", "DAILY_TOTAL_COST"]
            ]
            .sum()
            .sort_values("BIZ_DT")
        )
        fig = px.line(
            daily_cost,
            x="BIZ_DT",
            y=["STORAGE_COST", "TOTAL_HANDLING_COST", "CAPITAL_COST", "DAILY_TOTAL_COST"],
            labels={"BIZ_DT": "일자", "value": "비용", "variable": "비용 항목"},
            color_discrete_sequence=["#6EC6E8", "#8DD7A5", "#F8C77E", "#7B61FF"],
            height=420,
        )
        fig.for_each_trace(
            lambda trace: trace.update(
                name={
                    "STORAGE_COST": "보관비",
                    "TOTAL_HANDLING_COST": "하역비",
                    "CAPITAL_COST": "자본비용",
                    "DAILY_TOTAL_COST": "총 재고비용",
                }.get(trace.name, trace.name)
            )
        )
        style_figure(fig)
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        pie_data = pd.DataFrame(
            {
                "비용 항목": ["보관비", "하역비", "자본비용"],
                "금액": [total_storage, total_handling, total_capital],
            }
        )
        left, right = st.columns(2)
        with left:
            fig_pie = px.pie(
                pie_data,
                values="금액",
                names="비용 항목",
                color_discrete_sequence=["#BFE7F5", "#CBEFD6", "#FFE0A8"],
                hole=0.55,
                height=340,
            )
            style_figure(fig_pie)
            st.plotly_chart(fig_pie, use_container_width=True)
        with right:
            sensitivity = []
            for rate in [0.8, 1.0, 1.2]:
                sensitivity.append(
                    {
                        "시나리오": f"보관/하역 단가 {rate:.0%}",
                        "총 비용": total_storage * rate + total_handling * rate + total_capital,
                    }
                )
            fig_s = px.bar(
                pd.DataFrame(sensitivity),
                x="시나리오",
                y="총 비용",
                color="시나리오",
                color_discrete_sequence=["#DCEBFF", "#BFE7F5", "#F8D7DA"],
                height=340,
            )
            style_figure(fig_s)
            fig_s.update_layout(showlegend=False)
            st.plotly_chart(fig_s, use_container_width=True)

    with tab2:
        center_cost = (
            filtered.groupby("CENTER_NM", as_index=False)["DAILY_TOTAL_COST"]
            .sum()
            .sort_values("DAILY_TOTAL_COST", ascending=False)
        )
        item_cost = (
            filtered.groupby(["ITEM_CODE", "ITEM_NM", "ITEM_MDDV_NM", "ITEM_SMDV_NM"], as_index=False)[
                ["BOOK_END_QTY", "DAILY_TOTAL_COST", "INVENTORY_VALUE"]
            ]
            .sum()
            .sort_values("DAILY_TOTAL_COST", ascending=False)
            .head(50)
        )
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            fig_center = px.bar(
                center_cost.head(20),
                x="CENTER_NM",
                y="DAILY_TOTAL_COST",
                labels={"CENTER_NM": "센터", "DAILY_TOTAL_COST": "총 재고비용"},
                color_discrete_sequence=["#7B61FF"],
                height=420,
            )
            style_figure(fig_center)
            fig_center.update_xaxes(tickangle=-35)
            st.plotly_chart(fig_center, use_container_width=True)
        with right:
            st.dataframe(
                item_cost.rename(
                    columns={
                        "ITEM_CODE": "상품코드",
                        "ITEM_NM": "상품명",
                        "ITEM_MDDV_NM": "중분류",
                        "ITEM_SMDV_NM": "소분류",
                        "BOOK_END_QTY": "누적 기말재고",
                        "DAILY_TOTAL_COST": "총 재고비용",
                        "INVENTORY_VALUE": "재고금액",
                    }
                ),
                use_container_width=True,
                height=420,
                hide_index=True,
            )

    with tab3:
        latest_date = filtered["BIZ_DT"].max()
        latest_stock = (
            filtered[filtered["BIZ_DT"] == latest_date]
            .groupby(["ITEM_CODE", "ITEM_NM", "ITEM_MDDV_NM", "ITEM_SMDV_NM"], as_index=False)
            .agg(
                현재재고=("BOOK_END_QTY", "sum"),
                재고금액=("INVENTORY_VALUE", "sum"),
                일재고비용=("DAILY_TOTAL_COST", "sum"),
            )
        )
        velocity = (
            filtered.groupby(["ITEM_CODE"], as_index=False)["DAILY_OUTBOUND_QTY"]
            .sum()
            .rename(columns={"DAILY_OUTBOUND_QTY": "기간출고량"})
        )
        days_count = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1, 1)
        stagnant = latest_stock.merge(velocity, on="ITEM_CODE", how="left")
        stagnant["기간출고량"] = stagnant["기간출고량"].fillna(0)
        stagnant["일평균출고량"] = stagnant["기간출고량"] / days_count
        stagnant["재고소진예상일"] = np.where(
            stagnant["일평균출고량"] > 0,
            stagnant["현재재고"] / stagnant["일평균출고량"],
            np.inf,
        )
        stagnant["재고상태"] = np.select(
            [
                stagnant["현재재고"] <= 0,
                stagnant["재고소진예상일"] >= 60,
                stagnant["재고소진예상일"] >= 30,
            ],
            ["재고 없음", "부진재고", "주의"],
            default="정상",
        )
        stagnant["재고소진예상일"] = stagnant["재고소진예상일"].replace(np.inf, np.nan)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("현재 재고 상품", f"{(stagnant['현재재고'] > 0).sum():,}")
        s2.metric("부진재고 상품", f"{(stagnant['재고상태'] == '부진재고').sum():,}")
        s3.metric("주의 상품", f"{(stagnant['재고상태'] == '주의').sum():,}")
        s4.metric("부진재고 금액", format_won(stagnant.loc[stagnant["재고상태"] == "부진재고", "재고금액"].sum()))

        status_order = ["부진재고", "주의", "정상", "재고 없음"]
        status_color = {
            "부진재고": "#F8C7D0",
            "주의": "#FFE7A8",
            "정상": "#CDEFD8",
            "재고 없음": "#E5EAF2",
        }
        status_summary = (
            stagnant.groupby("재고상태", as_index=False)
            .agg(상품수=("ITEM_CODE", "count"), 재고금액=("재고금액", "sum"), 일재고비용=("일재고비용", "sum"))
        )
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            fig_status = px.bar(
                status_summary,
                x="재고상태",
                y="상품수",
                color="재고상태",
                category_orders={"재고상태": status_order},
                color_discrete_map=status_color,
                height=360,
            )
            style_figure(fig_status)
            fig_status.update_layout(showlegend=False)
            st.plotly_chart(fig_status, use_container_width=True)
        with right:
            fig_scatter = px.scatter(
                stagnant[stagnant["현재재고"] > 0],
                x="일평균출고량",
                y="현재재고",
                color="재고상태",
                size="재고금액",
                hover_name="ITEM_NM",
                color_discrete_map=status_color,
                category_orders={"재고상태": status_order},
                labels={"일평균출고량": "일평균 출고량", "현재재고": "현재 재고"},
                height=360,
            )
            style_figure(fig_scatter)
            st.plotly_chart(fig_scatter, use_container_width=True)

        stagnant_view = stagnant.sort_values(["재고상태", "재고금액"], ascending=[True, False]).rename(
            columns={
                "ITEM_CODE": "상품코드",
                "ITEM_NM": "상품명",
                "ITEM_MDDV_NM": "중분류",
                "ITEM_SMDV_NM": "소분류",
            }
        )
        st.dataframe(
            stagnant_view[
                [
                    "상품코드",
                    "상품명",
                    "중분류",
                    "소분류",
                    "현재재고",
                    "기간출고량",
                    "일평균출고량",
                    "재고소진예상일",
                    "재고금액",
                    "일재고비용",
                    "재고상태",
                ]
            ],
            use_container_width=True,
            height=360,
            hide_index=True,
        )

    with tab4:
        subtab_agg, subtab_unit = st.tabs(["전체 비용 (일별)", "EA당 단위 비용"])

        with subtab_agg:
            detail_cols = [
                "BIZ_DT",
                "CENTER_NM",
                "ITEM_CODE",
                "ITEM_NM",
                "BOOK_END_QTY",
                "PALLET_COUNT",
                "CALC_EA_PER_PALLET",
                "DAILY_OUTBOUND_QTY",
                "DAILY_INBOUND_QTY",
                "STORAGE_COST",
                "TOTAL_HANDLING_COST",
                "CAPITAL_COST",
                "STOCKOUT_OPP_COST",
                "DAILY_TOTAL_COST",
            ]
            detail = filtered[[col for col in detail_cols if col in filtered.columns]].copy()
            detail = detail.rename(
                columns={
                    "BIZ_DT": "일자",
                    "CENTER_NM": "센터",
                    "ITEM_CODE": "상품코드",
                    "ITEM_NM": "상품명",
                    "BOOK_END_QTY": "기말재고",
                    "PALLET_COUNT": "파렛트수",
                    "CALC_EA_PER_PALLET": "파렛트당EA",
                    "DAILY_OUTBOUND_QTY": "출고수량",
                    "DAILY_INBOUND_QTY": "입고수량",
                    "STORAGE_COST": "보관비",
                    "TOTAL_HANDLING_COST": "하역비",
                    "CAPITAL_COST": "자본비용",
                    "STOCKOUT_OPP_COST": "결품기회비용",
                    "DAILY_TOTAL_COST": "총 재고비용",
                }
            )
            detail_view = detail.head(INVENTORY_DETAIL_DISPLAY_ROWS)
            st.caption(
                f"전체 {len(detail):,}행 중 화면에는 최대 {INVENTORY_DETAIL_DISPLAY_ROWS:,}행만 표시합니다. "
                f"CSV는 최대 {INVENTORY_DETAIL_DOWNLOAD_ROWS:,}행까지 내려받을 수 있습니다."
            )
            st.dataframe(detail_view, use_container_width=True, height=460, hide_index=True)
            detail_download = detail.head(INVENTORY_DETAIL_DOWNLOAD_ROWS)
            st.download_button(
                "전체 비용 CSV 다운로드",
                data=detail_download.to_csv(index=False).encode("utf-8-sig"),
                file_name="inventory_cost_detail.csv",
                mime="text/csv",
                width="stretch",
            )

        with subtab_unit:
            st.caption(
                "상품 1EA를 기준으로 한 단위 비용입니다. 보관비·자본비용은 1일 기준, 하역비는 편도 1회 기준입니다."
            )
            unit_cols = [
                "ITEM_CODE",
                "ITEM_NM",
                "ST_CPM_AMT",
                "ST_SLEM_AMT",
                "CALC_EA_PER_PALLET",
                "CALC_OB_QTY",
                "STORAGE_COST_PER_EA",
                "HANDLING_COST_PER_EA",
                "CAPITAL_COST_PER_EA",
                "TOTAL_COST_PER_EA_PER_DAY",
                "LOGISTICS_TO_SLEM_PCT",
            ]
            unit_df = (
                filtered[[col for col in unit_cols if col in filtered.columns]]
                .drop_duplicates("ITEM_CODE")
                .sort_values("TOTAL_COST_PER_EA_PER_DAY", ascending=False)
            )
            unit_df = unit_df.rename(
                columns={
                    "ITEM_CODE": "상품코드",
                    "ITEM_NM": "상품명",
                    "ST_CPM_AMT": "원가(EA)",
                    "ST_SLEM_AMT": "매가(EA)",
                    "CALC_EA_PER_PALLET": "파렛트당EA",
                    "CALC_OB_QTY": "외박스당EA",
                    "STORAGE_COST_PER_EA": "보관비/EA/일(원)",
                    "HANDLING_COST_PER_EA": "하역비/EA/편도(원)",
                    "CAPITAL_COST_PER_EA": "자본비용/EA/일(원)",
                    "TOTAL_COST_PER_EA_PER_DAY": "총물류비/EA/일(원)",
                    "LOGISTICS_TO_SLEM_PCT": "매가대비일물류비(%)",
                }
            )
            unit_view = unit_df.head(INVENTORY_DETAIL_DISPLAY_ROWS)
            if len(unit_df) > INVENTORY_DETAIL_DISPLAY_ROWS:
                st.caption(
                    f"전체 {len(unit_df):,}행 중 화면에는 최대 {INVENTORY_DETAIL_DISPLAY_ROWS:,}행만 표시합니다."
                )
            st.dataframe(
                unit_view,
                use_container_width=True,
                height=460,
                hide_index=True,
                column_config={
                    "원가(EA)": st.column_config.NumberColumn("원가(EA)", format="%,.0f"),
                    "매가(EA)": st.column_config.NumberColumn("매가(EA)", format="%,.0f"),
                    "파렛트당EA": st.column_config.NumberColumn("파렛트당EA", format="%.0f"),
                    "외박스당EA": st.column_config.NumberColumn("외박스당EA", format="%.0f"),
                    "보관비/EA/일(원)": st.column_config.NumberColumn("보관비/EA/일", format="%.2f"),
                    "하역비/EA/편도(원)": st.column_config.NumberColumn("하역비/EA/편도", format="%.2f"),
                    "자본비용/EA/일(원)": st.column_config.NumberColumn("자본비용/EA/일", format="%.2f"),
                    "총물류비/EA/일(원)": st.column_config.NumberColumn("총물류비/EA/일", format="%.2f"),
                    "매가대비일물류비(%)": st.column_config.NumberColumn("매가대비 일물류비(%)", format="%.3f"),
                },
            )
            unit_download = unit_df.head(INVENTORY_DETAIL_DOWNLOAD_ROWS)
            st.download_button(
                "EA당 단위비용 CSV 다운로드",
                data=unit_download.to_csv(index=False).encode("utf-8-sig"),
                file_name="unit_cost_per_ea.csv",
                mime="text/csv",
                width="stretch",
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
    f1, f2, f3, f4, f5 = st.columns([1, 0.9, 1, 1, 1.6])
    selected_center = f1.selectbox(
        "센터",
        ["전체"] + sorted(preorder_df["CENTER_NM"].dropna().astype(str).unique().tolist()),
        key="past_raw_center",
    )
    f2.selectbox("대분류", ["과자"], key="past_raw_top")
    selected_mddv = f3.selectbox(
        "중분류",
        ["전체"] + sorted(preorder_df["ITEM_MDDV_NM"].dropna().astype(str).unique().tolist()),
        key="past_raw_mddv",
    )
    mddv_pool = preorder_df if selected_mddv == "전체" else preorder_df[preorder_df["ITEM_MDDV_NM"] == selected_mddv]
    selected_smdv = f4.selectbox(
        "소분류",
        ["전체"] + sorted(mddv_pool["ITEM_SMDV_NM"].dropna().astype(str).unique().tolist()),
        key="past_raw_smdv",
    )
    raw_keyword = f5.text_input(
        "제품코드 또는 제품명",
        key="past_raw_data_search",
        placeholder="제품코드 또는 제품명을 입력하세요",
    ).strip()

    allowed_items = preorder_df.copy()
    if selected_center != "전체":
        allowed_items = allowed_items[allowed_items["CENTER_NM"].astype(str) == selected_center]
    if selected_mddv != "전체":
        allowed_items = allowed_items[allowed_items["ITEM_MDDV_NM"] == selected_mddv]
    if selected_smdv != "전체":
        allowed_items = allowed_items[allowed_items["ITEM_SMDV_NM"] == selected_smdv]
    allowed_item_codes = allowed_items["ITEM_CODE"].astype(str).unique().tolist()
    allowed_center_codes = allowed_items["CENTER_CODE"].map(normalize_center_code).unique().tolist()
    allowed_center_names = allowed_items["CENTER_NM"].astype(str).unique().tolist()

    datasets = [
        ("센터 발주 Raw", center_order_df, "center_order_filtered.csv", "ORD_DATE"),
        ("센터 재고 Raw", stock_df, "center_stock_filtered.csv", "BIZ_DT"),
        ("매출/수요 Raw", sales_df, "sales_filtered.csv", "SALE_DATE"),
        ("예약주문 Raw", preorder_df, "preorder_filtered.csv", "NP_RLSE_DATE"),
    ]
    for title, frame, filename, date_col in datasets:
        st.markdown(f"##### {title}")
        if frame.empty:
            st.info(f"{title} 데이터가 없습니다.")
            continue
        filtered = frame.copy()
        if date_col in filtered.columns:
            filtered = filtered[filtered[date_col].isna() | (filtered[date_col] <= base_date)]

        if "ITEM_CODE" in filtered.columns:
            filtered = filtered[filtered["ITEM_CODE"].astype(str).isin(allowed_item_codes)]
        elif "ITEM_CD" in filtered.columns:
            filtered = filtered[filtered["ITEM_CD"].astype(str).isin(allowed_item_codes)]
        if "CENTER_CODE" in filtered.columns:
            filtered = filtered[filtered["CENTER_CODE"].map(normalize_center_code).isin(allowed_center_codes)]
        elif "CENT_CD" in filtered.columns:
            filtered = filtered[filtered["CENT_CD"].map(normalize_center_code).isin(allowed_center_codes)]
        if "CENTER_NM" in filtered.columns:
            filtered = filtered[filtered["CENTER_NM"].astype(str).isin(allowed_center_names)]
        elif "CENT_NM" in filtered.columns:
            filtered = filtered[filtered["CENT_NM"].astype(str).isin(allowed_center_names)]

        if raw_keyword:
            searchable_columns = [
                col
                for col in ["ITEM_CODE", "ITEM_CD", "ITEM_NM", "상품명", "제품명", "CENTER_CODE", "CENT_CD", "CENTER_NM", "CENT_NM"]
                if col in filtered.columns
            ]
            if searchable_columns:
                search_mask = pd.Series(False, index=filtered.index)
                for col in searchable_columns:
                    search_mask = search_mask | filtered[col].astype(str).str.contains(
                        raw_keyword, case=False, na=False
                    )
                filtered = filtered[search_mask]

        sort_options = filtered.columns.tolist()
        toolbar_left, toolbar_right = st.columns([1.6, 0.6])
        sort_col = toolbar_left.selectbox("정렬 기준", sort_options, key=f"sort_{filename}")
        if sort_col:
            filtered = filtered.sort_values(sort_col, ascending=False)
        toolbar_right.download_button(
            label="CSV",
            data=filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name=filename,
            mime="text/csv",
            width="stretch",
            key=f"download_{filename}",
        )
        st.caption(f"조회 건수 {len(filtered):,} · 컬럼 {len(filtered.columns):,}")
        st.dataframe(filtered.head(300), width="stretch", height=330)


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
        scatter_df = build_prediction_initial_outflow_scatter(
            predictions_df,
            preorder_df,
            filtered["ITEM_CODE"].astype(str).unique().tolist(),
            selected_center,
        )
        if not scatter_df.empty:
            scatter_df["출시일자_str"] = pd.to_datetime(
                scatter_df["NP_RLSE_DATE"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            scatter_df = scatter_df[
                (scatter_df["INITIAL_ORD_QTY"] > 0) & (scatter_df["OUTFLOW_7D"] > 0)
            ].copy()
        if scatter_df.empty:
            st.info("로그 스케일 그래프를 그릴 양수 데이터가 없습니다.")
        else:
            fig_sc = px.scatter(
                scatter_df,
                x="OUTFLOW_7D", y="INITIAL_ORD_QTY",
                color="상태",
                category_orders={"상태": OUTFLOW_STATUS_ORDER},
                color_discrete_map=OUTFLOW_STATUS_COLORS,
                hover_data={
                    "ITEM_NM": True, "출시일자_str": True,
                    "ITEM_MDDV_NM": True, "ITEM_SMDV_NM": True,
                    "OUTFLOW_7D": ":,.0f", "INITIAL_ORD_QTY": ":,.0f", "MD/OPTIMAL 배수": ":.2f", "실출고율(%)": ":.1f", "상태": True,
                },
                labels={
                    "ITEM_NM": "제품명", "출시일자_str": "출시일자",
                    "ITEM_MDDV_NM": "중분류", "ITEM_SMDV_NM": "소분류",
                    "OUTFLOW_7D": "OPTIMAL = OUTFLOW_7D (log)",
                    "INITIAL_ORD_QTY": "MD 실제 초도발주량 = INITIAL_ORD_QTY (log)",
                },
                height=420,
            )
            min_axis = max(1, min(scatter_df["OUTFLOW_7D"].min(), scatter_df["INITIAL_ORD_QTY"].min()))
            max_axis = max(scatter_df["OUTFLOW_7D"].max(), scatter_df["INITIAL_ORD_QTY"].max())
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


def build_past_item_status_df(
    preorder_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    if preorder_df.empty:
        return pd.DataFrame()

    item_info = preorder_df.groupby("ITEM_CODE").agg(
        ITEM_NM=("ITEM_NM", "first"),
        NP_RLSE_YMD=("NP_RLSE_YMD", "first"),
        BRAND=("BRAND", "first"),
        ITEM_MDDV_NM=("ITEM_MDDV_NM", "first"),
        ITEM_SMDV_NM=("ITEM_SMDV_NM", "first"),
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
            sales_df.groupby("ITEM_CODE")[["CENTER_SALE_QTY", "CENTER_SALE_AMT_VAT"]]
            .sum()
            .reset_index()
            .rename(columns={"CENTER_SALE_QTY": "실수요", "CENTER_SALE_AMT_VAT": "실수요금액"})
        )
    else:
        item_sales = pd.DataFrame(columns=["ITEM_CODE", "실수요", "실수요금액"])

    item_df = item_info.merge(item_orders, on="ITEM_CODE", how="left").merge(item_sales, on="ITEM_CODE", how="left")
    for col in ["실출고량", "실수요", "실수요금액"]:
        item_df[col] = item_df[col].fillna(0)
    item_df["출시일자"] = pd.to_datetime(item_df["NP_RLSE_YMD"].astype(str), format="%Y%m%d", errors="coerce")
    item_df["용량"] = item_df["ITEM_NM"].map(extract_capacity_from_name)
    safe_initial = item_df["초도발주량"].replace(0, pd.NA)
    item_df["초도출고율(%)"] = (item_df["실출고량"] / safe_initial * 100).round(1)
    item_df["출고율"] = item_df["실출고량"] / safe_initial
    item_df["상태"] = classify_outflow_status(item_df["출고율"])
    item_df["결품여부"] = item_df["상태"].isin(["결품 위험", "결품"])
    item_df["부진여부"] = item_df["상태"].isin(["부진재고", "과발주 위험"])
    return item_df


def build_past_center_raw_table(
    preorder_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    if preorder_df.empty:
        return pd.DataFrame()
    ten_day_cols = [col for col in PREORDER_DAY_COLUMNS[:10] if col in preorder_df.columns]
    raw = preorder_df.copy()
    raw["예약주문(10일치)"] = raw[ten_day_cols].sum(axis=1) if ten_day_cols else raw["total_pre_order_qty(D-11~D-8)"]
    base = raw.groupby(["ITEM_CODE", "CENTER_CODE", "CENTER_NM"], as_index=False).agg(
        ITEM_NM=("ITEM_NM", "first"),
        NP_RLSE_DATE=("NP_RLSE_DATE", "first"),
        BRAND=("BRAND", "first"),
        ITEM_MDDV_NM=("ITEM_MDDV_NM", "first"),
        ITEM_SMDV_NM=("ITEM_SMDV_NM", "first"),
        ST_SLEM_AMT=("ST_SLEM_AMT", "first"),
        예약주문4일치=("total_pre_order_qty(D-11~D-8)", "sum"),
        예약주문10일치=("예약주문(10일치)", "sum"),
        초도발주량=("INITIAL_ORD_QTY", "sum"),
    )
    base["CENTER_CODE"] = base["CENTER_CODE"].map(normalize_center_code)

    if not predictions_df.empty:
        shipped = build_outflow_7d_summary(predictions_df, ["ITEM_CODE", "CENTER_CODE"])
        base = base.merge(shipped, on=["ITEM_CODE", "CENTER_CODE"], how="left")
    else:
        base["실출고량"] = 0

    if not sales_df.empty:
        sales = (
            sales_df.groupby(["ITEM_CODE", "CENTER_NM"], as_index=False)["CENTER_SALE_QTY"]
            .sum()
            .rename(columns={"CENTER_SALE_QTY": "실수요"})
        )
        base = base.merge(sales, on=["ITEM_CODE", "CENTER_NM"], how="left")
    else:
        base["실수요"] = 0

    base["실출고량"] = base["실출고량"].fillna(0)
    base["실수요"] = base["실수요"].fillna(0)
    safe_initial = base["초도발주량"].replace(0, pd.NA)
    base["초도출고율(%)"] = (base["실출고량"] / safe_initial * 100).round(1)
    base["용량"] = base["ITEM_NM"].map(extract_capacity_from_name)
    return base


def render_past_lookup_overview(
    preorder_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> None:
    st.subheader("과거 신상품 조회")
    raw_df = build_past_center_raw_table(preorder_df, sales_df, predictions_df)
    if raw_df.empty:
        st.info("조회할 과거 신상품 데이터가 없습니다.")
        return

    min_date = raw_df["NP_RLSE_DATE"].min().date()
    max_date = raw_df["NP_RLSE_DATE"].max().date()
    f1, f2, f3, f4, f5, f6 = st.columns([1.3, 1, 0.8, 1, 1, 1.5])
    selected_range = f1.date_input("기간", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="past_overview_date")
    selected_center = f2.selectbox("센터", ["전체"] + sorted(raw_df["CENTER_NM"].dropna().astype(str).unique().tolist()), key="past_overview_center")
    f3.selectbox("대분류", ["과자"], key="past_overview_top")
    selected_mddv = f4.selectbox("중분류", ["전체"] + sorted(raw_df["ITEM_MDDV_NM"].dropna().astype(str).unique().tolist()), key="past_overview_mddv")
    smdv_pool = raw_df if selected_mddv == "전체" else raw_df[raw_df["ITEM_MDDV_NM"] == selected_mddv]
    selected_smdv = f5.selectbox("소분류", ["전체"] + sorted(smdv_pool["ITEM_SMDV_NM"].dropna().astype(str).unique().tolist()), key="past_overview_smdv")
    keyword = f6.text_input("제품코드 또는 제품명", key="past_overview_keyword")

    filtered = raw_df.copy()
    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        filtered = filtered[filtered["NP_RLSE_DATE"].dt.date.between(selected_range[0], selected_range[1])]
    if selected_center != "전체":
        filtered = filtered[filtered["CENTER_NM"] == selected_center]
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

    display = filtered.rename(columns={
        "CENTER_NM": "센터명",
        "ITEM_CODE": "제품코드",
        "ITEM_NM": "제품명",
        "NP_RLSE_DATE": "출시일자",
        "BRAND": "브랜드",
        "ITEM_MDDV_NM": "중분류",
        "ITEM_SMDV_NM": "소분류",
        "ST_SLEM_AMT": "가격",
        "예약주문4일치": "예약주문(4일치)",
        "예약주문10일치": "예약주문(10일치)",
        "실출고량": "실출고량(7일치)",
        "초도출고율(%)": "초도출고율",
    })
    ordered_cols = ["센터명", "제품코드", "제품명", "출시일자", "브랜드", "중분류", "소분류", "가격", "용량", "예약주문(4일치)", "예약주문(10일치)", "초도발주량", "실출고량(7일치)", "초도출고율"]
    available_cols = [col for col in ordered_cols if col in display.columns]
    selected_cols = st.multiselect("컬럼 선택", available_cols, default=available_cols, key="past_overview_columns")
    product_options = display[["제품코드", "제품명"]].drop_duplicates()
    product_labels = (product_options["제품코드"].astype(str) + " | " + product_options["제품명"].astype(str)).tolist()
    selected_products = st.multiselect("상품 선택(옵션)", product_labels, default=[], key="past_overview_products")
    if selected_products:
        selected_codes = [label.split(" | ", 1)[0] for label in selected_products]
        picked = display[display["제품코드"].astype(str).isin(selected_codes)]
        st.markdown("##### 선택 상품 모아보기")
        st.dataframe(picked[selected_cols], use_container_width=True, height=220, hide_index=True)
        display["_선택"] = display["제품코드"].astype(str).isin(selected_codes).astype(int)
        display = display.sort_values(["_선택", "실출고량(7일치)"], ascending=[False, False]).drop(columns="_선택")
    else:
        display = display.sort_values("실출고량(7일치)", ascending=False)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("행 수", f"{len(display):,}")
    k2.metric("초도발주량", f"{display['초도발주량'].sum():,.0f}")
    k3.metric("실출고량(7일치)", f"{display['실출고량(7일치)'].sum():,.0f}")
    k4.metric("평균 초도출고율", f"{display['초도출고율'].mean():.1f}%" if not display.empty else "-")
    st.dataframe(display[selected_cols], use_container_width=True, height=520, hide_index=True)


def render_past_product_data_detail(
    preorder_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> None:
    st.subheader("상품별 데이터")
    item_df = build_past_item_status_df(preorder_df, sales_df, predictions_df)
    if item_df.empty:
        st.info("상품별 데이터를 만들 수 없습니다.")
        return
    keyword = st.text_input("제품코드 또는 제품명", key="past_product_detail_search")
    options = item_df.copy()
    if keyword.strip():
        kw = keyword.strip()
        options = options[
            options["ITEM_CODE"].astype(str).str.contains(kw, case=False, na=False)
            | options["ITEM_NM"].astype(str).str.contains(kw, case=False, na=False)
        ]
    if options.empty:
        st.info("검색 조건에 해당하는 상품이 없습니다.")
        return
    labels = (options["ITEM_CODE"].astype(str) + " | " + options["ITEM_NM"].astype(str)).tolist()
    selected_label = st.selectbox("상품 선택", labels, key="past_product_detail_select")
    selected_code = selected_label.split(" | ", 1)[0]
    selected_item = item_df[item_df["ITEM_CODE"].astype(str) == selected_code].iloc[0]
    st.markdown(
        f"""
        <div class="insight-card" style="border-left-color:#4f6df5;">
            <strong>{selected_item['ITEM_NM']}</strong><br>
            {selected_item['BRAND']} · {selected_item['ITEM_MDDV_NM']} · {selected_item['ITEM_SMDV_NM']} · 출시일 {selected_item['출시일자'].date()}
        </div>
        """,
        unsafe_allow_html=True,
    )
    center_table = build_past_center_raw_table(preorder_df[preorder_df["ITEM_CODE"].astype(str) == selected_code], sales_df, predictions_df)
    if center_table.empty:
        st.info("센터별 데이터가 없습니다.")
        return
    center_view = center_table.rename(columns={
        "CENTER_NM": "센터명",
        "예약주문4일치": "예약주문(4일치)",
        "예약주문10일치": "예약주문(10일치)",
        "실출고량": "실출고량(7일치)",
        "초도출고율(%)": "초도출고율",
    })
    fig = go.Figure()
    for name, color in [("예약주문(4일치)", "#A7F3D0"), ("예약주문(10일치)", "#34D399"), ("초도발주량", "#7b4cf3"), ("실출고량(7일치)", "#35c8e8")]:
        fig.add_trace(go.Bar(name=name, x=center_view["센터명"], y=center_view[name], marker_color=color))
    fig.update_layout(barmode="group", height=340, margin=dict(l=0, r=0, t=20, b=80), plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        center_view[["센터명", "예약주문(4일치)", "예약주문(10일치)", "초도발주량", "실출고량(7일치)", "초도출고율"]],
        use_container_width=True,
        height=360,
        hide_index=True,
    )


def render_past_status_analysis_tab(
    preorder_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> None:
    st.subheader("상품별 상태 분석")
    item_df = build_past_item_status_df(preorder_df, sales_df, predictions_df)
    if item_df.empty:
        st.info("상태 분석 데이터가 없습니다.")
        return
    min_date = item_df["출시일자"].min().date()
    max_date = item_df["출시일자"].max().date()
    selected_range = st.date_input("기간", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="past_status_date")
    filtered = item_df.copy()
    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        filtered = filtered[filtered["출시일자"].dt.date.between(selected_range[0], selected_range[1])]
    status_filter = st.radio("상태 필터", ["전체", *OUTFLOW_STATUS_ORDER], horizontal=True, key="past_status_filter")
    if status_filter != "전체":
        filtered = filtered[filtered["상태"] == status_filter]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("상품 수", f"{len(filtered):,}")
    r2.metric("정상", f"{(filtered['상태'] == '정상').sum():,}")
    r3.metric("결품/위험", f"{filtered['상태'].isin(['결품 위험', '결품']).sum():,}")
    r4.metric("과발주/부진", f"{filtered['상태'].isin(['부진재고', '과발주 위험']).sum():,}")

    res_tbl = filtered[[
        "ITEM_CODE", "ITEM_NM", "출시일자", "BRAND", "ITEM_MDDV_NM", "ITEM_SMDV_NM",
        "초도발주량", "초기예약발주", "실출고량", "초도출고율(%)", "실수요", "상태",
    ]].rename(columns={
        "ITEM_CODE": "제품코드", "ITEM_NM": "제품명", "BRAND": "브랜드",
        "ITEM_MDDV_NM": "중분류", "ITEM_SMDV_NM": "소분류",
        "실출고량": "실출고량(7일치)", "초도출고율(%)": "초도출고율",
    }).copy()
    st.dataframe(res_tbl, use_container_width=True, height=360, hide_index=True)

    st.markdown("#### 중/소분류 집계")
    agg_level = st.radio("집계 기준", ["중분류", "소분류"], horizontal=True, key="past_status_agg")
    agg_col = "ITEM_MDDV_NM" if agg_level == "중분류" else "ITEM_SMDV_NM"
    cat_agg = filtered.groupby(agg_col).agg(
        상품수=("ITEM_CODE", "count"),
        총초도발주량=("초도발주량", "sum"),
        총초기예약발주=("초기예약발주", "sum"),
        총실출고량=("실출고량", "sum"),
        총실수요=("실수요", "sum"),
    ).reset_index().rename(columns={agg_col: agg_level})
    st.dataframe(cat_agg, use_container_width=True, height=260, hide_index=True)

    status_counts = filtered.groupby([agg_col, "상태"]).size().reset_index(name="상품수")
    fig_status = px.bar(
        status_counts,
        x=agg_col,
        y="상품수",
        color="상태",
        category_orders={"상태": OUTFLOW_STATUS_ORDER},
        color_discrete_map=OUTFLOW_STATUS_COLORS,
        height=340,
    )
    style_figure(fig_status)
    st.plotly_chart(fig_status, use_container_width=True)

    scatter_df = build_prediction_initial_outflow_scatter(
        predictions_df,
        preorder_df,
        filtered["ITEM_CODE"].astype(str).unique().tolist(),
        "전체",
    )
    if not scatter_df.empty:
        scatter_df = scatter_df[(scatter_df["INITIAL_ORD_QTY"] > 0) & (scatter_df["OUTFLOW_7D"] > 0)].copy()
    if scatter_df.empty:
        st.info("로그 스케일 그래프에 표시할 데이터가 없습니다.")
        return
    fig_sc = px.scatter(
        scatter_df,
        x="OUTFLOW_7D",
        y="INITIAL_ORD_QTY",
        color="상태",
        category_orders={"상태": OUTFLOW_STATUS_ORDER},
        color_discrete_map=OUTFLOW_STATUS_COLORS,
        hover_name="ITEM_NM",
        labels={"OUTFLOW_7D": "OPTIMAL = OUTFLOW_7D (log)", "INITIAL_ORD_QTY": "MD 초도발주량 (log)"},
        height=420,
    )
    min_axis = max(1, min(scatter_df["OUTFLOW_7D"].min(), scatter_df["INITIAL_ORD_QTY"].min()))
    max_axis = max(scatter_df["OUTFLOW_7D"].max(), scatter_df["INITIAL_ORD_QTY"].max())
    line_x = np.geomspace(min_axis, max_axis, 80)
    for label, multiplier, dash in [("y=x", 1.0, "dash"), ("MD = 2 x OPTIMAL", 2.0, "dot"), ("MD = 0.5 x OPTIMAL", 0.5, "dot")]:
        fig_sc.add_trace(go.Scatter(x=line_x, y=line_x * multiplier, mode="lines", line=dict(color="#9BA6BD", dash=dash), name=label, hoverinfo="skip"))
    fig_sc.update_xaxes(type="log")
    fig_sc.update_yaxes(type="log")
    style_figure(fig_sc)
    st.plotly_chart(fig_sc, use_container_width=True)


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
        ["과거 신상품 조회", "과거 Raw Data", "상품별 데이터", "상품별 상태 분석"]
    )
    with tabs[0]:
        render_past_lookup_overview(preorder_df, sales_df, predictions_df)
    with tabs[1]:
        render_past_raw_data_tab(preorder_df, sales_df, center_order_df, stock_df, base_date)
    with tabs[2]:
        render_past_product_data_detail(preorder_df, sales_df, predictions_df)
    with tabs[3]:
        render_past_status_analysis_tab(preorder_df, sales_df, predictions_df)


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

role_label = "MASTER" if is_master_user else "MD"
st.caption(f"접속 사용자: `{st.session_state.get('login_user', 'unknown')}` | 권한: `{role_label}`")

page_options = ["금주 신상품", "과거 신상품 조회", "MD 발주 시뮬레이션", "재고비용 시뮬레이션"]
with st.sidebar:
    logout_left, logout_right = st.columns([1, 0.72])
    with logout_right:
        if st.button("로그아웃", key="sidebar_logout_button", help="로그아웃", width="stretch"):
            st.session_state["is_logged_in"] = False
            st.session_state["is_master_user"] = False
            st.session_state.pop("login_user", None)
            st.rerun()
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
                st.info(f"{label}: 연결됨")
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

if selected_page == "MD 발주 시뮬레이션":
    with st.sidebar:
        st.header("파일 연결 상태")
        status_items = [
            ("예약주문 Raw", not full_preorder_df.empty),
            ("OUTFLOW_7D 예측", not full_predictions_df.empty),
        ]
        for label, connected in status_items:
            if connected:
                st.info(f"{label}: 연결됨")
            else:
                st.warning(f"{label}: 없음")

    render_md_order_simulation_tab(full_preorder_df, full_predictions_df)
    st.stop()

if selected_page == "재고비용 시뮬레이션":
    render_inventory_cost_page(
        full_stock_df,
        full_sales_df,
        full_center_order_df,
        full_preorder_df,
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

    excluded_labels = st.multiselect(
        "제외할 상품 선택",
        options=item_candidates["LABEL"].tolist(),
        default=[],
        help="검색 후 선택한 상품은 대시보드 집계에서 제외됩니다.",
    )
    excluded_items = [label.split(" | ", 1)[0] for label in excluded_labels]

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
    selected_items=[],
    selected_centers=selected_centers,
    selected_brands=selected_brands,
    date_range=selected_date_range,
)
if excluded_items:
    filtered_preorder = filtered_preorder[~filtered_preorder["ITEM_CODE"].astype(str).isin(excluded_items)].copy()

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

if st.session_state.get("weekly_view_mode", "list") != "detail":
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_kpi_card("누적 판매금액", format_won(kpis["누적 판매금액"]), "판매 실적")
    with kpi_cols[1]:
        render_kpi_card("초도 발주량", format_int(kpis["초도 발주량"]), f"{format_int(kpis['상품 수'])}개 상품")
    with kpi_cols[2]:
        render_kpi_card("사전 예약량", format_int(kpis["사전 예약량"]), f"{format_pct(kpis['예약/초도 비율'])}")
    with kpi_cols[3]:
        render_kpi_card("센터 커버리지", format_int(kpis["센터 수"]), f"평균 재고 {format_int(kpis['평균 재고'])}")

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
                {f"<br><span style='display:block;margin-top:0.65rem;color:#4f6df5;font-size:1.02rem;font-weight:800;line-height:1.55;'>&ldquo;{item_summary_text}&rdquo;</span>" if item_summary_text else ""}
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

    trend_options = ["전체"] + center_options
    trend_center = st.selectbox("예약주문 추이 센터", trend_options, key=f"weekly_trend_center_{selected_weekly_item}")
    if trend_center == "전체":
        trend_source = selected_profile.copy()
    else:
        trend_source = selected_center_profile[selected_center_profile["CENTER_NM"] == trend_center].copy()
    if not trend_source.empty:
        fig = px.line(
            trend_source,
            x="예약일자 표시",
            y="예약 수량",
            markers=True,
            title="예약주문 추이",
            labels={"예약일자 표시": "예약일자", "예약 수량": "예약 수량"},
            color_discrete_sequence=[TABLEAU_COLORS[0]],
            category_orders={"예약일자 표시": trend_source["예약일자 표시"].tolist()},
        )
        style_figure(fig)
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True, config={})

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
            fig.update_layout(coloraxis_showscale=False, height=420)
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

    detail_table = selected_center_plan.copy()
    if not detail_table.empty:
        detail_table["LDU"] = detail_table["CENTER_CODE"].map(
            lambda code: CENTER_WEIGHT_CONFIG.get(normalize_center_code(code), {}).get("ldu", "")
        ).replace("", pd.NA).fillna(detail_table["CENTER_NM"])
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
