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
            --blue: #0f7a4b;
            --orange: #ff6f1f;
            --green: #0f7a4b;
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
            background: linear-gradient(135deg, #0f7a4b, #ff6f1f);
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
            color: #0f7a4b;
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
            color: #0f7a4b;
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
            border-color: #0f7a4b !important;
            box-shadow: 0 0 0 1px #0f7a4b !important;
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
            background: linear-gradient(90deg, #0f7a4b, #ff6f1f) !important;
            color: white !important;
            border: none !important;
        }
        @media (max-width: 900px) {
            .login-title {
                font-size: 1.8rem;
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
        margin=dict(l=10, r=10, t=56, b=10),
        legend_title_text="",
        font=dict(size=13, color="#16202a"),
        title=dict(font=dict(size=18, color="#16202a")),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(217,221,231,0.65)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(217,221,231,0.65)", zeroline=False)
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
    plan["CENTER_CODE"] = plan["CENTER_CODE"].astype(str).str.strip()
    plan["센터 가중치"] = plan["CENTER_CODE"].map(
        lambda code: CENTER_WEIGHT_CONFIG.get(code, {}).get("weight", 1.0)
    )
    plan["기준 점포수"] = plan["CENTER_CODE"].map(
        lambda code: CENTER_WEIGHT_CONFIG.get(code, {}).get("store_count")
    )
    plan["LDU"] = plan["CENTER_CODE"].map(
        lambda code: CENTER_WEIGHT_CONFIG.get(code, {}).get("ldu", plan.loc[plan["CENTER_CODE"] == code, "CENTER_NM"].iloc[0])
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
    df["CENTER_CODE"] = df["CENTER_CODE"].astype(str).str.strip()
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
    df["CENTER_CODE"] = df["CENTER_CODE"].astype(str).str.strip()
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
        df["CENTER_CODE"] = df["CENT_CD"].astype(str).str.strip()
    if "SUM(A.CONV_QTY)" in df.columns:
        df["CONV_QTY"] = clean_numeric(df["SUM(A.CONV_QTY)"]).fillna(0)
    else:
        df["CONV_QTY"] = 0
    return df


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
    center_order_df: pd.DataFrame,
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

    if not center_order_df.empty:
        shipped = (
            center_order_df.groupby("ITEM_CODE", as_index=False)["CONV_QTY"]
            .sum()
            .rename(columns={"CONV_QTY": "실출고량"})
        )
        item_df = item_df.merge(shipped, on="ITEM_CODE", how="left")
    item_df["실출고량"] = item_df.get("실출고량", 0).fillna(0)

    safe_initial = item_df["초도발주량"].replace(0, pd.NA)
    safe_ship = item_df["실출고량"].replace(0, pd.NA)
    item_df["실제출고율(%)"] = (item_df["실출고량"] / safe_initial * 100).round(1)
    item_df["결품여부"] = item_df["실수요량"] > item_df["실출고량"]
    item_df["부진여부"] = (item_df["실출고량"] > 0) & ((item_df["실수요량"] / safe_ship) < 0.5)
    item_df["상태"] = item_df.apply(
        lambda row: "결품" if row["결품여부"] else ("부진" if row["부진여부"] else "정상"),
        axis=1,
    )
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
    base_date: pd.Timestamp,
) -> None:
    st.subheader("예약/수요 비교")

    item_df = build_past_reference_item_analysis(preorder_df, sales_df, center_order_df)
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
                color_discrete_sequence=["#1D4ED8"],
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
        if not center_order_df.empty:
            center_ship = (
                center_order_df[center_order_df["ITEM_CODE"].astype(str) == selected_item_code]
                .groupby("CENTER_CODE", as_index=False)["CONV_QTY"]
                .sum()
                .rename(columns={"CENTER_CODE": "센터코드", "CONV_QTY": "실출고량"})
            )
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
            go.Bar(name="사전예약발주", x=center_merged["센터명"], y=center_merged["사전예약발주"], marker_color="#1D4ED8")
        )
        fig_c.add_trace(
            go.Bar(name="실수요량", x=center_merged["센터명"], y=center_merged["실수요량"], marker_color="#93C5FD")
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


def render_past_simple_lookup(preorder_df: pd.DataFrame, center_order_df: pd.DataFrame) -> None:
    st.subheader("과거 신상품/유사 사례 조회")
    target = preorder_df if not preorder_df.empty else center_order_df
    if target.empty:
        st.info("조회할 과거 상품 데이터가 없습니다.")
        return

    query = st.text_input(
        "상품명 또는 키워드",
        placeholder="예: 초코, 감자, 젤리",
        key="past_simple_lookup_query",
    )
    filtered = target.copy()
    if query.strip() and "ITEM_NM" in filtered.columns:
        filtered = filtered[filtered["ITEM_NM"].astype(str).str.contains(query, case=False, na=False)]

    show_columns = [
        column
        for column in [
            "ITEM_CODE",
            "ITEM_NM",
            "CENTER_NM",
            "ITEM_MDDV_NM",
            "ITEM_SMDV_NM",
            "NP_RLSE_YMD",
            "INITIAL_ORD_QTY",
            "total_pre_order_qty(D-11~D-8)",
        ]
        if column in filtered.columns
    ]
    st.dataframe(filtered[show_columns], width="stretch", height=320, hide_index=True)


def render_past_category_compare(preorder_df: pd.DataFrame, sales_df: pd.DataFrame, base_date: pd.Timestamp) -> None:
    st.subheader("중/소분류별 예약주문 · 초도발주 · 실수요 비교")
    st.caption("상품군별 흐름을 살펴본 뒤, 선택 상품의 센터 상세를 함께 확인합니다.")

    analysis = build_preorder_sales_analysis(preorder_df, sales_df)
    if analysis.empty:
        st.info("카테고리 비교를 만들기 위한 데이터가 부족합니다.")
        return
    analysis = analysis[analysis["NP_RLSE_DATE"].le(base_date)].copy()
    if analysis.empty:
        st.info("기준일 이전 데이터가 없습니다.")
        return

    mddv_options = sorted(analysis["ITEM_MDDV_NM"].dropna().astype(str).unique().tolist())
    selected_mddv = st.selectbox("중분류", mddv_options, key="past_category_compare_mddv")
    scoped = analysis[analysis["ITEM_MDDV_NM"].astype(str) == selected_mddv].copy()

    smdv_options = sorted(scoped["ITEM_SMDV_NM"].dropna().astype(str).unique().tolist())
    selected_smdv = st.selectbox("소분류", smdv_options, key="past_category_compare_smdv")
    scoped = scoped[scoped["ITEM_SMDV_NM"].astype(str) == selected_smdv].copy()

    item_options = (
        scoped[["ITEM_CODE", "ITEM_NM"]]
        .drop_duplicates()
        .sort_values(["ITEM_NM", "ITEM_CODE"])
        .assign(label=lambda df: df["ITEM_NM"].astype(str) + " (" + df["ITEM_CODE"].astype(str) + ")")
    )
    selected_item_label = st.selectbox("상품", item_options["label"].tolist(), key="past_category_compare_item")
    selected_item_code = str(item_options.loc[item_options["label"] == selected_item_label, "ITEM_CODE"].iloc[0])

    item_scoped = scoped[scoped["ITEM_CODE"].astype(str) == selected_item_code].copy()
    metrics = item_scoped[["preorder_qty", "initial_order_qty", "actual_sales_qty_7d", "over_order_gap"]].sum()
    top_cols = st.columns(4)
    top_cols[0].metric("예약주문량", f"{metrics['preorder_qty']:,.0f}")
    top_cols[1].metric("초도발주량", f"{metrics['initial_order_qty']:,.0f}")
    top_cols[2].metric("매출 기반 실수요량", f"{metrics['actual_sales_qty_7d']:,.0f}")
    top_cols[3].metric("과발주 갭", f"{metrics['over_order_gap']:,.0f}")

    left, right = st.columns([1.2, 1])
    chart_source = pd.DataFrame(
        {
            "지표": ["예약주문량", "초도발주량", "매출 기반 실수요량"],
            "수량": [metrics["preorder_qty"], metrics["initial_order_qty"], metrics["actual_sales_qty_7d"]],
        }
    )
    left.bar_chart(chart_source.set_index("지표"), width="stretch")
    category_summary = (
        scoped.groupby(["ITEM_MDDV_NM", "ITEM_SMDV_NM"], as_index=False)[
            ["preorder_qty", "initial_order_qty", "actual_sales_qty_7d"]
        ]
        .sum()
        .rename(
            columns={
                "ITEM_MDDV_NM": "중분류",
                "ITEM_SMDV_NM": "소분류",
                "preorder_qty": "예약주문량",
                "initial_order_qty": "초도발주량",
                "actual_sales_qty_7d": "매출 기반 실수요량(7일)",
            }
        )
    )
    right.dataframe(category_summary, width="stretch", height=220, hide_index=True)

    detail_display = item_scoped[
        ["CENTER_NM", "NP_RLSE_DATE", "preorder_qty", "initial_order_qty", "actual_sales_qty_7d", "over_order_gap"]
    ].rename(
        columns={
            "CENTER_NM": "센터",
            "NP_RLSE_DATE": "출시일",
            "preorder_qty": "예약주문량",
            "initial_order_qty": "초도발주량",
            "actual_sales_qty_7d": "매출 기반 실수요량(7일)",
            "over_order_gap": "과발주 갭",
        }
    )
    st.dataframe(detail_display, width="stretch", height=320, hide_index=True)


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
    base_date: pd.Timestamp,
) -> None:
    st.markdown("## 신상품 과거 Raw Data")
    tabs = st.tabs(
        ["제품별 데이터", "예약/수요 비교", "과거 Raw Data", "과거 신상품 조회", "카테고리 비교", "기준일 신상품"]
    )
    with tabs[0]:
        render_past_product_data_tab(preorder_df, sales_df, base_date)
    with tabs[1]:
        render_past_product_lookup(preorder_df, sales_df, center_order_df, base_date)
    with tabs[2]:
        render_past_raw_data_tab(preorder_df, sales_df, center_order_df, stock_df, base_date)
    with tabs[3]:
        render_past_simple_lookup(preorder_df, center_order_df)
    with tabs[4]:
        render_past_category_compare(preorder_df, sales_df, base_date)
    with tabs[5]:
        render_past_current_release_focus(preorder_df, base_date)


preorder_df = load_preorder()
sales_df = load_sales()
stock_df = load_stock()
center_order_df = load_center_order()
item_master = build_item_master(preorder_df)
center_master = build_center_master(preorder_df)

full_preorder_df = preorder_df.copy()
full_sales_df = sales_df.copy()
full_stock_df = stock_df.copy()
full_center_order_df = center_order_df.copy()
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

page_options = ["현재 신상품", "과거 신상품 조회"]
selected_page = st.radio(
    "메뉴 선택",
    options=page_options,
    horizontal=True,
    key="app_page_selector",
    label_visibility="collapsed",
)

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
        past_base_date,
    )
    st.stop()

st.markdown(
    """
    <div class="hero">
        <h1>신상품 센터 운영 대시보드</h1>
        <p>예약 반응, 초도 발주, 실제 판매, 잔여 재고를 한 화면에서 겹쳐 보며 센터별 발주 적정성을 빠르게 판단할 수 있게 구성했습니다.</p>
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
            x="예약일자",
            y="예약 수량",
            markers=True,
            title="전체 예약주문 추이",
            labels={"예약일자": "예약일자", "예약 수량": "예약 수량"},
            color_discrete_sequence=[TABLEAU_COLORS[0]],
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
                color_continuous_scale=["#d7efe4", "#008061"],
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
                    "적정": "#0f7a4b",
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
                x="예약일자",
                y="예약 수량",
                markers=True,
                title="",
                labels={"예약일자": "예약일자", "예약 수량": "예약 수량"},
                color_discrete_sequence=[TABLEAU_COLORS[1]],
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
