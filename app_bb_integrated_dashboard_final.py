import os
import json
import re
import html
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import streamlit as st

import requests
import circlify
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# Fixed file names
# =========================================================
BUBBLE_CSV = "year_tag_counts_all_ranked.csv"
ELEMENTS_CSV = "year_element_mentions_with_examples_type8.csv"
GAMES_CSV = "year_game_top10_with_examples.csv"
BEAVERROCKS_URL = "https://beaverrocks.com/"
FULLTEXT_CSV = "burningbeaver_final_fulltext.csv"
OFFICIAL_GAMES_CSV = "BB2022_2025_games.csv"
OFFICIAL_ELEMENTS_CSV = "BB2022_2025_elements_final.csv"


# =========================================================
# Page
# =========================================================
st.set_page_config(page_title="Burning Beaver (Beaver Rocks) — Dashboard", layout="wide")

CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

:root{
  --bb-red: #E53935;
  --bb-orange: #FF6D00;
  --bb-bg: #FFF5EF;
  --bb-card: rgba(255,255,255,0.94);
  --bb-border: rgba(0,0,0,0.10);
  --bb-text: #171717;
  --bb-muted: rgba(23,23,23,0.66);
}

html, body, [class*="css"], [data-testid="stAppViewContainer"]  {
  font-family: "Pretendard", system-ui, -apple-system, "Segoe UI", Roboto, Arial, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif !important;
  color: var(--bb-text);
  font-size: 18px;
}

[data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 600px at 10% 0%, rgba(255,109,0,0.10), transparent 60%),
              radial-gradient(900px 500px at 90% 10%, rgba(229,57,53,0.10), transparent 55%),
              linear-gradient(180deg, var(--bb-bg), #FFFFFF 45%);
}

.hero{
  padding: 16px 18px;
  border: 1px solid var(--bb-border);
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(255,109,0,0.20), rgba(229,57,53,0.16));
  box-shadow: 0 10px 24px rgba(0,0,0,0.05);
  margin-bottom: 10px;
}
.hero h1{
  margin: 0;
  font-size: 32px;
  font-weight: 900;
  letter-spacing: -0.3px;
}
.hero .sub{
  margin-top: 8px;
  color: var(--bb-muted);
  font-weight: 800;
  font-size: 16px;
}
.chips{
  margin-top: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.chip{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 11px;
  border-radius: 999px;
  border: 1px solid var(--bb-border);
  background: rgba(255,255,255,0.78);
  font-size: 13px;
  font-weight: 900;
}
.chip a{ color: inherit; text-decoration: none; }
.chip:hover{ transform: translateY(-1px); transition: 120ms ease; }

.card{
  border: 1px solid var(--bb-border);
  border-radius: 18px;
  background: var(--bb-card);
  box-shadow: 0 10px 22px rgba(0,0,0,0.04);
  padding: 12px 14px;
}
.kicker{
  color: var(--bb-muted);
  font-size: 13px;
  font-weight: 900;
  margin-bottom: 6px;
}
.section-title{
  font-size: 21px;
  font-weight: 900;
  margin: 6px 0 10px;
}
.hl{
  background: rgba(255,109,0,0.30);
  padding: 0 4px;
  border-radius: 7px;
  font-weight: 900;
}
.small-note{
  color: rgba(23,23,23,0.72);
  font-size: 13px;
  font-weight: 700;
}

.emoji, .emoji *{
  font-family: "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", "Pretendard", system-ui, sans-serif !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =========================================================
# Secrets / Key loading
# =========================================================
def get_secret(name: str) -> Optional[str]:
    try:
        v = st.secrets.get(name)
        if v:
            return str(v)
    except Exception:
        pass
    v = os.environ.get(name)
    return str(v) if v else None


DEFAULT_OPENAI_MODEL = get_secret("DEFAULT_OPENAI_MODEL") or "gpt-5-mini"

AI_SYSTEM = """너는 Burning Beaver(Beaver Rocks) 행사의 기획/운영진을 돕는 데이터 분석 보조자다.

반드시 지킬 규칙:
- 출력은 한국어로만 작성한다.
- 모든 문장은 존댓말(합니다/됩니다/세요)로 끝나야 한다. 반말 금지.
- 제공된 데이터(연도별 키워드/언급 글 수/예시 일부/공식 리스트 정보/원문 스니펫)만 근거로 쓴다.
- 과장/단정 금지. 추측이면 '가설'이라고 명시한다.
- AI 요약/2026 방향성 버튼 출력은 반드시 한 줄(1문장)로 작성하고, 다른 섹션 내용을 덧붙이지 않는다.
- 챗봇 답변은 쉬운 한국어로 작성하되, 가능한 3~7줄 내로 정리한다.
"""

AI_ACCESS_CODE = get_secret("AI_ACCESS_CODE")  # optional gate


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ 설정")

    st.markdown("### 🫧 태그 버블")
    bubble_top_n = st.slider("Top N", min_value=3, max_value=8, value=6, step=1)
    bubble_show_count = st.checkbox("라벨에 언급량 표시", value=True)

    st.divider()

    st.markdown("### 🧩 운영요소")
    elem_top_n = st.slider("연도별 Top N", 5, 10, 10, 1)
    elem_type = st.selectbox(
        "분류(type)",
        ["전체", "연사/인플루언서", "굿즈/리워드", "장소/공간", "파트너쉽", "이벤트", "운영인력", "게임/IP", "네트워킹"]
    )
    elem_q = st.text_input("키워드 검색", value="").strip()

    st.divider()

    st.markdown("### 🎮 전시 게임")
    game_top_n = st.slider("연도별 Top N(트리맵)", min_value=5, max_value=10, value=10, step=1)

    st.divider()

    st.markdown("### <span class=\"emoji\">🤖</span> AI (요약/추천/챗봇)", unsafe_allow_html=True)
    ai_enabled = st.toggle("AI 기능 켜기", value=False)

    # ✅ 요청: 사이드바에서 키 입력(공유 평가용)
    if "openai_key_input" not in st.session_state:
        st.session_state.openai_key_input = ""

    openai_key_input = st.text_input(
        "OpenAI API Key (선택)",
        type="password",
        value=st.session_state.openai_key_input,
        help="공유용으로 secrets/env에 키를 넣지 않을 때만 사용하세요. (코드/깃허브에 절대 하드코딩 금지)"
    )
    st.session_state.openai_key_input = openai_key_input

    ai_model = st.text_input("모델", value=DEFAULT_OPENAI_MODEL)
    # ✅ 고정 파라미터(공유/평가용): 사용자가 조정 불가
    ai_char_limit = 500
    ai_temp = 0.4

    # Optional access gate for public sharing
    ai_unlocked = True
    if AI_ACCESS_CODE:
        st.caption("🔒 AI 잠금(공유용): 접근 코드가 필요합니다.")
        entered = st.text_input("AI 접근 코드", type="password")
        ai_unlocked = bool(entered) and entered == AI_ACCESS_CODE


def get_openai_api_key() -> Optional[str]:
    # 우선순위: 사이드바 입력 > secrets/env
    if st.session_state.get("openai_key_input"):
        return str(st.session_state["openai_key_input"]).strip()
    return get_secret("OPENAI_API_KEY")


# =========================================================
# Loaders
# =========================================================
@st.cache_data(show_spinner=False)
def load_bubble(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    df["tag"] = df["tag"].astype(str)
    df["MentionedPosts"] = pd.to_numeric(df["MentionedPosts"], errors="coerce").fillna(0).astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_elements(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "element_display" in df.columns:
        df["keyword"] = df["element_display"].astype(str)
    elif "keyword" in df.columns:
        df["keyword"] = df["keyword"].astype(str)
    else:
        df["keyword"] = df["element"].astype(str)

    if "examples_json" not in df.columns:
        df["examples_json"] = "[]"
    if "type" not in df.columns:
        df["type"] = "unknown"

    df["year"] = df["year"].astype(int)
    df["MentionedPosts"] = pd.to_numeric(df["MentionedPosts"], errors="coerce").fillna(0).astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_games(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["year"] = df["year"].astype(int)
    df["MentionedPosts"] = pd.to_numeric(df["MentionedPosts"], errors="coerce").fillna(0).astype(int)
    df["game"] = df["game"].astype(str)
    return df


# =========================================================
# Header
# =========================================================
st.markdown(
    f"""
    <div class="hero">
      <h1>🦫🔥 Burning Beaver (Beaver Rocks) — 후기 통합 대시보드</h1>
      <div class="sub">네이버 블로그 후기 기반 #이미지 키워드 #운영요소 #전시 게임</div>
      <div class="chips">
        <span class="chip">🗓️ 2022–2025</span>
        <span class="chip">🌐 <a href="{BEAVERROCKS_URL}" target="_blank" rel="noreferrer">Beaver Rocks 홈페이지 바로가기</a></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Helpers
# =========================================================
TYPE_EMOJI = {
    "연사/인플루언서": "🎤",
    "굿즈/리워드": "🎁",
    "장소/공간": "🗺️",
    "파트너쉽": "🤝",
    "이벤트": "🎪",
    "운영인력": "🔧",
    "게임/IP": "🎮",
    "네트워킹": "🌐",
}



@st.cache_data(show_spinner=False)
def load_fulltext(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["title","link","full_text","year"])
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    for c in ["title", "link", "full_text"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_official_games(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["No","Name","Year"])
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)
    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str)
    return df

@st.cache_data(show_spinner=False)
def load_official_elements(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def em(t: str) -> str:
    return TYPE_EMOJI.get(str(t).strip(), "✨")

def parse_examples(examples_json: str) -> List[Dict[str, str]]:
    try:
        arr = json.loads(examples_json) if isinstance(examples_json, str) else []
    except Exception:
        arr = []
    cleaned = []
    for ex in arr[:5]:
        cleaned.append({
            "context": (ex.get("context") or "").strip(),
            "title": (ex.get("title") or "").strip(),
            "link": (ex.get("link") or "").strip(),
        })
    return cleaned

def highlight(text: str) -> str:
    safe = (text or "").replace("\n", " ").strip()
    safe = re.sub(r"\s+", " ", safe)
    safe = safe.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = safe.replace("[[", "<span class='hl'>").replace("]]", "</span>")
    return safe

def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)

def lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)

def color_by_value(val: int, vmin: int, vmax: int) -> str:
    start = hex_to_rgb("#FFCC80")  # light orange
    mid   = hex_to_rgb("#FF6D00")  # orange
    end   = hex_to_rgb("#C62828")  # deep red

    if vmax <= vmin:
        t = 1.0
    else:
        t = (val - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))

    if t < 0.55:
        tt = t / 0.55
        rgb = (lerp(start[0], mid[0], tt), lerp(start[1], mid[1], tt), lerp(start[2], mid[2], tt))
    else:
        tt = (t - 0.55) / 0.45
        rgb = (lerp(mid[0], end[0], tt), lerp(mid[1], end[1], tt), lerp(mid[2], end[2], tt))
    return rgb_to_hex(rgb)

def wrap_label(s: str, width: int = 9) -> str:
    s = s.strip()
    if len(s) <= width:
        return s
    if " " in s:
        parts, cur = [], ""
        for w in s.split():
            if len(cur) + len(w) + 1 <= width:
                cur = (cur + " " + w).strip()
            else:
                parts.append(cur)
                cur = w
        if cur:
            parts.append(cur)
        parts = parts[:2]
        if len(parts) == 2:
            parts[-1] += "…"
        return "<br>".join(parts)
    return s[:width] + "<br>" + s[width:width*2] + ("…" if len(s) > width*2 else "")
def normalize_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_light_sentences(text: str, max_sent: int = 10) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) > 13:
        lines = lines[13:]
    joined = " ".join(lines)
    parts = re.split(r"(?<=[\.\!\?\。\！\？…])\s+|(?<=[다요죠임함])\s+", joined)
    parts = [p.strip() for p in parts if p and len(p.strip()) >= 12]
    return parts[:max_sent]

def retrieve_fulltext_snippets(df_full: pd.DataFrame, query: str, year: Optional[int], max_posts: int = 2) -> List[str]:
    q = (query or "").strip()
    if not q or df_full is None or df_full.empty or "full_text" not in df_full.columns:
        return []
    tokens = [t for t in re.split(r"[^\w가-힣]+", q) if len(t) >= 2][:6]
    if not tokens:
        return []
    pat = "(" + "|".join([re.escape(t) for t in tokens]) + ")"
    sub = df_full
    if year is not None and "year" in sub.columns:
        sub = sub[sub["year"] == int(year)]
    cand = sub[sub["full_text"].astype(str).str.contains(pat, case=False, na=False, regex=True)].head(18)
    snippets = []
    for _, r in cand.iterrows():
        sents = split_light_sentences(str(r.get("full_text", "")), max_sent=12)
        chosen = [ss for ss in sents if re.search(pat, ss, flags=re.IGNORECASE)][:2]
        if chosen:
            snippets.append(" / ".join(chosen)[:420])
        if len(snippets) >= max_posts:
            break
    return snippets

def detect_official_game_mentions(df_off: pd.DataFrame, query: str, year: Optional[int]) -> List[str]:
    if df_off is None or df_off.empty or "Name" not in df_off.columns:
        return []
    qn = normalize_text(query)
    if not qn:
        return []
    sub = df_off
    if year is not None and "Year" in sub.columns:
        sub = sub[sub["Year"] == int(year)]
    hits = []
    for name in sub["Name"].dropna().astype(str).tolist():
        n = normalize_text(name)
        if len(n) >= 3 and n in qn:
            hits.append(name)
        if len(hits) >= 8:
            break
    return hits


def detect_official_element_mentions(df_off_el: pd.DataFrame, query: str, year: Optional[int]) -> List[str]:
    if df_off_el is None or df_off_el.empty or "element" not in df_off_el.columns:
        return []
    qn = normalize_text(query)
    if not qn:
        return []
    sub = df_off_el
    if year is not None and "Year" in sub.columns:
        sub = sub[sub["Year"] == int(year)]
    hits = []
    for el in sub["element"].dropna().astype(str).tolist():
        n = normalize_text(el)
        if len(n) >= 2 and n in qn:
            hits.append(el)
        if len(hits) >= 10:
            break
    return hits


def bubble_plotly(year_df: pd.DataFrame, show_count: bool, height: int = 330):
    """
    ✅ 요청 반영:
    - 버블 라벨 배경 네모 제거 (bgcolor/border 삭제)
    - 글자는 검은색
    - 글자 크기 작게
    """
    year_df = year_df.sort_values(["MentionedPosts", "tag"], ascending=[False, True]).copy()
    year_df = year_df[year_df["MentionedPosts"] > 0].reset_index(drop=True)

    data = [{"id": str(t), "datum": int(v)} for t, v in zip(year_df["tag"], year_df["MentionedPosts"])]
    if not data:
        return go.Figure(), year_df

    circles = circlify.circlify(
        data,
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1),
    )

    leaves = [c for c in circles if isinstance(getattr(c, "ex", None), dict) and "id" in c.ex and "datum" in c.ex]
    leaves = sorted(leaves, key=lambda c: c.ex["datum"], reverse=True)

    values = [c.ex["datum"] for c in leaves]
    vmin, vmax = min(values), max(values)

    fig = go.Figure()
    for c in leaves:
        x, y, r = c.x, c.y, c.r
        tag = str(c.ex["id"])
        cnt = int(c.ex["datum"])

        fill = color_by_value(cnt, vmin, vmax)

        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=x - r, y0=y - r, x1=x + r, y1=y + r,
            line=dict(color="rgba(255,255,255,0.60)", width=2),
            fillcolor=fill,
            opacity=0.97,
            layer="below",
        )

        base = r * 34 + 5
        penalty = max(0, len(tag) - 7)
        font_size = int(base - penalty * 1.0)
        font_size = max(8, min(14, font_size))

        safe_tag = html.escape(tag)
        safe_tag = wrap_label(safe_tag, width=9)

        show_cnt_here = bool(show_count and r >= 0.16)
        label = f"{safe_tag}<br>({cnt})" if show_cnt_here else f"{safe_tag}"

        fig.add_annotation(
            x=x, y=y,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=font_size, color="#111111", family="Pretendard, sans-serif"),
            align="center",
        )

    fig.update_xaxes(visible=False, range=[-1.05, 1.05])
    fig.update_yaxes(visible=False, range=[-1.05, 1.05], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
    )
    return fig, year_df


# =========================================================
# OpenAI (HTTP) - Responses API
# =========================================================
def extract_output_text(resp_json: Dict[str, Any]) -> str:
    """Responses API 응답에서 사람이 읽을 텍스트만 안정적으로 추출."""
    if not isinstance(resp_json, dict):
        return ""

    ot = resp_json.get("output_text")
    if isinstance(ot, str) and ot.strip():
        return ot.strip()

    out = resp_json.get("output")
    if isinstance(out, list):
        parts = []
        for item in out:
            if not isinstance(item, dict):
                continue

            if item.get("type") == "message":
                content = item.get("content") or []
                if isinstance(content, list):
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                        if c.get("type") in ("output_text", "text"):
                            txt = c.get("text")
                            if isinstance(txt, str) and txt.strip():
                                parts.append(txt.strip())

            if item.get("type") in ("output_text", "text"):
                txt = item.get("text")
                if isinstance(txt, str) and txt.strip():
                    parts.append(txt.strip())

        if parts:
            return "\n".join(parts).strip()

    return ""


def supports_temperature(model: str) -> bool:
    m = str(model).strip()
    # 일부 모델은 temperature 전달 시 400이 날 수 있어 안전하게 제한
    return m.startswith("gpt-5.2") or m.startswith("gpt-4")

def supports_reasoning_param(model: str) -> bool:
    m = str(model).strip()
    return m.startswith("gpt-5") or m.startswith("o")

def call_openai_responses(
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_output_tokens: int = 1800,
    temperature: Optional[float] = None,
    timeout: int = 90,
    reasoning_effort: str = "low",
    retries: int = 2,
) -> Tuple[str, Dict[str, Any]]:
    """
    - 텍스트가 비거나(incomplete) JSON만 보이는 문제 방지:
      max_output_tokens를 넉넉히 주고, 텍스트가 없으면 자동 재시도.
    - temperature 미지원 모델은 자동으로 제거하고 재시도.
    """
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def _post(use_temp: bool, mot: int, use_reasoning: bool) -> Tuple[str, Dict[str, Any], int, str]:
        payload: Dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_output_tokens": int(mot),
            "text": {"format": {"type": "text"}},
        }

        if use_reasoning and supports_reasoning_param(model):
            payload["reasoning"] = {"effort": reasoning_effort}

        if use_temp and (temperature is not None) and supports_temperature(model):
            payload["temperature"] = float(temperature)

        r = requests.post(url, headers=headers, json=payload, timeout=timeout)

        try:
            parsed = r.json() if r.content else {}
        except Exception:
            parsed = {}

        data: Dict[str, Any] = parsed if isinstance(parsed, dict) else {}

        err_obj = data.get("error")
        msg = err_obj.get("message") if isinstance(err_obj, dict) else None
        err_text = msg or (r.text or "")

        text = extract_output_text(data)
        return text, data, r.status_code, err_text

    mot = int(max_output_tokens)
    use_reasoning = True

    text, data, status, err = _post(True, mot, use_reasoning)

    # temperature 미지원 자동 재시도
    if status >= 400 and "Unsupported parameter" in str(err) and "temperature" in str(err):
        text, data, status, err = _post(False, mot, use_reasoning)

    if status >= 400:
        raise RuntimeError(f"OpenAI API error ({status}): {err}")

    for _ in range(int(retries)):
        if text and text.strip():
            break

        resp_status = str(data.get("status") or "").lower()
        inc = data.get("incomplete_details") or {}
        reason = (inc.get("reason") if isinstance(inc, dict) else "") or ""

        if resp_status == "incomplete" and "max_output_tokens" in str(reason):
            mot = int(mot * 2.0)
            text, data, status, err = _post(False, mot, use_reasoning)
            continue

        if use_reasoning:
            use_reasoning = False
            text, data, status, err = _post(False, mot, use_reasoning)
            continue

        break

    if not text or not text.strip():
        resp_status = data.get("status")
        inc = data.get("incomplete_details")
        return f"AI 응답 텍스트가 비어있습니다. (status={resp_status}, incomplete_details={inc})", data

    return text.strip(), data


def ai_available() -> bool:
    return bool(ai_enabled and ai_unlocked and get_openai_api_key())


def ai_guardrail_banner():
    if not ai_enabled:
        st.info("AI 기능은 사이드바에서 켤 수 있어요. (기본 OFF)")
        return
    if AI_ACCESS_CODE and not ai_unlocked:
        st.warning("AI 접근 코드가 필요합니다.")
        return
    if not get_openai_api_key():
        st.warning("OpenAI API Key가 없습니다. (사이드바에 입력하거나 secrets/env에 설정)")
        return


def rate_limit_ok(max_calls: int = 30) -> bool:
    if "ai_calls" not in st.session_state:
        st.session_state.ai_calls = 0
    return st.session_state.ai_calls < max_calls


def bump_calls():
    st.session_state.ai_calls = int(st.session_state.get("ai_calls", 0)) + 1


def build_context_pack(
    df_b: pd.DataFrame,
    df_e: pd.DataFrame,
    df_g: pd.DataFrame,
    bubble_n: int,
    elem_n: int,
    game_n: int,
    elem_type_filter: str,
    elem_q_filter: str,
    df_off_games: Optional[pd.DataFrame] = None,
    df_off_elements: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    years = [2022, 2023, 2024, 2025]

    tags = {}
    for y in years:
        d = df_b[df_b["year"] == y].sort_values(["MentionedPosts", "tag"], ascending=[False, True]).head(bubble_n)
        tags[str(y)] = [{"tag": r["tag"], "posts": int(r["MentionedPosts"])} for _, r in d.iterrows()]

    ebase = df_e.copy()
    if elem_type_filter != "전체":
        ebase = ebase[ebase["type"] == elem_type_filter]
    if elem_q_filter:
        ebase = ebase[ebase["keyword"].str.contains(elem_q_filter, case=False, na=False)]

    elements = {}
    for y in years:
        d = ebase[ebase["year"] == y].sort_values("MentionedPosts", ascending=False).head(elem_n).copy()
        rows = []
        for _, r in d.iterrows():
            ex = parse_examples(r.get("examples_json", "[]"))
            rows.append({
                "type": str(r.get("type", "")),
                "keyword": str(r.get("keyword", "")),
                "posts": int(r.get("MentionedPosts", 0)),
                "examples": [x.get("context", "")[:200] for x in ex[:2] if x.get("context")],
            })
        elements[str(y)] = rows

    games = {}
    for y in years:
        d = df_g[df_g["year"] == y].sort_values(["MentionedPosts", "game"], ascending=[False, True]).head(game_n)
        games[str(y)] = [{"game": r["game"], "posts": int(r["MentionedPosts"])} for _, r in d.iterrows()]

    official_meta: Dict[str, Any] = {}

    if df_off_games is not None and not df_off_games.empty and "Year" in df_off_games.columns:
        for y in years:
            official_meta.setdefault(str(y), {})
            official_meta[str(y)]["official_games_cnt"] = int((df_off_games["Year"] == y).sum())

    if df_off_elements is not None and not df_off_elements.empty:
        # expected cols: Year, element, type
        if "Year" in df_off_elements.columns:
            for y in years:
                official_meta.setdefault(str(y), {})
                official_meta[str(y)]["official_elements_cnt"] = int((df_off_elements["Year"] == y).sum())
                if "type" in df_off_elements.columns:
                    vc = df_off_elements[df_off_elements["Year"] == y]["type"].astype(str).value_counts()
                    official_meta[str(y)]["official_elements_by_type"] = vc.to_dict()
        official_meta["official_elements_rows"] = int(len(df_off_elements))

    return {"years": years, "tags": tags, "elements": elements, "games": games, "official_meta": official_meta}


def enforce_char_limit(text: str, limit: int) -> str:
    t = (text or "").strip()
    if len(t) <= limit:
        return t
    return t[:max(0, limit-1)] + "…"




def pick_one_liner(raw_text: str, mode: str) -> str:
    """
    mode:
      - "summary": '2022→2025 변천사:' 한 문장만
      - "plan":    '2026 제안:' 한 문장만
    모델이 지시를 어기고 다른 섹션을 덧붙일 때를 대비해 방어적으로 잘라냅니다.
    """
    t = (raw_text or "").strip()

    # 1) 우선 prefix 기반으로 추출
    if mode == "summary":
        pref = "2022→2025 변천사:"
        # 다양한 공백/화살표 변형 허용
        m = re.search(r"(2022\s*→\s*2025\s*변천사\s*:\s*.+)", t)
        if m:
            t = m.group(1).strip()
    elif mode == "plan":
        pref = "2026 제안:"
        m = re.search(r"(2026\s*제안\s*:\s*.+)", t)
        if m:
            t = m.group(1).strip()
    else:
        pref = ""

    # 2) 다른 섹션/메타 문구가 뒤에 붙는 경우 잘라내기
    cut_markers = [
        "AI 요약", "2026 방향성", "AI 챗봇", "버튼:", "버튼 :", "incomplete", '{"id"', '"object":', "본 요약",
    ]
    # summary에만 적용(2026은 요약에서 특히 문제)
    if mode == "summary":
        cut_markers = ["2026 제안", "2026 방향성", "2026년"] + cut_markers

    # 가장 먼저 등장하는 marker 위치에서 잘라냄(단, prefix 자체는 보존)
    earliest = None
    for mk in cut_markers:
        pos = t.find(mk)
        if pos != -1:
            # prefix가 marker 안에 있는 건 제외
            if pref and pos <= t.find(pref) + len(pref):
                continue
            if earliest is None or pos < earliest:
                earliest = pos
    if earliest is not None and earliest > 0:
        t = t[:earliest].rstrip()

    # 3) 첫 문장만 남기기(마침표/물음표/느낌표 기준)
    # 한국어 문장은 보통 '.'로 끝나므로 '.' 기준 우선, 없으면 그대로 둠.
    # 다만 "2022→2025 변천사:" 는 한 문장만 요구하므로 첫 '.'까지 유지
    sentence_end = None
    for p in [".", "!", "?"]:
        idx = t.find(p)
        if idx != -1:
            sentence_end = idx
            break
    if sentence_end is not None:
        t = t[:sentence_end+1].strip()

    # 4) 줄바꿈/여백 정리
    t = t.replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t).strip()

    # 5) 존댓말 종결 방어(끝이 너무 잘리면 '입니다.' 붙임)
    if t and not re.search(r"(습니다\.?$|세요\.?$|해요\.?$|돼요\.?$|입니다\.?$|합니다\.?$|됩니다\.?$)", t):
        t = t.rstrip(" ,;:")
        if t.endswith("."):
            # 이미 문장부호가 있으면 단순히 존댓말 보정만 붙입니다.
            t = t[:-1].rstrip() + "입니다."
        else:
            t += "입니다."
    return t
def ai_panel(title: str, user_prompt: str, context_obj: Dict[str, Any], key_prefix: str):
    st.markdown(f"### <span class=\"emoji\">🤖</span> {title}", unsafe_allow_html=True)
    ai_guardrail_banner()
    if not ai_available():
        return
    if not rate_limit_ok():
        st.warning("AI 호출 한도(세션)가 초과되었습니다. (새로고침하면 초기화)")
        return

    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        btn_summary = st.button("AI 요약", key=f"{key_prefix}_sum")
    with colB:
        btn_plan = st.button("2026 방향성", key=f"{key_prefix}_plan")
    with colC:
        with st.expander("AI에 전달되는 요약 데이터(검증용)", expanded=False):
            st.json(context_obj)

    if not (btn_summary or btn_plan):
        st.caption("버튼을 눌렀을 때만 호출합니다. (기본 OFF / 비용 보호)")
        return

    bump_calls()

    if btn_summary:
        user = user_prompt + f"\n\n출력: '2022→2025 변천사: ...' 형식의 한 줄(1문장)만 작성하세요. 다른 섹션(예: 2026 제안/방향성) 내용, 메타 설명, 주의문, 가설 문구를 덧붙이지 마세요. 총 {ai_char_limit}자 이내."
        kicker = "AI 요약"
    else:
        user = user_prompt + f"\n\n출력: '2026 제안: ...' 형식의 한 줄(1문장)만 작성하세요. '2022→2025 변천사:' 문구를 출력하거나, AI 요약 버튼 관련 문장을 덧붙이지 마세요. (2022→2025 흐름은 문장 안에서 근거로 반영) 총 {ai_char_limit}자 이내."
        kicker = "2026 방향성"

    with st.spinner("AI 생성 중..."):
        try:
            key = get_openai_api_key()
            text, _meta = call_openai_responses(
                api_key=key,
                model=ai_model,
                system=AI_SYSTEM,
                user=user,
                max_output_tokens=1800,
                temperature=ai_temp,
                timeout=90,
                reasoning_effort="low",
                retries=2,
            )
            raw_text = (text or "").strip()
            text = pick_one_liner(raw_text, mode="summary" if btn_summary else "plan")
            text = enforce_char_limit(text, ai_char_limit)

            st.markdown(
                f"""
                <div class="card">
                  <div class="kicker">{html.escape(kicker)}</div>
                  <div style="font-size:18px; line-height:1.75; white-space:pre-wrap;">{html.escape(text)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(str(e))

# =========================================================
# Load data once
# =========================================================
try:
    df_bubble_all = load_bubble(BUBBLE_CSV)
    df_elements_all = load_elements(ELEMENTS_CSV)
    df_games_all = load_games(GAMES_CSV)

    df_fulltext = load_fulltext(FULLTEXT_CSV)
    df_off_games = load_official_games(OFFICIAL_GAMES_CSV)
    df_off_elements = load_official_elements(OFFICIAL_ELEMENTS_CSV)
except Exception as e:
    st.error(f"CSV 로드 실패: {e}")
    st.stop()


# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(["🫧 행사 이미지 키워드", "🧩 운영요소", "🎮 전시 게임", "💬 AI 챗봇"])


with tab1:
    st.markdown('<div class="section-title">🫧 연도별 “행사 이미지” 키워드</div>', unsafe_allow_html=True)

    years = sorted(df_bubble_all["year"].unique().tolist())
    years = [y for y in [2022, 2023, 2024, 2025] if y in years] or years
    years = years[:4]

    cols = st.columns(4)
    for col, y in zip(cols, years):
        with col:
            st.markdown(f"#### {y}")
            yd = df_bubble_all[df_bubble_all["year"] == int(y)].copy()
            yd = yd.sort_values(["MentionedPosts", "tag"], ascending=[False, True]).head(bubble_top_n).copy()
            if yd.empty:
                st.caption("데이터 없음")
                continue

            fig, _ = bubble_plotly(yd, bubble_show_count, height=320)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    ctx = build_context_pack(
        df_b=df_bubble_all,
        df_e=df_elements_all,
        df_g=df_games_all,
        df_off_games=df_off_games,
        df_off_elements=df_off_elements,
        bubble_n=bubble_top_n,
        elem_n=min(6, elem_top_n),
        game_n=min(8, game_top_n),
        elem_type_filter=elem_type,
        elem_q_filter=elem_q,
    )
    prompt = "다음은 네이버 블로그 후기 텍스트마이닝 기반 연도별 데이터 요약이다.\n" \
             f"- 섹션: 이미지 키워드(태그)\n- 데이터(JSON):\n{json.dumps({'years': ctx['years'], 'tags': ctx['tags'], 'official_meta': ctx.get('official_meta', {})}, ensure_ascii=False)}"
    ai_panel("이미지 키워드 인사이트", prompt, {"years": ctx["years"], "tags": ctx["tags"]}, "ai_tags")


with tab2:
    st.markdown('<div class="section-title">🧩 연도별 운영요소 키워드 & 언급 맥락</div>', unsafe_allow_html=True)

    base = df_elements_all.copy()
    if elem_type != "전체":
        base = base[base["type"] == elem_type]
    if elem_q:
        base = base[base["keyword"].str.contains(elem_q, case=False, na=False)]

    years = sorted(df_elements_all["year"].unique().tolist())
    years = [y for y in [2022, 2023, 2024, 2025] if y in years] or years
    year_tabs = st.tabs([f"{y}년" for y in years])

    for tab, y in zip(year_tabs, years):
        with tab:
            v = base[base["year"] == int(y)].copy()
            if v.empty:
                st.info("이 연도는 데이터가 없습니다.")
                continue

            v = v.sort_values("MentionedPosts", ascending=False).head(elem_top_n).reset_index(drop=True)
            v["label"] = v.apply(lambda r: f"{em(r['type'])} {r['keyword']}", axis=1)

            chart_df = v.sort_values("MentionedPosts", ascending=True)
            fig = px.bar(chart_df, x="MentionedPosts", y="label", orientation="h")
            fig.update_traces(
                marker_color="#FF6D00",
                hovertemplate="<b>%{y}</b><br>언급 글 수: %{x}<extra></extra>",
            )
            fig.update_layout(
                height=max(540, 46 * len(chart_df) + 160),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="",
                yaxis_title="",
                yaxis=dict(tickfont=dict(size=21)),
                xaxis=dict(tickfont=dict(size=13)),
                font=dict(family="Pretendard, sans-serif", size=18),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            for _, r in v.iterrows():
                header = f"{em(r['type'])} {r['keyword']}  ·  🧾 {int(r['MentionedPosts'])}"
                with st.expander(header, expanded=False):
                    examples = parse_examples(r.get("examples_json", "[]"))
                    if not examples:
                        st.info("예시 문장이 없습니다.")
                        continue

                    for j, ex in enumerate(examples, 1):
                        ctx_html = highlight(ex["context"])
                        title = ex["title"] if ex["title"] else "원문"
                        link = ex["link"]

                        if link:
                            shown = title if len(title) <= 90 else title[:90] + "…"
                            link_html = f"🔗 <a href='{link}' target='_blank' rel='noreferrer'>{shown}</a>"
                        else:
                            link_html = "🔗 (링크 없음)"

                        st.markdown(
                            f"""
                            <div class="card">
                              <div class="kicker">{em(r['type'])} 예시 {j}</div>
                              <div style="font-size:18px; line-height:1.75;">{ctx_html}</div>
                              <div style="margin-top:8px; font-size:13px; color: rgba(23,23,23,0.75);">{link_html}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    ctx = build_context_pack(
        df_b=df_bubble_all,
        df_e=df_elements_all,
        df_g=df_games_all,
        df_off_games=df_off_games,
        df_off_elements=df_off_elements,
        bubble_n=min(6, bubble_top_n),
        elem_n=elem_top_n,
        game_n=min(8, game_top_n),
        elem_type_filter=elem_type,
        elem_q_filter=elem_q,
    )
    prompt = "다음은 네이버 블로그 후기 텍스트마이닝 기반 연도별 데이터 요약이다.\n" \
             f"- 섹션: 운영요소(키워드+일부 예시 문장)\n- 필터: type={elem_type}, query='{elem_q or ''}'\n- 데이터(JSON):\n{json.dumps({'years': ctx['years'], 'elements': ctx['elements'], 'official_meta': ctx.get('official_meta', {})}, ensure_ascii=False)}"
    ai_panel("운영요소 인사이트", prompt, {"years": ctx["years"], "elements": ctx["elements"]}, "ai_elements")


with tab3:
    st.markdown('<div class="section-title">🎮 연도별 전시 게임 언급량 TOP</div>', unsafe_allow_html=True)

    df_g = (
        df_games_all.sort_values(["year", "MentionedPosts", "game"], ascending=[True, False, True])
            .groupby("year", as_index=False)
            .head(game_top_n)
    )

    RANK_COLORS = [
        "#B00020", "#C21807", "#D32F2F", "#E53935", "#F4511E",
        "#FB5E00", "#FF6D00", "#FF7A1A", "#FF8F00", "#FFA000",
    ]

    def year_ranked(df_all: pd.DataFrame, year: int) -> pd.DataFrame:
        d = df_all[df_all["year"] == year].copy()
        if d.empty:
            return d
        d = d.sort_values(["MentionedPosts", "game"], ascending=[False, True]).reset_index(drop=True)
        d["rank"] = range(1, len(d) + 1)
        d["color"] = d["rank"].apply(lambda r: RANK_COLORS[r-1] if 1 <= r <= len(RANK_COLORS) else RANK_COLORS[-1])
        return d

    cols = st.columns(4)
    for i, year in enumerate([2022, 2023, 2024, 2025]):
        with cols[i]:
            st.markdown(f"### {year}")
            d = year_ranked(df_g, year)
            if d.empty:
                st.caption("데이터 없음")
                continue

            fig = go.Figure(go.Treemap(
                labels=d["game"].astype(str).tolist(),
                parents=[""] * len(d),
                values=d["MentionedPosts"].astype(int).tolist(),
                marker=dict(colors=d["color"].tolist(), line=dict(width=2, color="rgba(255,255,255,0.35)")),
                texttemplate="<b>%{label}</b><br>%{value}",
                textfont=dict(size=38, color="white"),
                hovertemplate="<b>%{label}</b><br>언급량=%{value}<extra></extra>",
                branchvalues="total",
            ))
            fig.update_layout(
                height=560,
                margin=dict(l=4, r=4, t=6, b=4),
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    ctx = build_context_pack(
        df_b=df_bubble_all,
        df_e=df_elements_all,
        df_g=df_games_all,
        df_off_games=df_off_games,
        df_off_elements=df_off_elements,
        bubble_n=min(6, bubble_top_n),
        elem_n=min(6, elem_top_n),
        game_n=game_top_n,
        elem_type_filter=elem_type,
        elem_q_filter=elem_q,
    )
    prompt = "다음은 네이버 블로그 후기 텍스트마이닝 기반 연도별 데이터 요약이다.\n" \
             f"- 섹션: 전시 게임 Top\n- 데이터(JSON):\n{json.dumps({'years': ctx['years'], 'games': ctx['games'], 'official_meta': ctx.get('official_meta', {})}, ensure_ascii=False)}"
    ai_panel("전시 게임 인사이트", prompt, {"years": ctx["years"], "games": ctx["games"]}, "ai_games")


with tab4:
    st.markdown('<div class="section-title">💬 AI 챗봇 — “Ask Beaver Rocks”</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-note">질문에 대해, 현재 대시보드 데이터(태그/운영요소/게임 Top + 운영요소 예시 일부)를 근거로 답합니다.</div>', unsafe_allow_html=True)

    ai_guardrail_banner()
    if not ai_available():
        st.stop()

    if not rate_limit_ok(max_calls=60):
        st.warning("AI 호출 한도(세션)가 초과되었습니다. (새로고침하면 초기화)")
        st.stop()

    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant", "content": "안녕하세요! 2022~2025 후기 기반으로 운영/기획 질문을 도와드릴게요. 예: “2026 운영 우선순위 뭐부터?”"}]

    c1, c2 = st.columns([2, 1])
    with c1:
        scope = st.selectbox("근거 범위", ["전체", "이미지 키워드", "운영요소", "전시 게임"])
    with c2:
        year_scope = st.selectbox("연도", ["전체", "2022", "2023", "2024", "2025"])

    ctx = build_context_pack(
        df_b=df_bubble_all,
        df_e=df_elements_all,
        df_g=df_games_all,
        df_off_games=df_off_games,
        df_off_elements=df_off_elements,
        bubble_n=bubble_top_n,
        elem_n=min(elem_top_n, 10),
        game_n=game_top_n,
        elem_type_filter=elem_type,
        elem_q_filter=elem_q,
    )

    def ctx_text_for(scope_: str, year_: str) -> str:
        years = [str(y) for y in ctx["years"]]
        if year_ != "전체":
            years = [year_]
        parts = []
        # 공식 리스트(게임/운영요소) 요약
        omap = ctx.get("official_meta", {}) or {}
        if omap:
            parts.append("### 공식 목록(참고)")
            for y in years:
                yy = str(y)
                om = omap.get(yy, {}) if isinstance(omap, dict) else {}
                if not isinstance(om, dict):
                    continue
                g = om.get("official_games_cnt")
                e = om.get("official_elements_cnt")
                t = om.get("official_elements_by_type") or {}
                line = f"- {yy}: 공식게임 {g}개, 공식운영요소 {e}개"
                if isinstance(t, dict) and t:
                    top4 = list(t.items())[:4]
                    line += " (" + ", ".join([f"{k}:{v}" for k, v in top4]) + ")"
                parts.append(line)
        if scope_ in ("전체", "이미지 키워드"):
            parts.append("### 이미지 키워드(태그) Top")
            for y in years:
                items = ctx["tags"].get(y, [])
                if items:
                    parts.append(f"- {y}: " + ", ".join([f"{it['tag']}({it['posts']})" for it in items]))
        if scope_ in ("전체", "운영요소"):
            parts.append("### 운영요소 Top + 예시 일부")
            for y in years:
                items = ctx["elements"].get(y, [])
                if items:
                    parts.append(f"- {y}:")
                    for it in items[:6]:
                        ex = it.get("examples") or []
                        ex_txt = (" / ".join(ex[:1])[:160]) if ex else ""
                        parts.append(f"  - {it['type']} | {it['keyword']} ({it['posts']})" + (f" | 예: {ex_txt}" if ex_txt else ""))
        if scope_ in ("전체", "전시 게임"):
            parts.append("### 전시 게임 Top")
            for y in years:
                items = ctx["games"].get(y, [])
                if items:
                    parts.append(f"- {y}: " + ", ".join([f"{it['game']}({it['posts']})" for it in items]))
        return "\n".join(parts)

    retrieval = ctx_text_for(scope, year_scope)

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("질문을 입력하세요 (예: 2026 운영 우선순위 뭐부터?)")
    if user_msg:
        st.session_state.chat.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        bump_calls()

        y_int = None if year_scope == "전체" else int(year_scope)
        snippets = retrieve_fulltext_snippets(df_fulltext, user_msg, y_int, max_posts=2)
        matched_games = detect_official_game_mentions(df_off_games, user_msg, y_int)
        matched_elems = detect_official_element_mentions(df_off_elements, user_msg, y_int)

        extra_ctx_lines = []
        if matched_games:
            extra_ctx_lines.append("공식 전시 게임(질문 매칭): " + ", ".join(matched_games))
        if matched_elems:
            extra_ctx_lines.append("공식 운영요소(질문 매칭): " + ", ".join(matched_elems))
        if snippets:
            extra_ctx_lines.append("원문 스니펫(근거 일부):")
            extra_ctx_lines += [f"- {sn}" for sn in snippets]
        extra_ctx = "\n".join(extra_ctx_lines)

        system = AI_SYSTEM + "\n추가 규칙: 반드시 한국어로 답하고, 모든 문장은 존댓말로 끝내며, 근거를 포함하고, 글자 수 제한을 지켜라."
        user = (
            f"컨텍스트(요약):\n{retrieval}\n"
            + (f"\n{extra_ctx}\n" if extra_ctx else "\n")
            + f"질문: {user_msg}\n\n"
            + f"출력(총 {ai_char_limit}자 이내):\n"
            + "- 답변\n- 근거(불릿 2~4개)\n- 다음 액션(불릿 2~3개)"
        )

        with st.chat_message("assistant"):
            with st.spinner("AI 답변 생성 중..."):
                try:
                    key = get_openai_api_key()
                    text, meta = call_openai_responses(
                        api_key=key,
                        model=ai_model,
                        system=system,
                        user=user,
                        max_output_tokens=1800,
                        temperature=ai_temp,
                        timeout=90,
                    )
                    text = enforce_char_limit(text, ai_char_limit)
                    st.markdown(text)
                    st.session_state.chat.append({"role": "assistant", "content": text})
                except Exception as e:
                    err = f"에러: {e}"
                    st.error(err)
                    st.session_state.chat.append({"role": "assistant", "content": err})

    with st.expander("챗봇 근거 데이터(검증용)", expanded=False):
        st.text(retrieval)

    if st.button("대화 초기화"):
        st.session_state.chat = [{"role": "assistant", "content": "대화를 초기화했어요. 무엇을 도와드릴까요?"}]
