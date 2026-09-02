"""종편 4사 메인뉴스 시청률 예측 대시보드 (Streamlit).

실행:  .venv/bin/streamlit run app.py
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore")

from ratings import (backtest, data as rdata, forecast as rfc, kbo as rkbo,
                     model as rmodel)  # noqa: E402
from ratings.config import (  # noqa: E402
    CHANNELS, COLS_2049, DATA_DIR, HOUSEHOLD_COLS, MODEL_DIR, PALETTE,
    RATINGS_CSV, REPORT_DIR,
)
from ratings.features import EVENT_TYPES, feature_columns  # noqa: E402
from ratings.labels import (  # noqa: E402
    humanize, humanize_group, humanize_target,
)

st.set_page_config(page_title="종편 4사 메인뉴스 시청률 예측",
                   page_icon="📺", layout="wide", initial_sidebar_state="expanded")

HORIZON = 7


def day_label(d, today) -> str:
    """오늘 기준 상대 표기. 데이터가 전날까지 들어오므로 첫 예측일은 보통 '오늘'이다."""
    n = (pd.Timestamp(d).normalize() - pd.Timestamp(today).normalize()).days
    if n < 0:
        return f"{-n}일 전"
    return {0: "오늘", 1: "내일", 2: "모레"}.get(n, f"{n}일 후")


# --------------------------------------------------------------------------
# 데이터 · 모델 로딩 (캐시)
# --------------------------------------------------------------------------
def _candidate_files() -> list[Path]:
    files = [RATINGS_CSV] if RATINGS_CSV.exists() else []
    files += sorted(p for p in DATA_DIR.glob("*.csv")
                    if p != RATINGS_CSV and p.name != "kbo_schedule.csv")
    files += sorted(DATA_DIR.glob("*.xlsx"))
    return files


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return rdata.load_ratings(path)


@st.cache_resource(show_spinner=False)
def load_models(path: str, mtime: float, horizon: int, bundle_mtime: float = 0.0):
    """저장된 번들이 현재 데이터와 같은 시점이면 재사용, 아니면 즉시 학습.

    `bundle_mtime` 은 캐시 키 전용이다 — CLI(`train.py`)로 다시 학습하면 값이 바뀌어
    실행 중인 앱도 새 모델을 집어 든다.
    """
    df = rdata.load_ratings(path)
    data_end = rdata.coverage(df)["end"]

    bundle_path = MODEL_DIR / "models.pkl"
    if bundle_path.exists():
        try:
            with bundle_path.open("rb") as fh:
                bundle = pickle.load(fh)
            if bundle.get("data_end") == data_end and bundle.get("horizon", 0) >= horizon:
                return bundle["models"]
        except (pickle.UnpicklingError, EOFError, KeyError, AttributeError):
            pass

    drop = rmodel.DEFAULT_DROP_GROUPS
    bar = st.progress(0.0, text="모델 학습 중… (최초 1회, 약 2분)")
    models = rmodel.fit_all(
        df, horizons=range(1, horizon + 1), events=load_events(), drop_groups=drop,
        progress=lambda p, msg: bar.progress(min(p, 1.0), text=f"모델 학습 중… {msg}"),
    )
    bar.empty()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with bundle_path.open("wb") as fh:
        pickle.dump({"models": models, "horizon": horizon, "data_end": data_end,
                     "objective": "l1", "drop_groups": drop}, fh)
    return models


def refresh_data() -> str:
    """구글 시트에서 최신 시청률을 받아오고 캐시를 비운다.

    새 데이터가 들어오면 `load_models` 가 data_end 불일치를 감지해 자동으로
    재학습한다(진행 막대가 뜬다). 여기서는 내려받기까지만 한다.
    """
    import subprocess
    r = subprocess.run([sys.executable, "scripts/sync_data.py"],
                       cwd=str(Path(__file__).resolve().parent),
                       capture_output=True, text=True, timeout=180)
    load_data.clear()
    return (r.stdout or r.stderr or "").strip()


@st.cache_data(show_spinner=False, ttl=300)
def kbo_live(date_str: str):
    """당일 KBO 편성/취소 현황 (네이버 실시간). 5분 캐시."""
    try:
        return rkbo.live_status(date_str)
    except Exception as exc:                       # noqa: BLE001
        return {"error": str(exc)}


@st.cache_data(show_spinner=False)
def load_events():
    """data/events.csv (start,end,label,weight) 가 있으면 이벤트 변수로 학습에 반영."""
    ev_path = DATA_DIR / "events.csv"
    if not ev_path.exists():
        return None
    ev = pd.read_csv(ev_path)
    return ev if {"start", "end"}.issubset(ev.columns) else None


@st.cache_data(show_spinner=False)
def run_forecast(path: str, mtime: float, horizon: int,
                 kbo: str, temp: float, event: str,
                 bundle_mtime: float = 0.0, same_day: bool = False,
                 cancelled: int | None = None,
                 start_time: float | None = None) -> pd.DataFrame:
    """시나리오 조합별 예측 캐시 — 슬라이더를 움직여도 같은 조합은 다시 계산하지 않는다.

    same_day 면 D+1 대상일의 외생 조건(우천취소)까지 알고 있다고 보고 예측을 고친다.
    """
    df = load_data(path)
    models = load_models(path, mtime, horizon, bundle_mtime)
    sc = rfc.Scenario(kbo=kbo, temp_offset=temp, news_event=event,
                      cancelled_games=cancelled, start_time=start_time)
    known = df["Date"].max() + pd.Timedelta(days=1) if same_day else None
    return rfc.forecast(df, models, horizon, events=load_events(), scenario=sc,
                        known_through=known)


@st.cache_data(show_spinner=False)
def load_importance(path: str, mtime: float, target: str, horizon: int) -> pd.DataFrame:
    df = rdata.load_ratings(path)
    frame = rmodel.training_frame(df, horizon, events=load_events())
    holdout = frame.iloc[-120:]
    m = rmodel.fit_target(frame.iloc[:-120], target, horizon,
                          feature_columns(frame), with_quantiles=False)
    return rmodel.permutation_importance(m, holdout, n_repeats=3, top=12)


@st.cache_data(show_spinner=False)
def load_backtest(path: str, mtime: float, n_folds: int, horizon: int) -> pd.DataFrame:
    df = rdata.load_ratings(path)
    bar = st.progress(0.0, text="백테스트 실행 중…")
    res = backtest.run(df, horizon=horizon, n_folds=n_folds,
                       progress=lambda p, msg: bar.progress(min(p, 1.0),
                                                            text=f"백테스트… {msg}"))
    bar.empty()
    return res


# --------------------------------------------------------------------------
# 사이드바
# --------------------------------------------------------------------------
files = _candidate_files()
if not files:
    # Streamlit Cloud 등 원본이 없는 환경: 시트에서 바로 받아온다.
    # 저장소에 시청률 CSV 를 올리지 않으므로 데이터가 git 히스토리에 남지 않는다.
    with st.spinner("구글 시트에서 시청률 데이터를 받아오는 중…"):
        try:
            refresh_data()
        except Exception as exc:                       # noqa: BLE001
            st.error(f"시트에서 데이터를 받지 못했습니다.\n\n{exc}")
            st.stop()
    files = _candidate_files()

if not files:
    st.error(
        f"### 데이터 파일이 없습니다\n\n"
        f"`{DATA_DIR}` 에 **종편_4사_메인_시청률.csv** 를 넣어 주세요.\n\n"
        f"먼저 구조만 확인하려면 샘플을 만들 수 있습니다:\n"
        f"```bash\n.venv/bin/python scripts/make_sample_data.py\n```"
    )
    st.stop()

# 시나리오를 URL 쿼리 파라미터에 실어 링크로 공유할 수 있게 한다.
# (?kbo=none&metric=2049 처럼 — "우천취소면 어떻게 되냐"를 링크 하나로 넘길 수 있다)
qp = st.query_params
KBO_OPTIONS = ["auto", "none", "weekday_1830", "weekend_1700"]
KBO_LABELS = {"auto": "실제 일정 그대로",
              "none": "전 경기 우천취소 / 경기 없음",
              "weekday_1830": "평일 18:30 정상 진행",
              "weekend_1700": "주말 17:00 경기"}


def _qp_float(key: str, default: float, lo: float, hi: float) -> float:
    try:
        return min(max(float(qp.get(key, default)), lo), hi)
    except (TypeError, ValueError):
        return default


st.sidebar.title("⚙️ 설정")
choice = st.sidebar.selectbox("데이터 소스", files, format_func=lambda p: p.name)
metric = st.sidebar.radio("시청률 지표", ["가구", "2049"], horizontal=True,
                          index=1 if qp.get("metric") == "2049" else 0)

st.sidebar.divider()
st.sidebar.subheader("🕑 운용 모드")
same_day = st.sidebar.toggle(
    "당일 수정 (우천취소 확정 반영)",
    value=qp.get("sameday") == "1",
    help="KBO 는 보통 경기 1~2시간 전(17시 전후)에 우천취소를 발표하고 메인뉴스는 "
         "18:50~21:00 에 나간다. 그 사이에 확정된 취소를 반영해 D+1 예측을 고친다. "
         "과거 시청률은 그대로 마지막 관측일까지만 쓰므로, 미리 안 것이 아니다.")

# 당일 우천취소는 데이터 로드 후 실제 대상일을 알아야 조회할 수 있으므로
# 사이드바 자리만 잡아 두고 아래에서 채운다.
kbo_slot = st.sidebar.container() if same_day else None
cancelled = None
start_time = None

st.sidebar.divider()
st.sidebar.subheader("🔮 What-If 시나리오")
kbo_mode = st.sidebar.selectbox(
    "⚾ KBO 편성", KBO_OPTIONS, format_func=lambda k: KBO_LABELS[k],
    index=KBO_OPTIONS.index(qp["kbo"]) if qp.get("kbo") in KBO_OPTIONS else 0,
)
temp_offset = st.sidebar.slider(
    "🌡️ 평년 대비 기온 (°C)", -10.0, 10.0, _qp_float("temp", 0.0, -10.0, 10.0), 0.5,
    help="Open-Meteo 실측/예보 기온을 쓴다. 다만 요일·계절을 통제하고 나면 "
         "기온이 뉴스 시청률에 미치는 영향 자체가 작아서, ±8°C 를 줘도 "
         "변화는 0.01%p 수준이다.")
EVENT_OPTIONS = ["", *EVENT_TYPES]
EVENT_LABELS = {
    "": "평시",
    "윤석열_사법": "윤석열 사법 국면 (계엄·탄핵·내란재판)",
    "이재명_사법": "이재명 사법 국면 (체포동의안·영장)",
    "대선_정국": "대선 정국 (후보 단일화 등)",
}
news_event = st.sidebar.selectbox(
    "🚨 속보 국면", EVENT_OPTIONS, format_func=lambda k: EVENT_LABELS.get(k, k),
    index=EVENT_OPTIONS.index(qp["event"]) if qp.get("event") in EVENT_OPTIONS else 0,
    help="과거 같은 성격의 국면이 재현된다고 가정한다. 이 효과만은 모델 예측이 아니라 "
         "**과거 실측 배율**을 곱한 값이다 — 사례가 유형당 5~21일뿐이라 모델이 "
         "학습할 수 없기 때문이다.")

scenario = rfc.Scenario(kbo=kbo_mode, temp_offset=temp_offset, news_event=news_event)

st.sidebar.divider()
if st.sidebar.button("🔄 모델 재학습", width="stretch"):
    load_models.clear()
    load_importance.clear()
    load_backtest.clear()
    st.rerun()

path, mtime = str(choice), choice.stat().st_mtime
bundle_file = MODEL_DIR / "models.pkl"
bundle_mtime = bundle_file.stat().st_mtime if bundle_file.exists() else 0.0

df = load_data(path)
cov = rdata.coverage(df)
models = load_models(path, mtime, HORIZON, bundle_mtime)

cols = HOUSEHOLD_COLS if metric == "가구" else COLS_2049
label_of = {c.household_col if metric == "가구" else c.col_2049: c.label for c in CHANNELS}

# --------------------------------------------------------------------------
# 헤더
# --------------------------------------------------------------------------
st.title("📺 종편 4사 메인뉴스 시청률 예측")
head = st.columns([2, 1, 1, 1])
head[0].caption(f"**데이터** `{choice.name}`  ·  {cov['start']} ~ {cov['end']}")
head[1].metric("관측 일수", f"{cov['days']:,}일")
TODAY = pd.Timestamp.today().normalize()
LAST_OBS = df["Date"].max().normalize()
LAG_DAYS = (TODAY - LAST_OBS).days
head[2].metric("데이터 시차", f"{LAG_DAYS}일",
               delta="정상" if LAG_DAYS <= 1 else f"{LAG_DAYS - 1}일 지연",
               delta_color="off" if LAG_DAYS <= 1 else "inverse")
head[2].caption(f"마지막 관측 {LAST_OBS:%m/%d}")
head[3].metric("예측 모델", f"{len(models)}개")
head[3].caption("지표 8개 × 예측일 7일")

# --- 데이터 갱신: 새 시청률이 들어오면 예측 기준일이 하루 넘어간다 -------------
first_day = LAST_OBS + pd.Timedelta(days=1)
banner = st.container()
with banner:
    c1, c2 = st.columns([4, 1])
    with c1:
        if LAG_DAYS <= 1:
            st.info(
                f"**{LAST_OBS:%m/%d}** 까지 반영됨 → **{day_label(first_day, TODAY)}"
                f"({first_day:%m/%d}) 부터 7일** 예측 중. "
                "시트에 새 날짜가 들어왔다면 오른쪽 버튼으로 받아오세요."
            )
        else:
            st.error(
                f"⚠️ **데이터가 {LAG_DAYS}일 지연됐습니다.** 마지막 관측이 "
                f"{LAST_OBS:%m/%d} 라 예측이 {first_day:%m/%d}(이미 지난 날)부터 "
                "시작합니다. 갱신하면 예측 기준일이 오늘로 옮겨집니다."
            )
    with c2:
        if st.button("⬇️ 최신 데이터 받기", type="primary", width="stretch"):
            with st.spinner("구글 시트에서 내려받는 중…"):
                out = refresh_data()
            if "갱신" in out:
                st.success(out.splitlines()[-1])
                st.info("새 데이터가 들어왔습니다. 모델을 다시 학습합니다 (약 2분)…")
            else:
                st.caption(out.splitlines()[-1] if out else "변경 없음")
            st.rerun()
if choice.name.startswith("SAMPLE"):
    st.warning("현재 **합성 샘플 데이터**로 동작 중입니다. "
               "실제 `종편_4사_메인_시청률.csv` 를 `data/` 에 넣으면 자동으로 선택지에 나타납니다.")

events = load_events()
# --- 당일 수정: 네이버에서 대상일 KBO 현황을 실시간 조회 ----------------------
if same_day and kbo_slot is not None:
    target_day = (LAST_OBS + pd.Timedelta(days=1)).date().isoformat()
    live = kbo_live(target_day)
    with kbo_slot:
        if live.get("error"):
            st.warning(f"KBO 실시간 조회 실패 — 수동 입력하세요\n\n{live['error'][:90]}")
            auto = 0
        else:
            auto = int(live["cancelled"])
            if live["scheduled"] == 0:
                st.info(f"{target_day} 경기 없음 (월요일·비시즌)")
            elif auto == 0:
                st.success(f"편성 {live['scheduled']}경기 · **취소 없음** "
                           f"(조회 {live['fetched_at']})")
            elif auto >= live["scheduled"]:
                st.error(f"**전 경기 취소** ({auto}/{live['scheduled']}) "
                         f"· 조회 {live['fetched_at']}")
            else:
                st.warning(f"**{auto}경기 취소** / 편성 {live['scheduled']} "
                           f"· 조회 {live['fetched_at']}")
            with st.expander("경기별 현황"):
                st.table(pd.DataFrame(live["games"])[["시각", "경기", "구장", "상태"]]
                         .set_index("시각"))
        override = st.number_input(
            "취소 경기 수 (직접 보정)", min_value=-1, max_value=10, value=-1, step=1,
            help="-1 이면 위 실시간 조회값을 그대로 쓴다. 발표 전 시나리오를 "
                 "보려면 직접 숫자를 넣는다.")
        cancelled = auto if override < 0 else int(override)
        if not live.get("error") and live.get("main_start", -1) > 0:
            start_time = float(live["main_start"])
            hh, mm = int(start_time), round(start_time % 1 * 60)
            st.caption(f"적용값: **{cancelled}경기 취소** · 시작 **{hh:02d}:{mm:02d}** "
                       "(실시간 확인)")
        else:
            st.caption(f"적용값: **{cancelled}경기 취소**")

pred = run_forecast(path, mtime, HORIZON, kbo_mode, temp_offset, news_event,
                    bundle_mtime, same_day, cancelled, start_time)

with st.expander("이 대시보드는 어떻게 예측하나 (용어 설명)"):
    st.markdown(
        """
**한 줄 요약** — 각 채널의 과거 시청률 흐름에, 그날의 요일·공휴일·편성·야구·날씨를
더해서 앞으로 7일을 예측한다.

**예측 모델 56개.** 시청률 지표 8개(4사 × 가구/2049) × 예측일 7일 = 56 조합마다
전용 모델을 따로 학습한다. '내일'을 맞히는 모델과 '일주일 뒤'를 맞히는 모델은
쓸 수 있는 정보가 다르기 때문이다 — 일주일 뒤를 예측하는 시점엔 어제 시청률을
아직 모른다. 모델은 LightGBM(의사결정나무를 순차적으로 쌓는 방식)이다.

**모델이 보는 것**

| 종류 | 예 |
|---|---|
| 과거 시청률 | 어제·지난주 같은 요일·최근 7일 평균, 4사 서로의 값 |
| 달력 | 요일, 공휴일, 연휴 길이, 징검다리 평일 |
| 편성 | 각 채널 방송 시각, 동시간대 경쟁 강도·점유율 |
| 야구 | 편성 경기 수, 시작 시각, 내 방송 시간과 겹치는 정도 |
| 날씨 | 기온, 저녁 강수, 우천취소 위험 |

**80% 구간** — 예측값 하나가 아니라 '10번 중 8번은 이 범위 안에 들어온다'는 폭이다.

**사전 예측 / 당일 수정** — 전날 내는 예측은 우천취소를 모른다. 당일 낮에 취소가
확정되면 사이드바에서 모드를 바꿔 예측을 고칠 수 있다.
        """
    )
pred_m = pred[pred["metric"] == metric]

# 탭(st.tabs) 대신 세그먼트 — 숨겨진 탭 안에서는 폭이 0으로 잡혀 표가 눌리고
# plotly 차트가 아예 그려지지 않는다. 활성 섹션만 렌더링하면 그 문제가 사라진다.
VIEWS = ["📈 예측", "🔮 What-If", "🎯 모델 성능", "🗂️ 데이터"]
_v = qp.get("view")
view = st.segmented_control(
    "보기", VIEWS, default=VIEWS[int(_v)] if (_v or "").isdigit() and int(_v) < len(VIEWS)
    else VIEWS[0], label_visibility="collapsed", key="view")
view = view or VIEWS[0]

# 현재 상태를 URL 에 반영 (쿼리 파라미터 갱신은 rerun 을 일으키지 않는다)
_state = {"view": str(VIEWS.index(view)), "metric": metric, "kbo": kbo_mode,
          "temp": f"{temp_offset:g}", "event": news_event,
          "sameday": "1" if same_day else "0"}
if dict(qp) != _state:
    st.query_params.update(_state)

# --------------------------------------------------------------------------
# 예측 탭
# --------------------------------------------------------------------------
if view == VIEWS[0]:
    st.subheader(f"{day_label(first_day, TODAY)}({first_day:%m/%d %a}) 예상 {metric} 시청률")
    d1 = pred_m[pred_m["horizon"] == 1].set_index("label")
    last_obs = df.iloc[-1]
    kpi = st.columns(4)
    for i, ch in enumerate(CHANNELS):
        col = ch.household_col if metric == "가구" else ch.col_2049
        if ch.label not in d1.index:
            continue
        row = d1.loc[ch.label]
        prev = float(last_obs[col])
        kpi[i].metric(ch.label, f"{row['pred']:.3f}%",
                      delta=f"{row['pred'] - prev:+.3f}%p")
        kpi[i].caption(f"80% 구간 {row['lo']:.3f} ~ {row['hi']:.3f}%")

    if same_day:
        advance = run_forecast(path, mtime, HORIZON, kbo_mode, temp_offset,
                               news_event, bundle_mtime, False, None)
        adv1 = advance[(advance["metric"] == metric) & (advance["horizon"] == 1)]
        cmp_df = (adv1[["label", "pred"]].rename(columns={"pred": "사전 예측"})
                  .merge(d1.reset_index()[["label", "pred"]]
                         .rename(columns={"pred": "당일 수정"}), on="label"))
        cmp_df["변화"] = (cmp_df["당일 수정"] - cmp_df["사전 예측"]).round(3)
        if cmp_df["변화"].abs().max() > 0.0005:
            st.info(
                "**당일 수정 적용됨** — 대상일의 우천취소가 확정된 뒤 낸 예측이다. "
                "과거 시청률은 그대로 마지막 관측일까지만 쓴다."
            )
            st.table(cmp_df.round(3).set_index("label"))
        else:
            st.caption("대상일에 확정된 우천취소가 없어 사전 예측과 동일하다.")

    risk_rows = rfc.future_frames(df, HORIZON, events=events, scenario=scenario)
    risky = [(pd.Timestamp(r["Date"]), float(r.get("wx_precip_evening", 0) or 0),
              float(r.get("kbo_games", 0) or 0))
             for r in risk_rows.values()
             if float(r.get("kbo_games", 0) or 0) > 0
             and float(r.get("wx_precip_evening", 0) or 0) >= 0.5]
    if risky:
        lines = " · ".join(f"{d:%m/%d}(저녁 {mm:.1f}mm, {int(g)}경기)" for d, mm, g in risky)
        st.warning(
            f"⚾🌧️ **우천취소 위험** — {lines}\n\n"
            "저녁 강수 예보가 있고 야구가 편성된 날이다. 실측 기준 저녁 강수 10mm 이상이면 "
            "전 경기 취소 확률이 15%(무강수 1%)까지 오른다. 취소가 확정되면 "
            "사이드바 **당일 수정**을 켜서 예측을 고칠 수 있다."
        )

    st.divider()
    st.subheader("최근 실측 + 향후 7일 예측")

    recent = df.tail(35)
    fig = go.Figure()
    for ch in CHANNELS:
        col = ch.household_col if metric == "가구" else ch.col_2049
        color = PALETTE[ch.label]
        sub = pred_m[pred_m["label"] == ch.label].sort_values("Date")

        fig.add_trace(go.Scatter(
            x=recent["Date"], y=recent[col], name=ch.label, mode="lines",
            line=dict(color=color, width=2), legendgroup=ch.label))
        # 실측 마지막 점과 예측 첫 점을 잇는다
        bridge = pd.concat([recent.tail(1)[["Date", col]].rename(columns={col: "pred"}),
                            sub[["Date", "pred"]]])
        fig.add_trace(go.Scatter(
            x=bridge["Date"], y=bridge["pred"], name=f"{ch.label} 예측", mode="lines+markers",
            line=dict(color=color, width=2, dash="dot"), marker=dict(size=6),
            legendgroup=ch.label, showlegend=False))
        fig.add_trace(go.Scatter(
            x=list(sub["Date"]) + list(sub["Date"][::-1]),
            y=list(sub["hi"]) + list(sub["lo"][::-1]),
            fill="toself", fillcolor=color, opacity=0.13,
            line=dict(width=0), hoverinfo="skip", showlegend=False,
            legendgroup=ch.label))

    fig.add_vline(x=df["Date"].max(), line_dash="dash", line_color="gray",
                  annotation_text="예측 시작", annotation_position="top")
    fig.update_layout(height=460, hovermode="x unified",
                      yaxis_title=f"{metric} 시청률 (%)", xaxis_title=None,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        target_day = pd.to_datetime(pred_m["Date"]).min()
        comp = rfc.competition_frame(pred_m, metric, 1, when=target_day)
        st.subheader(f"동시간대 경쟁 구조 ({target_day:%m/%d %a})")

        # 방송 구간을 타임라인으로 — 겹치는 만큼이 곧 경쟁이다
        rows = []
        for _, r in comp.iterrows():
            ch = next(c for c in CHANNELS if c.label == r["방송사"])
            a0, a1 = ch.window(target_day)
            rows.append({"방송사": r["방송사"], "시작": a0, "종료": a1,
                         "예측": r["예측"]})
        tl = pd.DataFrame(rows)
        fig_tl = go.Figure()
        for _, r in tl.iterrows():
            fig_tl.add_trace(go.Bar(
                y=[r["방송사"]], x=[r["종료"] - r["시작"]], base=[r["시작"]],
                orientation="h", name=r["방송사"],
                marker_color=PALETTE.get(r["방송사"], "#888"),
                text=f'{r["예측"]:.2f}%', textposition="inside",
                hovertemplate=f'{r["방송사"]}<br>%{{base:.2f}}시 시작<br>'
                              f'예측 {r["예측"]:.3f}%<extra></extra>'))
        fig_tl.update_layout(
            height=300, showlegend=False, barmode="overlay",
            xaxis_title="방송 시각",
            xaxis=dict(range=[18.0, 22.2], dtick=0.5,
                       tickvals=[18, 18.5, 19, 19.5, 20, 20.5, 21, 21.5, 22],
                       ticktext=["18:00", "18:30", "19:00", "19:30", "20:00",
                                 "20:30", "21:00", "21:30", "22:00"]),
            margin=dict(l=10, r=10, t=10, b=40))
        st.plotly_chart(fig_tl, width="stretch")
        st.table(comp.set_index("방송사"))
        st.caption(
            "겹치는 시간이 곧 경쟁이다. 방송 길이가 달라 겹침은 **비대칭**이다 — "
            "짧은 프로그램이 긴 프로그램에 먹히는 쪽이 더 큰 값을 갖는다.\n\n"
            "**평일과 주말이 다른 경기다.** 평일엔 TV조선(21:00)이 완전 단독이고 "
            "채널A·MBN·JTBC 가 19시 전후에서 붙는다. 주말엔 TV조선이 18:57 뉴스7 로 "
            "내려오고 MBN 은 19:30 뉴스센터로 빠져, **MBN 의 실제 경쟁자가 "
            "채널A(0%)가 아니라 TV조선(77%)이 된다.**"
        )
    with right:
        st.subheader("향후 7일 예측표")
        short = {"채널A 뉴스A": "채널A", "JTBC 뉴스룸": "JTBC",
                 "MBN 뉴스7": "MBN", "TV조선 뉴스9": "TV조선"}
        table = (pred_m.pivot(index="Date", columns="label", values="pred")
                 .round(3).reset_index())
        table["요일"] = pd.to_datetime(table["Date"]).dt.strftime("%a")
        table["날짜"] = [f"{d:%m/%d}" for d in pd.to_datetime(table["Date"])]
        table["시점"] = [day_label(d, TODAY) for d in pd.to_datetime(table["Date"])]
        order = [c.label for c in CHANNELS if c.label in table.columns]
        table = table[["날짜", "요일", "시점", *order]].rename(columns=short)
        st.table(table.set_index("날짜"))

    with st.expander("예측 구간에 반영된 외생 조건 보기"):
        rows = rfc.future_frames(df, HORIZON, events=events, scenario=scenario)
        DOW = ["월", "화", "수", "목", "금", "토", "일"]

        def _hm(v):
            """18.5 -> '18:30', 음수/0 -> '-'"""
            try:
                v = float(v)
            except (TypeError, ValueError):
                return "-"
            return "-" if v <= 0 else f"{int(v):02d}:{round(v % 1 * 60):02d}"

        disp = []
        for r in rows.values():
            games = float(r.get("kbo_games", 0) or 0)
            disp.append({
                "날짜": pd.Timestamp(r["Date"]).strftime("%m/%d"),
                "요일": DOW[int(r["dayofweek"])],
                "공휴일": "○" if r["is_holiday"] else "",
                "연휴 길이": f"{int(r['consecutive_off_days'])}일"
                          if r["consecutive_off_days"] else "",
                "일몰": _hm(r["sunset_hour"]),
                "기온": f"{float(r['temp_effective']):.1f}°C",
                "저녁 강수 예보": f"{float(r.get('wx_precip_evening', 0) or 0):.1f}mm",
                "야구": f"{int(games)}경기" if games else "없음",
                "야구 시작": _hm(r.get("kbo_main_start", -1)),
                "우천취소 위험": ("높음" if float(r.get("kbo_rain_risk", 0) or 0) >= 1.0
                            else "보통" if float(r.get("kbo_rain_risk", 0) or 0) > 0
                            else ""),
                "속보 국면": "○" if float(r.get("is_news_event", 0) or 0) else "",
            })
        st.table(pd.DataFrame(disp).set_index("날짜"))
        st.caption(
            "모델이 미래 각 날짜에 대해 **확정적으로 알고 있는** 조건들이다. "
            "우천취소 실적은 사전 예측 시점에 알 수 없어 비워 두고, 대신 "
            "'야구 편성 × 저녁 비 예보'로 위험도만 표시한다. "
            "취소가 확정되면 사이드바에서 **당일 수정**을 켜면 반영된다."
        )

# --------------------------------------------------------------------------
# What-If 탭
# --------------------------------------------------------------------------
if view == VIEWS[1]:
    st.subheader("시나리오 효과")
    if scenario.is_default:
        st.info("사이드바에서 KBO 편성·기온·속보 강도를 바꾸면 "
                "기본 예측 대비 변화량이 여기에 나타납니다.")
    else:
        base_pred = run_forecast(path, mtime, HORIZON, "auto", 0.0, "", bundle_mtime,
                                 same_day, cancelled)
        delta = base_pred.merge(
            pred, on=["Date", "horizon", "target", "channel", "label", "metric"],
            suffixes=("_base", "_alt"))
        delta["delta"] = delta["pred_alt"] - delta["pred_base"]
        dm = delta[delta["metric"] == metric]
        summary = (dm.groupby("label")
                   .agg(기본예측=("pred_base", "mean"), 시나리오=("pred_alt", "mean"),
                        변화=("delta", "mean")).round(3).reset_index())
        summary["변화율%"] = (100 * summary["변화"] / summary["기본예측"]).round(1)

        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.table(summary.set_index("label"))
        with c2:
            fig_d = px.bar(summary, x="변화", y="label", orientation="h",
                           color="label", color_discrete_map=PALETTE,
                           labels={"변화": "7일 평균 변화 (%p)", "label": ""})
            fig_d.update_layout(height=320, showlegend=False)
            fig_d.add_vline(x=0, line_color="gray")
            st.plotly_chart(fig_d, width="stretch")

        st.subheader("일자별 변화")
        daily = dm.pivot(index="Date", columns="label", values="delta").round(4)
        daily.index = pd.to_datetime(daily.index).strftime("%m/%d (%a)")
        st.dataframe(daily, width="stretch")

    if news_event:
        mult, per_win = rfc.event_multipliers(df, events, detail=True)
        mult = mult.get(news_event, {})
        if mult:
            def _rng(col):
                v = [x for _, x in per_win.get((news_event, col), [])]
                return f"{min(v):.2f}~{max(v):.2f}" if len(v) > 1 else "-"
            mt = pd.DataFrame({
                "방송사": [c.label for c in CHANNELS],
                "가구 배율": [round(mult.get(c.household_col, 1.0), 2) for c in CHANNELS],
                "구간별 범위": [_rng(c.household_col) for c in CHANNELS],
                "2049 배율": [round(mult.get(c.col_2049, 1.0), 2) for c in CHANNELS],
            })
            st.markdown(f"##### 적용된 실측 배율 — {EVENT_LABELS.get(news_event, news_event)}")
            st.table(mt.set_index("방송사"))
            st.caption(
                "과거 같은 유형 국면에서 **국면 직전 8주 같은 요일** 대비 관측된 평균 배율이다. "
                "모델 예측값에 이 배율을 곱한다. **모델이 학습한 값이 아니다** — "
                "사례가 너무 적어 학습이 불가능하다.\n\n"
                "구간별 범위가 넓다는 건 같은 유형 안에서도 사건 크기가 제각각이라는 뜻이다 "
                "— 평균 하나로 대표하기 어렵다는 신호로 읽어야 한다."
            )

    st.divider()
    st.markdown("##### 각 노브를 어디까지 믿을 수 있나")
    st.markdown(
        "- **⚾ KBO 편성 — 신뢰 가능.** 3년치 실제 일정·시작시각·우천취소로 학습했다. "
        "19시대 3사가 21시대 TV조선보다 크게 반응하는 것도 데이터에서 나온 결과다.\n"
        "- **🌡️ 기온 — 작동하지만 효과가 작다.** Open-Meteo 실측/예보 기온을 쓴다"
        "(과거엔 계절 근사치라 아예 무효였다). 방향은 맞다 — 추우면 오른다. "
        "다만 ±8°C 를 줘도 0.01%p 수준이라, 요일·계절을 통제하면 기온 자체가 "
        "뉴스 시청률을 거의 안 움직인다는 게 이 데이터의 답이다.\n"
        "- **🚨 속보 국면 — 모델이 아니라 과거 실측 배율이다.** 유형당 사례가 5~21일뿐이라 "
        "`min_child_samples=20` 에 걸려 트리가 분기조차 못 한다(split 횟수 0으로 확인). "
        "그래서 모델에 맡기지 않고, 과거 같은 성격 국면에서 **관측된 채널별 배율**을 "
        "예측값에 곱한다. 아래 표가 그 배율이다.\n"
        "- **채널별 우위가 뒤바뀐다.** 윤석열 사법 국면에선 JTBC 만 1.63배로 뛰고 "
        "나머지 3사는 1.0 근처에 머문다. 이재명 사법 국면에선 MBN(1.47)·채널A(1.33)가 "
        "앞서고 JTBC(1.21)가 뒤진다. (한때 '정반대'로 봤는데, 그건 전 기간 평균을 "
        "기준선으로 써서 3년치 추세가 섞인 탓이었다. 직전 8주 대비로는 대체로 다 오른다.)\n"
        "- **강도 조절이 없는 이유** — 배율이 과거 평균 하나뿐이라 눈금을 만들 근거가 없다."
    )

# --------------------------------------------------------------------------
# 모델 성능 탭
# --------------------------------------------------------------------------
if view == VIEWS[2]:
    st.subheader("어떤 변수가 예측을 좌우하나")
    imp_target = st.selectbox("대상 지표", cols, format_func=humanize_target)
    imp = load_importance(path, mtime, imp_target, 1).copy()
    imp["설명"] = imp["feature"].map(humanize)
    fig_imp = px.bar(imp.iloc[::-1], x="importance", y="설명", orientation="h",
                     hover_data={"feature": True, "설명": False},
                     labels={"importance": "이 값을 섞으면 오차가 얼마나 커지나", "설명": ""},
                     color="importance", color_continuous_scale="Blues")
    fig_imp.update_layout(height=430, coloraxis_showscale=False)
    st.plotly_chart(fig_imp, width="stretch")
    st.caption(
        "**읽는 법** — 그 변수 값만 무작위로 뒤섞은 뒤 예측이 얼마나 나빠지는지 잰다. "
        "많이 나빠질수록 모델이 그 변수에 크게 기대고 있다는 뜻이다. "
        "학습에 쓰지 않은 최근 120일에서 측정했다.\n\n"
        "⚠️ 이 측정은 **과거 구간에서 우천취소를 아는 상태**로 계산된다. 따라서 "
        "`kbo_cancelled` 의 높은 순위는 *당일 수정* 모드에 해당하는 이야기다. "
        "사전 예측 시점엔 이 값이 항상 0이라 같은 기여를 하지 못한다.\n\n"
        "그리고 '모델이 얼마나 의존하는가'와 '표본 밖에서 실제로 도움이 되는가'는 "
        "다른 질문이다. 후자는 바로 아래에서 재는데, 결론이 갈릴 수 있다."
    )

    sd_path = REPORT_DIR / "backtest_raw_sameday.csv"
    adv_path = REPORT_DIR / "backtest_raw.csv"
    if sd_path.exists() and adv_path.exists():
        st.divider()
        st.subheader("당일 수정이 얼마나 도움이 되나")
        st.caption("같은 기준일에서 두 방식을 나란히 채점했다. 차이가 곧 "
                   "'우천취소를 미리 아는 것의 값어치'다.")
        a = pd.read_csv(adv_path, parse_dates=["cutoff", "Date"])
        b = pd.read_csv(sd_path, parse_dates=["cutoff", "Date"])
        key = ["cutoff", "Date", "horizon", "target"]
        mm = (a[key + ["actual", "model"]]
              .merge(b[key + ["model"]], on=key, suffixes=("_adv", "_sd")))
        mm["e_adv"] = (mm["model_adv"] - mm["actual"]).abs()
        mm["e_sd"] = (mm["model_sd"] - mm["actual"]).abs()

        sched = rkbo.load_daily(pd.Series(sorted(mm["Date"].unique())),
                                as_of=pd.Timestamp("2100-01-01")).set_index("Date")
        mm["취소"] = mm["Date"].map(sched["kbo_cancelled"]).fillna(0)

        rows = []
        for lo, hi, lab in [(0, 0, "취소 0경기"), (1, 2, "취소 1~2경기"),
                            (3, 99, "취소 3경기 이상")]:
            g = mm[(mm["취소"] >= lo) & (mm["취소"] <= hi)]
            if g.empty:
                continue
            rows.append({"대상일 유형": lab, "일수": g["Date"].nunique(),
                         "사전 예측": round(g["e_adv"].mean(), 4),
                         "당일 수정": round(g["e_sd"].mean(), 4),
                         "개선": f"{(1 - g['e_sd'].mean() / g['e_adv'].mean()) * 100:+.1f}%"})
        rows.append({"대상일 유형": "전체", "일수": mm["Date"].nunique(),
                     "사전 예측": round(mm["e_adv"].mean(), 4),
                     "당일 수정": round(mm["e_sd"].mean(), 4),
                     "개선": f"{(1 - mm['e_sd'].mean() / mm['e_adv'].mean()) * 100:+.1f}%"})
        st.table(pd.DataFrame(rows).set_index("대상일 유형"))
        st.caption("야구 취소가 없는 날엔 결과가 **완전히 동일**하고, 전 구장 취소 날에만 크게 "
                   "좋아진다. 전체 평균만 보면 이 효과가 묻히므로 나눠 봐야 한다.")

    abl_path = REPORT_DIR / "ablation.csv"
    if abl_path.exists():
        st.divider()
        st.subheader("변수 묶음을 빼면 예측이 나빠지나")
        st.caption("위쪽은 **모델이 얼마나 기대는가**, 여기는 **빼면 실제로 나빠지는가**를 잰다. "
                   "둘은 갈릴 수 있다 — 요일·계절과 사실상 같은 정보를 담은 변수는 "
                   "모델이 많이 쓰면서도 새 정보를 주지는 않기 때문이다.")
        abl = pd.read_csv(abl_path)
        abl["악화폭"] = abl["악화폭%"].map("{:+.2f}%".format)
        abl["제외 그룹"] = abl["제외 그룹"].map(humanize_group)
        c_tbl, c_fig = st.columns([1, 1.3])
        with c_tbl:
            st.table(abl[["제외 그룹", "MAE", "악화폭"]].set_index("제외 그룹"))
        with c_fig:
            plot = abl[abl["제외 그룹"] != "(없음 · 전체 피처)"].copy()
            fig_a = px.bar(plot, x="악화폭%", y="제외 그룹", orientation="h",
                           color="악화폭%", color_continuous_scale="RdBu_r",
                           color_continuous_midpoint=0,
                           labels={"악화폭%": "그 묶음을 뺐을 때 오차 변화 (%)", "제외 그룹": ""})
            fig_a.add_vline(x=0, line_color="gray")
            fig_a.update_layout(height=260, coloraxis_showscale=False)
            st.plotly_chart(fig_a, width="stretch")
        st.caption("양수 = 그 묶음이 예측에 도움이 된다 · 음수 = 빼는 편이 낫다")
        st.warning(
            "**이 표로는 아무것도 판단하면 안 된다.** 같은 방법으로 세 번 쟀는데 "
            "세 번 다 다른 답이 나왔다 — 1차(기준일 5개)는 '동시간대 경쟁만 기여', "
            "2차(10개)는 그 반대, 3차(6개)는 '모든 그룹을 빼는 게 낫다' 였다.\n\n"
            "폴드 세트가 바뀌면 절대 수준도 크게 달라진다(같은 모델이 6폴드 0.1270, "
            "26폴드 0.1654). 이 데이터·이 폴드 수로는 변수 묶음의 기여를 판정할 "
            "검정력이 없다는 뜻이다.\n\n"
            "실제로 판정하려면 `run_backtest.py --drop <그룹> --folds 45` 처럼 폴드를 "
            "충분히 늘리고 부트스트랩 신뢰구간까지 봐야 한다."
        )

        per_path = REPORT_DIR / "ablation_by_target.csv"
        if per_path.exists():
            with st.expander("지표별 상세"):
                per = pd.read_csv(per_path).copy()
                per["지표"] = per["지표"].map(humanize_target)
                per.columns = [humanize_group(c.replace(" 제외 악화폭%", "")) + " 제외"
                               if c.endswith("제외 악화폭%") else c for c in per.columns]
                st.table(per.set_index("지표"))

    st.divider()
    st.subheader("과거로 돌아가 실제로 맞혔는지 검증")
    st.caption("과거 어느 날로 돌아가 그 시점까지의 데이터만으로 이후 7일을 예측한 뒤, "
               "실제값과 대조한다. 기준일 이후 데이터는 학습에도 변수 계산에도 쓰지 않는다. "
               "시청률은 요일 효과가 워낙 커서 `지난주 같은 요일 값`만 써도 꽤 맞는다 — "
               "그래서 이 단순한 방법들을 이겨야 모델을 쓸 이유가 생긴다.")

    saved = REPORT_DIR / "backtest_raw.csv"
    res = None
    if "bt" in st.session_state:
        res = st.session_state["bt"]
    elif saved.exists():
        res = pd.read_csv(saved, parse_dates=["cutoff", "Date"])
        st.caption(f"저장된 리포트 사용 — `{saved.name}` "
                   f"({pd.Timestamp(saved.stat().st_mtime, unit='s'):%Y-%m-%d %H:%M})")

    with st.expander("여기서 다시 돌리기 (기준일 하나당 약 40초)"):
        n_folds = st.slider("검증에 쓸 기준일 개수", 3, 20, 6)
        if st.button("검증 실행", type="primary"):
            st.session_state["bt"] = load_backtest(path, mtime, n_folds, HORIZON)
            st.rerun()
        st.code("# 전체 검증은 터미널이 빠릅니다\n"
                ".venv/bin/python run_backtest.py --folds 10 --step 21", language="bash")

    if res is None or res.empty:
        st.info("아직 검증 결과가 없습니다. 위에서 실행하거나 터미널에서 리포트를 만드세요.")
    else:
        by_target = backtest.summarize(res, ("target",))
        by_h = backtest.summarize(res, ("horizon",))

        overall_mae = float((res["model"] - res["actual"]).abs().mean())
        base_mae = float((res["snaive_7"] - res["actual"]).abs().mean())
        m1, m2, m3 = st.columns(3)
        m1.metric("모델 평균 오차", f"{overall_mae:.4f}%p")
        m2.metric("지난주 같은 요일 값", f"{base_mae:.4f}%p")
        m3.metric("개선폭", f"{(1 - overall_mae / base_mae) * 100:+.1f}%")

        st.markdown("**지표별 평균 오차(%p)** — 개선폭이 양수면 그 단순 방법보다 낫다는 뜻")
        show = pd.DataFrame({
            "지표": by_target["target"].map(humanize_target),
            "표본": by_target["n"],
            "모델": by_target["model_MAE"],
            "지난주 동요일": by_target["snaive_7_MAE"],
            "4주 동요일평균": by_target["dow_mean_4_MAE"],
            "vs 지난주": by_target["vs_snaive7_MAE%"].map("{:+.1f}%".format),
            "vs 4주평균": by_target["vs_dow4_MAE%"].map("{:+.1f}%".format),
        })
        st.table(show.set_index("지표"))

        fig_h = go.Figure()
        for c, name in [("model_MAE", "모델"), ("snaive_7_MAE", "지난주 동요일"),
                        ("dow_mean_4_MAE", "최근 4주 동요일 평균"),
                        ("naive_last_MAE", "마지막 관측값")]:
            if c in by_h.columns:
                fig_h.add_trace(go.Scatter(x=by_h["horizon"], y=by_h[c],
                                           name=name, mode="lines+markers"))
        fig_h.update_layout(height=360, xaxis_title="예측 지평 (일)",
                            yaxis_title="평균 오차 (%p)", hovermode="x unified",
                            title="예측 지평이 길어질수록 오차가 어떻게 커지는가")
        st.plotly_chart(fig_h, width="stretch")

# --------------------------------------------------------------------------
# 데이터 탭
# --------------------------------------------------------------------------
if view == VIEWS[3]:
    st.subheader("기술 통계")
    desc = rdata.describe(df).copy()
    desc["지표"] = desc["지표"].map(humanize_target)
    st.table(desc.set_index("지표"))
    if cov["imputed_dates"]:
        st.caption("선형 보간으로 채운 날: " + ", ".join(cov["imputed_dates"])
                   + " — 학습 타깃에서는 제외된다.")

    st.subheader("전체 시계열")
    long = df.melt(id_vars="Date", value_vars=cols, var_name="지표", value_name="시청률")
    long["방송사"] = long["지표"].map(label_of)
    fig_all = px.line(long, x="Date", y="시청률", color="방송사",
                      color_discrete_map=PALETTE)
    fig_all.update_layout(height=420, hovermode="x unified", yaxis_title="시청률 (%)")
    st.plotly_chart(fig_all, width="stretch")

    st.subheader("이상치 (이동 로버스트 z > 3)")
    oc = st.selectbox("대상", cols, format_func=humanize_target, key="outlier")
    out = rdata.outlier_dates(df, oc)
    if out.empty:
        st.info("탐지된 이상치가 없습니다.")
    else:
        odf = out.reset_index()
        odf.columns = ["날짜", "z-score"]
        odf["날짜"] = odf["날짜"].dt.strftime("%Y-%m-%d (%a)")
        st.dataframe(odf.round(2), hide_index=True, width="stretch", height=260)
        st.caption("대형 이슈 국면으로 판단되면 `data/events.csv` 에 구간을 등록하세요 "
                   "(컬럼: start, end, label, weight). 모델이 별도 변수로 학습합니다.")
