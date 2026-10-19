
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.write(
    "영화의 흥행 관련 수치를 분석해 비슷한 특징을 가진 "
    "영화 세 그룹으로 나누어 봅니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/kobis_movies.csv"
)

FEATURES = [
    "first_scrn",
    "first_show",
    "total_audi",
    "days_in_top10",
]

KOREAN_NAMES = {
    "first_scrn": "개봉 첫날 스크린 수",
    "first_show": "개봉 첫날 상영 횟수",
    "total_audi": "누적 관객 수",
    "days_in_top10": "박스오피스 TOP 10 진입 일수",
}

GROUP_NAMES = {
    0: "㉠",
    1: "㉡",
    2: "㉢",
}

# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    required = [
        "movieCd",
        "movieNm",
        "openDt",
        "first_scrn",
        "first_show",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    missing_columns = [
        col for col in required if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSV에 필요한 열이 없습니다: "
            + ", ".join(missing_columns)
        )

    numeric_cols = [
        "first_scrn",
        "first_show",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


try:
    original_df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.info("데이터 URL과 CSV 열 이름을 확인해 주세요.")
    st.stop()

st.caption(f"불러온 전체 영화 수: {len(original_df):,}편")

# -----------------------------
# 데이터 전처리
# -----------------------------
# 분석에 필요한 수치가 없거나 첫 주 관객이 0인 영화 제외
valid_mask = (
    original_df[FEATURES].notna().all(axis=1)
    & original_df["first_week_audi"].notna()
    & (original_df["first_week_audi"] > 0)
)

df = original_df.loc[valid_mask].copy()

# K-평균은 수치형 속성만 사용
if len(df) < 3:
    st.error("군집 분석을 하려면 분석 가능한 영화가 최소 3편 필요합니다.")
    st.stop()

st.write(
    f"분석 가능한 영화: **{len(df):,}편** "
    f"/ 제외된 영화: **{len(original_df) - len(df):,}편**"
)

# -----------------------------
# 사용할 속성 선택
# -----------------------------
st.subheader("1. 군집 분석에 사용할 속성")

selected_features = st.multiselect(
    "영화 유형을 나누는 데 사용할 속성을 선택하세요.",
    options=FEATURES,
    default=FEATURES,
    format_func=lambda x: KOREAN_NAMES[x],
)

if not selected_features:
    st.warning("분석에 사용할 속성을 하나 이상 선택해 주세요.")
    st.stop()

# -----------------------------
# 표준화 및 K-평균 군집화
# -----------------------------
X = df[selected_features].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["군집"] = kmeans.fit_predict(X_scaled)

# 군집 번호는 누적 관객 평균이 작은 그룹부터 큰 그룹 순으로
# ㉠, ㉡, ㉢에 대응시킴
cluster_order = (
    df.groupby("군집")["total_audi"]
    .mean()
    .sort_values()
    .index
    .tolist()
)

cluster_labels = {
    cluster_id: GROUP_NAMES[i]
    for i, cluster_id in enumerate(cluster_order)
}

df["영화 유형"] = df["군집"].map(cluster_labels)

# -----------------------------
# 군집 결과 요약
# -----------------------------
st.subheader("2. 영화 유형별 결과")

summary_cols = [
    "first_scrn",
    "first_show",
    "total_audi",
    "days_in_top10",
]

summary = (
    df.groupby("영화 유형")[summary_cols]
    .mean()
    .rename(columns=KOREAN_NAMES)
    .reindex(["㉠", "㉡", "㉢"])
    .round(1)
)

summary.index.name = "영화 유형"

counts = (
    df["영화 유형"]
    .value_counts()
    .reindex(["㉠", "㉡", "㉢"], fill_value=0)
)

st.write("각 유형에 속한 영화 수")
c1, c2, c3 = st.columns(3)

for col, label in zip([c1, c2, c3], ["㉠", "㉡", "㉢"]):
    col.metric(f"{label} 유형", f"{counts[label]:,}편")

st.write("유형별 수치 평균 (원래 단위)")
st.dataframe(summary, use_container_width=True)

st.caption(
    "유형 기호는 누적 관객 평균이 낮은 그룹부터 ㉠, ㉡, ㉢ 순으로 붙였습니다. "
    "각 유형은 정답이나 영화의 공식 장르가 아닌 데이터 기반 군집입니다."
)

# -----------------------------
# 2차원 산점도
# -----------------------------
st.subheader("3. 2차원 산점도")

left, right = st.columns(2)

with left:
    x_feature = st.selectbox(
        "가로축 속성",
        selected_features,
        index=0,
        format_func=lambda x: KOREAN_NAMES[x],
        key="x_feature",
    )

with right:
    y_default = min(1, len(selected_features) - 1)
    y_feature = st.selectbox(
        "세로축 속성",
        selected_features,
        index=y_default,
        format_func=lambda x: KOREAN_NAMES[x],
        key="y_feature",
    )

plot_df = df.copy()
plot_df["영화 유형"] = pd.Categorical(
    plot_df["영화 유형"],
    categories=["㉠", "㉡", "㉢"],
    ordered=True,
)

fig2 = px.scatter(
    plot_df,
    x=x_feature,
    y=y_feature,
    color="영화 유형",
    hover_name="movieNm",
    hover_data={
        "openDt": True,
        "first_week_audi": ":,",
        "total_audi": ":,",
        "영화 유형": True,
    },
    category_orders={"영화 유형": ["㉠", "㉡", "㉢"]},
    labels={
        x_feature: KOREAN_NAMES[x_feature],
        y_feature: KOREAN_NAMES[y_feature],
        "영화 유형": "영화 유형",
        "openDt": "개봉일",
        "first_week_audi": "첫 주 관객 수",
        "total_audi": "누적 관객 수",
    },
    title="영화 유형별 2차원 분포",
)

fig2.update_layout(
    height=600,
    legend_title_text="영화 유형",
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 3차원 산점도
# -----------------------------
st.subheader("4. 3차원 산점도")

if len(selected_features) < 3:
    st.info("3차원 산점도를 그리려면 분석 속성을 3개 이상 선택하세요.")
else:
    col_x, col_y, col_z = st.columns(3)

    with col_x:
        x3 = st.selectbox(
            "X축",
            selected_features,
            index=0,
            format_func=lambda x: KOREAN_NAMES[x],
            key="x3",
        )

    with col_y:
        y3 = st.selectbox(
            "Y축",
            selected_features,
            index=1,
            format_func=lambda x: KOREAN_NAMES[x],
            key="y3",
        )

    with col_z:
        z3 = st.selectbox(
            "Z축",
            selected_features,
            index=2,
            format_func=lambda x: KOREAN_NAMES[x],
            key="z3",
        )

    fig3 = px.scatter_3d(
        plot_df,
        x=x3,
        y=y3,
        z=z3,
        color="영화 유형",
        hover_name="movieNm",
        hover_data={
            "total_audi": ":,",
            "first_week_audi": ":,",
        },
        category_orders={"영화 유형": ["㉠", "㉡", "㉢"]},
        labels={
            x3: KOREAN_NAMES[x3],
            y3: KOREAN_NAMES[y3],
            z3: KOREAN_NAMES[z3],
            "영화 유형": "영화 유형",
        },
        title="영화 유형별 3차원 분포",
    )

    fig3.update_layout(
        height=700,
        legend_title_text="영화 유형",
    )

    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# 유형별 영화 다섯 편
# -----------------------------
st.subheader("5. 유형별 누적 관객 수 TOP 5")

for label in ["㉠", "㉡", "㉢"]:
    st.markdown(f"### {label} 유형")

    group = (
        df[df["영화 유형"] == label]
        .sort_values("total_audi", ascending=False)
        .head(5)
        .copy()
    )

    if group.empty:
        st.write("해당 유형에 속한 영화가 없습니다.")
        continue

    display = group[
        [
            "movieNm",
            "openDt",
            "first_scrn",
            "first_show",
            "first_week_audi",
            "total_audi",
            "days_in_top10",
        ]
    ].rename(
        columns={
            "movieNm": "영화명",
            "openDt": "개봉일",
            "first_scrn": "개봉 첫날 스크린 수",
            "first_show": "개봉 첫날 상영 횟수",
            "first_week_audi": "첫 주 관객 수",
            "total_audi": "누적 관객 수",
            "days_in_top10": "TOP 10 진입 일수",
        }
    )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
    )

# -----------------------------
# 분석 데이터 다운로드
# -----------------------------
st.subheader("6. 분석 결과 다운로드")

download_df = df.copy()
download_df["군집"] = download_df["영화 유형"]

csv_data = download_df.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    label="군집 분석 결과 CSV 다운로드",
    data=csv_data,
    file_name="movie_clusters.csv",
    mime="text/csv",
)

with st.expander("분석 방법 알아보기"):
    st.markdown(
        """
        1. 분석에 필요한 수치가 누락된 영화와 첫 주 관객 수가 0인 영화를 제외합니다.
        2. 선택한 속성들의 단위와 규모 차이를 줄이기 위해 표준화합니다.
        3. K-평균 알고리즘으로 영화 데이터를 세 그룹으로 분류합니다.
        4. 산점도로 그룹의 분포를 확인하고, 유형별 평균과 대표 영화를 비교합니다.

        **주의:** 군집 번호 자체에는 순서나 우열의 의미가 없습니다.
        이 앱에서는 결과를 읽기 쉽게 하기 위해 누적 관객 평균을 기준으로
        ㉠, ㉡, ㉢을 표시합니다.
        """
    )
