
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================================
# 1. 페이지 설정
# =====================================
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 영화 유형 나누기")
st.write(
    "영화의 흥행 관련 수치를 분석하여 "
    "비슷한 특징을 가진 영화 세 그룹으로 나눕니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/kobis_movies.csv"
)

FEATURE_NAMES = {
    "first_scrn": "첫 관측일 스크린 수",
    "first_show": "첫 관측일 상영 횟수",
    "first_week_audi": "개봉 첫 주 관객 수",
    "total_audi": "누적 관객 수",
    "days_in_top10": "TOP 10 진입 일수",
}

GROUP_NAMES = {0: "㉠", 1: "㉡", 2: "㉢"}

# =====================================
# 2. 데이터 불러오기
# =====================================
@st.cache_data(ttl=3600)
def load_from_url(url):
    return pd.read_csv(url, encoding="utf-8-sig")


uploaded_file = st.file_uploader(
    "CSV 파일 업로드 (URL 오류가 발생하면 사용하세요)",
    type=["csv"],
)

df = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        st.success("업로드한 CSV 파일을 불러왔습니다.")
    except Exception as e:
        st.error(f"CSV 파일을 읽지 못했습니다: {e}")
        st.stop()
else:
    with st.spinner("온라인 데이터 주소를 확인하고 있습니다..."):
        try:
            df = load_from_url(DATA_URL)
            st.success("온라인 데이터를 불러왔습니다.")
        except Exception:
            st.warning(
                "온라인 데이터 주소에 접속할 수 없습니다. "
                "원본 CSV 파일을 아래 업로더에 넣어 주세요."
            )
            st.markdown(
                "원본 파일을 준비할 수 없다면, "
                "GitHub 저장소의 데이터 파일이 복구될 때까지 기다려야 합니다."
            )
            st.stop()

# =====================================
# 3. 데이터 확인 및 전처리
# =====================================
st.subheader("1. 데이터 확인")

df.columns = df.columns.astype(str).str.strip()

required_columns = ["movieNm"]
if "movieNm" not in df.columns:
    st.error(
        "CSV에 영화명 열인 movieNm이 없습니다. "
        "올바른 영화 데이터 파일인지 확인해 주세요."
    )
    st.stop()

if "movieNm" not in df.columns:
    st.stop()

# 영화명 결측값 제거
df = df.dropna(subset=["movieNm"]).copy()
df["movieNm"] = df["movieNm"].astype(str)

# 숫자형 속성 변환
available_features = [
    col for col in FEATURE_NAMES if col in df.columns
]

for col in available_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if len(available_features) < 2:
    st.error(
        "군집 분석에 필요한 수치형 열이 부족합니다. "
        "first_scrn, first_show, first_week_audi, "
        "total_audi, days_in_top10 열이 있는지 확인해 주세요."
    )
    st.stop()

st.write(f"전체 영화 수: **{len(df):,}편**")
st.write("데이터 미리보기")
st.dataframe(df.head(10), use_container_width=True)

# =====================================
# 4. 분석할 속성 선택
# =====================================
st.divider()
st.subheader("2. 분석할 속성 선택")

selected_features = st.multiselect(
    "군집 분석에 사용할 속성을 선택하세요.",
    options=available_features,
    default=available_features,
    format_func=lambda col: FEATURE_NAMES[col],
)

if len(selected_features) < 2:
    st.warning("분석할 속성을 2개 이상 선택해 주세요.")
    st.stop()

# 선택한 속성의 결측값이 있는 영화 제외
analysis_df = df.dropna(subset=selected_features).copy()

# 세 그룹을 만들기 위해 최소 3편 필요
if len(analysis_df) < 3:
    st.error("분석 가능한 영화가 3편 이상이어야 합니다.")
    st.stop()

# 모든 값이 같은 열은 군집 분석에 의미가 없으므로 제거
constant_features = [
    col for col in selected_features
    if analysis_df[col].nunique() <= 1
]

if constant_features:
    st.warning(
        "모든 영화의 값이 같아 분석에 사용할 수 없는 속성: "
        + ", ".join(FEATURE_NAMES[c] for c in constant_features)
    )
    selected_features = [
        col for col in selected_features
        if col not in constant_features
    ]

if len(selected_features) < 2:
    st.error("서로 다른 값을 가진 속성이 2개 이상 필요합니다.")
    st.stop()

X = analysis_df[selected_features].astype(float)

# =====================================
# 5. 표준화 및 K-평균 군집 분석
# =====================================
st.divider()
st.subheader("3. K-평균 군집 분석")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

analysis_df["군집번호"] = kmeans.fit_predict(X_scaled)

# 누적 관객 평균이 낮은 군집부터 ㉠, ㉡, ㉢ 표시
if "total_audi" in analysis_df.columns:
    cluster_order = (
        analysis_df.groupby("군집번호")["total_audi"]
        .mean()
        .sort_values()
        .index.tolist()
    )
else:
    cluster_order = sorted(analysis_df["군집번호"].unique())

label_map = {
    cluster_id: GROUP_NAMES[i]
    for i, cluster_id in enumerate(cluster_order)
}

analysis_df["영화 유형"] = analysis_df["군집번호"].map(label_map)

st.write(
    f"군집 분석에 사용한 영화: **{len(analysis_df):,}편**"
)
st.write(
    f"분석 속성: **{', '.join(FEATURE_NAMES[c] for c in selected_features)}**"
)

# =====================================
# 6. 유형별 영화 수
# =====================================
st.divider()
st.subheader("4. 유형별 영화 수")

counts = (
    analysis_df["영화 유형"]
    .value_counts()
    .reindex(["㉠", "㉡", "㉢"], fill_value=0)
)

c1, c2, c3 = st.columns(3)
for col, label in zip([c1, c2, c3], ["㉠", "㉡", "㉢"]):
    col.metric(f"{label} 유형", f"{counts[label]:,}편")

count_df = counts.rename_axis("영화 유형").reset_index(name="영화 수")

fig_count = px.bar(
    count_df,
    x="영화 유형",
    y="영화 수",
    color="영화 유형",
    category_orders={"영화 유형": ["㉠", "㉡", "㉢"]},
    title="영화 유형별 편수",
    text="영화 수",
)
fig_count.update_layout(showlegend=False)
st.plotly_chart(fig_count, use_container_width=True)

st.text_input(
    "이 그래프로 알 수 있는 것",
    key="note_count",
    placeholder="예: 어떤 유형에 속한 영화가 가장 많은가?",
)

# =====================================
# 7. 유형별 속성 평균
# =====================================
st.divider()
st.subheader("5. 유형별 속성 평균")

summary = (
    analysis_df.groupby("영화 유형")[selected_features]
    .mean()
    .reindex(["㉠", "㉡", "㉢"])
    .round(1)
    .rename(columns=FEATURE_NAMES)
)

st.dataframe(summary, use_container_width=True)

st.text_input(
    "유형별 평균에서 알 수 있는 것",
    key="note_mean",
    placeholder="예: 누적 관객 평균은 어느 유형에서 가장 높은가?",
)

# =====================================
# 8. 2차원 산점도
# =====================================
st.divider()
st.subheader("6. 2차원 산점도")

col1, col2 = st.columns(2)

with col1:
    x_feature = st.selectbox(
        "가로축",
        selected_features,
        index=0,
        format_func=lambda col: FEATURE_NAMES[col],
        key="scatter_x",
    )

with col2:
    y_feature = st.selectbox(
        "세로축",
        selected_features,
        index=1,
        format_func=lambda col: FEATURE_NAMES[col],
        key="scatter_y",
    )

fig_scatter = px.scatter(
    analysis_df,
    x=x_feature,
    y=y_feature,
    color="영화 유형",
    hover_name="movieNm",
    hover_data={
        "영화 유형": True,
        "군집번호": False,
    },
    category_orders={"영화 유형": ["㉠", "㉡", "㉢"]},
    labels={
        x_feature: FEATURE_NAMES[x_feature],
        y_feature: FEATURE_NAMES[y_feature],
        "영화 유형": "영화 유형",
    },
    title=(
        f"{FEATURE_NAMES[x_feature]}과 "
        f"{FEATURE_NAMES[y_feature]}에 따른 영화 유형"
    ),
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.text_input(
    "이 산점도로 알 수 있는 것",
    key="note_scatter",
    placeholder="예: 비슷한 속성을 가진 영화들이 가까이 모이는가?",
)

# =====================================
# 9. 3차원 산점도
# =====================================
st.divider()
st.subheader("7. 3차원 산점도")

if len(selected_features) >= 3:
    col_x, col_y, col_z = st.columns(3)

    with col_x:
        x3 = st.selectbox(
            "X축 속성",
            selected_features,
            index=0,
            format_func=lambda col: FEATURE_NAMES[col],
            key="x3",
        )

    with col_y:
        y3 = st.selectbox(
            "Y축 속성",
            selected_features,
            index=1,
            format_func=lambda col: FEATURE_NAMES[col],
            key="y3",
        )

    with col_z:
        z3 = st.selectbox(
            "Z축 속성",
            selected_features,
            index=2,
            format_func=lambda col: FEATURE_NAMES[col],
            key="z3",
        )

    fig_3d = px.scatter_3d(
        analysis_df,
        x=x3,
        y=y3,
        z=z3,
        color="영화 유형",
        hover_name="movieNm",
        category_orders={"영화 유형": ["㉠", "㉡", "㉢"]},
        labels={
            x3: FEATURE_NAMES[x3],
            y3: FEATURE_NAMES[y3],
            z3: FEATURE_NAMES[z3],
            "영화 유형": "영화 유형",
        },
        title="세 가지 속성으로 살펴본 영화 유형",
    )

    fig_3d.update_layout(height=700)
    st.plotly_chart(fig_3d, use_container_width=True)

    st.text_input(
        "이 3차원 그래프로 알 수 있는 것",
        key="note_3d",
        placeholder="그래프에서 발견한 특징을 한 문장으로 적어 보세요.",
    )
else:
    st.info(
        "3차원 산점도를 보려면 서로 다른 분석 속성을 "
        "3개 이상 선택하세요."
    )

# =====================================
# 10. 유형별 누적 관객 TOP 5
# =====================================
st.divider()
st.subheader("8. 유형별 대표 영화")

if "total_audi" in analysis_df.columns:
    for label in ["㉠", "㉡", "㉢"]:
        st.markdown(f"#### {label} 유형")

        group = (
            analysis_df[analysis_df["영화 유형"] == label]
            .sort_values("total_audi", ascending=False)
            .head(5)
        )

        display_columns = ["movieNm"]

        for col in [
            "openDt",
            "first_scrn",
            "first_show",
            "first_week_audi",
            "total_audi",
            "days_in_top10",
        ]:
            if col in group.columns:
                display_columns.append(col)

        display_df = group[display_columns].rename(
            columns={
                "movieNm": "영화명",
                "openDt": "개봉일",
                "first_scrn": "첫 관측일 스크린 수",
                "first_show": "첫 관측일 상영 횟수",
                "first_week_audi": "개봉 첫 주 관객 수",
                "total_audi": "누적 관객 수",
                "days_in_top10": "TOP 10 진입 일수",
            }
        )

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
        )
else:
    st.info("누적 관객 수 열이 없어 대표 영화를 정렬할 수 없습니다.")

# =====================================
# 11. 결과 다운로드
# =====================================
st.divider()
st.subheader("9. 분석 결과 다운로드")

csv_bytes = analysis_df.to_csv(
    index=False,
    encoding="utf-8-sig",
).encode("utf-8-sig")

st.download_button(
    label="분석 결과 CSV 다운로드",
    data=csv_bytes,
    file_name="movie_clusters.csv",
    mime="text/csv",
)

with st.expander("분석 방법과 주의점"):
    st.markdown(
        """
        - **표준화:** 단위가 다른 속성들을 비슷한 척도로 바꿉니다.
        - **K-평균:** 선택한 수치가 비슷한 영화끼리 3개 그룹으로 묶습니다.
        - **군집 번호:** ㉠, ㉡, ㉢은 데이터에서 발견한 그룹을 구분하는 이름입니다.
        - **주의:** 군집은 영화의 공식 장르나 작품성 평가가 아닙니다.
          어떤 속성을 선택하느냐에 따라 결과가 달라질 수 있습니다.
        - **결측값:** 선택한 분석 속성에 값이 없는 영화는 분석에서 제외됩니다.
        """
    )
