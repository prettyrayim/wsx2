summary = (
    analysis.groupby("영화 유형")[summary_cols]
    .mean()
    .reindex(GROUP_LABELS)
    .round(1)
    .rename(columns=FEATURE_LABELS)
)
st.dataframe(summary, use_container_width=True)

st.divider()
st.subheader("4. 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_feature = st.selectbox(
        "가로축", selected_features, index=0,
        format_func=lambda c: FEATURE_LABELS[c], key="scatter_x"
    )
with col2:
    y_feature = st.selectbox(
        "세로축", selected_features, index=1,
        format_func=lambda c: FEATURE_LABELS[c], key="scatter_y"
    )

plot_data = analysis.copy()
fig2 = px.scatter(
    plot_data,
    x=x_feature,
    y=y_feature,
    color="영화 유형",
    hover_name="영화명",
    hover_data=[c for c in ["개봉일", "장르", "국적"] if c in plot_data.columns],
    category_orders={"영화 유형": GROUP_LABELS},
    labels={
        x_feature: FEATURE_LABELS[x_feature],
        y_feature: FEATURE_LABELS[y_feature],
        "영화 유형": "영화 유형",
    },
    title="선택한 속성으로 본 영화 유형",
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.subheader("5. 3차원 산점도")
if len(selected_features) >= 3:
    c1, c2, c3 = st.columns(3)
    with c1:
        x3 = st.selectbox(
            "X축", selected_features, index=0,
            format_func=lambda c: FEATURE_LABELS[c], key="x3"
        )
    with c2:
        y3 = st.selectbox(
            "Y축", selected_features, index=1,
            format_func=lambda c: FEATURE_LABELS[c], key="y3"
        )
    with c3:
        z3 = st.selectbox(
            "Z축", selected_features, index=2,
            format_func=lambda c: FEATURE_LABELS[c], key="z3"
        )
    fig3 = px.scatter_3d(
        plot_data, x=x3, y=y3, z=z3,
        color="영화 유형", hover_name="영화명",
        category_orders={"영화 유형": GROUP_LABELS},
        labels={
            x3: FEATURE_LABELS[x3],
            y3: FEATURE_LABELS[y3],
            z3: FEATURE_LABELS[z3],
            "영화 유형": "영화 유형",
        },
        title="세 가지 속성으로 본 영화 유형",
    )
    fig3.update_layout(height=650)
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("3차원 산점도를 보려면 분석 속성을 3개 이상 선택하세요.")

st.divider()
st.subheader("6. 유형별 대표 영화 TOP 5")
rank_col = (
    "누적관객(분석기간)"
    if "누적관객(분석기간)" in analysis.columns
    else selected_features[0]
)

for label in GROUP_LABELS:
    st.markdown(f"#### {label} 유형")
    group = (
        analysis[analysis["영화 유형"] == label]
        .sort_values(rank_col, ascending=False)
        .head(5)
    )
    display_cols = [
        c for c in [
            "영화명", "개봉일", "장르", "국적",
            "누적관객(분석기간)", "최대월관객",
            "최대스크린수", "관측월수", "누적매출(분석기간)"
        ]
        if c in group.columns
    ]
    display = group[display_cols].rename(columns=FEATURE_LABELS)
    st.dataframe(display, hide_index=True, use_container_width=True)

st.divider()
st.subheader("7. 분석 결과 저장")
csv_data = analysis.to_csv(index=False, encoding="utf-8-sig")
st.download_button(
    "분석 결과 CSV 다운로드",
    data=csv_data,
    file_name="movie_clusters.csv",
    mime="text/csv",
)

with st.expander("분석 방법과 주의점"):
    st.markdown(
        """
        - 월별 CSV의 영화별 관객 수와 스크린 수를 모아 분석용 수치를 만들었습니다.
        - 표준화로 속성별 단위 차이를 줄인 뒤 K-평균으로 세 그룹을 찾습니다.
        - ㉠, ㉡, ㉢은 누적 관객 평균 순서로 붙인 구분 기호이며 공식 장르가 아닙니다.
        - 이 데이터는 2024년 1월~2026년 4월 월별 자료를 바탕으로 합니다.
          원래 사용하려던 216편짜리 `kobis_movies.csv`와는 데이터 범위와 변수 정의가 다릅니다.
        """
    )
