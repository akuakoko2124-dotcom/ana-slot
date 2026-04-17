"""
パチスロ末尾別データ解析ダッシュボード
====================================
- 画像アップロード → Gemini 1.5 Flash で表を抽出
- SQLite に蓄積
- Plotly でダッシュボード表示
"""

import os
import json
import sqlite3
import re
from datetime import date
from typing import Optional, List

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from google.genai import types as genai_types
from PIL import Image

# ─────────────────────────────────────────
# 定数・設定
# ─────────────────────────────────────────
DB_PATH = "slot_data.db"
TERMINAL_OPTIONS = [str(i) for i in range(10)] + ["ゾロ目"]

GEMINI_PROMPT = (
    "この画像はパチスロの末尾別データです。"
    "表から『末尾』『末尾別差枚数』『平均差枚』『平均G数』『勝率』を抽出し、"
    "純粋なJSON形式で出力してください。"
    "数値のカンマや『+』記号は除去し、純粋な数値型としてください。"
    "勝率は文字列のままで構いません。"
    "出力形式の例:\n"
    '[\n'
    '  {"末尾": "0", "末尾別差枚数": 12345, "平均差枚": 234, "平均G数": 5678, "勝率": "52.3%"},\n'
    '  ...\n'
    ']\n'
    "JSON以外のテキストは一切出力しないでください。"
)

# ─────────────────────────────────────────
# DB 初期化
# ─────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS terminal_data (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            date            TEXT    NOT NULL,
            terminal_number TEXT    NOT NULL,
            total_diff      REAL,
            avg_diff        REAL,
            avg_game        REAL,
            win_rate        TEXT,
            UNIQUE(date, terminal_number)
        )
    """)
    con.commit()
    con.close()

def save_to_db(date_str: str, rows: List[dict]):
    """rows: Gemini が返した JSON リスト"""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    inserted = 0
    skipped = 0
    for row in rows:
        try:
            cur.execute(
                """
                INSERT OR REPLACE INTO terminal_data
                    (date, terminal_number, total_diff, avg_diff, avg_game, win_rate)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    date_str,
                    str(row.get("末尾", "")),
                    _to_float(row.get("末尾別差枚数")),
                    _to_float(row.get("平均差枚")),
                    _to_float(row.get("平均G数")),
                    str(row.get("勝率", "")),
                ),
            )
            inserted += 1
        except Exception as e:
            skipped += 1
            st.warning(f"行の保存をスキップ: {row} → {e}")
    con.commit()
    con.close()
    return inserted, skipped

def load_all_data() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM terminal_data ORDER BY date, terminal_number", con
    )
    con.close()
    return df

# ─────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────
def _to_float(val):
    if val is None:
        return None
    try:
        cleaned = re.sub(r"[^\d.\-]", "", str(val))
        return float(cleaned) if cleaned not in ("", "-") else None
    except Exception:
        return None

def _win_rate_to_float(val: str) -> Optional[float]:
    """'52.3%' → 52.3"""
    try:
        return float(str(val).replace("%", "").strip())
    except Exception:
        return None

def get_gemini_client():
    api_key = None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "Gemini APIキーが見つかりません。\n"
            "`.streamlit/secrets.toml` に `GEMINI_API_KEY = '...'` を記載するか、"
            "環境変数 `GEMINI_API_KEY` を設定してください。"
        )
        st.stop()
    return genai.Client(
        api_key=api_key,
        http_options=genai_types.HttpOptions(api_version="v1beta"),
    )

def parse_gemini_response(text: str) -> List[dict]:
    """Gemini レスポンスから JSON 部分だけ抽出してパース"""
    # コードブロックを除去
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.strip("`").strip()
    # 配列部分だけ取り出す
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)

# ─────────────────────────────────────────
# ページ共通スタイル
# ─────────────────────────────────────────
def apply_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Noto+Sans+JP:wght@300;400;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Noto Sans JP', 'Inter', sans-serif;
        }

        /* ダークグラデーション背景 */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
        }

        /* サイドバー */
        section[data-testid="stSidebar"] {
            background: rgba(255,255,255,0.04) !important;
            border-right: 1px solid rgba(255,255,255,0.08);
            backdrop-filter: blur(12px);
        }

        /* カード風コンテナ */
        .card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 1.5rem 2rem;
            margin-bottom: 1.2rem;
            backdrop-filter: blur(10px);
        }

        /* メトリクスカード */
        div[data-testid="metric-container"] {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }

        /* ボタン */
        .stButton > button {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(102,126,234,0.5);
        }

        /* DataFrameテーブル */
        .dataframe { border-radius: 10px; overflow: hidden; }

        /* ヘッダー装飾 */
        .page-title {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
        }
        .page-subtitle {
            color: rgba(255,255,255,0.5);
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }

        /* タブ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 8px 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────
# ページ: データ入力
# ─────────────────────────────────────────
def page_input():
    st.markdown('<div class="page-title">📥 データ入力</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">スクリーンショットをアップロードして末尾別データを抽出します</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        selected_date = st.date_input("📅 対象日付", value=date.today(), key="input_date")
        uploaded = st.file_uploader(
            "🖼️ スクリーンショット",
            type=["png", "jpg", "jpeg", "webp"],
            key="uploaded_image",
        )
        analyze_btn = st.button("🔍 AIで解析する", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="アップロード画像", use_container_width=True)

    with col2:
        if "extracted_rows" not in st.session_state:
            st.session_state.extracted_rows = None
        if "extraction_date" not in st.session_state:
            st.session_state.extraction_date = None

        if analyze_btn:
            if not uploaded:
                st.warning("画像をアップロードしてください。")
            else:
                with st.spinner("AIで解析中..."):
                    try:
                        client = get_gemini_client()
                        img = Image.open(uploaded)
                        response = client.models.generate_content(
                            model="gemini-flash-latest",
                            contents=[GEMINI_PROMPT, img],
                        )
                        raw_text = response.text
                        rows = parse_gemini_response(raw_text)
                        st.session_state.extracted_rows = rows
                        st.session_state.extraction_date = str(selected_date)
                        st.success(f"✅ {len(rows)} 件のデータを抽出しました！")
                    except json.JSONDecodeError as e:
                        st.error(f"JSONパースエラー: {e}\n\nGemini 生レスポンス:\n```\n{raw_text}\n```")
                    except Exception as e:
                        st.error(f"解析エラー: {e}")

        if st.session_state.extracted_rows:
            st.markdown("### 📋 抽出結果プレビュー")
            df_preview = pd.DataFrame(st.session_state.extracted_rows)
            st.dataframe(df_preview, use_container_width=True)

            st.markdown(
                f"**対象日付:** `{st.session_state.extraction_date}`"
            )

            col_save, col_clear = st.columns(2)
            with col_save:
                if st.button("💾 DBに保存する", use_container_width=True):
                    ins, skp = save_to_db(
                        st.session_state.extraction_date,
                        st.session_state.extracted_rows,
                    )
                    st.success(f"保存完了: {ins} 件追加 / {skp} 件スキップ")
                    st.session_state.extracted_rows = None
                    st.session_state.extraction_date = None
            with col_clear:
                if st.button("🗑️ クリア", use_container_width=True):
                    st.session_state.extracted_rows = None
                    st.session_state.extraction_date = None
                    st.rerun()

# ─────────────────────────────────────────
# ページ: ダッシュボード
# ─────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(color="rgba(255,255,255,0.85)", family="Noto Sans JP, Inter"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.15)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.15)"),
)

def page_dashboard():
    st.markdown('<div class="page-title">📊 ダッシュボード</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">蓄積データから末尾別トレンドを分析します</div>',
        unsafe_allow_html=True,
    )

    df = load_all_data()
    if df.empty:
        st.info("データがまだありません。まず「データ入力」ページで画像を解析・保存してください。")
        return

    # 勝率を数値化
    df["win_rate_num"] = df["win_rate"].apply(_win_rate_to_float)

    # ─── KPI サマリー ───────────────────────
    total_records = len(df)
    days_count = df["date"].nunique()
    best_terminal = (
        df.groupby("terminal_number")["win_rate_num"].mean().idxmax()
        if not df["win_rate_num"].isna().all()
        else "N/A"
    )
    total_diff_sum = df["total_diff"].sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("📅 データ日数", f"{days_count} 日")
    k2.metric("📝 総レコード数", f"{total_records} 件")
    k3.metric("🏆 平均勝率 最高末尾", f"末尾 {best_terminal}")
    k4.metric("💴 累計差枚合計", f"{total_diff_sum:,.0f} 枚")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🎰 末尾別比較", "📈 時系列推移", "🗃️ 生データ"])

    # ─── Tab1: 末尾別比較 ────────────────────
    with tab1:
        agg = (
            df.groupby("terminal_number")
            .agg(
                平均勝率=("win_rate_num", "mean"),
                累計差枚=("total_diff", "sum"),
                平均差枚=("avg_diff", "mean"),
                平均G数=("avg_game", "mean"),
                データ件数=("id", "count"),
            )
            .reset_index()
            .rename(columns={"terminal_number": "末尾"})
        )

        col_l, col_r = st.columns(2)

        with col_l:
            fig1 = px.bar(
                agg,
                x="末尾",
                y="平均勝率",
                color="平均勝率",
                color_continuous_scale="plasma",
                title="末尾別 平均勝率 (%)",
                text_auto=".1f",
            )
            fig1.update_layout(**PLOTLY_THEME, title_font_size=16, showlegend=False)
            fig1.update_traces(marker_line_width=0)
            st.plotly_chart(fig1, use_container_width=True)

        with col_r:
            fig2 = px.bar(
                agg,
                x="末尾",
                y="累計差枚",
                color="累計差枚",
                color_continuous_scale="RdYlGn",
                title="末尾別 累計差枚数",
                text_auto=",.0f",
            )
            fig2.update_layout(**PLOTLY_THEME, title_font_size=16, showlegend=False)
            fig2.update_traces(marker_line_width=0)
            st.plotly_chart(fig2, use_container_width=True)

        # レーダーチャート
        radar_cols = ["平均勝率", "平均差枚", "平均G数"]
        agg_radar = agg[["末尾"] + radar_cols].dropna()
        if not agg_radar.empty:
            # 正規化 (0-1)
            for c in radar_cols:
                mn, mx = agg_radar[c].min(), agg_radar[c].max()
                agg_radar[c] = (
                    (agg_radar[c] - mn) / (mx - mn) if mx != mn else 0.5
                )
            fig_radar = go.Figure()
            for _, row in agg_radar.iterrows():
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=[row[c] for c in radar_cols] + [row[radar_cols[0]]],
                        theta=radar_cols + [radar_cols[0]],
                        fill="toself",
                        name=f"末尾 {row['末尾']}",
                        opacity=0.6,
                    )
                )
            fig_radar.update_layout(
                **PLOTLY_THEME,
                title="末尾別 総合レーダー（正規化）",
                polar=dict(
                    bgcolor="rgba(255,255,255,0.04)",
                    radialaxis=dict(visible=True, color="rgba(255,255,255,0.4)"),
                    angularaxis=dict(color="rgba(255,255,255,0.4)"),
                ),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.dataframe(agg, use_container_width=True)

    # ─── Tab2: 時系列推移 ───────────────────
    with tab2:
        terminals = sorted(df["terminal_number"].unique())
        selected_terminals = st.multiselect(
            "末尾を選択（複数可）",
            options=terminals,
            default=terminals[:3] if len(terminals) >= 3 else terminals,
        )
        metric_choice = st.selectbox(
            "指標を選択",
            ["win_rate_num", "avg_diff", "total_diff", "avg_game"],
            format_func=lambda x: {
                "win_rate_num": "勝率 (%)",
                "avg_diff": "平均差枚",
                "total_diff": "末尾別差枚数",
                "avg_game": "平均G数",
            }[x],
        )

        df_filtered = df[df["terminal_number"].isin(selected_terminals)].copy()
        df_filtered["date"] = pd.to_datetime(df_filtered["date"])
        df_filtered = df_filtered.sort_values("date")

        if df_filtered.empty:
            st.info("選択した末尾のデータがありません。")
        else:
            fig_line = px.line(
                df_filtered,
                x="date",
                y=metric_choice,
                color="terminal_number",
                markers=True,
                title=f"末尾別 時系列推移 — {metric_choice}",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig_line.update_layout(
                **PLOTLY_THEME,
                title_font_size=16,
                legend_title_text="末尾",
                xaxis_title="日付",
            )
            st.plotly_chart(fig_line, use_container_width=True)

    # ─── Tab3: 生データ ──────────────────────
    with tab3:
        date_options = ["すべて"] + sorted(df["date"].unique().tolist(), reverse=True)
        sel_date = st.selectbox("日付フィルター", date_options)
        df_show = df if sel_date == "すべて" else df[df["date"] == sel_date]
        st.dataframe(
            df_show[
                ["date", "terminal_number", "total_diff", "avg_diff", "avg_game", "win_rate"]
            ].rename(
                columns={
                    "date": "日付",
                    "terminal_number": "末尾",
                    "total_diff": "末尾別差枚数",
                    "avg_diff": "平均差枚",
                    "avg_game": "平均G数",
                    "win_rate": "勝率",
                }
            ),
            use_container_width=True,
        )
        csv = df_show.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ CSVダウンロード",
            data=csv,
            file_name=f"slot_data_{sel_date}.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────
# メイン
# ─────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="パチスロ末尾解析ダッシュボード",
        page_icon="🎰",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_css()
    init_db()

    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem 0 1.5rem;">
                <div style="font-size:2.5rem;">🎰</div>
                <div style="font-weight:700; font-size:1.1rem; color:white;">末尾別データ解析</div>
                <div style="font-size:0.75rem; color:rgba(255,255,255,0.4); margin-top:4px;">Powered by Gemini 1.5 Flash</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "ナビゲーション",
            ["📥 データ入力", "📊 ダッシュボード"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        db_df = load_all_data()
        st.markdown(
            f"""
            <div style="font-size:0.8rem; color:rgba(255,255,255,0.5);">
            📦 DB 内レコード数: <b style="color:white;">{len(db_df)}</b><br>
            📅 取込済み日数: <b style="color:white;">{db_df['date'].nunique() if not db_df.empty else 0}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if page == "📥 データ入力":
        page_input()
    else:
        page_dashboard()


if __name__ == "__main__":
    main()
