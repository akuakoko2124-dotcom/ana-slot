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
    "表から『末尾』『末尾別差枚数』『平均差枚』『平均G数』『勝率』を抽出してください。"
    "数値のカンマや『+』記号は除去し、純粋な数値型として出力してください。"
    "勝率は文字列のままで構いません。"
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
    
    if not df.empty:
        # 日付変換
        df["date_dt"] = pd.to_datetime(df["date"])
        # 日付の1の位
        df["date_last_digit"] = df["date_dt"].dt.day % 10
        # 曜日
        df["day_of_week"] = df["date_dt"].apply(get_day_of_week_jp)
        # 勝率の数値化
        df["win_rate_num"] = df["win_rate"].apply(_win_rate_to_float)
        
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

def get_day_of_week_jp(dt):
    """datetime オブジェクトを日本語の曜日名に変換"""
    days = ["月", "火", "水", "木", "金", "土", "日"]
    return days[dt.weekday()]

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
            font-family: 'Inter', 'Noto Sans JP', sans-serif !important;
        }

        /* 超リッチなダーク・メッシュグラデーション背景 */
        .stApp {
            background-color: #0b0f19;
            background-image: 
                radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
                radial-gradient(at 50% 0%, hsla(225,39%,30%,0.2) 0, transparent 50%), 
                radial-gradient(at 100% 0%, hsla(339,49%,30%,0.2) 0, transparent 50%);
            background-attachment: fixed;
            min-height: 100vh;
        }

        /* サイドバー (グラスモーフィズム) */
        section[data-testid="stSidebar"] {
            background: rgba(17, 24, 39, 0.4) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
        }

        /* グラデーションテキストの装飾 */
        .page-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #a855f7 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
            letter-spacing: -0.5px;
        }
        .page-subtitle {
            color: #9ca3af;
            font-size: 0.95rem;
            font-weight: 400;
            margin-bottom: 1.8rem;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            padding-bottom: 1rem;
        }

        /* アップロードや入力のカード (超グラスモーフィズム) */
        .card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(20px);
            box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
            transition: transform 0.3s ease;
        }

        /* KPI メトリクスカード (ネオンシャドウ) */
        div[data-testid="metric-container"] {
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.5) 0%, rgba(15, 23, 42, 0.5) 100%);
            border: 1px solid rgba(139, 92, 246, 0.2);
            border-radius: 16px;
            padding: 1.2rem;
            box-shadow: 0 4px 20px rgba(139, 92, 246, 0.1);
            backdrop-filter: blur(12px);
            transition: all 0.3s ease;
        }
        div[data-testid="metric-container"]:hover {
            border-color: rgba(139, 92, 246, 0.5);
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.25);
            transform: translateY(-2px);
        }

        /* ボタンデザイン (サイバーパンク風) */
        .stButton > button {
            background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 2rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        }
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6);
            background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
            color: #ffffff;
        }

        /* タブのスタイリング */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding: 10px 24px;
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-bottom: none;
            color: #9ca3af;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(180deg, rgba(139, 92, 246, 0.15) 0%, rgba(30, 41, 59, 0) 100%);
            color: #f3f4f6;
            border-top: 2px solid #8b5cf6;
        }

        /* テーブル (DataFrame) */
        .dataframe {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.05);
        }
        /* 予測ランキングカード */
        .prediction-card {
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.15);
            transition: all 0.3s ease;
        }
        .prediction-card:hover {
            border-color: #8b5cf6;
            box-shadow: 0 0 30px rgba(139, 92, 246, 0.3);
            transform: scale(1.02);
        }
        .rank-s { border: 2px solid #ff00ff; box-shadow: 0 0 20px #ff00ff44; }
        .rank-a { border: 2px solid #00ffff; box-shadow: 0 0 20px #00ffff44; }
        .rank-b { border: 2px solid #00ff00; box-shadow: 0 0 20px #00ff0044; }

        .rank-badge {
            font-size: 2rem;
            font-weight: 900;
            margin-bottom: 0.5rem;
            display: block;
        }
        .rank-s .rank-badge { color: #ff00ff; text-shadow: 0 0 10px #ff00ff; }
        .rank-a .rank-badge { color: #00ffff; text-shadow: 0 0 10px #00ffff; }
        .rank-b .rank-badge { color: #00ff00; text-shadow: 0 0 10px #00ff00; }
        
        .prediction-label {
            font-size: 0.8rem;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .prediction-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: white;
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
                with st.spinner("🚀 超高速AIで解析中..."):
                    try:
                        client = get_gemini_client()
                        
                        # 【高速化1】画像を送信前に縮小する (最大1200px)
                        img = Image.open(uploaded)
                        img.thumbnail((1200, 1200))

                        # 【高速化2 & 3】最軽量モデル ＋ JSON強制モード
                        response = client.models.generate_content(
                            model="gemini-flash-lite-latest",
                            contents=[GEMINI_PROMPT, img],
                            config=genai_types.GenerateContentConfig(
                                response_mime_type="application/json",
                            )
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

    # ─── 🔮 明日の狙い目予測 ───────────────────
    st.markdown("### 🔮 狙い目予測")
    
    # 予測対象日の選択
    col_target, _ = st.columns([1, 2])
    with col_target:
        target_date = st.date_input("予測対象日を選んでください", value=date.today() + pd.Timedelta(days=1))
    
    t_last_digit = target_date.day % 10
    t_dow = get_day_of_week_jp(target_date)
    
    st.info(f"📅 **解析条件:** 日付末尾: `{t_last_digit}` | 曜日: `{t_dow}` (重み: 日付末尾={st.session_state.weights['date']} / 曜日={st.session_state.weights['dow']} / トレンド={st.session_state.weights['trend']})")
    
    # スコア計算
    terminal_list = sorted(df["terminal_number"].unique())
    scores = []
    
    for t_num in terminal_list:
        sub = df[df["terminal_number"] == t_num]
        
        # A: 同日付末尾の平均勝率
        score_a = sub[sub["date_last_digit"] == t_last_digit]["win_rate_num"].mean()
        if pd.isna(score_a): score_a = 0
            
        # B: 同曜日の平均勝率
        score_b = sub[sub["day_of_week"] == t_dow]["win_rate_num"].mean()
        if pd.isna(score_b): score_b = 0
            
        # C: 直近3回のトレンド (平均差枚をスケーリング: 1000枚で10点相当)
        last_3 = sub.sort_values("date_dt", ascending=False).head(3)
        score_c = last_3["avg_diff"].mean() / 100 if not last_3.empty else 0
        if pd.isna(score_c): score_c = 0
        
        # 荷重平均
        total_score = (
            score_a * st.session_state.weights["date"] +
            score_b * st.session_state.weights["dow"] +
            score_c * st.session_state.weights["trend"]
        )
        
        scores.append({
            "terminal": t_num,
            "score": total_score,
            "details": {"a": score_a, "b": score_b, "c": score_c}
        })
        
    # ランキング
    ranking = sorted(scores, key=lambda x: x["score"], reverse=True)
    
    c1, c2, c3 = st.columns(3)
    ranks = [("S", "rank-s", c1), ("A", "rank-a", c2), ("B", "rank-b", c3)]
    
    for i, (label, css_class, col) in enumerate(ranks):
        if i < len(ranking):
            item = ranking[i]
            with col:
                st.markdown(f"""
                <div class="prediction-card {css_class}">
                    <span class="rank-badge">{label}</span>
                    <div class="prediction-label">推奨末尾</div>
                    <div class="prediction-value">{item['terminal']}</div>
                    <div style="font-size:0.7rem; color:rgba(255,255,255,0.4); margin-top:8px;">
                        スコア: {item['score']:.1f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    tab_h, tab1, tab2, tab3 = st.tabs(["🗺️ ターゲット分析", "🎰 末尾別比較", "📈 時系列推移", "🗃️ 生データ"])

    # ─── Tab_H: ターゲット・デイ分析 (ヒートマップ) ────────
    with tab_h:
        st.markdown("#### 「日付末尾 × 台番号末尾」の相関ヒートマップ")
        
        metric_h = st.radio(
            "表示指標",
            ["平均勝率", "平均差枚"],
            horizontal=True,
            key="hm_metric"
        )
        val_col = "win_rate_num" if metric_h == "平均勝率" else "avg_diff"
        
        # ピボットテーブル作成
        # Y軸: 日付末尾 (0-9), X軸: 台番号末尾 (0-9/ゾロ目)
        pivot_df = df.pivot_table(
            index="date_last_digit",
            columns="terminal_number",
            values=val_col,
            aggfunc="mean"
        ).reindex(index=range(10))
        
        # 表示用に並べ替え (0-9, ゾロ目の順)
        cols = [str(i) for i in range(10)]
        if "ゾロ目" in pivot_df.columns:
            cols.append("ゾロ目")
        pivot_df = pivot_df.reindex(columns=cols)
        
        fig_hm = px.imshow(
            pivot_df,
            labels=dict(x="台末尾", y="日付末尾", color=metric_h),
            x=pivot_df.columns,
            y=pivot_df.index,
            color_continuous_scale="Viridis" if metric_h == "平均勝率" else "RdYlGn",
            aspect="auto",
            text_auto=".1f" if metric_h == "平均勝率" else ".0f"
        )
        fig_hm.update_layout(
            **PLOTLY_THEME,
            xaxis=dict(title="台番号末尾 / ゾロ目", dtick=1),
            yaxis=dict(title="日付末尾 (1の位)", dtick=1),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_hm, use_container_width=True)
        
        st.markdown("""
        <div style="font-size:0.85rem; color:#9ca3af; padding: 10px; border-left: 3px solid #8b5cf6; background: rgba(139, 92, 246, 0.05);">
        💡 <b>見方:</b> 特定の日付末尾（縦軸）において。どの末尾（横軸）が強いかという「店のクセ」を可視化しています。<br>
        色が明るい/緑に近いほどパフォーマンスが良いことを示します。
        </div>
        """, unsafe_allow_html=True)

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
        st.markdown("### ⚙️ 予測重み付け設定")
        w_date = st.slider("📅 日付末尾の重み", 0.0, 1.0, 0.4, 0.1, help="翌日と同じ日付末尾の過去の実績を重視")
        w_dow = st.slider("🗓️ 曜日の重み", 0.0, 1.0, 0.3, 0.1, help="翌日と同じ曜日の過去の実績を重視")
        w_trend = st.slider("📈 トレンドの重み", 0.0, 1.0, 0.3, 0.1, help="直近の勢いを重視")
        
        # セッション状態に保存
        st.session_state.weights = {
            "date": w_date,
            "dow": w_dow,
            "trend": w_trend
        }

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
