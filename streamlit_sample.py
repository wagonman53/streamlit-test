import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np


#弾性グラフ関数
def plot_elasticity(df, x_column, y_column, bin_size=10, bin_threshold=20):
    # binの作成
    bins = np.arange(0, 201, bin_size)
    df = df.copy()
    df['x_bin'] = pd.cut(df[x_column], bins=bins)

    # bin毎の確率を計算
    grouped = df.groupby('x_bin', observed=True)
    total_counts = grouped.size()
    call_or_raise_counts = grouped[y_column].apply(lambda x: ((x == 'call') | (x == 'raise')).sum())
    raise_counts = grouped[y_column].apply(lambda x: (x == 'raise').sum())

    probability_call_or_raise = (call_or_raise_counts / total_counts).reset_index()
    probability_call_or_raise.columns = ['x_bin', 'probability_call_or_raise']

    probability_raise = (raise_counts / total_counts).reset_index()
    probability_raise.columns = ['x_bin', 'probability_raise']

    # binの中央値を計算
    probability_call_or_raise['x_value'] = probability_call_or_raise['x_bin'].apply(lambda x: x.mid)
    probability_raise['x_value'] = probability_raise['x_bin'].apply(lambda x: x.mid)

    # 各ビンのサンプル数を計算
    bin_counts = total_counts.reset_index()
    bin_counts.columns = ['x_bin', 'count']
    bin_counts['x_value'] = bin_counts['x_bin'].apply(lambda x: x.mid)

    # 閾値未満のビンを除外
    valid_bins = bin_counts[bin_counts['count'] >= bin_threshold]['x_bin']
    probability_call_or_raise = probability_call_or_raise[probability_call_or_raise['x_bin'].isin(valid_bins)]
    probability_raise = probability_raise[probability_raise['x_bin'].isin(valid_bins)]
    bin_counts = bin_counts[bin_counts['x_bin'].isin(valid_bins)]

    # サブプロットの作成（2行1列）
    fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                           row_heights=[0.7, 0.3])

    # 折れ線グラフの追加（call or raise）
    fig.add_trace(go.Scatter(x=probability_call_or_raise['x_value'], y=probability_call_or_raise['probability_call_or_raise'], 
                             mode='lines+markers', name='Call+Raise freq'), row=1, col=1)

    # 折れ線グラフの追加（raise only）
    fig.add_trace(go.Scatter(x=probability_raise['x_value'], y=probability_raise['probability_raise'], 
                             mode='lines+markers', name='Raise freq'), row=1, col=1)

    # y = 1 / (1 + x / 100) の曲線を追加
    x_range = np.linspace(0, 200, 1000)
    y_curve = 1 / (1 + x_range / 100)
    fig.add_trace(go.Scatter(x=x_range, y=y_curve, mode='lines', name="MDF"),
                  row=1, col=1)

    # 棒グラフの追加
    fig.add_trace(go.Bar(x=bin_counts['x_value'], y=bin_counts['count'], name="Data Count"),
                  row=2, col=1)

    # レイアウトの設定
    fig.update_layout(
        title='Flop elasticity',
        yaxis_title='Ratio',
        yaxis_tickformat='.0%',
        xaxis2_title='Bet Size (%)',
        yaxis2_title='Data Count',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=800  # グラフの高さを調整
    )

    # Y軸の範囲を0-1に設定（上部のグラフ）
    fig.update_yaxes(range=[0, 1], row=1, col=1)

    return fig


def plot_action_distribution(df, column_name, title):
    # カテゴリ列を作成する関数
    def categorize(value):
        if pd.isna(value):
            return 'check'
        elif value >= 200:
            return '200%over'
        elif 0 <= value < 200:
            return f'{int(value // 10) * 10}%-{int((value // 10) + 1) * 10}%'

    # カテゴリ列を作成
    df['category'] = df[column_name].apply(categorize)

    # カテゴリごとの割合を計算
    category_counts = df['category'].value_counts()
    category_percentages = (category_counts / len(df)) * 100

    # カテゴリを適切な順序でソート
    def sort_key(category):
        if category == 'check':
            return -1
        elif category == '200%over':
            return 201
        else:
            return float(category.split('-')[0][:-1])

    sorted_categories = sorted(category_percentages.index, key=sort_key)
    sorted_categories.reverse()

    # グラフの作成
    fig = go.Figure(go.Bar(
        y=sorted_categories,
        x=[category_percentages[cat] for cat in sorted_categories],
        orientation='h',
        text=[f'{category_percentages[cat]:.1f}%' for cat in sorted_categories],
        textposition='outside',
        textfont=dict(size=12, color='#E0E0E0'),
        marker=dict(color='#5D8AA8', line=dict(color='#B0C4DE', width=1))
    ))

    # レイアウトの設定
    fig.update_layout(
        title=dict(text=title, font=dict(color='#E0E0E0')),
        xaxis_title=dict(text='割合 (%)', font=dict(color='#E0E0E0')),
        yaxis_title=dict(text='カテゴリ', font=dict(color='#E0E0E0')),
        height=max(500, len(sorted_categories) * 30),
        margin=dict(l=150, r=20, t=50, b=50),
        yaxis=dict(color='#E0E0E0'),
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#121212',
        font=dict(color='#E0E0E0')
    )

    return fig


def plot_hand_distribution(df, category_column, title="Category Proportions"):
    # 空の値を除外し、カテゴリごとの割合を計算
    value_counts = df[category_column].dropna().value_counts()
    total = value_counts.sum()
    proportions = (value_counts / total * 100).sort_values(ascending=True)

    # プロットの作成
    fig = go.Figure(go.Bar(
        y=proportions.index,
        x=proportions.values,
        orientation='h',
        text=[f'{value:.1f}%' for value in proportions.values],
        textposition='auto',
        hoverinfo='text',
        hovertext=[f'{index}: {value:.1f}%' for index, value in zip(proportions.index, proportions.values)]
    ))

    # レイアウトの設定
    fig.update_layout(
        title=title,
        xaxis_title="割合 (%)",
        yaxis_title=category_column,
        height=max(500, len(proportions) * 30),  # グラフの高さを動的に調整
        xaxis=dict(range=[0, 100])
    )

    return fig


# CSVファイルの読み込み関数
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)  # CSVファイルのパスを指定してください
    df = df[df['Flop action 1'] == 'check']
    return df


# ページ設定
st.set_page_config(layout="wide")

selected_file = st.sidebar.selectbox("プールデータの選択", ["50-100 1M mda.csv"])
df = load_data(selected_file)

#サイドバーの設定
pot_range = st.sidebar.slider("Postflop開始時のpotsize($)", 2, 50,(5, 10))

aggressor = st.sidebar.selectbox("Preflop Aggressor", ['All'] + list(df['Aggressor'].unique()))

oop_positions = st.sidebar.multiselect("OOP Position", 
                                       ['All'] + list(df['OOP_position'].unique()),
                                       default=['All'])

ip_positions = st.sidebar.multiselect("IP Position", 
                                      ['All'] + list(df['IP_position'].unique()),
                                       default=['All'])

oop_rank = st.sidebar.selectbox("OOP Player Rank", ['All'] + list(df['OOP_player_rank'].unique()))

ip_rank = st.sidebar.selectbox("IP Player Rank", ['All'] + list(df['IP_player_rank'].unique()))

high_card = st.sidebar.slider("ハイカードの範囲", 2, 14,(2, 14))

flop_type = st.sidebar.selectbox("Flopのタイプ", ['All'] + list(df['Flop_type'].unique()))


# フィルタリングを適用
filtered_df = df[
    (df['Pot'] >= pot_range[0]) & (df['Pot'] <= pot_range[1]) &
    (df['Flop_high'] >= high_card[0]) & (df['Flop_high'] <= high_card[1])
]

if aggressor != 'All':
    filtered_df = filtered_df[filtered_df['Aggressor'] == aggressor]

if 'All' not in oop_positions:
    filtered_df = filtered_df[filtered_df['OOP_position'].isin(oop_positions)]

if 'All' not in ip_positions:
    filtered_df = filtered_df[filtered_df['IP_position'].isin(ip_positions)]

if oop_rank != 'All':
    filtered_df = filtered_df[filtered_df['OOP_player_rank'] == oop_rank]

if ip_rank != 'All':
    filtered_df = filtered_df[filtered_df['IP_player_rank'] == ip_rank]

if flop_type != 'All':
    filtered_df = filtered_df[filtered_df['Flop_type'] == flop_type]

# タイトル
st.title("root-check ノードの分析画面")

# 2カラムレイアウトの作成
col1, col2 = st.columns(2)

with col1:
    st.subheader("Bet検討用の情報")
    
    bet_df = filtered_df[filtered_df['Flop action 2'] == "bet"]
    fig1 = plot_elasticity(bet_df, "Flop size 2", "Flop action 3")
    st.plotly_chart(fig1, use_container_width=True)

    flop_bet_size = st.slider("Flopのbet size", 10, 200,(30, 100))
    bet_df_turn = bet_df[(bet_df["Flop size 2"] >= flop_bet_size[0]) & 
                     (bet_df["Flop size 2"] <= flop_bet_size[1]) & 
                     (bet_df["Turn action 1"] == "check") &
                     (bet_df["Turn action 2"] == "bet")]
    fig2 = plot_elasticity(bet_df_turn, "Turn size 2", "Turn action 3")
    st.plotly_chart(fig2, use_container_width=True)
    


with col2:
    st.subheader("Check検討用の情報")

    check_df = filtered_df[filtered_df['Flop action 2'] == "check"]
    fig1 = plot_action_distribution(df,"Turn size 1","Turn OOPbet freq")
    st.plotly_chart(fig1, use_container_width=True)

    bmcb_size = st.slider("Flopのbet size", 10, 200,(40, 60))
    check_df_turn = check_df[(check_df["Turn action 1"] == "bet") &
                             (check_df["Turn size 1"] >= bmcb_size[0]) &
                             (check_df["Turn size 1"] <= bmcb_size[1])]
    fig2 = plot_hand_distribution(check_df_turn,"OOP_Turn_hand_rank","hand rank")
    st.plotly_chart(fig2, use_container_width=True)
