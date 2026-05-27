"""
2_📊_Dashboard.py
-----------------
Página de dashboard analítico da aplicação Streamlit — Obesidade.

Apresenta uma visão analítica completa dos dados e do desempenho do modelo,
dividida em três blocos principais:

    1. KPIs do Modelo   : cards com total de registros, acurácia e AUC
    2. Matriz de Confusão: desempenho do modelo por classe
    3. Gráficos Analíticos: distribuições e relações entre variáveis

Os valores de acurácia, AUC e matriz de confusão são carregados diretamente
do artefato 'obesidade.pkl', calculados uma única vez no notebook de treino
com SEED=42, garantindo consistência entre o dashboard e o modelo em produção.

Dependências externas:
    - obesidade.pkl      : artefato com modelo, pipeline e métricas serializados
    - obesidade_clean.csv: dataset limpo para os gráficos analíticos
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from utils import DropFeatures, MinMax, CustomOrdinalEncoder, CustomOneHotEncoder

st.set_page_config(page_title='Dashboard - Obesidade', layout='wide')

st.title('📊 Dashboard Analítico — Obesidade')
st.write('Visão analítica dos dados utilizados no modelo preditivo de obesidade.')


# ── Carregamento do Artefato ───────────────────────────────────────────────────
# Métricas e matriz foram calculadas no notebook com SEED=42 e salvas no joblib
# para garantir que os valores exibidos sejam sempre idênticos ao modelo em produção
@st.cache_resource
def carregar_artefatos():
    return joblib.load('obesidade.pkl')

artefatos    = carregar_artefatos()
acuracia     = artefatos['acuracia']
auc          = artefatos['auc']
matriz       = artefatos['matriz']
n_registros  = artefatos['n_registros']
classes      = artefatos['classes']

# Nomes das classes em português para exibição na matriz
classes_pt = [
    'Peso Insuficiente', 'Peso Normal',
    'Sobrepeso Grau I',  'Sobrepeso Grau II',
    'Obesidade Tipo I',  'Obesidade Tipo II', 'Obesidade Tipo III'
]


# ── Carregamento dos Dados ─────────────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    df = pd.read_csv('obesidade_clean.csv')

    for col in ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']:
        df[col] = df[col].round().astype(int)

    df['Obesity'] = df['Obesity'].map({
        'Insufficient_Weight': 'Peso Insuficiente',
        'Normal_Weight':       'Peso Normal',
        'Overweight_Level_I':  'Sobrepeso Grau I',
        'Overweight_Level_II': 'Sobrepeso Grau II',
        'Obesity_Type_I':      'Obesidade Tipo I',
        'Obesity_Type_II':     'Obesidade Tipo II',
        'Obesity_Type_III':    'Obesidade Tipo III'
    })

    df['Gender']         = df['Gender'].map({'Male': 'Masculino', 'Female': 'Feminino'})
    df['family_history'] = df['family_history'].map({'yes': 'Sim', 'no': 'Não'})
    df['FAVC']           = df['FAVC'].map({'yes': 'Sim', 'no': 'Não'})
    df['SMOKE']          = df['SMOKE'].map({'yes': 'Sim', 'no': 'Não'})
    df['SCC']            = df['SCC'].map({'yes': 'Sim', 'no': 'Não'})
    df['CAEC']           = df['CAEC'].map({'no': 'Não', 'Sometimes': 'Às vezes', 'Frequently': 'Frequentemente', 'Always': 'Sempre'})
    df['CALC']           = df['CALC'].map({'no': 'Não', 'Sometimes': 'Às vezes', 'Frequently': 'Frequentemente', 'Always': 'Sempre'})
    df['MTRANS']         = df['MTRANS'].map({
        'Public_Transportation': 'Transporte Público',
        'Automobile': 'Automóvel',
        'Walking': 'Caminhada',
        'Motorbike': 'Moto',
        'Bike': 'Bicicleta'
    })

    df['faixa_etaria'] = pd.cut(df['Age'],
        bins=[0, 18, 30, 45, 60, 100],
        labels=['Até 18', '19-30', '31-45', '46-60', '60+']
    )

    df['IMC'] = df['Weight'] / (df['Height'] ** 2)

    return df

df = carregar_dados()

ordem_obesidade = [
    'Peso Insuficiente', 'Peso Normal',
    'Sobrepeso Grau I',  'Sobrepeso Grau II',
    'Obesidade Tipo I',  'Obesidade Tipo II', 'Obesidade Tipo III'
]

# Paleta sequencial de azul — do mais claro (menor gravidade) ao mais escuro
mapa_cores = {
    'Peso Insuficiente': '#c6dbef',
    'Peso Normal':       '#9ecae1',
    'Sobrepeso Grau I':  '#6baed6',
    'Sobrepeso Grau II': '#4292c6',
    'Obesidade Tipo I':  '#2171b5',
    'Obesidade Tipo II': '#08519c',
    'Obesidade Tipo III':'#08306b'
}

# Paleta divergente para o scatter — maximiza contraste entre categorias adjacentes
cores_scatter = {
    'Peso Insuficiente': '#4575b4',
    'Peso Normal':       '#91bfdb',
    'Sobrepeso Grau I':  '#fee090',
    'Sobrepeso Grau II': '#fc8d59',
    'Obesidade Tipo I':  '#d73027',
    'Obesidade Tipo II': '#a50026',
    'Obesidade Tipo III':'#67001f'
}


# ── Filtros ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('Filtros')
    genero_sel = st.multiselect(
        'Gênero', df['Gender'].unique().tolist(),
        default=df['Gender'].unique().tolist()
    )
    faixa_sel = st.multiselect(
        'Faixa etária', ['Até 18', '19-30', '31-45', '46-60', '60+'],
        default=['Até 18', '19-30', '31-45', '46-60', '60+']
    )

df_f = df[df['Gender'].isin(genero_sel) & df['faixa_etaria'].isin(faixa_sel)]

st.markdown('---')


# ── Cards de KPI ───────────────────────────────────────────────────────────────
st.subheader('📈 Desempenho do Modelo')
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(
        label='📋 Total de Registros',
        value=f'{n_registros:,}'.replace(',', '.'),
        help='Total de pacientes no dataset utilizado para treino e teste'
    )

with kpi2:
    st.metric(
        label='🎯 Acurácia do Modelo',
        value=f'{acuracia}%',
        help='Percentual de classificações corretas no conjunto de teste (Voting RF + CatBoost + LGBM)'
    )

with kpi3:
    st.metric(
        label='📊 AUC (weighted OvR)',
        value=f'{auc}',
        help='Área sob a curva ROC com estratégia one-vs-rest e média ponderada entre as 7 classes'
    )

st.markdown('---')


# ── Matriz de Confusão ─────────────────────────────────────────────────────────
st.subheader('🔲 Matriz de Confusão Normalizada')
st.caption('Proporção de acertos e erros por classe no conjunto de teste. Diagonal principal = acertos.')

# Constrói heatmap interativo com Plotly a partir da matriz numpy salva no joblib
z_text = [[f'{v:.2f}' for v in row] for row in matriz]

fig_cm = ff.create_annotated_heatmap(
    z=np.round(matriz, 2).tolist(),
    x=classes_pt,
    y=classes_pt,
    annotation_text=z_text,
    colorscale='Blues',
    showscale=True
)
fig_cm.update_layout(
    xaxis_title='Label Predita',
    yaxis_title='Label Verdadeira',
    xaxis=dict(tickangle=-30),
    height=500
)
# Eixo Y invertido para manter a convenção padrão (classe 0 no topo)
fig_cm['layout']['yaxis']['autorange'] = 'reversed'

st.plotly_chart(fig_cm, use_container_width=True, key='matriz_confusao')

st.markdown('---')


# ── Gráficos Analíticos ────────────────────────────────────────────────────────
st.subheader('📊 Análise Exploratória dos Dados')

# Linha 1: Distribuição % + Boxplot Idade
col1, col2 = st.columns(2)

with col1:
    st.subheader('Distribuição dos Níveis de Obesidade (%)')
    df_pct = df_f['Obesity'].value_counts(normalize=True).reset_index()
    df_pct.columns = ['Obesity', 'Percentual']
    df_pct['Percentual'] = df_pct['Percentual'] * 100
    df_pct['Obesity'] = pd.Categorical(df_pct['Obesity'], categories=ordem_obesidade, ordered=True)
    df_pct = df_pct.sort_values('Obesity')
    fig = px.bar(df_pct, x='Obesity', y='Percentual',
                 color='Obesity', color_discrete_map=mapa_cores,
                 category_orders={'Obesity': ordem_obesidade},
                 text_auto='.1f')
    fig.update_layout(showlegend=False, xaxis_title='Categoria de Peso',
                      yaxis_title='Percentual (%)', xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True, key='distribuicao')

with col2:
    st.subheader('Faixa de Idade por Nível de Peso')
    fig = px.box(df_f, x='Obesity', y='Age',
                 color='Obesity', color_discrete_map=mapa_cores,
                 category_orders={'Obesity': ordem_obesidade},
                 labels={'Obesity': 'Categoria de Peso', 'Age': 'Idade'})
    fig.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True, key='boxplot_idade')

# Linha 2: Scatter Altura vs Peso + Histograma IMC
col3, col4 = st.columns(2)

with col3:
    st.subheader('Explorador: Altura vs Peso')
    fig = px.scatter(df_f, x='Height', y='Weight',
                     color='Obesity',
                     color_discrete_map=cores_scatter,
                     category_orders={'Obesity': ordem_obesidade},
                     hover_data=['Age', 'Gender'],
                     labels={'Height': 'Altura (m)', 'Weight': 'Peso (kg)', 'Obesity': 'Nível'})
    st.plotly_chart(fig, use_container_width=True, key='scatter_imc')

with col4:
    st.subheader('Distribuição do IMC na População')
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    sns.histplot(df_f['IMC'], bins=30, kde=True, color='#6baed6', edgecolor='black', ax=ax)
    ax.set_xlabel('IMC (kg/m²)', color='black')
    ax.set_ylabel('Frequência', color='black')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# Linha 3: FAVC + Gênero empilhado
col5, col6 = st.columns(2)

with col5:
    st.subheader('Consumo de Alimentos Calóricos por Nível de Peso')
    fig = px.histogram(df_f, x='Obesity', color='FAVC',
                       category_orders={'Obesity': ordem_obesidade, 'FAVC': ['Sim', 'Não']},
                       barmode='group',
                       text_auto=True,
                       color_discrete_map={'Sim': '#08306b', 'Não': '#9ecae1'},
                       labels={'Obesity': 'Categoria de Peso', 'FAVC': 'Come calóricos?',
                                'count': 'Quantidade'})
    fig.update_layout(xaxis_title='Categoria de Peso', yaxis_title='Quantidade',
                      xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True, key='favc')

with col6:
    st.subheader('Proporção de Níveis de Peso por Gênero (100% Empilhado)')
    df_gender = df_f.groupby(['Gender', 'Obesity']).size().reset_index(name='Count')
    df_gender['Percentual'] = df_gender.groupby('Gender')['Count'].transform(
        lambda x: (x / x.sum()) * 100
    )
    fig = px.bar(df_gender, x='Gender', y='Percentual',
                 color='Obesity',
                 color_discrete_map=mapa_cores,
                 category_orders={'Obesity': ordem_obesidade},
                 text_auto='.1f',
                 barmode='stack',
                 labels={'Gender': 'Gênero', 'Percentual': 'Percentual (%)', 'Obesity': 'Nível'})
    fig.update_layout(xaxis_title='Gênero', yaxis_title='Percentual (%)')
    st.plotly_chart(fig, use_container_width=True, key='genero_empilhado')
