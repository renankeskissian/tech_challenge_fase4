"""
1_🏥_Simulador.py
-----------------
Página principal da aplicação Streamlit — Simulador de Avaliação de Obesidade.

Permite que a equipe médica preencha os dados comportamentais e clínicos
de um paciente e receba a previsão do nível de obesidade com a
confiabilidade associada, gerada pelo modelo de Machine Learning treinado.

Fluxo da aplicação:
    1. Coleta dos inputs do usuário via widgets Streamlit
    2. Conversão dos valores exibidos (português) para os valores originais do modelo (inglês)
    3. Recriação das features de engenharia na mesma lógica do notebook de treino
    4. Montagem do DataFrame com a mesma estrutura e ordem de colunas do treino
    5. Transformação via pipeline e predição via modelo
    6. Exibição do resultado com classificação, confiabilidade e distribuição de probabilidades

Dependências externas:
    - obesidade.pkl  : artefato contendo pipeline e modelo serializados
    - utils.py       : transformadores customizados da pipeline
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from utils import DropFeatures, MinMax, CustomOrdinalEncoder, CustomOneHotEncoder


# ── Carregamento do Artefato ───────────────────────────────────────────────────
# Pipeline e modelo são carregados de um único arquivo para garantir
# que as mesmas transformações do treino sejam aplicadas na predição
artefatos = joblib.load('obesidade.pkl')
pipeline = artefatos['pipeline']
modelo   = artefatos['modelo']


# ── Mapeamento do Target ───────────────────────────────────────────────────────
# Converte a predição numérica (0–6) de volta para o rótulo clínico legível
mapa_obesidade = {
    0: 'Peso Insuficiente',
    1: 'Peso Normal',
    2: 'Sobrepeso Grau I',
    3: 'Sobrepeso Grau II',
    4: 'Obesidade Tipo I',
    5: 'Obesidade Tipo II',
    6: 'Obesidade Tipo III'
}


# ── Interface ──────────────────────────────────────────────────────────────────
st.title('🏥 Simulador de Avaliação de Obesidade')
st.write('Preencha os dados do paciente para obter a previsão do nível de obesidade.')

col1, col2 = st.columns(2)

with col1:
    st.subheader('Dados Pessoais')
    input_gender         = st.radio('Gênero', ['Masculino', 'Feminino'])
    input_age            = float(st.slider('Idade', 14, 61, 25))
    input_height         = float(st.number_input('Altura (metros)', 1.40, 2.00, 1.70, step=0.01))
    input_weight         = float(st.number_input('Peso (kg)', 30.0, 200.0, 70.0, step=0.5))
    input_family_history = st.radio('Histórico familiar de excesso de peso?', ['Sim', 'Não'])
    input_smoke          = st.radio('Fumante?', ['Sim', 'Não'])
    input_scc            = st.radio('Monitora calorias diariamente?', ['Sim', 'Não'])
    input_favc           = st.radio('Come alimentos calóricos com frequência?', ['Sim', 'Não'])

with col2:
    st.subheader('Hábitos Alimentares e Atividade')
    input_fcvc   = st.slider('Frequência de consumo de vegetais (1=raramente, 3=sempre)', 1, 3, 2)
    input_ncp    = st.slider('Número de refeições principais por dia', 1, 4, 3)
    input_caec   = st.selectbox('Come entre as refeições?', ['Não', 'Às vezes', 'Frequentemente', 'Sempre'])
    input_ch2o   = st.slider('Consumo diário de água (1=<1L, 2=1-2L, 3=>2L)', 1, 3, 2)
    input_faf    = st.slider('Frequência de atividade física semanal (0=nenhuma, 3=diária)', 0, 3, 1)
    input_tue    = st.slider('Tempo em dispositivos eletrônicos (0=<2h, 1=3-5h, 2=>5h)', 0, 2, 1)
    input_calc   = st.selectbox('Frequência de consumo de álcool?', ['Não', 'Às vezes', 'Frequentemente', 'Sempre'])
    input_mtrans = st.selectbox('Meio de transporte habitual?', [
        'Transporte Público', 'Automóvel', 'Caminhada', 'Moto', 'Bicicleta'
    ])


# ── Dicionários de Conversão ───────────────────────────────────────────────────
# Os inputs são exibidos em português para o usuário, mas o modelo foi treinado
# com os valores originais em inglês — a conversão garante compatibilidade
mapa_genero  = {'Masculino': 'Male',        'Feminino': 'Female'}
mapa_sim_nao = {'Sim': 'yes',               'Não': 'no'}
mapa_freq    = {
    'Não':            'no',
    'Às vezes':       'Sometimes',
    'Frequentemente': 'Frequently',
    'Sempre':         'Always'
}
mapa_mtrans  = {
    'Transporte Público': 'Public_Transportation',
    'Automóvel':          'Automobile',
    'Caminhada':          'Walking',
    'Moto':               'Motorbike',
    'Bicicleta':          'Bike'
}


# ── Feature Engineering ────────────────────────────────────────────────────────
# Recria as mesmas features derivadas geradas no notebook de treino.
# É obrigatório que a lógica aqui seja idêntica à do notebook para que
# o modelo receba os dados no mesmo formato com que foi treinado.

# Faixa etária categorizada em 5 grupos (0–4)
#input_faixa_etaria = pd.cut([input_age], bins=[0, 18, 30, 45, 60, 100], labels=[0, 1, 2, 3, 4]).astype(int)[0]

# Versões binárias/numéricas das variáveis necessárias para as features derivadas
input_favc_bin   = 1 if mapa_sim_nao[input_favc] == 'yes' else 0
input_family_bin = 1 if mapa_sim_nao[input_family_history] == 'yes' else 0
input_calc_num   = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}[mapa_freq[input_calc]]

# Score de qualidade da dieta: frequência de vegetais × refeições - consumo calórico
input_qualidade_dieta    = input_fcvc * input_ncp - input_favc_bin

# Risco combinado: predisposição genética + hábito alimentar de risco
input_risco_genetico     = input_family_bin * input_favc_bin

# Interação entre sedentarismo tecnológico e consumo de álcool
input_alcool_sedentario  = input_calc_num * input_tue


# ── Montagem do DataFrame ──────────────────────────────────────────────────────
# A ordem das colunas deve ser idêntica à do train_df no momento do pipeline.fit_transform
# Obesity=0 é um placeholder — a coluna é removida antes da predição
novo_paciente = pd.DataFrame([{
    'Age':                   input_age,
    'Height':                input_height,
    'Weight':                input_weight,
    'FCVC':                  float(input_fcvc),
    'NCP':                   float(input_ncp),
    'CH2O':                  float(input_ch2o),
    'FAF':                   float(input_faf),
    'TUE':                   float(input_tue),
    'Gender':                mapa_genero[input_gender],
    'family_history':        mapa_sim_nao[input_family_history],
    'FAVC':                  mapa_sim_nao[input_favc],
    'SMOKE':                 mapa_sim_nao[input_smoke],
    'SCC':                   mapa_sim_nao[input_scc],
    'CAEC':                  mapa_freq[input_caec],
    'CALC':                  mapa_freq[input_calc],
    'MTRANS':                mapa_mtrans[input_mtrans],
    'Obesity':               0,
    'qualidade_dieta':       float(input_qualidade_dieta),
    'risco_genetico_habito': float(input_risco_genetico),
    'alcool_sedentario':     float(input_alcool_sedentario)
}])


# ── Predição ───────────────────────────────────────────────────────────────────
if st.button('🔍 Avaliar Paciente'):

    # IMC calculado apenas para exibição informativa — não entra no modelo
    imc = input_weight / (input_height ** 2)

    # Aplica apenas transform (pipeline já foi fitada no treino)
    paciente_transformado = pipeline.transform(novo_paciente)

    # Remove o placeholder do target antes de passar ao modelo
    paciente_transformado = paciente_transformado.drop(columns=['Obesity'])

    predicao_bruta = modelo.predict(paciente_transformado)
    predicao = np.ravel(predicao_bruta)[0]
    predicao = int(predicao)
    probabilidades = modelo.predict_proba(paciente_transformado)[0]

    # Confiabilidade = probabilidade atribuída à classe predita
    confianca = probabilidades[predicao] * 100
    resultado = mapa_obesidade[predicao]

    st.divider()
    st.subheader('📊 Resultado da Avaliação')

    # Dados antropométricos do paciente para contexto clínico
    st.caption(f'Altura: {input_height:.2f}m | Peso: {input_weight:.1f}kg | IMC: {imc:.1f}')

    # Cor do alerta varia conforme a gravidade da classificação
    if predicao <= 1:
        st.info(f'**Classificação: {resultado}**')
    elif predicao <= 3:
        st.warning(f'**Classificação: {resultado}**')
    else:
        st.error(f'**Classificação: {resultado}**')

    # Faixas de confiabilidade para orientar a interpretação clínica
    if confianca >= 70:
        st.metric('Confiabilidade da previsão', f'{confianca:.1f}%', 'Alta confiança')
    elif confianca >= 50:
        st.metric('Confiabilidade da previsão', f'{confianca:.1f}%', 'Confiança moderada')
    else:
        st.metric('Confiabilidade da previsão', f'{confianca:.1f}%', 'Resultado inconclusivo — consulte um médico')

    # Distribuição completa de probabilidades por classe para análise detalhada
    st.subheader('Probabilidade por classe')
    prob_df = pd.DataFrame({
        'Classificação':    list(mapa_obesidade.values()),
        'Probabilidade (%)': [f'{p*100:.1f}%' for p in probabilidades]
    })
    st.dataframe(prob_df, hide_index=True, use_container_width=True)
