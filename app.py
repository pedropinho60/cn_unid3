from typing import Callable
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from methods import (
    _linear_regression_base,
    linear_regression,
    exponential_regression,
    geometric_regression,
    power_regression,
    logarithmic_regression,
    polinomial_regression
)

# FUNÇÕES DE AJUSTE MMQ

def r2_score(y, y_pred):
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean)**2)
    ss_res = np.sum((y - y_pred)**2)
    return 1 - ss_res/ss_tot

# LEITURA E LIMPEZA DOS DADOS

def ler_dados_co2(caminho_arquivo):
    dados_anuais = {}
    with open(caminho_arquivo, "r") as f:
        for linha in f:
            if linha.startswith("#") or linha.strip() == "":
                continue
            partes = linha.strip().split()
            if len(partes) < 5:
                continue
            try:
                ano = int(partes[0])
                valor = float(partes[4])
                if ano not in dados_anuais:
                    dados_anuais[ano] = []
                dados_anuais[ano].append(valor)
            except Exception:
                continue
    medias_anuais = []
    for ano in sorted(dados_anuais.keys()):
        valores = dados_anuais[ano]
        media = sum(valores)/len(valores)
        medias_anuais.append((ano, round(media, 3)))
    return pd.DataFrame(medias_anuais, columns=["Ano", "CO2 (ppm)"]) # pyright: ignore[reportArgumentType]

def ler_dados_temperatura(caminho_arquivo):
    anos = []
    temp = []
    with open(caminho_arquivo, "r", encoding="latin1") as f:
        for linha in f:
            if linha.startswith("Year") or linha.startswith("----") or linha.strip() == "":
                continue
            partes = linha.strip().split()
            if len(partes) < 2:
                continue
            try:
                ano = int(partes[0])
                valor = float(partes[1])
                anos.append(ano)
                temp.append(valor)
            except Exception:
                continue
    return pd.DataFrame({"Ano": anos, "Temperatura (°C)": temp})

# CÁLCULO E PLOTAGEM

def rodar_ajustes(df, eixo_y):
    x = df["Ano"].values
    y = df[eixo_y].values
    resultados = {}

    coef = _linear_regression_base(x, y)
    lr = linear_regression(x, y)
    y_pred = lr(x)
    if coef is not None:
        resultados["Linear"] = {"coef": coef, "y_pred": y_pred, "r2": r2_score(y, y_pred)}

    coef = _linear_regression_base(x, y)
    le = exponential_regression(x, y)
    y_pred = le(x)
    if coef is not None:
        resultados["Exponencial"] = {"coef": coef, "y_pred": y_pred, "r2": r2_score(y, y_pred)}

    coef = _linear_regression_base(x, y)
    lg = logarithmic_regression(x, y)
    y_pred = lg(x)
    if coef is not None:
        resultados["Logarítmico"] = {"coef": coef, "y_pred": y_pred, "r2": r2_score(y, y_pred)}

    coef = _linear_regression_base(x, y)
    lp = power_regression(x, y)
    y_pred = lp(x)
    if coef is not None:
        resultados["Potência"] = {"coef": coef, "y_pred": y_pred, "r2": r2_score(y, y_pred)}

    coef = _linear_regression_base(x, y)
    lg = geometric_regression(x, y)
    y_pred = lg(x)
    if coef is not None:
        resultados["Geométrico"] = {"coef": coef, "y_pred": y_pred, "r2": r2_score(y, y_pred)}

    coef = _linear_regression_base(x, y)
    lq = polinomial_regression(x, y)
    y_pred = lq(x)
    resultados["Polinomial"] = {"coef": coef, "y_pred": y_pred, "r2": r2_score(y, y_pred)}

    return resultados

def plotar_ajuste_individual(df, eixo_y, nome, res):
    x = df["Ano"].values
    y = df[eixo_y].values

    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(x, y, label="Dados Reais", color="black")
    ax.plot(x, res["y_pred"], label=f"{nome} (R²={res['r2']:.3f})", color="red")
    ax.set_xlabel("Ano")
    ax.set_ylabel(eixo_y)
    ax.set_title(f"Ajuste: {nome}")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plotar_ajustes(df, eixo_y, resultados, titulo):
    x = df["Ano"].values
    y = df[eixo_y].values

    fig, ax = plt.subplots(figsize=(10,5))
    ax.scatter(x, y, label="Dados Reais", color="black")

    for nome, res in resultados.items():
        ax.plot(x, res["y_pred"], label=f"{nome} (R²={res['r2']:.3f})")

    ax.set_xlabel("Ano")
    ax.set_ylabel(eixo_y)
    ax.set_title(titulo)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def escolher_melhor_ajuste(resultados):
    return max(resultados.items(), key=lambda item: item[1]["r2"])

def prever_futuro(df, nome_ajuste, ano_futuro):
    x = df["Ano"].values
    y = df[eixo_y].values

    methods = {
        "Linear": linear_regression,
        "Exponencial": exponential_regression,
        "Logarítmico": logarithmic_regression,
        "Potência": power_regression,
        "Geométrico": geometric_regression,
        "Polinomial": polinomial_regression
    }

    regression = methods[nome_ajuste]

    reg = regression(x, y)

    return reg(ano_futuro)

# STREAMLIT UI

st.title("Dashboard Climático: Temperatura e CO₂")

st.markdown("""
Este dashboard mostra dados históricos e análises de **Concentração de CO₂** e **Anomalia da Temperatura Global**,
com ajustes por Mínimos Quadrados para diferentes modelos matemáticos.

Fonte dos dados:
- [Temperatura (NASA)](https://climate.nasa.gov/vital-signs/global-temperature/?intent=121)
- [CO₂ (NASA)](https://climate.nasa.gov/vital-signs/carbon-dioxide/?intent=121)
""")

tipo_dado = st.selectbox("Escolha o conjunto de dados:", ("CO₂ Atmosférico (NOAA)", "Temperatura Global (NASA GISS)"))

if tipo_dado == "CO₂ Atmosférico (NOAA)":
    caminho = "co2_mm_mlo.txt"
    df = ler_dados_co2(caminho)
    eixo_y = "CO2 (ppm)"
    titulo = "Concentração Anual de CO₂ Atmosférico (ppm)"
else:
    caminho = "temperature.txt"
    df = ler_dados_temperatura(caminho)
    eixo_y = "Temperatura (°C)"
    titulo = "Anomalia da Temperatura Global (°C)"

st.subheader("Dados Brutos")
st.dataframe(df)

st.subheader("Ajustes via mínimos quadrados")

resultados = rodar_ajustes(df, eixo_y)

st.write("Coeficiente de determinação R² para cada ajuste:")
r2s = {k: v["r2"] for k,v in resultados.items()}
st.table(pd.DataFrame.from_dict(r2s, orient="index", columns=["R²"]))

st.subheader("Ajustes individuais")
for nome, res in resultados.items():
    if nome in ["Exponencial", "Potência", "Geométrico"] and tipo_dado != "CO₂ Atmosférico (NOAA)":
        continue
    plotar_ajuste_individual(df, eixo_y, nome, res)

st.subheader("Todos os ajustes")
plotar_ajustes(df, eixo_y, resultados, titulo)

melhor_nome, melhor_res = escolher_melhor_ajuste(resultados)
st.markdown(f"### Melhor ajuste: **{melhor_nome}** (R² = {melhor_res['r2']:.4f})")

st.subheader("Estimativa futura")
ano_futuro = st.slider("Escolha o ano para previsão", min_value=df["Ano"].max()+1, max_value=2100, value=df["Ano"].max()+10)

pred = prever_futuro(df, melhor_nome, ano_futuro)
if pred is not None:
    st.markdown(f"Previsão para {ano_futuro}: **{pred:.3f} {eixo_y.split(' ')[0]}** em relação aos 30 anos anteriores")
else:
    st.markdown("Previsão indisponível para este modelo e ano.")
