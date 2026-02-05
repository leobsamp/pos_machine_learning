
import streamlit as st
import pandas as pd
import requests
from datetime import date
import plotly.graph_objects as go
from scr_pipeline import pipeline_scrdata

# ==============================
# PTAX - DÓLAR
# ==============================
BASE = "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata"

def _fmt_mmddyyyy(d: date) -> str:
    return d.strftime("%m-%d-%Y")

@st.cache_data(ttl=3600)
def cotacao_dolar_periodo_df(data_ini: date, data_fim: date) -> pd.DataFrame:
    url = f"{BASE}/CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)"
    params = {
        "@dataInicial": f"'{_fmt_mmddyyyy(data_ini)}'",
        "@dataFinalCotacao": f"'{_fmt_mmddyyyy(data_fim)}'",
        "$format": "json",
        "$select": "cotacaoCompra,cotacaoVenda,dataHoraCotacao",
        "$top": 10000,
    }

    r = requests.get(url, params=params, timeout=30)
    if not r.ok:
        raise requests.HTTPError(f"{r.status_code} - {r.text}", response=r)

    payload = r.json()
    df = pd.DataFrame(payload.get("value", []))

    if df.empty:
        return df

    df["dataHoraCotacao"] = pd.to_datetime(df["dataHoraCotacao"])
    df = df.sort_values("dataHoraCotacao").reset_index(drop=True)
    return df


def dolar_diario(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["data"] = df["dataHoraCotacao"].dt.date
    out = (
        df.groupby("data", as_index=False)[["cotacaoCompra", "cotacaoVenda"]]
          .mean()
          .rename(columns={
              "cotacaoCompra": "compra",
              "cotacaoVenda": "venda"
          })
    )
    out["data"] = pd.to_datetime(out["data"])
    return out.sort_values("data").reset_index(drop=True)


# ==============================
# SCR
# ==============================
@st.cache_data(show_spinner=True)
def carregar_scr(ano: int) -> pd.DataFrame:
    resultado = pipeline_scrdata(
        ano,
        salvar_parquet=True,
        salvar_csv=False,
        verbose=False
    )
    return resultado["df_processed"]


# ==============================
# UI
# ==============================
st.set_page_config(
    page_title="Painel BCB - Dólar e SCR",
    layout="wide"
)

st.title("📊 Painel BCB — Dólar (PTAX) e SCR.data")

with st.sidebar:
    st.header("Fonte de dados")
    fonte = st.selectbox(
        "Escolha o dado",
        ["Dólar comercial (PTAX)", "SCR.data"]
    )


# ==============================
# DÓLAR
# ==============================
if fonte == "Dólar comercial (PTAX)":

    st.subheader("💵 Dólar comercial — PTAX")

    hoje = date.today()

    with st.sidebar:
        st.header("Parâmetros")
        data_ini = st.date_input("Data inicial", value=date(hoje.year, 1, 1))
        data_fim = st.date_input("Data final", value=hoje)
        consultar = st.button("Consultar")

    if consultar:
        if data_ini > data_fim:
            st.error("A data inicial não pode ser maior que a data final.")
            st.stop()

        with st.spinner("Consultando API do Banco Central..."):
            df_raw = cotacao_dolar_periodo_df(data_ini, data_fim)
            df = dolar_diario(df_raw)

        if df.empty:
            st.warning("Nenhuma cotação encontrada para o período.")
            st.stop()

        ultima = df.iloc[-1]
        c1, c2 = st.columns(2)
        c1.metric("Última compra", f"R$ {ultima['compra']:.4f}")
        c2.metric("Última venda", f"R$ {ultima['venda']:.4f}")

        st.subheader("📈 Evolução diária do dólar")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["data"], y=df["compra"], name="Compra"))
        fig.add_trace(go.Scatter(x=df["data"], y=df["venda"], name="Venda"))
        fig.update_layout(
            height=520,
            xaxis_title="Data",
            yaxis_title="R$",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Baixar CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="cotacao_dolar_ptax.csv",
            mime="text/csv"
        )

    else:
        st.info("Selecione o período e clique em **Consultar**.")


# ==============================
# SCR
# ==============================
else:
    st.subheader("📊 SCR.data — Sistema de Informações de Crédito")

    with st.sidebar:
        ano = st.number_input("Ano", min_value=2010, max_value=2100, value=2025, step=1)

    df = carregar_scr(int(ano))

    if df.empty:
        st.warning("Nenhum dado disponível para o ano selecionado.")
        st.stop()

    df["data_base"] = pd.to_datetime(df["data_base"])

    with st.sidebar:
        st.header("Filtros")
        uf = st.selectbox("UF", ["(Todas)"] + sorted(df["uf"].dropna().unique()))
        cliente = st.selectbox("Cliente", ["(Todos)"] + sorted(df["cliente"].dropna().unique()))
        modalidade = st.selectbox("Modalidade", ["(Todas)"] + sorted(df["modalidade"].dropna().unique()))

    df_f = df.copy()
    if uf != "(Todas)": df_f = df_f[df_f["uf"] == uf]
    if cliente != "(Todos)": df_f = df_f[df_f["cliente"] == cliente]
    if modalidade != "(Todas)": df_f = df_f[df_f["modalidade"] == modalidade]

    st.caption(f"Registros após filtros: {len(df_f):,}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Carteira ativa", f"{df_f['carteira_ativa'].sum():,.2f}")
    c2.metric("Inadimplência", f"{df_f['carteira_inadimplencia'].sum():,.2f}")

    taxa = (
        (df_f["taxa_inadimplencia"] * df_f["carteira_ativa"]).sum()
        / df_f["carteira_ativa"].sum()
        if df_f["carteira_ativa"].sum() > 0 else 0
    )
    c3.metric("Taxa inadimplência", f"{taxa:.2%}")

    st.subheader("📈 Evolução da carteira ativa")
    serie_ativa = df_f.groupby("data_base")["carteira_ativa"].sum()
    st.line_chart(serie_ativa, use_container_width=True)

    st.subheader("📈 Evolução da inadimplência")
    serie_inad = df_f.groupby("data_base")["carteira_inadimplencia"].sum()
    st.line_chart(serie_inad, use_container_width=True)

    st.subheader("📊 Top modalidades por carteira ativa")
    top_mod = df_f.groupby("modalidade")["carteira_ativa"].sum().sort_values(ascending=False).head(15)
    st.bar_chart(top_mod, use_container_width=True)

    st.subheader("📊 Top UFs por carteira ativa")
    top_uf = df_f.groupby("uf")["carteira_ativa"].sum().sort_values(ascending=False).head(15)
    st.bar_chart(top_uf, use_container_width=True)

    st.subheader("📥 Exportar dados filtrados")
    st.download_button(
        "Baixar CSV",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name=f"scr_filtrado_{ano}.csv",
        mime="text/csv"
    )
