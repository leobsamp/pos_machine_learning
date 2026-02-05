
import streamlit as st
import pandas as pd
import requests
from datetime import date

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


# ---------------- UI ----------------

st.set_page_config(
    page_title="PTAX - Cotação do Dólar",
    layout="wide"
)

st.title("💵 PTAX — Cotação do Dólar (Banco Central)")
st.write("Consulta automática via API oficial do BCB.")

with st.sidebar:
    st.header("Parâmetros")

    hoje = date.today()

    data_ini = st.date_input(
        "Data inicial",
        value=date(hoje.year, 1, 1)
    )

    data_fim = st.date_input(
        "Data final",
        value=hoje
    )

    consultar = st.button("Consultar")


if consultar:

    if data_ini > data_fim:
        st.error("A data inicial não pode ser maior que a data final.")
        st.stop()

    with st.spinner("Consultando API do Banco Central..."):
        try:
            df = cotacao_dolar_periodo_df(data_ini, data_fim)
        except Exception as e:
            st.error(f"Erro ao consultar API: {e}")
            st.stop()

    if df.empty:
        st.warning("Nenhuma cotação encontrada para o período.")
        st.stop()

    # ---------- Métrica principal ----------
    ultima = df.iloc[-1]

    col1, col2 = st.columns(2)

    col1.metric(
        "Última cotação de compra",
        f"R$ {ultima['cotacaoCompra']:.4f}"
    )

    col2.metric(
        "Última cotação de venda",
        f"R$ {ultima['cotacaoVenda']:.4f}"
    )

    # ---------- Gráfico ----------
    st.subheader("📈 Evolução do dólar")

    df_plot = df.set_index("dataHoraCotacao")[[
        "cotacaoCompra",
        "cotacaoVenda"
    ]]

    st.line_chart(df_plot)

    # ---------- Tabela ----------
    st.subheader("📊 Dados completos")

    st.dataframe(
        df,
        use_container_width=True
    )

    # ---------- Download ----------
    st.download_button(
        "⬇️ Baixar CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="cotacao_dolar_ptax.csv",
        mime="text/csv",
    )

else:
    st.info("Selecione o período e clique em **Consultar**.")
