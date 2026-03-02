import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import s3fs
from datetime import date
import pyarrow.parquet as pq


# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Painel BCB — PTAX & SCR", layout="wide")

S3_BUCKET = "projeto-bcb-scr-datalake"
S3_REGION = "us-east-2"  # ajuste se o bucket estiver em outra região

# Regras de versionamento
V1_YEARS = list(range(2012, 2024))  # 2012..2023
V2_YEARS = [2024, 2025]

S3_PREFIX_V1_KEY = "scr/processed/versao=v1/"
S3_PREFIX_V2_KEY = "scr/processed/versao=v2/"

PTAX_BASE = "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata"


# ==============================
# Helpers
# ==============================
def versao_por_ano(ano: int) -> str:
    if ano in V1_YEARS:
        return "v1"
    if ano in V2_YEARS:
        return "v2"
    raise ValueError("Ano fora do intervalo suportado (2012–2025).")


def pick_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _fmt_mmddyyyy(d: date) -> str:
    return d.strftime("%m-%d-%Y")


@st.cache_data(ttl=3600)
def cotacao_dolar_periodo_df(data_ini: date, data_fim: date) -> pd.DataFrame:
    url = f"{PTAX_BASE}/CotacaoDolarPeriodo(dataInicial=@dataInicial,dataFinalCotacao=@dataFinalCotacao)"
    params = {
        "@dataInicial": f"'{_fmt_mmddyyyy(data_ini)}'",
        "@dataFinalCotacao": f"'{_fmt_mmddyyyy(data_fim)}'",
        "$format": "json",
        "$select": "cotacaoCompra,cotacaoVenda,dataHoraCotacao",
        "$top": 10000,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    df = pd.DataFrame(payload.get("value", []))
    if df.empty:
        return df
    df["dataHoraCotacao"] = pd.to_datetime(df["dataHoraCotacao"])
    return df.sort_values("dataHoraCotacao").reset_index(drop=True)


def dolar_diario(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["dia"] = d["dataHoraCotacao"].dt.date
    d = d.groupby("dia", as_index=False)[["cotacaoCompra", "cotacaoVenda"]].mean()
    d["dia"] = pd.to_datetime(d["dia"])
    return d


def _anon_fs() -> s3fs.S3FileSystem:
    # Bucket público: acesso anônimo (sem IAM, sem secrets)
    return s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": S3_REGION})


@st.cache_data(ttl=3600)
def carregar_scr_parquet_publico(ano: int) -> pd.DataFrame:
    versao = versao_por_ano(ano)
    prefix = S3_PREFIX_V1_KEY if versao == "v1" else S3_PREFIX_V2_KEY

    base_prefix = f"{S3_BUCKET}/{prefix}ano={ano}/scrdata_{ano}.parquet"
    fs = _anon_fs()

    paths = []
    try:
        paths = [p for p in fs.ls(base_prefix) if p.endswith(".parquet")]
    except Exception:
        paths = []

    def _normalizar_ano(df: pd.DataFrame) -> pd.DataFrame:
        """Garante que a coluna 'ano' seja sempre Int64, independente do tipo original."""
        if "ano" in df.columns:
            df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
        return df

    if not paths:
        file_path = f"s3://{S3_BUCKET}/{prefix}ano={ano}/scrdata_{ano}.parquet"
        # Lê com unified_schema para evitar conflito de tipos entre row groups
        table = pq.read_table(file_path, filesystem=fs)
        df = table.cast(
            _schema_normalizado(table.schema)
        ).to_pandas()
    else:
        import pyarrow as pa
        tables = []
        for p in paths:
            with fs.open(p, "rb") as f:
                t = pq.read_table(f)
                # Converte colunas dictionary para seus value_type antes de concatenar
                t = _cast_dict_columns(t)
                tables.append(t)
        # Agora todos os schemas são compatíveis
        df = pa.concat_tables(tables, promote_options="default").to_pandas()

    return _normalizar_ano(df)


def _cast_dict_columns(table):
    """Converte colunas do tipo dictionary para o tipo base (ex: dict<int32> -> int32)."""
    import pyarrow as pa
    new_fields = []
    new_columns = []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if pa.types.is_dictionary(field.type):
            # Decodifica o dictionary para o tipo dos values
            col = col.cast(field.type.value_type)
            field = field.with_type(field.type.value_type)
        new_fields.append(field)
        new_columns.append(col)
    new_schema = pa.schema(new_fields)
    return pa.table(dict(zip(table.schema.names, new_columns)), schema=new_schema)


def _schema_normalizado(schema):
    """Retorna um schema onde todas as colunas dictionary são convertidas para seu value_type."""
    import pyarrow as pa
    new_fields = []
    for field in schema:
        if pa.types.is_dictionary(field.type):
            new_fields.append(field.with_type(field.type.value_type))
        else:
            new_fields.append(field)
    return pa.schema(new_fields)

# ==============================
# UI
# ==============================
st.title("📊 Painel Analítico — Banco Central (PTAX & SCR)")
st.caption("PTAX via API; SCR via Parquet em bucket S3 público (leitura anônima).")

with st.sidebar:
    st.header("Fonte de dados")
    fonte = st.selectbox("Escolha", ["PTAX (Dólar)", "SCR (Crédito)"])

# ------------------------------
# PTAX
# ------------------------------
if fonte == "PTAX (Dólar)":
    st.subheader("💵 PTAX — Cotação do Dólar")

    with st.sidebar:
        st.divider()
        st.subheader("Parâmetros PTAX")
        hoje = date.today()
        data_ini = st.date_input("Data inicial", value=date(hoje.year, 1, 1))
        data_fim = st.date_input("Data final", value=hoje)
        consultar = st.button("Consultar PTAX")

    if consultar:
        if data_ini > data_fim:
            st.error("A data inicial não pode ser maior que a data final.")
            st.stop()

        with st.spinner("Consultando API do Banco Central..."):
            df = cotacao_dolar_periodo_df(data_ini, data_fim)

        if df.empty:
            st.warning("Nenhuma cotação encontrada para o período.")
            st.stop()

        df_d = dolar_diario(df)
        ultima = df.iloc[-1]

        c1, c2 = st.columns(2)
        c1.metric("Última cotação de compra", f"R$ {ultima['cotacaoCompra']:.4f}")
        c2.metric("Última cotação de venda", f"R$ {ultima['cotacaoVenda']:.4f}")

        st.subheader("📈 Evolução do dólar (média diária)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_d["dia"], y=df_d["cotacaoCompra"], mode="lines", name="Compra"))
        fig.add_trace(go.Scatter(x=df_d["dia"], y=df_d["cotacaoVenda"], mode="lines", name="Venda"))
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Data", yaxis_title="R$")
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Dados completos (intradiário)")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Baixar CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="cotacao_dolar_ptax.csv",
            mime="text/csv",
        )
    else:
        st.info("Selecione o período e clique em **Consultar PTAX**.")

# ------------------------------
# SCR
# ------------------------------
else:
    st.subheader("🏦 SCR — Sistema de Informações de Crédito (Parquet em S3 público)")

    with st.sidebar:
        st.divider()
        st.subheader("Parâmetros SCR")
        ano_sel = st.selectbox("Ano", list(range(2012, 2026)), index=(2025 - 2012))

    with st.spinner(f"Carregando SCR {ano_sel} do S3 público..."):
        try:
            df_scr = carregar_scr_parquet_publico(ano_sel)
        except Exception as e:
            v = versao_por_ano(ano_sel)
            prefix = S3_PREFIX_V1_KEY if v == "v1" else S3_PREFIX_V2_KEY
            st.error(
                "Falha ao carregar parquet do S3 público.\n\n"
                f"Ano selecionado: {ano_sel}\n"
                f"Versão esperada: {v}\n"
                f"Key esperada: {prefix}ano={ano_sel}/scrdata_{ano_sel}.parquet\n\n"
                f"Erro: {e}"
            )
            st.stop()

    if df_scr.empty:
        st.warning("Arquivo carregado, mas o DataFrame veio vazio.")
        st.stop()

    st.write(f"Linhas: **{len(df_scr):,}** | Colunas: **{len(df_scr.columns)}**")

    # Métricas (best-effort)
    col_ativa = pick_first_col(df_scr, ["carteira_ativa", "saldo_carteira_ativa", "carteira_total", "saldo_total"])
    col_inad = pick_first_col(df_scr, ["carteira_inadimplencia", "saldo_inadimplencia", "inadimplencia"])
    col_taxa = pick_first_col(df_scr, ["taxa_inadimplencia", "inadimplencia_pct", "pct_inadimplencia"])

    c1, c2, c3 = st.columns(3)

    if col_ativa:
        c1.metric(
            "Carteira ativa (soma)",
            f"{pd.to_numeric(df_scr[col_ativa], errors='coerce').fillna(0).sum():,.2f}",
        )
    else:
        c1.metric("Carteira ativa (soma)", "N/D")

    if col_inad:
        c2.metric(
            "Carteira inadimplência (soma)",
            f"{pd.to_numeric(df_scr[col_inad], errors='coerce').fillna(0).sum():,.2f}",
        )
    else:
        c2.metric("Carteira inadimplência (soma)", "N/D")

    if col_taxa:
        s = pd.to_numeric(df_scr[col_taxa], errors="coerce").dropna()
        if not s.empty:
            val = float(s.mean())
            taxa_pct = val if val > 1.5 else val * 100
            c3.metric("Taxa inadimplência (média)", f"{taxa_pct:.2f}%")
        else:
            c3.metric("Taxa inadimplência (média)", "N/D")
    else:
        c3.metric("Taxa inadimplência (média)", "N/D")

    st.subheader("📊 Amostra de dados")
    st.dataframe(df_scr.head(500), use_container_width=True)

    st.download_button(
        "⬇️ Baixar amostra (CSV)",
        data=df_scr.head(50000).to_csv(index=False).encode("utf-8"),
        file_name=f"scr_{ano_sel}_amostra.csv",
        mime="text/csv",
    )
