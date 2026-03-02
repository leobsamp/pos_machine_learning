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
    import pyarrow as pa

    versao = versao_por_ano(ano)
    prefix = S3_PREFIX_V1_KEY if versao == "v1" else S3_PREFIX_V2_KEY
    fs = _anon_fs()
    file_path = f"s3://{S3_BUCKET}/{prefix}ano={ano}/scrdata_{ano}.parquet"

    # Tenta listar se for "diretório" de parquets
    paths = []
    try:
        base_prefix = f"{S3_BUCKET}/{prefix}ano={ano}/scrdata_{ano}.parquet"
        paths = [p for p in fs.ls(base_prefix) if p.endswith(".parquet")]
    except Exception:
        paths = []

    if not paths:
        paths = [file_path.replace("s3://", "")]  # remove prefixo para uso com fs.open

    tables = []
    for p in paths:
        s3_path = p if p.startswith(S3_BUCKET) else p
        with fs.open(s3_path, "rb") as f:
            pf = pq.ParquetFile(f)
            for i in range(pf.num_row_groups):
                rg = pf.read_row_group(i)
                rg = _cast_dict_columns(rg)
                tables.append(rg)

    if not tables:
        return pd.DataFrame()

    # Unifica schemas antes de concatenar
    schemas = [t.schema for t in tables]
    unified_schema = _unify_schemas(schemas)
    tables = [_align_to_schema(t, unified_schema) for t in tables]

    df = pa.concat_tables(tables).to_pandas()
    return _normalizar_tipos(df)


def _cast_dict_columns(table):
    """Converte todas as colunas dictionary para o tipo base."""
    import pyarrow as pa
    new_columns = {}
    new_fields = []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        if pa.types.is_dictionary(field.type):
            target_type = field.type.value_type
            col = col.cast(target_type)
            field = field.with_type(target_type)
        new_columns[field.name] = col
        new_fields.append(field)
    return pa.table(new_columns, schema=pa.schema(new_fields))


def _unify_schemas(schemas):
    """
    Constrói um schema unificado: para cada campo, escolhe o tipo
    mais 'largo' encontrado entre os schemas (int32 < int64 < float64).
    Strings e tipos incompatíveis são resolvidos como string (large_utf8).
    """
    import pyarrow as pa

    NUMERIC_RANK = {
        pa.int8(): 0, pa.int16(): 1, pa.int32(): 2, pa.int64(): 3,
        pa.float32(): 4, pa.float64(): 5,
    }

    field_types: dict[str, pa.DataType] = {}

    for schema in schemas:
        for field in schema:
            name = field.name
            t = field.type
            if name not in field_types:
                field_types[name] = t
            else:
                existing = field_types[name]
                if existing == t:
                    continue
                # Ambos numéricos: promove para o maior
                if t in NUMERIC_RANK and existing in NUMERIC_RANK:
                    field_types[name] = t if NUMERIC_RANK[t] > NUMERIC_RANK[existing] else existing
                else:
                    # Fallback: usa string para tipos incompatíveis
                    field_types[name] = pa.large_utf8()

    return pa.schema([pa.field(name, t) for name, t in field_types.items()])


def _align_to_schema(table, target_schema):
    """
    Alinha uma tabela ao schema alvo: faz cast de colunas existentes,
    adiciona colunas ausentes como null, ignora colunas extras.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    columns = {}
    for field in target_schema:
        if field.name in table.schema.names:
            col = table.column(field.name)
            try:
                col = col.cast(field.type, safe=False)
            except Exception:
                # Cast impossível: converte para string
                col = col.cast(pa.large_utf8(), safe=False)
            columns[field.name] = col
        else:
            # Coluna ausente: preenche com nulls
            columns[field.name] = pa.nulls(len(table), type=field.type)

    return pa.table(columns, schema=target_schema)


def _normalizar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos problemáticos conhecidos após conversão para pandas."""
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    if "mes" in df.columns:
        df["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")
    if "data_base" in df.columns:
        df["data_base"] = pd.to_datetime(df["data_base"], errors="coerce")
    return df

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
