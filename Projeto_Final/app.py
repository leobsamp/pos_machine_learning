import pandas as pd
import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import s3fs
from datetime import date, datetime
import pyarrow.parquet as pq

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Painel BCB — Educação Financeira", layout="wide")

S3_BUCKET = "projeto-bcb-scr-datalake"
S3_REGION = "us-east-2"

V1_YEARS = list(range(2012, 2024))
V2_YEARS = [2024, 2025]

S3_PREFIX_V1_KEY = "scr/processed/versao=v1/"
S3_PREFIX_V2_KEY = "scr/processed/versao=v2/"

PTAX_BASE = "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata"
BCB_SERIES_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados"

# Séries do BCB usadas no painel
BCB_SERIES = {
    "Selic (% a.a.)":       432,
    "IPCA (% a.m.)":        433,
    "Juros Crédito PF (% a.m.)": 20714,
}


# ==============================
# HELPERS GERAIS
# ==============================
def versao_por_ano(ano: int) -> str:
    if ano in V1_YEARS:
        return "v1"
    if ano in V2_YEARS:
        return "v2"
    raise ValueError("Ano fora do intervalo suportado (2012–2025).")


def pick_first_col(df: pd.DataFrame, candidates: list) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _fmt_mmddyyyy(d: date) -> str:
    return d.strftime("%m-%d-%Y")


def _anon_fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": S3_REGION})


# ==============================
# LEITURA S3 (com normalização de schema)
# ==============================
def _cast_dict_columns(table):
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
    import pyarrow as pa
    NUMERIC_RANK = {
        pa.int8(): 0, pa.int16(): 1, pa.int32(): 2, pa.int64(): 3,
        pa.float32(): 4, pa.float64(): 5,
    }
    field_types: dict = {}
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
                if t in NUMERIC_RANK and existing in NUMERIC_RANK:
                    field_types[name] = t if NUMERIC_RANK[t] > NUMERIC_RANK[existing] else existing
                else:
                    field_types[name] = pa.large_utf8()
    return pa.schema([pa.field(name, t) for name, t in field_types.items()])


def _align_to_schema(table, target_schema):
    import pyarrow as pa
    columns = {}
    for field in target_schema:
        if field.name in table.schema.names:
            col = table.column(field.name)
            try:
                col = col.cast(field.type, safe=False)
            except Exception:
                col = col.cast(pa.large_utf8(), safe=False)
            columns[field.name] = col
        else:
            columns[field.name] = pa.nulls(len(table), type=field.type)
    return pa.table(columns, schema=target_schema)


def _normalizar_tipos(df: pd.DataFrame) -> pd.DataFrame:
    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    if "mes" in df.columns:
        df["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")
    if "data_base" in df.columns:
        df["data_base"] = pd.to_datetime(df["data_base"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def carregar_scr_parquet_publico(ano: int) -> pd.DataFrame:
    import pyarrow as pa

    versao = versao_por_ano(ano)
    prefix = S3_PREFIX_V1_KEY if versao == "v1" else S3_PREFIX_V2_KEY
    fs = _anon_fs()
    file_path = f"s3://{S3_BUCKET}/{prefix}ano={ano}/scrdata_{ano}.parquet"

    paths = []
    try:
        base_prefix = f"{S3_BUCKET}/{prefix}ano={ano}/scrdata_{ano}.parquet"
        paths = [p for p in fs.ls(base_prefix) if p.endswith(".parquet")]
    except Exception:
        paths = []

    if not paths:
        paths = [file_path.replace("s3://", "")]

    tables = []
    for p in paths:
        with fs.open(p, "rb") as f:
            pf = pq.ParquetFile(f)
            for i in range(pf.num_row_groups):
                rg = pf.read_row_group(i)
                rg = _cast_dict_columns(rg)
                tables.append(rg)

    if not tables:
        return pd.DataFrame()

    unified_schema = _unify_schemas([t.schema for t in tables])
    tables = [_align_to_schema(t, unified_schema) for t in tables]
    df = pa.concat_tables(tables).to_pandas()
    return _normalizar_tipos(df)


# ==============================
# PTAX
# ==============================
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
    df = pd.DataFrame(r.json().get("value", []))
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


# ==============================
# SÉRIES BCB (Selic, IPCA, Juros)
# ==============================
@st.cache_data(ttl=3600)
def carregar_serie_bcb(serie: int, data_ini: str, data_fim: str) -> pd.DataFrame:
    """
    Consulta série temporal do BCB via API SGS.
    data_ini / data_fim no formato DD/MM/YYYY.
    """
    url = BCB_SERIES_BASE.format(serie=serie)
    params = {
        "formato": "json",
        "dataInicial": data_ini,
        "dataFinal": data_fim,
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        if df.empty:
            return df
        df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        return df.sort_values("data").reset_index(drop=True)
    except Exception as e:
        st.warning(f"Não foi possível carregar a série {serie}: {e}")
        return pd.DataFrame()


def _periodo_para_sgsdates(data_ini: date, data_fim: date):
    return data_ini.strftime("%d/%m/%Y"), data_fim.strftime("%d/%m/%Y")


# ==============================
# AGREGAÇÕES SCR
# ==============================
def agregar_inadimplencia_por_ano(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna taxa média de inadimplência anual."""
    col_taxa = pick_first_col(df, ["taxa_inadimplencia", "inadimplencia_pct", "pct_inadimplencia"])
    col_ativa = pick_first_col(df, ["carteira_ativa"])
    col_inad = pick_first_col(df, ["carteira_inadimplencia", "saldo_inadimplencia", "inadimplencia"])

    if "ano" not in df.columns:
        return pd.DataFrame()

    if col_taxa:
        agg = (
            df.groupby("ano")[col_taxa]
            .mean()
            .reset_index()
            .rename(columns={col_taxa: "taxa_inadimplencia_media"})
        )
        agg["taxa_inadimplencia_media"] = pd.to_numeric(agg["taxa_inadimplencia_media"], errors="coerce")
        # Normaliza escala (0-1 vs 0-100)
        if agg["taxa_inadimplencia_media"].dropna().max() <= 1.5:
            agg["taxa_inadimplencia_media"] *= 100
        return agg

    if col_inad and col_ativa:
        grp = df.groupby("ano").agg(
            inad=(col_inad, "sum"),
            ativa=(col_ativa, "sum"),
        ).reset_index()
        grp["taxa_inadimplencia_media"] = (grp["inad"] / grp["ativa"].replace(0, np.nan)) * 100
        return grp[["ano", "taxa_inadimplencia_media"]]

    return pd.DataFrame()


def agregar_carteira_por_modalidade(df: pd.DataFrame) -> pd.DataFrame:
    col_ativa = pick_first_col(df, ["carteira_ativa", "saldo_carteira_ativa", "carteira_total"])
    col_mod = pick_first_col(df, ["modalidade", "submodalidade"])
    if not col_ativa or not col_mod:
        return pd.DataFrame()
    grp = (
        df.groupby(col_mod)[col_ativa]
        .sum()
        .reset_index()
        .rename(columns={col_mod: "modalidade", col_ativa: "carteira_ativa"})
        .sort_values("carteira_ativa", ascending=False)
        .head(15)
    )
    grp["carteira_ativa"] = pd.to_numeric(grp["carteira_ativa"], errors="coerce")
    return grp


def agregar_inadimplencia_por_uf(df: pd.DataFrame) -> pd.DataFrame:
    col_taxa = pick_first_col(df, ["taxa_inadimplencia", "inadimplencia_pct", "pct_inadimplencia"])
    col_ativa = pick_first_col(df, ["carteira_ativa"])
    col_inad = pick_first_col(df, ["carteira_inadimplencia", "saldo_inadimplencia", "inadimplencia"])

    if "uf" not in df.columns:
        return pd.DataFrame()

    if col_taxa:
        grp = (
            df.groupby("uf")[col_taxa]
            .mean()
            .reset_index()
            .rename(columns={col_taxa: "taxa_inadimplencia"})
            .sort_values("taxa_inadimplencia", ascending=False)
        )
        grp["taxa_inadimplencia"] = pd.to_numeric(grp["taxa_inadimplencia"], errors="coerce")
        if grp["taxa_inadimplencia"].dropna().max() <= 1.5:
            grp["taxa_inadimplencia"] *= 100
        return grp

    if col_inad and col_ativa:
        grp = df.groupby("uf").agg(
            inad=(col_inad, "sum"),
            ativa=(col_ativa, "sum"),
        ).reset_index()
        grp["taxa_inadimplencia"] = (grp["inad"] / grp["ativa"].replace(0, np.nan)) * 100
        return grp[["uf", "taxa_inadimplencia"]].sort_values("taxa_inadimplencia", ascending=False)

    return pd.DataFrame()


# ==============================
# UI
# ==============================
st.title("📊 Painel Analítico — Banco Central do Brasil")
st.caption("Dados abertos do BCB: PTAX, SCR, Selic, IPCA e Juros de Crédito PF.")

with st.sidebar:
    st.header("🔎 Navegação")
    fonte = st.selectbox(
        "Escolha a seção",
        [
            "PTAX (Dólar)",
            "SCR — Indicadores de Crédito",
            "Índices Macroeconômicos",
            "Correlações entre Indicadores",
        ],
    )

# ==============================================
# 1. PTAX
# ==============================================
if fonte == "PTAX (Dólar)":
    st.subheader("💵 PTAX — Cotação do Dólar Comercial")

    with st.sidebar:
        st.divider()
        st.subheader("Parâmetros")
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

        c1, c2, c3 = st.columns(3)
        c1.metric("Última compra", f"R$ {ultima['cotacaoCompra']:.4f}")
        c2.metric("Última venda", f"R$ {ultima['cotacaoVenda']:.4f}")
        spread = ultima["cotacaoVenda"] - ultima["cotacaoCompra"]
        c3.metric("Spread (venda - compra)", f"R$ {spread:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_d["dia"], y=df_d["cotacaoCompra"], mode="lines", name="Compra", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=df_d["dia"], y=df_d["cotacaoVenda"], mode="lines", name="Venda", line=dict(color="#ff7f0e")))
        fig.update_layout(height=460, xaxis_title="Data", yaxis_title="R$", hovermode="x unified")
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Dados intradiários")
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Baixar CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="cotacao_dolar_ptax.csv", mime="text/csv")
    else:
        st.info("Selecione o período e clique em **Consultar PTAX**.")


# ==============================================
# 2. SCR — INDICADORES DE CRÉDITO
# ==============================================
elif fonte == "SCR — Indicadores de Crédito":
    st.subheader("🏦 SCR — Sistema de Informações de Crédito")

    with st.sidebar:
        st.divider()
        st.subheader("Parâmetros")
        ano_sel = st.selectbox("Ano base", list(range(2012, 2026)), index=(2025 - 2012))

    with st.spinner(f"Carregando SCR {ano_sel}..."):
        try:
            df_scr = carregar_scr_parquet_publico(ano_sel)
        except Exception as e:
            v = versao_por_ano(ano_sel)
            prefix = S3_PREFIX_V1_KEY if v == "v1" else S3_PREFIX_V2_KEY
            st.error(
                f"Falha ao carregar parquet do S3.\n\n"
                f"Ano: {ano_sel} | Versão: {v}\n"
                f"Key: {prefix}ano={ano_sel}/scrdata_{ano_sel}.parquet\n\nErro: {e}"
            )
            st.stop()

    if df_scr.empty:
        st.warning("Arquivo carregado, mas o DataFrame veio vazio.")
        st.stop()

    st.caption(f"Linhas carregadas: **{len(df_scr):,}** | Colunas: **{len(df_scr.columns)}**")

    # ---- Métricas resumo ----
    col_ativa = pick_first_col(df_scr, ["carteira_ativa", "saldo_carteira_ativa", "carteira_total", "saldo_total"])
    col_inad  = pick_first_col(df_scr, ["carteira_inadimplencia", "saldo_inadimplencia", "inadimplencia"])
    col_taxa  = pick_first_col(df_scr, ["taxa_inadimplencia", "inadimplencia_pct", "pct_inadimplencia"])

    c1, c2, c3 = st.columns(3)
    if col_ativa:
        total_ativa = pd.to_numeric(df_scr[col_ativa], errors="coerce").fillna(0).sum()
        c1.metric("Carteira Ativa Total (R$)", f"{total_ativa:,.0f}")
    else:
        c1.metric("Carteira Ativa Total", "N/D")

    if col_inad:
        total_inad = pd.to_numeric(df_scr[col_inad], errors="coerce").fillna(0).sum()
        c2.metric("Carteira Inadimplente (R$)", f"{total_inad:,.0f}")
    else:
        c2.metric("Carteira Inadimplente", "N/D")

    if col_taxa:
        s = pd.to_numeric(df_scr[col_taxa], errors="coerce").dropna()
        if not s.empty:
            val = float(s.mean())
            taxa_pct = val if val > 1.5 else val * 100
            c3.metric("Taxa de Inadimplência Média", f"{taxa_pct:.2f}%")
    else:
        c3.metric("Taxa de Inadimplência Média", "N/D")

    st.divider()

    # ---- Aba 1: Série temporal de inadimplência ----
    st.subheader("📈 Série Temporal de Inadimplência por Ano")
    st.caption("Para construir a série completa, selecione múltiplos anos abaixo.")

    with st.expander("⚙️ Configurar anos para a série temporal", expanded=True):
        anos_disponiveis = list(range(2012, 2026))
        anos_serie = st.multiselect(
            "Anos a incluir na série",
            options=anos_disponiveis,
            default=[ano_sel],
        )

    if anos_serie:
        progresso = st.progress(0, text="Carregando dados para a série temporal...")
        registros = []
        for i, ano in enumerate(sorted(anos_serie)):
            progresso.progress((i + 1) / len(anos_serie), text=f"Carregando {ano}...")
            try:
                df_ano = carregar_scr_parquet_publico(ano)
                agg = agregar_inadimplencia_por_ano(df_ano)
                if not agg.empty:
                    agg["ano_ref"] = ano
                    registros.append(agg)
            except Exception:
                pass
        progresso.empty()

        if registros:
            df_serie = pd.concat(registros, ignore_index=True)
            df_serie = df_serie.sort_values("ano")

            fig_serie = go.Figure()
            fig_serie.add_trace(go.Scatter(
                x=df_serie["ano"].astype(str),
                y=df_serie["taxa_inadimplencia_media"],
                mode="lines+markers",
                name="Taxa de Inadimplência (%)",
                line=dict(color="#e74c3c", width=2),
                marker=dict(size=8),
            ))
            fig_serie.update_layout(
                height=400,
                xaxis_title="Ano",
                yaxis_title="Taxa de Inadimplência (%)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_serie, use_container_width=True)
        else:
            st.info("Não foi possível calcular a inadimplência para os anos selecionados.")

    st.divider()

    # ---- Aba 2: Ranking de UF ----
    st.subheader("🗺️ Ranking de UF por Inadimplência")
    df_uf = agregar_inadimplencia_por_uf(df_scr)

    if not df_uf.empty:
        df_uf = df_uf.dropna(subset=["taxa_inadimplencia"]).sort_values("taxa_inadimplencia", ascending=True)

        fig_uf = go.Figure(go.Bar(
            x=df_uf["taxa_inadimplencia"],
            y=df_uf["uf"],
            orientation="h",
            marker_color=df_uf["taxa_inadimplencia"],
            marker_colorscale="RdYlGn_r",
            text=df_uf["taxa_inadimplencia"].map(lambda v: f"{v:.2f}%"),
            textposition="outside",
        ))
        fig_uf.update_layout(
            height=max(400, len(df_uf) * 22),
            xaxis_title="Taxa de Inadimplência (%)",
            yaxis_title="UF",
            margin=dict(l=60, r=60, t=30, b=40),
        )
        st.plotly_chart(fig_uf, use_container_width=True)
    else:
        st.info("Coluna 'uf' não encontrada nos dados deste ano.")

    st.divider()

    # ---- Amostra ----
    with st.expander("📋 Amostra dos dados brutos"):
        st.dataframe(df_scr.head(500), use_container_width=True)
        st.download_button(
            "⬇️ Baixar amostra (CSV)",
            data=df_scr.head(50000).to_csv(index=False).encode("utf-8"),
            file_name=f"scr_{ano_sel}_amostra.csv",
            mime="text/csv",
        )


# ==============================================
# 3. ÍNDICES MACROECONÔMICOS
# ==============================================
elif fonte == "Índices Macroeconômicos":
    st.subheader("📉 Índices Macroeconômicos — BCB")
    st.caption("Séries históricas de Selic, IPCA e Taxa Média de Juros para Crédito PF.")

    with st.sidebar:
        st.divider()
        st.subheader("Parâmetros")
        hoje = date.today()
        data_ini_macro = st.date_input("Data inicial", value=date(hoje.year - 5, 1, 1), key="macro_ini")
        data_fim_macro = st.date_input("Data final", value=hoje, key="macro_fim")
        indices_sel = st.multiselect(
            "Índices a exibir",
            options=list(BCB_SERIES.keys()),
            default=list(BCB_SERIES.keys()),
        )
        buscar = st.button("Buscar índices")

    if buscar:
        if not indices_sel:
            st.warning("Selecione ao menos um índice.")
            st.stop()

        ini_str, fim_str = _periodo_para_sgsdates(data_ini_macro, data_fim_macro)

        dados_macros: dict = {}
        with st.spinner("Consultando API do Banco Central..."):
            for nome in indices_sel:
                serie_id = BCB_SERIES[nome]
                df_s = carregar_serie_bcb(serie_id, ini_str, fim_str)
                if not df_s.empty:
                    dados_macros[nome] = df_s

        if not dados_macros:
            st.error("Não foi possível carregar nenhum índice.")
            st.stop()

        # Um gráfico por índice (escala independente)
        for nome, df_idx in dados_macros.items():
            st.subheader(f"📊 {nome}")
            media = df_idx["valor"].mean()
            minimo = df_idx["valor"].min()
            maximo = df_idx["valor"].max()

            col1, col2, col3 = st.columns(3)
            col1.metric("Média do período", f"{media:.2f}")
            col2.metric("Mínimo", f"{minimo:.2f}")
            col3.metric("Máximo", f"{maximo:.2f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_idx["data"],
                y=df_idx["valor"],
                mode="lines",
                name=nome,
                fill="tozeroy",
                fillcolor="rgba(31,119,180,0.1)",
                line=dict(color="#1f77b4", width=2),
            ))
            fig.update_layout(
                height=350,
                xaxis_title="Data",
                yaxis_title=nome,
                hovermode="x unified",
                margin=dict(l=10, r=10, t=20, b=40),
            )
            fig.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Ver dados tabulares"):
                st.dataframe(df_idx, use_container_width=True)
                st.download_button(
                    f"⬇️ Baixar {nome} (CSV)",
                    data=df_idx.to_csv(index=False).encode("utf-8"),
                    file_name=f"{nome.replace(' ', '_').replace('/', '')}.csv",
                    mime="text/csv",
                    key=f"dl_{nome}",
                )
    else:
        st.info("Configure o período e os índices desejados, depois clique em **Buscar índices**.")


# ==============================================
# 4. CORRELAÇÕES ENTRE INDICADORES
# ==============================================
elif fonte == "Correlações entre Indicadores":
    st.subheader("🔗 Correlações entre Indicadores do BCB")
    st.caption(
        "Comparação visual entre séries macroeconômicas (Selic, IPCA, Juros PF) e "
        "a taxa de inadimplência do SCR ao longo do tempo, usando gráfico de linhas com eixo duplo."
    )

    with st.sidebar:
        st.divider()
        st.subheader("Parâmetros")
        hoje = date.today()
        data_ini_corr = st.date_input("Data inicial", value=date(hoje.year - 5, 1, 1), key="corr_ini")
        data_fim_corr = st.date_input("Data final", value=hoje, key="corr_fim")

        indice_eixo1 = st.selectbox(
            "Indicador — Eixo Esquerdo",
            options=list(BCB_SERIES.keys()),
            index=0,
        )
        indice_eixo2 = st.selectbox(
            "Indicador — Eixo Direito",
            options=list(BCB_SERIES.keys()) + ["Inadimplência SCR (%)"],
            index=len(BCB_SERIES),
        )

        if "Inadimplência SCR (%)" in [indice_eixo1, indice_eixo2]:
            anos_inad = st.multiselect(
                "Anos do SCR para inadimplência",
                options=list(range(2012, 2026)),
                default=list(range(max(2018, data_ini_corr.year), min(2026, data_fim_corr.year + 1))),
            )

        calcular = st.button("Calcular correlações")

    if calcular:
        ini_str, fim_str = _periodo_para_sgsdates(data_ini_corr, data_fim_corr)

        # Carrega eixo 1
        serie_e1 = None
        if indice_eixo1 in BCB_SERIES:
            with st.spinner(f"Carregando {indice_eixo1}..."):
                df_e1 = carregar_serie_bcb(BCB_SERIES[indice_eixo1], ini_str, fim_str)
            if not df_e1.empty:
                serie_e1 = df_e1.rename(columns={"valor": indice_eixo1}).set_index("data")[indice_eixo1]

        # Carrega eixo 2
        serie_e2 = None
        if indice_eixo2 in BCB_SERIES:
            with st.spinner(f"Carregando {indice_eixo2}..."):
                df_e2 = carregar_serie_bcb(BCB_SERIES[indice_eixo2], ini_str, fim_str)
            if not df_e2.empty:
                serie_e2 = df_e2.rename(columns={"valor": indice_eixo2}).set_index("data")[indice_eixo2]

        elif indice_eixo2 == "Inadimplência SCR (%)":
            with st.spinner("Calculando inadimplência SCR por ano..."):
                registros_inad = []
                for ano in sorted(anos_inad):
                    try:
                        df_ano = carregar_scr_parquet_publico(ano)
                        agg = agregar_inadimplencia_por_ano(df_ano)
                        if not agg.empty:
                            # Usa o meio do ano como referência temporal
                            agg["data"] = pd.to_datetime(agg["ano"].astype(str) + "-06-30")
                            registros_inad.append(agg[["data", "taxa_inadimplencia_media"]])
                    except Exception:
                        pass

            if registros_inad:
                df_inad_serie = pd.concat(registros_inad).sort_values("data")
                serie_e2 = df_inad_serie.set_index("data")["taxa_inadimplencia_media"]
                serie_e2.name = "Inadimplência SCR (%)"

        if serie_e1 is None and serie_e2 is None:
            st.error("Não foi possível carregar nenhuma série.")
            st.stop()

        # ---- Gráfico de linhas com eixo duplo ----
        fig_corr = make_subplots(specs=[[{"secondary_y": True}]])

        if serie_e1 is not None:
            fig_corr.add_trace(
                go.Scatter(
                    x=serie_e1.index,
                    y=serie_e1.values,
                    mode="lines",
                    name=indice_eixo1,
                    line=dict(color="#1f77b4", width=2),
                ),
                secondary_y=False,
            )

        if serie_e2 is not None:
            fig_corr.add_trace(
                go.Scatter(
                    x=serie_e2.index,
                    y=serie_e2.values,
                    mode="lines+markers" if indice_eixo2 == "Inadimplência SCR (%)" else "lines",
                    name=indice_eixo2,
                    line=dict(color="#e74c3c", width=2, dash="dot"),
                    marker=dict(size=7),
                ),
                secondary_y=True,
            )

        fig_corr.update_layout(
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=50, b=40),
        )
        fig_corr.update_xaxes(title_text="Data")
        fig_corr.update_yaxes(title_text=indice_eixo1 if serie_e1 is not None else "", secondary_y=False)
        fig_corr.update_yaxes(title_text=indice_eixo2 if serie_e2 is not None else "", secondary_y=True)

        st.plotly_chart(fig_corr, use_container_width=True)

        # ---- Correlação numérica ----
        if serie_e1 is not None and serie_e2 is not None:
            st.divider()
            st.subheader("📐 Correlação de Pearson")

            # Alinha as séries pela data (resample mensal)
            df_merged = pd.merge(
                serie_e1.resample("ME").mean().reset_index().rename(columns={serie_e1.name: "e1"}),
                serie_e2.resample("ME").mean().reset_index().rename(columns={serie_e2.name: "e2"}),
                on="data",
                how="inner",
            ).dropna()

            if len(df_merged) >= 3:
                corr = df_merged["e1"].corr(df_merged["e2"])
                cor_abs = abs(corr)
                interpretacao = (
                    "correlação forte" if cor_abs >= 0.7
                    else "correlação moderada" if cor_abs >= 0.4
                    else "correlação fraca"
                )
                direcao = "positiva" if corr >= 0 else "negativa"

                col_a, col_b = st.columns([1, 3])
                col_a.metric(
                    "Coeficiente de Pearson (r)",
                    f"{corr:.4f}",
                    help="Varia de -1 (correlação negativa perfeita) a +1 (correlação positiva perfeita).",
                )
                col_b.info(
                    f"Os dois indicadores apresentam **{interpretacao} {direcao}** "
                    f"(r = {corr:.4f}), calculada sobre {len(df_merged)} observações mensais comuns."
                )

                # Scatter plot auxiliar
                fig_scatter = px.scatter(
                    df_merged,
                    x="e1",
                    y="e2",
                    labels={"e1": indice_eixo1, "e2": indice_eixo2},
                    trendline="ols",
                    title=f"Dispersão: {indice_eixo1} × {indice_eixo2}",
                    color_discrete_sequence=["#1f77b4"],
                )
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("Período de sobreposição entre as séries muito curto para calcular correlação.")

    else:
        st.info("Configure os indicadores e o período, depois clique em **Calcular correlações**.")