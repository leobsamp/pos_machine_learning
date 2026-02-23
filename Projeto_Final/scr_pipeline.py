from __future__ import annotations

import requests
from datetime import date
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple


# ============================================================
# Sprint 3 — Padronização de Indicadores (Normalização V1 x V2)
# ============================================================

CORE_SCHEMA_COLS: List[str] = [
    "data_base",
    "ano",
    "mes",
    "uf",
    "cliente",
    "modalidade",
    "submodalidade",
    "segmento",
    "porte",
    # medidas padronizadas
    "carteira_ativa",
    "carteira_vencida",
    "carteira_inadimplencia",
    "ativo_problematico",
    # compatibilidade temporal
    "versao_scr",
]

# Colunas auxiliares (podem existir em uma versão e não em outra)
AUX_COLS: List[str] = [
    "taxa_inadimplencia",
    "perc_carteira_vencida",
    "taxa_ativo_problematico",
]

# Aliases por versão -> padrão único
# (Ajuste conforme observar nomes reais nos arquivos V1/V2)
SCHEMA_MAP: Dict[str, Dict[str, str]] = {
    "v1": {
        # carteira ativa
        "saldo_carteira_ativa": "carteira_ativa",
        "carteira_total": "carteira_ativa",
        "saldo_total": "carteira_ativa",
        # inadimplência (valor)
        "saldo_inadimplencia": "carteira_inadimplencia",
        "inadimplencia": "carteira_inadimplencia",
        # inadimplência (taxa)
        "pct_inadimplencia": "taxa_inadimplencia",
        "inadimplencia_pct": "taxa_inadimplencia",
        # vencida (taxa)
        "pct_carteira_vencida": "perc_carteira_vencida",
        # ativo problemático (valor/taxa)
        "ativo_prob": "ativo_problematico",
        "percentual_ativo_problematico": "taxa_ativo_problematico",
        "taxa_ativo_problematico": "taxa_ativo_problematico",
    },
    "v2": {
        "saldo_carteira_ativa": "carteira_ativa",
        "saldo_inadimplencia": "carteira_inadimplencia",
        "taxa_inadimplencia": "taxa_inadimplencia",
        "ativo_problematico": "ativo_problematico",
        "taxa_ativo_problematico": "taxa_ativo_problematico",
    },
}


def normalizar_schema_scr(df: pd.DataFrame, versao: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Normaliza schema V1/V2 para um padrão único.

    Objetivos:
      - Mapear nomes diferentes para um padrão único (via SCHEMA_MAP)
      - Criar colunas derivadas quando inexistentes (ex.: carteira_vencida)
      - Garantir que o dataset final tenha sempre as mesmas colunas (CORE_SCHEMA_COLS + AUX_COLS)
      - Incluir flag de versão (versao_scr) para compatibilidade temporal

    Retorna:
      (df_normalizado, alertas)
    """
    d = df.copy()
    alertas: List[str] = []

    v = (versao or "").lower().strip()
    if v not in ("v1", "v2"):
        v = "desconhecida"
        alertas.append("Versão do SCR não informada/invalidada; usando 'desconhecida'.")

    # 1) Rename por aliases
    if v in SCHEMA_MAP:
        rename_map = {c: SCHEMA_MAP[v][c] for c in d.columns if c in SCHEMA_MAP[v]}
        if rename_map:
            d = d.rename(columns=rename_map)

    # 2) Flag de versão
    d["versao_scr"] = v

    # 3) Garantir colunas CORE/AUX sempre presentes
    for col in CORE_SCHEMA_COLS + AUX_COLS:
        if col not in d.columns:
            d[col] = np.nan

    # 4) Derivações quando necessário/possível
    # 4.1) carteira_vencida
    if d["carteira_vencida"].isna().all():
        # Tenta derivar via % * carteira_ativa
        if (not d["perc_carteira_vencida"].isna().all()) and (not d["carteira_ativa"].isna().all()):
            d["carteira_vencida"] = d["perc_carteira_vencida"] * d["carteira_ativa"]
            alertas.append("Carteira vencida derivada via perc_carteira_vencida * carteira_ativa.")
        # Fallback: proxy por inadimplência (valor)
        elif not d["carteira_inadimplencia"].isna().all():
            d["carteira_vencida"] = d["carteira_inadimplencia"]
            alertas.append("Carteira vencida ausente; usando carteira_inadimplencia como proxy (atenção à interpretação).")
        else:
            alertas.append("Carteira vencida não disponível e não foi possível derivar.")

    # 4.2) carteira_inadimplencia (valor) via taxa_inadimplencia * carteira_ativa
    if d["carteira_inadimplencia"].isna().all():
        if (not d["taxa_inadimplencia"].isna().all()) and (not d["carteira_ativa"].isna().all()):
            d["carteira_inadimplencia"] = d["taxa_inadimplencia"] * d["carteira_ativa"]
            alertas.append("Carteira inadimplência derivada via taxa_inadimplencia * carteira_ativa.")
        else:
            alertas.append("Carteira inadimplência não disponível e não foi possível derivar.")

    # 4.3) ativo_problematico (valor) via taxa_ativo_problematico * carteira_ativa
    if d["ativo_problematico"].isna().all():
        if (not d["taxa_ativo_problematico"].isna().all()) and (not d["carteira_ativa"].isna().all()):
            d["ativo_problematico"] = d["taxa_ativo_problematico"] * d["carteira_ativa"]
            alertas.append("Ativo problemático derivado via taxa_ativo_problematico * carteira_ativa.")

    # 5) Sanitização de tipos numéricos (medidas)
    for col in ["carteira_ativa", "carteira_vencida", "carteira_inadimplencia", "ativo_problematico"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    return d, alertas


def criar_indicadores_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    Redefine o conjunto de indicadores "core" (comparáveis entre V1 e V2),
    calculando taxas de forma padronizada quando possível.

    Core:
      - Carteira Ativa Total (soma padronizada é feita na camada de agregação; aqui é coluna base)
      - Carteira Vencida (mantém/deriva na normalização)
      - Taxa de Inadimplência (calcula se possível)
      - Percentual de Carteira Problemática (calcula se possível)
    """
    d = df.copy()

    if {"carteira_inadimplencia", "carteira_ativa"}.issubset(d.columns):
        d["taxa_inadimplencia"] = (d["carteira_inadimplencia"] / d["carteira_ativa"]).replace([np.inf, -np.inf], np.nan)

    if {"carteira_vencida", "carteira_ativa"}.issubset(d.columns):
        d["perc_carteira_vencida"] = (d["carteira_vencida"] / d["carteira_ativa"]).replace([np.inf, -np.inf], np.nan)

    if {"ativo_problematico", "carteira_ativa"}.issubset(d.columns):
        d["taxa_ativo_problematico"] = (d["ativo_problematico"] / d["carteira_ativa"]).replace([np.inf, -np.inf], np.nan)

    if {"ativo_problematico", "carteira_ativa"}.issubset(d.columns):
        d["perc_carteira_problematica"] = (d["ativo_problematico"] / d["carteira_ativa"]).replace([np.inf, -np.inf], np.nan)

    return d


def validar_core_schema(df: pd.DataFrame) -> List[str]:
    """Valida presença do schema core e retorna alertas."""
    faltantes = [c for c in CORE_SCHEMA_COLS if c not in df.columns]
    alertas: List[str] = []
    if faltantes:
        alertas.append(f"Colunas CORE ausentes após normalização: {faltantes}")
    return alertas


def processar_scrdata(
    df: pd.DataFrame,
    versao: str = "desconhecida",
    remover_zeros: bool = True,
    criar_indicadores: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Processa automaticamente dados do SCR.data para análise e visualização.

    Etapas:
    - padroniza nomes das colunas
    - converte datas
    - converte valores monetários
    - cria colunas temporais
    - normaliza schema (V1/V2) e injeta flag de versão
    - cria indicadores core (comparáveis)
    - valida consistência
    """

    df = df.copy()

    if verbose:
        print("🔄 Iniciando processamento do SCR.data...")

    # ------------------------------------------------
    # 1. Padronizar nomes das colunas
    # ------------------------------------------------
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("ç", "c")
        .str.replace("ã", "a")
        .str.replace("á", "a")
        .str.replace("é", "e")
        .str.replace("í", "i")
        .str.replace("ó", "o")
        .str.replace("ú", "u")
    )

    # ------------------------------------------------
    # 2. Converter data
    # ------------------------------------------------
    if "data_base" in df.columns:
        df["data_base"] = pd.to_datetime(df["data_base"], errors="coerce")
        df["ano"] = df["data_base"].dt.year
        df["mes"] = df["data_base"].dt.month

    # ------------------------------------------------
    # 3. Converter valores monetários
    # Detecta automaticamente colunas numéricas
    # ------------------------------------------------
    colunas_texto = [
        "uf", "segmento", "cliente", "cnae_ocupacao",
        "porte", "modalidade", "submodalidade",
        "origem", "indexador"
    ]

    colunas_para_converter = [
        c for c in df.columns
        if c not in colunas_texto
        and c not in ["data_base", "ano", "mes", "__arquivo_origem"]
    ]

    for c in colunas_para_converter:
        if df[c].dtype == "object":
            try:
                df[c] = (
                    df[c]
                    .str.replace(".", "", regex=False)
                    .str.replace(",", ".", regex=False)
                )
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                pass

    # ------------------------------------------------
    # 4. Normalizar schema (V1/V2) + flag de versão
    # ------------------------------------------------
    df, alertas_schema = normalizar_schema_scr(df, versao=versao)
    df.attrs["alertas_schema"] = alertas_schema

    # ------------------------------------------------
    # 5. Criar indicadores core padronizados
    # ------------------------------------------------
    if criar_indicadores:
        df = criar_indicadores_core(df)

    # ------------------------------------------------
    # 6. Remover registros sem valor analítico
    # ------------------------------------------------
    if remover_zeros and "carteira_ativa" in df.columns:
        antes = len(df)
        df = df[df["carteira_ativa"] > 0]
        depois = len(df)

        if verbose:
            print(f"🧹 Removidas {antes - depois:,} linhas com carteira zerada.")

    # ------------------------------------------------
    # 7. Checagem de consistência
    # ------------------------------------------------
    if {"carteira_ativa", "carteira_vencida"}.issubset(df.columns):
        inconsistentes = df[df["carteira_vencida"] > df["carteira_ativa"]]
        if verbose and len(inconsistentes) > 0:
            print(f"⚠️ {len(inconsistentes)} registros inconsistentes encontrados (vencida > ativa).")

    # ------------------------------------------------
    # 8. Ordenar dataset
    # ------------------------------------------------
    if "data_base" in df.columns:
        df = df.sort_values("data_base")

    # ------------------------------------------------
    # 9. Reset index
    # ------------------------------------------------
    df = df.reset_index(drop=True)

    # ------------------------------------------------
    # 10. Validação final + alertas de compatibilidade
    # ------------------------------------------------
    alertas_final: List[str] = []
    alertas_final += df.attrs.get("alertas_schema", [])
    alertas_final += validar_core_schema(df)

    df.attrs["alertas_final"] = alertas_final

    if verbose:
        print("✅ SCR.data processado com sucesso!")
        print(f"Linhas: {len(df):,}")
        print(f"Colunas: {len(df.columns)}")
        if alertas_final:
            print("⚠️ Alertas de compatibilidade:")
            for a in alertas_final:
                print(f" - {a}")

    return df


def pipeline_scrdata(
    ano: int,
    base_dir: Union[str, Path] = "data/scrdata",
    forcar_download: bool = False,
    encoding: str = "utf-8",
    sep: Optional[str] = None,
    salvar_parquet: bool = True,
    salvar_csv: bool = False,
    adicionar_coluna_origem: bool = True,
    remover_zeros: bool = True,
    criar_indicadores: bool = True,
    versao_scr: str = "desconhecida",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline completo do SCR.data:
    - baixa ZIP por ano (com cache)
    - extrai (com cache)
    - concatena todos os CSVs mensais
    - processa (tipos, datas, normalização de schema, indicadores core)
    - salva em parquet/csv
    - retorna artefatos e DataFrame

    Retorna um dicionário com:
      - df_raw, df_processed
      - paths (zip, extracted_dir, parquet/csv)
      - metadata
    """
    if not (1900 <= int(ano) <= 2100):
        raise ValueError(f"Ano inválido: {ano}")

    base_dir = Path(base_dir)
    ano_dir = base_dir / str(ano)
    raw_dir = ano_dir / "raw"
    extracted_dir = ano_dir / "extracted"
    processed_dir = ano_dir / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://www.bcb.gov.br/pda/desig/scrdata_{ano}.zip"
    zip_path = raw_dir / f"scrdata_{ano}.zip"

    # ---------------------------
    # 1) Download (cache por ano)
    # ---------------------------
    if zip_path.exists() and not forcar_download:
        if verbose:
            print(f"ℹ️ ZIP {ano} já existe: {zip_path.resolve()}")
            print("➡️ Download não será realizado.")
    else:
        if verbose:
            print(f"⬇️ Baixando: {url}")
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
        if verbose:
            print(f"✅ ZIP salvo em: {zip_path.resolve()}")

    # ---------------------------
    # 2) Extração (cache)
    # ---------------------------
    ja_extraido = any(extracted_dir.rglob("*.csv"))
    if ja_extraido and not forcar_download:
        if verbose:
            print(f"ℹ️ CSVs já extraídos em: {extracted_dir.resolve()}")
            print("➡️ Extração não será realizada.")
    else:
        if forcar_download:
            # limpa extração para evitar mistura
            for p in sorted(extracted_dir.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    try:
                        p.rmdir()
                    except OSError:
                        pass

        if verbose:
            print(f"🗜️ Extraindo para: {extracted_dir.resolve()}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extracted_dir)
        if verbose:
            print("✅ Extração concluída.")

    # ---------------------------
    # 3) Localizar CSVs
    # ---------------------------
    csv_paths = sorted(extracted_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {extracted_dir.resolve()}")

    if verbose:
        print(f"📄 CSVs encontrados: {len(csv_paths)}")

    # ---------------------------
    # 4) Ler e concatenar CSVs
    # ---------------------------
    def inferir_sep(csv_path: Path, enc: str) -> str:
        sample = csv_path.read_bytes()[:50_000]
        text = sample.decode(enc, errors="ignore")
        return ";" if text.count(";") > text.count(",") else ","

    dfs: List[pd.DataFrame] = []
    for p in csv_paths:
        local_sep = sep or inferir_sep(p, encoding)

        try:
            df_part = pd.read_csv(p, encoding=encoding, sep=local_sep)
        except UnicodeDecodeError:
            df_part = pd.read_csv(p, encoding="latin1", sep=local_sep)

        if adicionar_coluna_origem:
            df_part["__arquivo_origem"] = p.name

        dfs.append(df_part)

    df_raw = pd.concat(dfs, ignore_index=True, sort=False)

    if verbose:
        print(f"✅ RAW consolidado: {len(df_raw):,} linhas | {len(df_raw.columns)} colunas")

    # ---------------------------
    # 5) Processar (normalização + indicadores core)
    # ---------------------------
    df_processed = processar_scrdata(
        df_raw,
        versao=versao_scr,
        remover_zeros=remover_zeros,
        criar_indicadores=criar_indicadores,
        verbose=verbose
    )

    # ---------------------------
    # 6) Salvar outputs
    # ---------------------------
    parquet_path = processed_dir / f"scrdata_{ano}.parquet"
    csv_path = processed_dir / f"scrdata_{ano}.csv"

    if salvar_parquet:
        df_processed.to_parquet(parquet_path, index=False)
        if verbose:
            print(f"💾 Parquet salvo em: {parquet_path.resolve()}")

    if salvar_csv:
        df_processed.to_csv(csv_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"💾 CSV salvo em: {csv_path.resolve()}")

    # Metadados
    df_processed.attrs["fonte_url"] = url
    df_processed.attrs["zip_path"] = str(zip_path.resolve())
    df_processed.attrs["extracted_dir"] = str(extracted_dir.resolve())
    df_processed.attrs["parquet_path"] = str(parquet_path.resolve()) if salvar_parquet else None
    df_processed.attrs["csv_path"] = str(csv_path.resolve()) if salvar_csv else None
    df_processed.attrs["csv_count"] = len(csv_paths)

    return {
        "df_raw": df_raw,
        "df_processed": df_processed,
        "paths": {
            "zip": zip_path.resolve(),
            "extracted_dir": extracted_dir.resolve(),
            "parquet": parquet_path.resolve() if salvar_parquet else None,
            "csv": csv_path.resolve() if salvar_csv else None,
        },
        "metadata": {
            "ano": ano,
            "url": url,
            "versao_scr": versao_scr,
            "csv_count": len(csv_paths),
            "rows_raw": len(df_raw),
            "rows_processed": len(df_processed),
            "cols_processed": len(df_processed.columns),
        }
    }