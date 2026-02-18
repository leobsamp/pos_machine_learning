"""
SCR -> S3 (raw + processed parquet)

Para cada ano:
- baixa ZIP do BCB (streaming)
- sobe ZIP no S3 (raw)
- extrai CSVs do ZIP
- concatena e processa (reusa processar_scrdata do seu scr_pipeline.py)
- salva Parquet (compressão)
- sobe Parquet no S3 (processed)

Uso:
  python scr_s3_pipeline.py --bucket SEU_BUCKET --prefix scr --start-year 2012 --end-year 2025

Credenciais AWS:
  export AWS_ACCESS_KEY_ID=...
  export AWS_SECRET_ACCESS_KEY=...
  export AWS_DEFAULT_REGION=sa-east-1   (ou sua região)
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, List

import boto3
import pandas as pd
import requests

# Reaproveita seu processamento (tipos + indicadores etc.)
# Certifique-se de que scr_pipeline.py está no mesmo diretório do script
from scr_pipeline import processar_scrdata


BCB_URL_TEMPLATE = "https://www.bcb.gov.br/pda/desig/scrdata_{ano}.zip"


# ----------------------------
# Utilitários
# ----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download_zip_streaming(url: str, out_path: Path, timeout: int = 300, chunk_size: int = 1024 * 1024) -> None:
    """Baixa arquivo grande sem carregar tudo em memória."""
    ensure_dir(out_path.parent)

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def safe_extract_zip(zip_path: Path, extract_to: Path) -> None:
    """
    Extração segura (evita zip-slip).
    """
    ensure_dir(extract_to)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_path = extract_to / member.filename
            # normaliza e garante que está dentro do diretório de extração
            resolved = member_path.resolve()
            if not str(resolved).startswith(str(extract_to.resolve())):
                raise RuntimeError(f"Entrada suspeita no zip (zip-slip): {member.filename}")
        zf.extractall(extract_to)


def infer_sep(csv_path: Path, encoding: str = "utf-8") -> str:
    sample = csv_path.read_bytes()[:50_000]
    text = sample.decode(encoding, errors="ignore")
    return ";" if text.count(";") > text.count(",") else ","


def read_and_concat_csvs(csv_paths: List[Path], encoding: str = "utf-8", sep: Optional[str] = None) -> pd.DataFrame:
    dfs = []
    for p in csv_paths:
        local_sep = sep or infer_sep(p, encoding=encoding)
        try:
            df_part = pd.read_csv(p, encoding=encoding, sep=local_sep)
        except UnicodeDecodeError:
            df_part = pd.read_csv(p, encoding="latin1", sep=local_sep)

        df_part["__arquivo_origem"] = p.name
        dfs.append(df_part)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True, sort=False)


def s3_upload_file(s3_client, local_path: Path, bucket: str, key: str) -> None:
    s3_client.upload_file(str(local_path), bucket, key)


# ----------------------------
# Pipeline por ano
# ----------------------------
def process_year_to_s3(
    ano: int,
    bucket: str,
    prefix: str = "scr",
    workdir: Path = Path("data_work"),
    remover_zeros: bool = True,
    criar_indicadores: bool = True,
    parquet_compression: str = "snappy",
    overwrite: bool = False,
) -> None:
    """
    Executa pipeline do SCR para um ano e publica raw + parquet no S3.
    """
    s3 = boto3.client("s3")

    url = BCB_URL_TEMPLATE.format(ano=ano)

    # Paths locais temporários
    ano_dir = workdir / f"ano={ano}"
    raw_dir = ano_dir / "raw"
    extracted_dir = ano_dir / "extracted"
    processed_dir = ano_dir / "processed"
    ensure_dir(raw_dir)
    ensure_dir(extracted_dir)
    ensure_dir(processed_dir)

    zip_local = raw_dir / f"scrdata_{ano}.zip"
    parquet_local = processed_dir / f"scrdata_{ano}.parquet"

    # Keys no S3 (pastas lógicas via prefixo)
    zip_key = f"{prefix}/raw/ano={ano}/scrdata_{ano}.zip"
    parquet_key = f"{prefix}/processed/ano={ano}/scrdata_{ano}.parquet"

    # Se não quiser sobrescrever, podemos tentar checar se já existe parquet no S3
    if not overwrite:
        try:
            s3.head_object(Bucket=bucket, Key=parquet_key)
            print(f"ℹ️ Parquet já existe no S3 para {ano}: s3://{bucket}/{parquet_key} (pulando)")
            return
        except s3.exceptions.ClientError:
            pass  # não existe -> continua

    print(f"\n=== Ano {ano} ===")
    print(f"⬇️ Download ZIP: {url}")

    # 1) Download ZIP (streaming)
    download_zip_streaming(url, zip_local)

    # 2) Upload raw ZIP para S3
    print(f"☁️ Upload raw ZIP -> s3://{bucket}/{zip_key}")
    s3_upload_file(s3, zip_local, bucket, zip_key)

    # 3) Extração segura
    # limpa extração para evitar mistura
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    ensure_dir(extracted_dir)

    print(f"🗜️ Extraindo ZIP local -> {extracted_dir}")
    safe_extract_zip(zip_local, extracted_dir)

    # 4) Localiza CSVs
    csv_paths = sorted(extracted_dir.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"Nenhum CSV encontrado após extração em {extracted_dir}")

    print(f"📄 CSVs encontrados: {len(csv_paths)}")

    # 5) Lê + concatena
    df_raw = read_and_concat_csvs(csv_paths, encoding="utf-8", sep=None)
    print(f"✅ RAW consolidado: {len(df_raw):,} linhas | {len(df_raw.columns)} colunas")

    # 6) Processa (reaproveita sua função)
    df_processed = processar_scrdata(
        df_raw,
        remover_zeros=remover_zeros,
        criar_indicadores=criar_indicadores,
        verbose=True,
    )

    # 7) Salva parquet (compressão)
    print(f"💾 Salvando Parquet local -> {parquet_local}")
    df_processed.to_parquet(parquet_local, index=False, compression=parquet_compression)

    # 8) Upload parquet para S3 (processed)
    print(f"☁️ Upload Parquet -> s3://{bucket}/{parquet_key}")
    s3_upload_file(s3, parquet_local, bucket, parquet_key)

    print(f"✅ Concluído ano {ano}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True, help="Nome do bucket S3")
    parser.add_argument("--prefix", default="scr", help="Prefixo base no bucket (ex.: scr)")
    parser.add_argument("--start-year", type=int, default=2012)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--workdir", default="data_work", help="Diretório local temporário")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescrever parquet no S3 se existir")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    ensure_dir(workdir)

    if args.start_year > args.end_year:
        raise ValueError("start-year não pode ser maior que end-year")

    for ano in range(args.start_year, args.end_year + 1):
        process_year_to_s3(
            ano=ano,
            bucket=args.bucket,
            prefix=args.prefix,
            workdir=workdir,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
