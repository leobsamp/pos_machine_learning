import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from botocore.config import Config
from botocore.exceptions import EndpointConnectionError, ConnectionClosedError, ReadTimeoutError, SSLError

def show_env():
    print("=== Variáveis de ambiente (parcial) ===")
    for k in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION", "AWS_REGION",
              "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY"]:
        v = os.getenv(k)
        if v:
            if "SECRET" in k:
                v = v[:4] + "..."  # mascara
            print(f"{k}={v}")
        else:
            print(f"{k}=<não definido>")
    print()

def test_s3(bucket_name: str, prefix: str = "scr/"):
    show_env()

    # Recomendação: no Windows, às vezes AWS_REGION é lido em vez de AWS_DEFAULT_REGION em certos setups
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "sa-east-1"
    print(f"🌎 Região usada: {region}")

    # Config com timeouts/retries melhores para diagnosticar
    cfg = Config(
        region_name=region,
        connect_timeout=10,
        read_timeout=20,
        retries={"max_attempts": 3, "mode": "standard"},
    )

    try:
        s3 = boto3.client("s3", config=cfg)

        # 1) Teste básico de autenticação (não precisa de bucket)
        sts = boto3.client("sts", config=cfg)
        ident = sts.get_caller_identity()
        print(f"✅ STS ok. Account={ident.get('Account')} Arn={ident.get('Arn')}")

        # 2) Teste de acesso ao bucket
        print(f"\n🔎 Testando head_bucket em: {bucket_name}")
        s3.head_bucket(Bucket=bucket_name)
        print("✅ head_bucket ok (bucket existe e você tem permissão).")

        # 3) Listagem simples
        print(f"\n📂 Listando até 5 objetos em Prefix='{prefix}':")
        resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=5)
        objs = resp.get("Contents", [])
        if not objs:
            print("⚠️ Nenhum objeto encontrado (ou prefix vazio).")
        else:
            for o in objs:
                print(" -", o["Key"])

        print("\n🎉 Conexão com S3 funcionando.")

    except NoCredentialsError:
        print("❌ Credenciais não encontradas. Refaça os SET no CMD.")
    except EndpointConnectionError as e:
        print("❌ Falha de conexão com o endpoint (rede/DNS/proxy/firewall).")
        print(f"   Detalhe: {e}")
        print("\n➡️ Se você estiver em rede corporativa, configure HTTP(S)_PROXY ou tente em outra rede/VPN.")
    except SSLError as e:
        print("❌ Erro SSL/TLS (muito comum com proxy corporativo / inspeção HTTPS).")
        print(f"   Detalhe: {e}")
        print("\n➡️ Se houver proxy corporativo com inspeção, você precisa do proxy configurado no ambiente.")
    except (ReadTimeoutError, ConnectionClosedError) as e:
        print("❌ Timeout/Conexão fechada (instabilidade ou proxy).")
        print(f"   Detalhe: {e}")
    except ClientError as e:
        code = e.response["Error"].get("Code")
        msg = e.response["Error"].get("Message")
        print(f"❌ ClientError: {code} - {msg}")
        if code in ("AccessDenied", "Forbidden"):
            print("➡️ Isso é IAM/policy: falta s3:ListBucket / s3:GetObject / s3:PutObject conforme sua necessidade.")
        elif code in ("NoSuchBucket",):
            print("➡️ Nome do bucket errado ou bucket em outra conta/região.")
    except Exception as e:
        print("❌ Erro inesperado:", repr(e))

if __name__ == "__main__":
    BUCKET_NAME = "SEU_BUCKET_AQUI"
    PREFIX = "scr/"
    test_s3(BUCKET_NAME, PREFIX)
