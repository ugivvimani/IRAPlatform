from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppSettings:
    app_env: str
    app_host: str
    app_port: int
    vector_backend: str
    pinecone_api_key: str
    pinecone_index: str
    pinecone_namespace: str
    db_backend: str
    postgres_dsn: str
    sqlite_db_path: str
    key_vault_url: str


def _load_keyvault_overrides() -> dict[str, str]:
    vault_url = os.getenv("AZURE_KEY_VAULT_URL", "")
    if not vault_url:
        return {}

    try:
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient

        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=credential)
    except Exception:
        return {}

    mapping = {
        "PINECONE_API_KEY": "pinecone-api-key",
        "OPENAI_API_KEY": "openai-api-key",
        "OPENROUTER_API_KEY": "openrouter-api-key",
        "AZURE_OPENAI_API_KEY": "azure-openai-api-key",
        "AZURE_OPENAI_ENDPOINT": "azure-openai-endpoint",
        "AZURE_OPENAI_DEPLOYMENT": "azure-openai-deployment",
        "OPENSANCTIONS_API_KEY": "opensanctions-api-key",
        "NEWS_API_KEY": "news-api-key",
        "ESG_API_KEY": "esg-api-key",
        "SEC_CONTACT_EMAIL": "sec-contact-email",
        "POSTGRES_DSN": "postgres-dsn",
        "APPLICATIONINSIGHTS_CONNECTION_STRING": "appinsights-connection-string",
    }

    loaded: dict[str, str] = {}
    for env_name, secret_name in mapping.items():
        try:
            loaded[env_name] = client.get_secret(secret_name).value
        except Exception:
            continue
    return loaded


def load_settings() -> AppSettings:
    keyvault_values = _load_keyvault_overrides()
    for name, value in keyvault_values.items():
        if value and not os.getenv(name):
            os.environ[name] = value

    return AppSettings(
        app_env=os.getenv("APP_ENV", "local"),
        app_host=os.getenv("APP_HOST", "127.0.0.1"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        vector_backend="pinecone",
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
        pinecone_index=os.getenv("PINECONE_INDEX", "ira-platform-memory"),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "default"),
        db_backend=os.getenv("DB_BACKEND", "sqlite"),
        postgres_dsn=os.getenv("POSTGRES_DSN", ""),
        sqlite_db_path=os.getenv("SQLITE_DB_PATH", "data/ira_platform.db"),
        key_vault_url=os.getenv("AZURE_KEY_VAULT_URL", ""),
    )

