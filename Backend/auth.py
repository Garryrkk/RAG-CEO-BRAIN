

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import httpx
import structlog
from redis.asyncio import Redis

logger = structlog.get_logger(__name__)


@dataclass
class TokenData:
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[float] = None  # Unix timestamp
    token_type: str = "Bearer"
    scope: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        # 60-second buffer before actual expiry
        return time.time() >= (self.expires_at - 60)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "token_type": self.token_type,
            "scope": self.scope,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenData":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class AuthProvider(ABC):
    """Abstract base for all authentication strategies."""

    def __init__(self, connector_id: str, redis: Redis):
        self.connector_id = connector_id
        self.redis = redis
        self._token_key = f"auth:tokens:{connector_id}"

    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> TokenData:
        """Perform initial authentication and return token data."""
        ...

    @abstractmethod
    async def refresh(self, token_data: TokenData) -> TokenData:
        """Refresh an expired token."""
        ...

    @abstractmethod
    async def get_headers(self) -> Dict[str, str]:
        """Return headers required for authenticated requests."""
        ...

    async def get_token(self) -> TokenData:
        """Get current token, refreshing if expired."""
        token_data = await self._load_token()
        if token_data is None:
            raise RuntimeError(f"No token found for connector {self.connector_id}. Run authenticate() first.")

        if token_data.is_expired:
            logger.info("Token expired, refreshing", connector_id=self.connector_id)
            token_data = await self.refresh(token_data)
            await self._save_token(token_data)

        return token_data

    async def _save_token(self, token_data: TokenData) -> None:
        await self.redis.set(
            self._token_key,
            json.dumps(token_data.to_dict()),
            ex=86400 * 30  # 30 days
        )

    async def _load_token(self) -> Optional[TokenData]:
        raw = await self.redis.get(self._token_key)
        if raw is None:
            return None
        return TokenData.from_dict(json.loads(raw))

    async def revoke(self) -> None:
        await self.redis.delete(self._token_key)


class OAuthProvider(AuthProvider):
    """
    OAuth2 authentication provider.
    Handles authorization code flow, token exchange, and refresh.
    """

    def __init__(
        self,
        connector_id: str,
        redis: Redis,
        client_id: str,
        client_secret: str,
        token_url: str,
        authorize_url: str,
        scopes: list[str],
        redirect_uri: str,
    ):
        super().__init__(connector_id, redis)
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.authorize_url = authorize_url
        self.scopes = scopes
        self.redirect_uri = redirect_uri

    def get_authorization_url(self, state: str) -> str:
        """Generate the OAuth authorization URL."""
        from urllib.parse import urlencode
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
        return f"{self.authorize_url}?{urlencode(params)}"

    async def authenticate(self, credentials: Dict[str, Any]) -> TokenData:
        """Exchange authorization code for tokens."""
        code = credentials.get("code")
        if not code:
            raise ValueError("Authorization code required for OAuth authentication")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": self.redirect_uri,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            data = response.json()

        token_data = self._parse_token_response(data)
        await self._save_token(token_data)
        logger.info("OAuth authentication successful", connector_id=self.connector_id)
        return token_data

    async def refresh(self, token_data: TokenData) -> TokenData:
        """Refresh access token using refresh token."""
        if not token_data.refresh_token:
            raise RuntimeError("No refresh token available. Re-authentication required.")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": token_data.refresh_token,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            data = response.json()

        new_token = self._parse_token_response(data)
        # Preserve refresh token if new one not provided
        if not new_token.refresh_token:
            new_token.refresh_token = token_data.refresh_token

        logger.info("Token refreshed", connector_id=self.connector_id)
        return new_token

    async def get_headers(self) -> Dict[str, str]:
        token = await self.get_token()
        return {"Authorization": f"{token.token_type} {token.access_token}"}

    def _parse_token_response(self, data: Dict[str, Any]) -> TokenData:
        expires_in = data.get("expires_in")
        expires_at = None
        if expires_in:
            expires_at = time.time() + int(expires_in)

        return TokenData(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            expires_at=expires_at,
            token_type=data.get("token_type", "Bearer"),
            scope=data.get("scope"),
            extra={k: v for k, v in data.items() if k not in {
                "access_token", "refresh_token", "expires_in", "token_type", "scope"
            }},
        )


class APIKeyProvider(AuthProvider):
    """Simple API key authentication."""

    def __init__(self, connector_id: str, redis: Redis, api_key: str, header_name: str = "Authorization"):
        super().__init__(connector_id, redis)
        self.api_key = api_key
        self.header_name = header_name

    async def authenticate(self, credentials: Dict[str, Any]) -> TokenData:
        token = TokenData(access_token=credentials.get("api_key", self.api_key))
        await self._save_token(token)
        return token

    async def refresh(self, token_data: TokenData) -> TokenData:
        # API keys don't expire in the traditional sense
        return token_data

    async def get_headers(self) -> Dict[str, str]:
        return {self.header_name: f"Bearer {self.api_key}"}


class ServiceAccountProvider(AuthProvider):
    """
    Service account (JWT-based) authentication.
    Used for Google APIs and similar service-to-service auth.
    """

    def __init__(
        self,
        connector_id: str,
        redis: Redis,
        service_account_key: Dict[str, Any],
        scopes: list[str],
    ):
        super().__init__(connector_id, redis)
        self.service_account_key = service_account_key
        self.scopes = scopes

    async def authenticate(self, credentials: Dict[str, Any]) -> TokenData:
        return await self._get_service_account_token()

    async def refresh(self, token_data: TokenData) -> TokenData:
        return await self._get_service_account_token()

    async def get_headers(self) -> Dict[str, str]:
        token = await self.get_token()
        return {"Authorization": f"Bearer {token.access_token}"}

    async def _get_service_account_token(self) -> TokenData:
        import jwt as pyjwt
        now = int(time.time())
        payload = {
            "iss": self.service_account_key["client_email"],
            "sub": self.service_account_key["client_email"],
            "aud": "https://oauth2.googleapis.com/token",
            "iat": now,
            "exp": now + 3600,
            "scope": " ".join(self.scopes),
        }

        signed_jwt = pyjwt.encode(
            payload,
            self.service_account_key["private_key"],
            algorithm="RS256",
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion": signed_jwt,
                },
            )
            response.raise_for_status()
            data = response.json()

        return TokenData(
            access_token=data["access_token"],
            expires_at=now + data.get("expires_in", 3600),
            token_type=data.get("token_type", "Bearer"),
        )


class MicrosoftOAuthProvider(OAuthProvider):
    """
    Microsoft-specific OAuth2 provider.
    Handles tenant-aware token endpoints.
    """

    AUTHORIZE_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize"
    TOKEN_URL = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

    def __init__(
        self,
        connector_id: str,
        redis: Redis,
        client_id: str,
        client_secret: str,
        tenant_id: str,
        scopes: list[str],
        redirect_uri: str,
    ):
        super().__init__(
            connector_id=connector_id,
            redis=redis,
            client_id=client_id,
            client_secret=client_secret,
            token_url=self.TOKEN_URL.format(tenant_id=tenant_id),
            authorize_url=self.AUTHORIZE_URL.format(tenant_id=tenant_id),
            scopes=scopes,
            redirect_uri=redirect_uri,
        )
        self.tenant_id = tenant_id
