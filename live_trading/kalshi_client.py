"""
Kalshi API client with RSA signature authentication.

Implements request signing and provides methods to interact with the Kalshi demo API.
"""

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.common.config import get_config
from src.common.logger import get_logger
from src.common.paths import get_repo_root


class KalshiClient:
    """
    Client for interacting with the Kalshi API.

    Handles RSA signature authentication and provides methods to fetch
    account information and market data.
    """

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the Kalshi client.

        Args:
            api_key_id: Kalshi API key ID. If None, loads from environment/config.
            private_key_path: Path to RSA private key file. If None, loads from environment/config.
            base_url: Kalshi API base URL. If None, loads from environment/config.
        """
        config = get_config()

        self.api_key_id = api_key_id or config.kalshi_api_key_id
        self.base_url = base_url or config.kalshi_base_url

        # Resolve private key path
        if private_key_path:
            self.private_key_path = Path(private_key_path)
        elif config.kalshi_private_key_path:
            # If path is relative, resolve from repo root
            key_path = Path(config.kalshi_private_key_path)
            if key_path.is_absolute():
                self.private_key_path = key_path
            else:
                self.private_key_path = get_repo_root() / key_path
        else:
            self.private_key_path = None

        # Validate credentials
        if not self.api_key_id:
            raise ValueError("KALSHI_API_KEY_ID is required. Set it in .env or pass as argument.")
        if not self.private_key_path or not self.private_key_path.exists():
            raise ValueError(
                f"Private key file not found at {self.private_key_path}. "
                "Set KALSHI_PRIVATE_KEY_PATH in .env or pass as argument."
            )

        # Load private key
        self._load_private_key()

        # Setup logger
        self.logger = get_logger(__name__)

        # Verify we're using demo API
        if "demo-api" not in self.base_url:
            self.logger.warning(
                f"Base URL does not contain 'demo-api': {self.base_url}. "
                "Are you sure you want to use production?"
            )
        else:
            self.logger.info(f"Using Kalshi demo API: {self.base_url}")

    def _load_private_key(self) -> None:
        """Load the RSA private key from file."""
        try:
            with open(self.private_key_path, "rb") as key_file:
                key_data = key_file.read()
                self.private_key = serialization.load_pem_private_key(
                    key_data, password=None, backend=default_backend()
                )
        except Exception as e:
            raise ValueError(f"Failed to load private key from {self.private_key_path}: {e}")

    def _sign_request(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """
        Sign a request using RSA-SHA256.

        The signature is computed over: timestamp + HTTP_METHOD + request_path + body

        Args:
            timestamp: Request timestamp in milliseconds (as string).
            method: HTTP method (GET, POST, etc.).
            path: API request path (e.g., "/trade-api/v2/portfolio/balance").
            body: Request body as string (empty for GET requests).

        Returns:
            Base64-encoded signature string.
        """
        # Create the message to sign: timestamp + method + path + body
        message = f"{timestamp}{method}{path}{body}".encode("utf-8")

        # Sign using RSA-PSS with SHA256
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        # Encode signature as base64
        signature_b64 = base64.b64encode(signature).decode("utf-8")
        return signature_b64

    def _make_request(
        self, method: str, path: str, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an authenticated request to the Kalshi API.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: API endpoint path (relative to base_url, e.g., "/portfolio/balance").
            params: Query parameters for GET requests.
            data: Request body data for POST/PUT requests.

        Returns:
            JSON response as dictionary.

        Raises:
            requests.HTTPError: If the API request fails.
            ValueError: If authentication fails.
        """
        # Generate timestamp in milliseconds
        timestamp = str(int(time.time() * 1000))

        # Prepare request body
        body = ""
        if data:
            body = json.dumps(data, separators=(",", ":"))

        # Extract the API path from base_url for signing
        # base_url is like "https://demo-api.kalshi.co/trade-api/v2"
        # We need "/trade-api/v2" + path for signing
        parsed_url = urlparse(self.base_url)
        api_path = parsed_url.path  # e.g., "/trade-api/v2"
        full_path = f"{api_path}{path}"  # e.g., "/trade-api/v2/portfolio/balance"

        # Sign the request using the full API path
        signature = self._sign_request(timestamp, method, full_path, body)

        # Build full URL
        url = f"{self.base_url}{path}"

        # Prepare headers
        headers = {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json",
        }

        # Log request (without sensitive data)
        self.logger.info(f"Making {method} request to {path}")

        # Make the request
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data, params=params, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Log response status
            self.logger.info(f"Response status: {response.status_code} for {path}")

            # Check for errors
            response.raise_for_status()

            # Parse and return JSON
            return response.json()

        except requests.exceptions.HTTPError as e:
            error_msg = f"API request failed: {e}"
            if hasattr(e.response, "text"):
                error_msg += f" - {e.response.text}"
            self.logger.error(error_msg)
            raise
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {e}"
            self.logger.error(error_msg)
            raise

    def get_account(self) -> Dict[str, Any]:
        """
        Fetch account information and balance.

        Returns:
            Dictionary containing account information including balance, equity, etc.
        """
        self.logger.info("Fetching account information...")
        response = self._make_request("GET", "/portfolio/balance")
        return response

    def list_markets(self, limit: int = 5) -> Dict[str, Any]:
        """
        Fetch a list of active markets.

        Args:
            limit: Maximum number of markets to return.

        Returns:
            Dictionary containing market list and metadata.
        """
        self.logger.info(f"Fetching {limit} active markets...")
        params = {"limit": limit}
        response = self._make_request("GET", "/markets", params=params)
        return response

