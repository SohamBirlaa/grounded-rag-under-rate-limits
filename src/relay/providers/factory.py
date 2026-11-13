"""Provider factory for The Relay.

Selects and instantiates either the real Groq client or the local stand-in engine
based strictly on explicit configuration or request source parameters.
Guarantees NO silent fallback from Groq to stand-in.
"""

from typing import Optional
import httpx

from relay.config.settings import RelayConfig
from relay.contracts.schemas import ProviderSourceType
from relay.providers.base import ModelProviderPort
from relay.providers.groq import GroqProviderAdapter
from relay.providers.standin import StandInProviderAdapter


def get_provider(
    config: RelayConfig,
    source: Optional[ProviderSourceType] = None,
    inject_delay: bool = True,
    http_client: Optional[httpx.AsyncClient] = None,
) -> ModelProviderPort:
    """Instantiate the configured model provider port.

    Args:
        config: Central RelayConfig instance.
        source: Explicit provider selection ('real' or 'stand-in').
                If None, defaults to config.provider.default_source.
        inject_delay: Whether stand-in executes real async delays (set False in fast unit tests).
        http_client: Optional custom httpx.AsyncClient for testing or transport injection.

    Returns:
        Instance implementing ModelProviderPort.
    """
    selected_source = source or config.provider.default_source

    if selected_source == "real":
        return GroqProviderAdapter(
            api_key=config.provider.api_key,
            base_url=config.provider.base_url,
            default_model=config.provider.model_id,
            request_timeout_s=config.provider.request_timeout_s,
            http_client=http_client,
        )
    elif selected_source == "stand-in":
        return StandInProviderAdapter(
            config=config.standin,
            inject_delay=inject_delay,
            simulated_model=f"{config.provider.model_id}-standin",
        )
    else:
        raise ValueError(f"Unknown provider source: {selected_source}")

