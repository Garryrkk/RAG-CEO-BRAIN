
from typing import Dict, Optional, Type
import structlog
from app.connectors.base.connector import BaseConnector, SourceType

logger = structlog.get_logger(__name__)

_REGISTRY: Dict[SourceType, Type[BaseConnector]] = {}


def register_connector(source_type: SourceType):
    """Class decorator to register a connector implementation."""
    def decorator(cls: Type[BaseConnector]):
        _REGISTRY[source_type] = cls
        logger.info("Connector registered", source_type=source_type.value, cls=cls.__name__)
        return cls
    return decorator


class ConnectorRegistry:
    """
    Registry of all connector implementations.
    Allows creating connectors by source type without knowing concrete classes.
    Adding a new connector: implement BaseConnector, decorate with @register_connector.
    Zero architectural changes required.
    """

    @staticmethod
    def get(source_type: SourceType) -> Optional[Type[BaseConnector]]:
        return _REGISTRY.get(source_type)

    @staticmethod
    def list_registered() -> list[SourceType]:
        return list(_REGISTRY.keys())

    @staticmethod
    def create(source_type: SourceType, config, redis, db) -> BaseConnector:
        cls = _REGISTRY.get(source_type)
        if cls is None:
            available = [s.value for s in _REGISTRY.keys()]
            raise ValueError(
                f"No connector registered for source type '{source_type}'. "
                f"Available: {available}"
            )
        return cls(config=config, redis=redis, db=db)
