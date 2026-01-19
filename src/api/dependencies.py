"""Shared dependencies for API routes.

Multi-tenancy: Organization context is extracted from headers and passed
through the request lifecycle for tenant isolation.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Optional, Dict, Any, TypeVar, Generic, Callable, Type

from fastapi import Depends, HTTPException, Header
from sqlalchemy import select, func

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Agent Factory (Thread-safe Singleton Pattern)
# =============================================================================

class AgentType(str, Enum):
    """Supported agent types."""
    DOCUMENT = "document"
    SHEETS = "sheets"
    EXTRACTOR = "extractor"


class AgentFactory:
    """
    Generic factory for creating and managing agent instances.

    Implements thread-safe singleton pattern with lazy initialization.
    Each agent type is created once and reused across requests.

    Usage:
        agent = await AgentFactory.get(AgentType.DOCUMENT)
        agent = await AgentFactory.get(AgentType.SHEETS)
        agent = await AgentFactory.get(AgentType.EXTRACTOR)

        # Or using convenience functions:
        agent = await get_document_agent()
    """

    _instances: Dict[AgentType, Any] = {}
    _lock: Optional[asyncio.Lock] = None

    @classmethod
    async def _get_lock(cls) -> asyncio.Lock:
        """Get or create the async lock (lazily initialized)."""
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    async def get(cls, agent_type: AgentType) -> Any:
        """
        Get or create an agent instance (thread-safe).

        Args:
            agent_type: The type of agent to get

        Returns:
            The agent instance

        Raises:
            ValueError: If agent type is not supported
        """
        if agent_type not in cls._instances:
            lock = await cls._get_lock()
            async with lock:
                # Double-check after acquiring lock
                if agent_type not in cls._instances:
                    cls._instances[agent_type] = cls._create_agent(agent_type)
                    logger.info(f"{agent_type.value.title()}Agent initialized")

        return cls._instances[agent_type]

    @classmethod
    def _create_agent(cls, agent_type: AgentType) -> Any:
        """
        Create a new agent instance based on type.

        Args:
            agent_type: The type of agent to create

        Returns:
            The new agent instance
        """
        if agent_type == AgentType.DOCUMENT:
            from src.agents.document import DocumentAgent
            return DocumentAgent()

        elif agent_type == AgentType.SHEETS:
            from src.agents.sheets import SheetsAgent, SheetsAgentConfig
            return SheetsAgent(SheetsAgentConfig())

        elif agent_type == AgentType.EXTRACTOR:
            from src.agents.extractor import ExtractorAgent, ExtractorAgentConfig
            return ExtractorAgent(ExtractorAgentConfig())

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    @classmethod
    async def initialize_all(cls) -> None:
        """
        Initialize all agents at startup (eager initialization).

        This ensures agents are warm and ready for requests immediately.
        Called during application startup in the lifespan handler.
        """
        logger.info("Initializing all agents at startup...")

        for agent_type in AgentType:
            try:
                await cls.get(agent_type)
            except Exception as e:
                logger.error(f"Failed to initialize {agent_type.value}Agent: {e}")
                raise

        logger.info("All agents initialized and ready")

    @classmethod
    async def shutdown_all(cls) -> None:
        """
        Shutdown all agent instances gracefully.

        This should be called during application shutdown to ensure
        all background tasks are completed and resources are released.
        """
        logger.info("Shutting down all agents...")

        for agent_type, agent in list(cls._instances.items()):
            if agent is not None:
                try:
                    agent.shutdown(wait=True)
                    logger.info(f"{agent_type.value.title()}Agent shutdown complete")
                except Exception as e:
                    logger.error(f"Error shutting down {agent_type.value}Agent: {e}")

        cls._instances.clear()
        logger.info("All agents shutdown complete")

    @classmethod
    def get_instance(cls, agent_type: AgentType) -> Optional[Any]:
        """
        Get an agent instance if it exists (non-async, for shutdown).

        Returns None if the agent hasn't been initialized.
        """
        return cls._instances.get(agent_type)


# =============================================================================
# Organization Lookup (for multi-tenancy)
# =============================================================================

async def lookup_organization(identifier: str) -> Optional[Any]:
    """
    Look up organization by ID or name.

    Tries exact ID match first, then falls back to case-insensitive name match.

    Args:
        identifier: Organization ID (UUID string) or name

    Returns:
        OrganizationModel instance or None if not found
    """
    try:
        from src.db.connection import db
        from src.db.models import OrganizationModel

        logger.info(f"Looking up organization: '{identifier}'")

        async with db.session() as session:
            if session is None:
                # Database disabled - return None (will use header as-is)
                logger.warning(f"Database session is None - database may be disabled. Cannot lookup org: '{identifier}'")
                return None

            # Try by ID first (exact string match - id column is VARCHAR)
            logger.debug(f"Trying org lookup by ID: '{identifier}'")
            stmt = select(OrganizationModel).where(OrganizationModel.id == identifier)
            result = await session.execute(stmt)
            org = result.scalar_one_or_none()
            if org:
                logger.info(f"Found organization by ID: '{org.name}' (id={org.id})")
                return org

            # Try by name (case-insensitive)
            logger.debug(f"Trying org lookup by name (case-insensitive): '{identifier.lower()}'")
            stmt = select(OrganizationModel).where(
                func.lower(OrganizationModel.name) == identifier.lower()
            )
            result = await session.execute(stmt)
            org = result.scalar_one_or_none()

            if org:
                logger.info(f"Found organization by name: '{org.name}' (id={org.id})")
            else:
                logger.warning(f"Organization not found by ID or name: '{identifier}'")

            return org

    except Exception as e:
        logger.error(f"Organization lookup failed for '{identifier}': {type(e).__name__}: {e}", exc_info=True)
        return None


async def validate_user_organization_membership(
    user_id: str,
    claimed_org_id: str,
) -> bool:
    """
    Validate that a user belongs to the claimed organization.

    SECURITY: This is critical for multi-tenant isolation. Users must only
    access data from organizations they belong to.

    Args:
        user_id: User ID from authentication
        claimed_org_id: Organization ID being claimed in the request

    Returns:
        True if user belongs to the organization, False otherwise
    """
    try:
        from src.db.connection import db
        from src.db.models import UserModel

        async with db.session() as session:
            if session is None:
                logger.warning("Database unavailable for user-org validation")
                return False

            # Look up user and verify organization membership
            stmt = select(UserModel).where(
                UserModel.id == user_id,
                UserModel.is_active == True,
            )
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()

            if not user:
                logger.warning(f"User not found or inactive: {user_id}")
                return False

            if user.organization_id != claimed_org_id:
                logger.warning(
                    f"SECURITY: User {user_id} (org={user.organization_id}) "
                    f"attempted to access org={claimed_org_id}"
                )
                return False

            logger.debug(f"User {user_id} validated for org {claimed_org_id}")
            return True

    except Exception as e:
        logger.error(f"User-org validation failed: {e}", exc_info=True)
        return False


async def lookup_user_by_email(email: str) -> Optional[Any]:
    """
    Look up user by email address.

    Args:
        email: User email address

    Returns:
        UserModel instance or None if not found
    """
    try:
        from src.db.connection import db
        from src.db.models import UserModel

        async with db.session() as session:
            if session is None:
                return None

            stmt = select(UserModel).where(
                func.lower(UserModel.email) == email.lower(),
                UserModel.is_active == True,
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    except Exception as e:
        logger.error(f"User lookup by email failed: {e}", exc_info=True)
        return None


# =============================================================================
# Multi-Tenancy Context
# =============================================================================

@dataclass
class OrgContext:
    """Organization context for multi-tenant operations."""
    org_id: str
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for passing to services."""
        return {
            "org_id": self.org_id,
            "user_id": self.user_id,
        }


async def get_org_id(
    x_organization_id: Optional[str] = Header(None, alias="X-Organization-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
) -> str:
    """
    Extract and resolve organization ID from request header with user validation.

    SECURITY: Validates that the authenticated user belongs to the claimed
    organization. Users cannot access data from other organizations.

    Looks up the organization by UUID or name and returns the actual UUID.
    Required for multi-tenant isolation. All data operations must be
    scoped to the organization.

    Args:
        x_organization_id: Organization identifier (UUID or name) from header
        x_user_id: User ID from authentication (required for validation)
        x_user_email: User email from authentication (fallback for user lookup)

    Returns:
        Organization UUID string

    Raises:
        HTTPException 400: If required headers are missing
        HTTPException 401: If user authentication is missing
        HTTPException 403: If user doesn't belong to the claimed organization
        HTTPException 404: If organization not found
    """
    if not x_organization_id:
        raise HTTPException(
            status_code=400,
            detail="X-Organization-ID header required for multi-tenant operation"
        )

    # Look up organization to get actual UUID
    org = await lookup_organization(x_organization_id)
    if not org:
        raise HTTPException(
            status_code=404,
            detail=f"Organization not found: {x_organization_id}"
        )

    org_id = str(org.id)

    # SECURITY: Validate user belongs to the claimed organization
    # User identity can come from X-User-ID or X-User-Email header
    user_id = x_user_id

    # If no user_id but have email, look up user by email
    if not user_id and x_user_email:
        user = await lookup_user_by_email(x_user_email)
        if user:
            user_id = str(user.id)

    if not user_id:
        # No user authentication provided - reject for security
        logger.warning(
            f"SECURITY: Request to org '{x_organization_id}' without user authentication"
        )
        raise HTTPException(
            status_code=401,
            detail="User authentication required. Provide X-User-ID or X-User-Email header."
        )

    # Validate user belongs to the claimed organization
    is_member = await validate_user_organization_membership(user_id, org_id)
    if not is_member:
        raise HTTPException(
            status_code=403,
            detail="Access denied: You do not belong to this organization"
        )

    return org_id


async def get_optional_org_id(
    x_organization_id: Optional[str] = Header(None, alias="X-Organization-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
) -> Optional[str]:
    """
    Extract and resolve optional organization ID from request header.

    Use for endpoints that can work without org context (e.g., health checks).
    Returns the actual UUID if org is found and user is authorized, None otherwise.

    SECURITY: If org is provided, user must also be provided and validated.
    """
    if not x_organization_id:
        return None

    # Look up organization to get actual UUID
    org = await lookup_organization(x_organization_id)
    if not org:
        return None

    org_id = str(org.id)

    # SECURITY: If org is provided, validate user belongs to it
    user_id = x_user_id
    if not user_id and x_user_email:
        user = await lookup_user_by_email(x_user_email)
        if user:
            user_id = str(user.id)

    if user_id:
        is_member = await validate_user_organization_membership(user_id, org_id)
        if not is_member:
            logger.warning(
                f"SECURITY: User {user_id} denied access to org {org_id} (optional context)"
            )
            return None
        return org_id

    # No user provided with org - return None for optional context
    logger.debug(f"Optional org context without user auth - returning None")
    return None


async def get_user_id(
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
) -> Optional[str]:
    """
    Extract user ID from request header.

    Can be provided directly via X-User-ID or looked up by X-User-Email.
    In production, this should ideally be extracted from a validated JWT token.
    """
    if x_user_id:
        return x_user_id

    if x_user_email:
        user = await lookup_user_by_email(x_user_email)
        if user:
            return str(user.id)

    return None


async def get_org_context(
    x_organization_id: Optional[str] = Header(None, alias="X-Organization-ID"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_user_email: Optional[str] = Header(None, alias="X-User-Email"),
) -> OrgContext:
    """
    Get full organization context for multi-tenant operations.

    Combines org_id and user_id into a single context object.
    User-organization membership is validated for security.
    """
    # Get org_id with user validation
    org_id = await get_org_id(x_organization_id, x_user_id, x_user_email)

    # Get user_id (already validated as part of get_org_id)
    user_id = await get_user_id(x_user_id, x_user_email)

    return OrgContext(org_id=org_id, user_id=user_id)


# =============================================================================
# Agent Dependencies (Convenience functions using AgentFactory)
# =============================================================================

async def get_document_agent():
    """Get or create DocumentAgent instance (thread-safe)."""
    return await AgentFactory.get(AgentType.DOCUMENT)


async def get_sheets_agent():
    """Get or create SheetsAgent instance (thread-safe)."""
    return await AgentFactory.get(AgentType.SHEETS)


async def get_extractor_agent():
    """Get or create ExtractorAgent instance (thread-safe)."""
    return await AgentFactory.get(AgentType.EXTRACTOR)


async def initialize_agents() -> None:
    """
    Initialize all agents at startup (eager initialization).

    This ensures agents are warm and ready for requests immediately.
    Called during application startup in the lifespan handler.

    Raises:
        Exception: If any agent fails to initialize (fail-fast behavior).
    """
    await AgentFactory.initialize_all()


async def shutdown_agents():
    """
    Shutdown all agent instances gracefully.

    This should be called during application shutdown to ensure
    all background tasks are completed and resources are released.
    """
    await AgentFactory.shutdown_all()


# =============================================================================
# Configuration Dependencies
# =============================================================================

@lru_cache()
def get_upload_directory() -> str:
    """Get the upload directory path."""
    base = os.getcwd()
    upload_dir = os.getenv("UPLOAD_DIRECTORY", "upload")
    return os.path.join(base, upload_dir)


@lru_cache()
def get_parsed_directory() -> str:
    """
    Get the parsed documents directory path.

    Returns GCS URI when GCS storage is configured.
    """
    try:
        from src.storage import get_storage_config

        config = get_storage_config()
        # Return GCS path in format gs://bucket/prefix/parsed
        return f"gs://{config.gcs_bucket}/{config.gcs_prefix}/{config.parsed_directory}"
    except Exception as e:
        logger.warning(f"Failed to get storage config, using local path: {e}")
        base = os.getcwd()
        parsed_dir = os.getenv("PARSED_DIRECTORY", "parsed")
        return os.path.join(base, parsed_dir)


@lru_cache()
def get_max_upload_size() -> int:
    """Get maximum upload size in bytes."""
    max_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    return max_mb * 1024 * 1024


# =============================================================================
# Optional API Key Authentication
# =============================================================================

def get_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[str]:
    """
    Optional API key authentication.

    If API_KEY_REQUIRED is set to 'true' in environment, validates the key.
    Otherwise, returns the key for logging purposes.
    """
    from src.utils.env_utils import parse_bool_env
    api_key_required = parse_bool_env("API_KEY_REQUIRED", False)
    expected_key = os.getenv("API_KEY", "")

    if api_key_required:
        if not x_api_key:
            raise HTTPException(
                status_code=401,
                detail="API key required. Provide X-API-Key header."
            )
        if x_api_key != expected_key:
            raise HTTPException(
                status_code=403,
                detail="Invalid API key"
            )

    return x_api_key


# =============================================================================
# Session Management
# =============================================================================

_active_sessions = {}


def get_session_manager():
    """Get the session manager (simple in-memory for now)."""
    return _active_sessions


async def validate_session(session_id: str) -> dict:
    """Validate and return session info."""
    sessions = get_session_manager()
    if session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found"
        )
    return sessions[session_id]


# =============================================================================
# RAG Dependencies
# =============================================================================

_file_search_store = None


async def get_file_search_store():
    """Get or create Gemini File Search store manager."""
    global _file_search_store
    if _file_search_store is None:
        try:
            from src.rag.gemini_file_store import GeminiFileStore
            _file_search_store = GeminiFileStore()
            logger.info("GeminiFileStore initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize GeminiFileStore: {e}")
            raise HTTPException(
                status_code=503,
                detail="RAG service unavailable"
            )
    return _file_search_store


# =============================================================================
# Database Dependencies
# =============================================================================

async def get_db_session():
    """Get database session for async operations."""
    try:
        from src.db.connection import db
        async with db.session() as session:
            if session:
                yield session
            else:
                yield None
    except Exception as e:
        logger.warning(f"Database session unavailable: {e}")
        yield None
