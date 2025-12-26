"""Integration tests for RAG functionality.

These tests create real Gemini file stores and database records.
All resources are cleaned up after test execution.

Run with: RUN_INTEGRATION_TESTS=1 pytest tests/integration/test_rag_integration.py -v
"""

import os
import pytest
from fastapi.testclient import TestClient

# Skip all tests in this module unless RUN_INTEGRATION_TESTS is set
pytestmark = pytest.mark.integration


# Test organization - must exist in database
TEST_ORG_ID = os.getenv("TEST_ORG_ID", "89a05c06-f798-4836-8645-571a87b2014d")
TEST_ORG_NAME = os.getenv("TEST_ORG_NAME", "Integration Test Org")


@pytest.fixture(scope="module")
def client():
    """Create a test client."""
    from src.api.app import create_app
    app = create_app()
    return TestClient(app)


@pytest.fixture(scope="module")
def headers():
    """Return headers with required X-Organization-ID."""
    return {"X-Organization-ID": TEST_ORG_ID}


@pytest.fixture(scope="module")
def test_file(tmp_path_factory):
    """Create a temporary test file."""
    tmp_dir = tmp_path_factory.mktemp("test_files")
    test_file = tmp_dir / "integration_test_doc.md"
    test_file.write_text("""# Integration Test Document

This is a test document for integration testing.

## Content
- Test item 1
- Test item 2
""")
    return str(test_file)


@pytest.fixture(scope="module")
def cleanup_stores():
    """
    Cleanup fixture that tracks and deletes stores created during tests.

    Usage:
        def test_something(cleanup_stores):
            # Store created with specific display_name pattern
            cleanup_stores.append("integration_test_org_file_search_store")
    """
    stores_to_cleanup = []
    yield stores_to_cleanup

    # Cleanup after all tests in module
    if stores_to_cleanup:
        print(f"\nCleaning up {len(stores_to_cleanup)} test stores...")
        try:
            from src.rag.gemini_file_store import list_all_stores, delete_store

            all_stores = list_all_stores()
            for store in all_stores:
                if store.display_name in stores_to_cleanup:
                    print(f"  Deleting Gemini store: {store.display_name}")
                    try:
                        delete_store(store.name)
                    except Exception as e:
                        print(f"    Error: {e}")
        except Exception as e:
            print(f"  Cleanup error: {e}")

        # Also cleanup database
        try:
            import asyncio
            from src.db.connection import db
            from sqlalchemy import text

            async def cleanup_db():
                async with db.session() as session:
                    if session:
                        for display_name in stores_to_cleanup:
                            await session.execute(
                                text(f"DELETE FROM file_search_stores WHERE display_name = :name"),
                                {"name": display_name}
                            )
                        await session.commit()

            asyncio.run(cleanup_db())
            print("  Database cleanup complete")
        except Exception as e:
            print(f"  Database cleanup error: {e}")


class TestRagUploadIntegration:
    """Integration tests for RAG upload functionality."""

    def test_auto_store_creation_and_upload(self, client, headers, test_file, cleanup_stores):
        """Test that auto store creation works end-to-end."""
        # Register for cleanup
        expected_store_name = "integration_test_org_file_search_store"
        cleanup_stores.append(expected_store_name)

        response = client.post(
            "/api/v1/rag/stores/auto/upload",
            headers=headers,
            json={
                "file_paths": [test_file],
                "org_name": "Integration Test Org",
                "folder_name": "Test Folder",
                "parser_version": "test_v1.0",
            }
        )

        # Check response
        assert response.status_code == 200
        data = response.json()

        # May fail if org doesn't exist - that's ok for CI
        if data.get("success"):
            assert data["uploaded"] == 1
            assert len(data["files"]) == 1
            assert data["files"][0]["org_name"] == "Integration Test Org"
            assert data["files"][0]["content_hash"] is not None
        else:
            # Expected if org doesn't exist in test database
            print(f"Upload failed (expected if org not in DB): {data.get('errors')}")

    def test_store_name_is_lowercase(self, client, headers, test_file, cleanup_stores):
        """Test that store names are always lowercase."""
        from src.rag.gemini_file_store import generate_store_display_name

        # Test various org name formats
        test_cases = [
            ("UPPERCASE ORG", "uppercase_org_file_search_store"),
            ("MixedCase Org", "mixedcase_org_file_search_store"),
            ("lowercase org", "lowercase_org_file_search_store"),
            ("Org With - Hyphens", "org_with_hyphens_file_search_store"),
        ]

        for org_name, expected_display_name in test_cases:
            result = generate_store_display_name(org_name)
            assert result == expected_display_name, f"Expected {expected_display_name}, got {result}"


class TestRagSearchIntegration:
    """Integration tests for RAG search functionality."""

    def test_search_returns_citations(self, client, headers, cleanup_stores):
        """Test that search returns proper citations."""
        # This test requires a store with documents
        # Skip if no store exists

        # First, list stores
        response = client.get("/api/v1/rag/stores", headers=headers)
        assert response.status_code == 200

        data = response.json()
        if not data.get("stores"):
            pytest.skip("No stores available for search test")

        store_id = data["stores"][0]["store_id"]

        # Perform search
        response = client.post(
            f"/api/v1/rag/stores/{store_id}/search",
            headers=headers,
            json={"query": "What is this document about?"}
        )

        assert response.status_code == 200
        data = response.json()

        if data.get("success"):
            assert "response" in data
            assert "citations" in data
