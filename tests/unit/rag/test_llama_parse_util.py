"""Unit tests for LlamaParse document parsing utility."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock


class TestValidateFile:
    """Tests for _validate_file function."""

    def test_validate_file_success(self, tmp_path):
        """Test validation of an existing file with supported extension."""
        from src.rag.llama_parse_util import _validate_file

        test_file = tmp_path / "test.pdf"
        test_file.write_text("dummy content")

        result = _validate_file(str(test_file))

        assert result == test_file
        assert result.exists()

    def test_validate_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        from src.rag.llama_parse_util import _validate_file

        with pytest.raises(FileNotFoundError) as exc_info:
            _validate_file("/nonexistent/path/file.pdf")

        assert "File not found" in str(exc_info.value)

    def test_validate_file_unsupported_extension(self, tmp_path):
        """Test ValueError for unsupported file extension."""
        from src.rag.llama_parse_util import _validate_file

        test_file = tmp_path / "test.xyz"
        test_file.write_text("dummy content")

        with pytest.raises(ValueError) as exc_info:
            _validate_file(str(test_file))

        assert "Unsupported file format" in str(exc_info.value)
        assert ".xyz" in str(exc_info.value)

    def test_validate_file_case_insensitive(self, tmp_path):
        """Test that extension validation is case-insensitive."""
        from src.rag.llama_parse_util import _validate_file

        test_file = tmp_path / "test.PDF"
        test_file.write_text("dummy content")

        result = _validate_file(str(test_file))

        assert result == test_file

    def test_validate_file_docx(self, tmp_path):
        """Test validation of docx file."""
        from src.rag.llama_parse_util import _validate_file

        test_file = tmp_path / "document.docx"
        test_file.write_text("dummy content")

        result = _validate_file(str(test_file))

        assert result == test_file

    def test_validate_file_image(self, tmp_path):
        """Test validation of image file."""
        from src.rag.llama_parse_util import _validate_file

        test_file = tmp_path / "image.png"
        test_file.write_text("dummy content")

        result = _validate_file(str(test_file))

        assert result == test_file


class TestGetParser:
    """Tests for _get_parser function."""

    def test_get_parser_success(self):
        """Test successful parser creation with API key."""
        from src.rag.llama_parse_util import _get_parser

        with patch("src.rag.llama_parse_util.LLAMA_CLOUD_API_KEY", "test-api-key"), \
             patch("src.rag.llama_parse_util.LlamaParse") as mock_parser_class:

            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser

            result = _get_parser()

            assert result == mock_parser
            mock_parser_class.assert_called_once()

            # Verify configuration
            call_kwargs = mock_parser_class.call_args.kwargs
            assert call_kwargs["api_key"] == "test-api-key"
            assert call_kwargs["use_vendor_multimodal_model"] is True
            assert call_kwargs["output_tables_as_HTML"] is True
            assert call_kwargs["take_screenshot"] is True

    def test_get_parser_missing_api_key(self):
        """Test ValueError when LLAMA_CLOUD_API_KEY is not set."""
        from src.rag.llama_parse_util import _get_parser

        with patch("src.rag.llama_parse_util.LLAMA_CLOUD_API_KEY", None):
            with pytest.raises(ValueError) as exc_info:
                _get_parser()

            assert "LLAMA_CLOUD_API_KEY" in str(exc_info.value)


class TestGetAgentParser:
    """Tests for _get_agent_parser function."""

    def test_get_agent_parser_normal_complexity(self):
        """Test agent parser with normal complexity uses openai model."""
        from src.rag.llama_parse_util import _get_agent_parser

        with patch("src.rag.llama_parse_util.LLAMA_CLOUD_API_KEY", "test-api-key"), \
             patch("src.rag.llama_parse_util.OPENAI_API_KEY", "openai-key"), \
             patch("src.rag.llama_parse_util.LlamaParse") as mock_parser_class:

            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser

            result = _get_agent_parser(complexity="normal")

            assert result == mock_parser
            call_kwargs = mock_parser_class.call_args.kwargs
            assert call_kwargs["vendor_multimodal_model_name"] == "openai-gpt-5-mini"
            assert call_kwargs["vendor_multimodal_api_key"] == "openai-key"

    def test_get_agent_parser_high_complexity(self):
        """Test agent parser with high complexity uses gemini model."""
        from src.rag.llama_parse_util import _get_agent_parser

        with patch("src.rag.llama_parse_util.LLAMA_CLOUD_API_KEY", "test-api-key"), \
             patch("src.rag.llama_parse_util.GOOGLE_API_KEY", "google-key"), \
             patch("src.rag.llama_parse_util.LlamaParse") as mock_parser_class:

            mock_parser = MagicMock()
            mock_parser_class.return_value = mock_parser

            result = _get_agent_parser(complexity="high")

            assert result == mock_parser
            call_kwargs = mock_parser_class.call_args.kwargs
            assert call_kwargs["vendor_multimodal_model_name"] == "gemini-2.5-pro"
            assert call_kwargs["vendor_multimodal_api_key"] == "google-key"

    def test_get_agent_parser_missing_llama_key(self):
        """Test ValueError when LLAMA_CLOUD_API_KEY is not set."""
        from src.rag.llama_parse_util import _get_agent_parser

        with patch("src.rag.llama_parse_util.LLAMA_CLOUD_API_KEY", None):
            with pytest.raises(ValueError) as exc_info:
                _get_agent_parser()

            assert "LLAMA_CLOUD_API_KEY" in str(exc_info.value)

    def test_get_agent_parser_missing_openai_key(self):
        """Test ValueError when OPENAI_API_KEY not set for normal complexity."""
        from src.rag.llama_parse_util import _get_agent_parser

        with patch("src.rag.llama_parse_util.LLAMA_CLOUD_API_KEY", "test-api-key"), \
             patch("src.rag.llama_parse_util.OPENAI_API_KEY", None):

            with pytest.raises(ValueError) as exc_info:
                _get_agent_parser(complexity="normal")

            assert "API key required" in str(exc_info.value)

    def test_get_agent_parser_missing_google_key(self):
        """Test ValueError when GOOGLE_API_KEY not set for high complexity."""
        from src.rag.llama_parse_util import _get_agent_parser

        with patch("src.rag.llama_parse_util.LLAMA_CLOUD_API_KEY", "test-api-key"), \
             patch("src.rag.llama_parse_util.GOOGLE_API_KEY", None):

            with pytest.raises(ValueError) as exc_info:
                _get_agent_parser(complexity="high")

            assert "API key required" in str(exc_info.value)


class TestParseDocument:
    """Tests for parse_document function."""

    @pytest.fixture
    def mock_parser(self):
        """Create a mock parser instance."""
        parser = MagicMock()
        parser.load_data.return_value = [
            MagicMock(text="# Document Title\n\nContent here."),
        ]
        return parser

    @pytest.fixture
    def test_pdf(self, tmp_path):
        """Create a temporary PDF file."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content")
        return str(pdf_file)

    def test_parse_document_success(self, mock_parser, test_pdf):
        """Test successful document parsing."""
        from src.rag.llama_parse_util import parse_document

        with patch("src.rag.llama_parse_util._get_parser", return_value=mock_parser):
            result = parse_document(test_pdf)

            assert result == "# Document Title\n\nContent here."
            mock_parser.load_data.assert_called_once()

    def test_parse_document_empty_result(self, test_pdf):
        """Test parsing when no content is extracted."""
        from src.rag.llama_parse_util import parse_document

        mock_parser = MagicMock()
        mock_parser.load_data.return_value = []

        with patch("src.rag.llama_parse_util._get_parser", return_value=mock_parser):
            result = parse_document(test_pdf)

            assert result == ""

    def test_parse_document_multiple_pages(self, test_pdf):
        """Test parsing document with multiple pages."""
        from src.rag.llama_parse_util import parse_document

        mock_parser = MagicMock()
        mock_parser.load_data.return_value = [
            MagicMock(text="Page 1 content"),
            MagicMock(text="Page 2 content"),
            MagicMock(text="Page 3 content"),
        ]

        with patch("src.rag.llama_parse_util._get_parser", return_value=mock_parser):
            result = parse_document(test_pdf)

            assert "Page 1 content" in result
            assert "Page 2 content" in result
            assert "Page 3 content" in result
            assert "---" in result  # Page separator

    def test_parse_document_with_output_dir(self, mock_parser, test_pdf):
        """Test parsing with output directory saves to storage."""
        from src.rag.llama_parse_util import parse_document

        with patch("src.rag.llama_parse_util._get_parser", return_value=mock_parser), \
             patch("src.rag.llama_parse_util._save_markdown") as mock_save:

            result = parse_document(test_pdf, output_dir="parsed/docs")

            mock_save.assert_called_once()
            assert result == "# Document Title\n\nContent here."

    def test_parse_document_agent_mode(self, test_pdf):
        """Test parsing with agent mode enabled."""
        from src.rag.llama_parse_util import parse_document

        mock_agent_parser = MagicMock()
        mock_agent_parser.load_data.return_value = [
            MagicMock(text="Agent parsed content"),
        ]

        with patch("src.rag.llama_parse_util._get_agent_parser", return_value=mock_agent_parser) as mock_get_agent:
            result = parse_document(test_pdf, use_agent_mode=True)

            mock_get_agent.assert_called_once_with("normal")
            assert result == "Agent parsed content"

    def test_parse_document_high_complexity(self, test_pdf):
        """Test parsing with high complexity mode."""
        from src.rag.llama_parse_util import parse_document

        mock_agent_parser = MagicMock()
        mock_agent_parser.load_data.return_value = [
            MagicMock(text="High complexity content"),
        ]

        with patch("src.rag.llama_parse_util._get_agent_parser", return_value=mock_agent_parser) as mock_get_agent:
            result = parse_document(test_pdf, use_agent_mode=True, complexity="high")

            mock_get_agent.assert_called_once_with("high")
            assert result == "High complexity content"

    def test_parse_document_exception_handling(self, test_pdf):
        """Test that exceptions are re-raised after logging."""
        from src.rag.llama_parse_util import parse_document

        mock_parser = MagicMock()
        mock_parser.load_data.side_effect = Exception("Parse error")

        with patch("src.rag.llama_parse_util._get_parser", return_value=mock_parser):
            with pytest.raises(Exception) as exc_info:
                parse_document(test_pdf)

            assert "Parse error" in str(exc_info.value)

    def test_parse_document_file_not_found(self):
        """Test parsing non-existent file raises FileNotFoundError."""
        from src.rag.llama_parse_util import parse_document

        with pytest.raises(FileNotFoundError):
            parse_document("/nonexistent/file.pdf")


class TestParseDocuments:
    """Tests for parse_documents function."""

    @pytest.fixture
    def test_files(self, tmp_path):
        """Create multiple test files."""
        files = []
        for i in range(3):
            f = tmp_path / f"doc{i}.pdf"
            f.write_bytes(b"%PDF-1.4 content")
            files.append(str(f))
        return files

    def test_parse_documents_multiple_files(self, test_files):
        """Test parsing multiple documents."""
        from src.rag.llama_parse_util import parse_documents

        mock_parser = MagicMock()
        mock_parser.load_data.side_effect = [
            [MagicMock(text=f"Content {i}")] for i in range(3)
        ]

        with patch("src.rag.llama_parse_util._get_parser", return_value=mock_parser):
            results = parse_documents(test_files)

            assert len(results) == 3
            assert results[0] == "Content 0"
            assert results[1] == "Content 1"
            assert results[2] == "Content 2"

    def test_parse_documents_with_output_dir(self, test_files):
        """Test parsing multiple documents with output directory."""
        from src.rag.llama_parse_util import parse_documents

        mock_parser = MagicMock()
        mock_parser.load_data.return_value = [MagicMock(text="Content")]

        with patch("src.rag.llama_parse_util._get_parser", return_value=mock_parser), \
             patch("src.rag.llama_parse_util._save_markdown") as mock_save:

            results = parse_documents(test_files, output_dir="parsed/docs")

            assert len(results) == 3
            assert mock_save.call_count == 3

    def test_parse_documents_validation_first(self, tmp_path):
        """Test that all files are validated before parsing starts."""
        from src.rag.llama_parse_util import parse_documents

        valid_file = tmp_path / "valid.pdf"
        valid_file.write_bytes(b"%PDF-1.4 content")

        # Mix of valid and invalid files
        with pytest.raises(FileNotFoundError):
            parse_documents([str(valid_file), "/nonexistent/file.pdf"])


class TestGetSupportedExtensions:
    """Tests for get_supported_extensions function."""

    def test_get_supported_extensions_returns_sorted_list(self):
        """Test that extensions are returned as a sorted list."""
        from src.rag.llama_parse_util import get_supported_extensions

        extensions = get_supported_extensions()

        assert isinstance(extensions, list)
        assert extensions == sorted(extensions)

    def test_supported_extensions_includes_pdf(self):
        """Test that .pdf is in supported extensions."""
        from src.rag.llama_parse_util import get_supported_extensions

        extensions = get_supported_extensions()

        assert ".pdf" in extensions

    def test_supported_extensions_includes_docx(self):
        """Test that .docx is in supported extensions."""
        from src.rag.llama_parse_util import get_supported_extensions

        extensions = get_supported_extensions()

        assert ".docx" in extensions

    def test_supported_extensions_includes_images(self):
        """Test that image formats are in supported extensions."""
        from src.rag.llama_parse_util import get_supported_extensions

        extensions = get_supported_extensions()

        assert ".jpg" in extensions
        assert ".png" in extensions
        assert ".jpeg" in extensions


class TestSaveMarkdown:
    """Tests for markdown saving functionality."""

    @pytest.mark.asyncio
    async def test_save_markdown_async(self, tmp_path):
        """Test async markdown saving to storage."""
        from src.rag.llama_parse_util import _save_markdown_async

        mock_storage = MagicMock()
        mock_storage.save = AsyncMock(return_value="gs://bucket/parsed/test.md")

        with patch("src.storage.get_storage", return_value=mock_storage):
            result = await _save_markdown_async(
                "# Content",
                Path("/path/to/test.pdf"),
                "parsed"
            )

            assert result == "gs://bucket/parsed/test.md"
            mock_storage.save.assert_called_once_with(
                "# Content",
                "test.md",
                directory="parsed"
            )
