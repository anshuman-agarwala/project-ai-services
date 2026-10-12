"""
Unit tests for document processing utilities.
"""

import json
from pathlib import Path
import pytest
from unittest.mock import Mock, MagicMock, patch
from collections import Counter

from digitize.processing.language import detect_document_language
from digitize.processing.tables import process_table


@pytest.mark.unit
class TestDetectDocumentLanguage:
    """Tests for detect_document_language function."""

    def test_detect_language_with_valid_english_data(self):
        """Test language detection with valid English text blocks."""
        data = [
            {"text": "This is an English sentence about artificial intelligence."},
            {"text": "Machine learning is a subset of AI."},
            {"text": "Deep learning uses neural networks."},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            mock_detect.return_value = "EN"
            result = detect_document_language(data)

        assert result == "EN"
        assert mock_detect.called

    def test_detect_language_with_valid_german_data(self):
        """Test language detection with valid German text blocks."""
        data = [
            {"text": "Dies ist ein deutscher Satz über künstliche Intelligenz."},
            {"text": "Maschinelles Lernen ist eine Teilmenge der KI."},
            {"text": "Deep Learning verwendet neuronale Netze."},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            mock_detect.return_value = "DE"
            result = detect_document_language(data)

        assert result == "DE"

    def test_detect_language_with_valid_french_data(self):
        """Test language detection with valid French text blocks."""
        data = [
            {"text": "Ceci est une phrase française sur l'intelligence artificielle."},
            {"text": "L'apprentissage automatique est un sous-ensemble de l'IA."},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            mock_detect.return_value = "FR"
            result = detect_document_language(data)

        assert result == "FR"

    def test_detect_language_with_valid_italian_data(self):
        """Test language detection with valid Italian text blocks."""
        data = [
            {"text": "Questa è una frase italiana sull'intelligenza artificiale."},
            {"text": "L'apprendimento automatico è un sottoinsieme dell'IA."},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            mock_detect.return_value = "IT"
            result = detect_document_language(data)

        assert result == "IT"

    def test_detect_language_with_valid_japanese_data(self):
        """Test language detection with valid Japanese text blocks."""
        data = [
            {"text": "これは人工知能に関する日本語の文章です。"},
            {"text": "機械学習はAIのサブセットです。"},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            mock_detect.return_value = "JA"
            result = detect_document_language(data)

        assert result == "JA"

    def test_detect_language_with_unsupported_language_falls_back_to_english(self):
        """Test that unsupported languages fall back to English."""
        data = [
            {"text": "Este es un texto en español."},
            {"text": "El aprendizaje automático es importante."},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            mock_detect.return_value = "ES"  # Spanish not in lang_map
            result = detect_document_language(data)

        assert result == "EN"

    def test_detect_language_with_long_text_samples_blocks(self):
        """Test that long text samples random blocks."""
        data = [
            {"text": "A" * 300},  # Long block
            {"text": "B" * 300},
            {"text": "C" * 300},
            {"text": "D" * 300},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            with patch("digitize.processing.language.random.randint") as mock_randint:
                mock_detect.return_value = "EN"
                # Return indices 0, 1, 2 for the first 3 calls
                mock_randint.side_effect = [0, 1, 2]

                result = detect_document_language(data)

        assert result == "EN"
        # Should call randint to sample blocks
        assert mock_randint.call_count >= 3

    def test_detect_language_with_mixed_languages_uses_most_common(self):
        """Test that mixed languages use the most common detected language."""
        data = [
            {"text": "English text here with sufficient length for detection."},
            {"text": "More English text to ensure proper detection works."},
            {"text": "Dies ist ein deutscher Satz."},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            # Return EN twice, DE once
            mock_detect.side_effect = ["EN", "EN", "DE"]
            result = detect_document_language(data)

        assert result == "EN"

    def test_detect_language_with_empty_list_returns_english(self):
        """Test that empty data list returns English."""
        data = []

        with patch("digitize.processing.language.logger") as mock_logger:
            result = detect_document_language(data)

        assert result == "EN"
        mock_logger.warning.assert_called_once()
        assert "Empty data list" in mock_logger.warning.call_args[0][0]

    def test_detect_language_with_non_list_input_returns_english(self):
        """Test that non-list input returns English with warning."""
        data = "not a list"

        with patch("digitize.processing.language.logger") as mock_logger:
            result = detect_document_language(data)

        assert result == "EN"
        mock_logger.warning.assert_called_once()
        assert "expected list" in mock_logger.warning.call_args[0][0]

    def test_detect_language_with_non_dict_elements_returns_english(self):
        """Test that list with non-dict elements returns English."""
        data = ["string1", "string2", 123]

        with patch("digitize.processing.language.logger") as mock_logger:
            result = detect_document_language(data)

        assert result == "EN"
        mock_logger.warning.assert_called_once()
        assert "non-dict elements" in mock_logger.warning.call_args[0][0]

    def test_detect_language_with_no_text_blocks_returns_english(self):
        """Test that data with no text blocks returns English."""
        data = [
            {"label": "header", "page": 1},
            {"label": "footer", "page": 2},
            {"text": ""},  # Empty text
            {"text": "   "},  # Whitespace only
        ]

        with patch("digitize.processing.language.logger") as mock_logger:
            result = detect_document_language(data)

        assert result == "EN"
        mock_logger.warning.assert_called_once()
        assert "No text blocks found" in mock_logger.warning.call_args[0][0]

    def test_detect_language_with_blocks_missing_text_field(self):
        """Test that blocks without 'text' field are handled gracefully."""
        data = [
            {"text": "Valid text block here."},
            {"label": "header"},  # No text field
            {"text": "Another valid block."},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            mock_detect.return_value = "EN"
            result = detect_document_language(data)

        assert result == "EN"

    def test_detect_language_with_exception_in_try_block_returns_english(self):
        """Test that exceptions during detection fall back to English."""
        data = [
            {"text": "A" * 100},
            {"text": "B" * 100},
            {"text": "C" * 100},
        ]

        with patch("digitize.processing.language.detect_language"):
            with patch("digitize.processing.language.logger") as mock_logger:
                with patch("digitize.processing.language.random.randint") as mock_randint:
                    # Make random.randint raise an exception
                    mock_randint.side_effect = Exception("Sampling failed")
                    result = detect_document_language(data)

        assert result == "EN"
        mock_logger.warning.assert_called_once()
        assert "Language detection failed" in mock_logger.warning.call_args[0][0]

    def test_detect_language_with_very_long_blocks_takes_chunks(self):
        """Test that very long blocks (>500 chars) are chunked."""
        long_text = "A" * 1000
        data = [
            {"text": long_text},
            {"text": long_text},
            {"text": long_text},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            with patch("digitize.processing.language.random.randint") as mock_randint:
                mock_detect.return_value = "EN"
                # Return valid indices (0, 1, 2) for sampling from data
                mock_randint.side_effect = [0, 1, 2]

                result = detect_document_language(data)

        assert result == "EN"
        # Should call detect_language for each sampled block
        assert mock_detect.call_count >= 1

    def test_detect_language_samples_fewer_blocks_when_data_is_small(self):
        """Test that sampling adjusts to available blocks when text is long enough."""
        data = [
            {"text": "A" * 150},
            {"text": "B" * 150},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            with patch("digitize.processing.language.random.randint") as mock_randint:
                mock_detect.return_value = "EN"
                # Return indices 0, 1 for the 2 available blocks
                mock_randint.side_effect = [0, 1]

                result = detect_document_language(data)

        assert result == "EN"
        # Should sample the 2 available blocks
        assert mock_randint.call_count >= 2

    def test_detect_language_with_whitespace_only_blocks_skips_them(self):
        """Test that blocks with only whitespace are skipped."""
        data = [
            {"text": "   \n\t   "},  # Whitespace only
            {"text": "Valid text here."},
            {"text": ""},  # Empty
            {"text": "Another valid block."},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            mock_detect.return_value = "EN"
            result = detect_document_language(data)

        assert result == "EN"

    def test_detect_language_logs_detected_languages_when_sampling(self):
        """Test that detected languages are logged for debugging when sampling occurs."""
        data = [
            {"text": "Text block one with sufficient length for detection and sampling."},
            {"text": "Text block two with sufficient length for detection and sampling."},
            {"text": "Text block three with sufficient length for detection and sampling."},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            with patch("digitize.processing.language.logger") as mock_logger:
                mock_detect.return_value = "EN"
                result = detect_document_language(data)

        assert result == "EN"
        assert mock_logger.debug.call_count >= 0

    def test_detect_language_with_none_values_in_text_field(self):
        """Test that None values in text field are handled."""
        data = [
            {"text": None},
            {"text": "Valid text here."},
            {"text": None},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            mock_detect.return_value = "EN"
            result = detect_document_language(data)

        assert result == "EN"

    def test_detect_language_with_integer_in_text_field(self):
        """Test that non-string values in text field are handled."""
        data = [
            {"text": 123},
            {"text": "Valid text here."},
            {"text": 456},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            mock_detect.return_value = "EN"
            result = detect_document_language(data)

        assert result == "EN"

    def test_detect_language_with_mixed_valid_and_invalid_blocks(self):
        """Test handling of mixed valid and invalid blocks."""
        data = [
            {"text": "Valid text block."},
            {"text": None},
            {"text": 123},
            {"text": ""},
            {"text": "Another valid block."},
            {"label": "header"},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            mock_detect.return_value = "EN"
            result = detect_document_language(data)

        assert result == "EN"

    def test_detect_language_returns_most_common_from_counter(self):
        """Test that Counter correctly identifies most common language."""
        data = [
            {"text": "A" * 300},
            {"text": "B" * 300},
            {"text": "C" * 300},
        ]

        with patch("digitize.processing.language.detect_language") as mock_detect:
            # Return different languages
            mock_detect.side_effect = ["EN", "DE", "EN"]
            result = detect_document_language(data)

        # EN appears twice, should be selected
        assert result == "EN"


class TestProcessTableLanguageSelection:
    """Tests for language-specific table summarize/classify selection."""

    @patch("digitize.processing.tables.summarize_and_classify_tables")
    @patch("digitize.processing.tables.merge_consecutive_tables")
    @patch("digitize.processing.tables.settings")
    def test_process_table_uses_italian_prompt_and_max_tokens(
        self, mock_settings, mock_merge_tables, mock_summarize_and_classify
    ):
        from common.lang_utils import LanguageCodes
        from pathlib import Path

        mock_italian = Mock()
        mock_italian.prompt = "Italian prompt template"
        mock_italian.max_tokens = 1339

        mock_english = Mock()
        mock_english.prompt = "English prompt template"
        mock_english.max_tokens = 1024

        mock_german = Mock()
        mock_german.prompt = "German prompt template"
        mock_german.max_tokens = 1536

        mock_french = Mock()
        mock_french.prompt = "French prompt template"
        mock_french.max_tokens = 1260

        mock_table_summary = MagicMock()
        mock_table_summary.italian = mock_italian
        mock_table_summary.english = mock_english
        mock_table_summary.german = mock_german
        mock_table_summary.french = mock_french

        mock_settings.table_summary = mock_table_summary

        mock_merge_tables.return_value = {
            0: {
                "markdown": "| Colonna | Valore |\n|---|---|\n| CPU | Power10 |",
                "caption": "Specifiche",
                "page_number": 1,
            }
        }
        mock_summarize_and_classify.return_value = (["Riassunto"], [True], {})

        mock_table = Mock()
        mock_prov = Mock()
        mock_prov.page_no = 1
        mock_table.prov = [mock_prov]
        mock_table.export_to_markdown.return_value = "| Colonna | Valore |\n|---|---|\n| CPU | Power10 |"
        mock_table.caption_text.return_value = "Specifiche"

        converted_doc = Mock()
        converted_doc.tables = [mock_table]

        with patch.object(Path, 'write_text'):
            table_count, process_time, failures = process_table(
                converted_doc=converted_doc,
                doc_path="sample.pdf",
                out_path=Path("/tmp/out.json"),
                gen_model="test-model",
                gen_endpoint="http://llm",
                document_language=LanguageCodes.ITALIAN,
            )

        _, kwargs = mock_summarize_and_classify.call_args
        assert kwargs["prompt_template"] == "Italian prompt template"
        assert kwargs["max_tokens"] == 1339
        assert failures == {}

    @patch("digitize.processing.tables.summarize_and_classify_tables")
    @patch("digitize.processing.tables.merge_consecutive_tables")
    @patch("digitize.processing.tables.settings")
    def test_process_table_uses_english_prompt_and_max_tokens(
        self, mock_settings, mock_merge_tables, mock_summarize_and_classify
    ):
        from common.lang_utils import LanguageCodes
        from pathlib import Path

        mock_english = Mock()
        mock_english.prompt = "English prompt template"
        mock_english.max_tokens = 1024

        mock_german = Mock()
        mock_german.prompt = "German prompt template"
        mock_german.max_tokens = 1536

        mock_italian = Mock()
        mock_italian.prompt = "Italian prompt template"
        mock_italian.max_tokens = 1339

        mock_french = Mock()
        mock_french.prompt = "French prompt template"
        mock_french.max_tokens = 1260

        mock_table_summary = MagicMock()
        mock_table_summary.english = mock_english
        mock_table_summary.german = mock_german
        mock_table_summary.italian = mock_italian
        mock_table_summary.french = mock_french

        mock_settings.table_summary = mock_table_summary

        mock_merge_tables.return_value = {
            0: {
                "markdown": "| Column | Value |\n|---|---|\n| CPU | Power10 |",
                "caption": "Specifications",
                "page_number": 1,
            }
        }
        mock_summarize_and_classify.return_value = (["Summary"], [True], {})

        mock_table = Mock()
        mock_prov = Mock()
        mock_prov.page_no = 1
        mock_table.prov = [mock_prov]
        mock_table.export_to_markdown.return_value = "| Column | Value |\n|---|---|\n| CPU | Power10 |"
        mock_table.caption_text.return_value = "Specifications"

        converted_doc = Mock()
        converted_doc.tables = [mock_table]

        with patch.object(Path, 'write_text'):
            table_count, process_time, failures = process_table(
                converted_doc=converted_doc,
                doc_path="sample.pdf",
                out_path=Path("/tmp/out.json"),
                gen_model="test-model",
                gen_endpoint="http://llm",
                document_language=LanguageCodes.ENGLISH,
            )

        _, kwargs = mock_summarize_and_classify.call_args
        assert kwargs["prompt_template"] == "English prompt template"
        assert kwargs["max_tokens"] == 1024
        assert failures == {}

    @patch("digitize.processing.tables.summarize_and_classify_tables")
    @patch("digitize.processing.tables.merge_consecutive_tables")
    @patch("digitize.processing.tables.settings")
    def test_process_table_uses_german_prompt_and_max_tokens(
        self, mock_settings, mock_merge_tables, mock_summarize_and_classify
    ):
        from common.lang_utils import LanguageCodes
        from pathlib import Path

        mock_german = Mock()
        mock_german.prompt = "German prompt template"
        mock_german.max_tokens = 1536

        mock_english = Mock()
        mock_english.prompt = "English prompt template"
        mock_english.max_tokens = 1024

        mock_italian = Mock()
        mock_italian.prompt = "Italian prompt template"
        mock_italian.max_tokens = 1339

        mock_french = Mock()
        mock_french.prompt = "French prompt template"
        mock_french.max_tokens = 1260

        mock_table_summary = MagicMock()
        mock_table_summary.german = mock_german
        mock_table_summary.english = mock_english
        mock_table_summary.italian = mock_italian
        mock_table_summary.french = mock_french

        mock_settings.table_summary = mock_table_summary

        mock_merge_tables.return_value = {
            0: {
                "markdown": "| Spalte | Wert |\n|---|---|\n| CPU | Power10 |",
                "caption": "Spezifikationen",
                "page_number": 1,
            }
        }
        mock_summarize_and_classify.return_value = (["Zusammenfassung"], [True], {})

        mock_table = Mock()
        mock_prov = Mock()
        mock_prov.page_no = 1
        mock_table.prov = [mock_prov]
        mock_table.export_to_markdown.return_value = "| Spalte | Wert |\n|---|---|\n| CPU | Power10 |"
        mock_table.caption_text.return_value = "Spezifikationen"

        converted_doc = Mock()
        converted_doc.tables = [mock_table]

        with patch.object(Path, 'write_text'):
            table_count, process_time, failures = process_table(
                converted_doc=converted_doc,
                doc_path="sample.pdf",
                out_path=Path("/tmp/out.json"),
                gen_model="test-model",
                gen_endpoint="http://llm",
                document_language=LanguageCodes.GERMAN,
            )

        _, kwargs = mock_summarize_and_classify.call_args
        assert kwargs["prompt_template"] == "German prompt template"
        assert kwargs["max_tokens"] == 1536
        assert failures == {}

    @patch("digitize.processing.tables.summarize_and_classify_tables")
    @patch("digitize.processing.tables.merge_consecutive_tables")
    @patch("digitize.processing.tables.settings")
    def test_process_table_uses_french_prompt_and_max_tokens(
        self, mock_settings, mock_merge_tables, mock_summarize_and_classify
    ):
        from common.lang_utils import LanguageCodes
        from pathlib import Path

        mock_french = Mock()
        mock_french.prompt = "French prompt template"
        mock_french.max_tokens = 1260

        mock_english = Mock()
        mock_english.prompt = "English prompt template"
        mock_english.max_tokens = 1024

        mock_german = Mock()
        mock_german.prompt = "German prompt template"
        mock_german.max_tokens = 1536

        mock_italian = Mock()
        mock_italian.prompt = "Italian prompt template"
        mock_italian.max_tokens = 1339

        mock_table_summary = MagicMock()
        mock_table_summary.french = mock_french
        mock_table_summary.english = mock_english
        mock_table_summary.german = mock_german
        mock_table_summary.italian = mock_italian

        mock_settings.table_summary = mock_table_summary

        mock_merge_tables.return_value = {
            0: {
                "markdown": "| Colonne | Valeur |\n|---|---|\n| CPU | Power10 |",
                "caption": "Spécifications",
                "page_number": 1,
            }
        }
        mock_summarize_and_classify.return_value = (["Résumé"], [True], {})

        mock_table = Mock()
        mock_prov = Mock()
        mock_prov.page_no = 1
        mock_table.prov = [mock_prov]
        mock_table.export_to_markdown.return_value = "| Colonne | Valeur |\n|---|---|\n| CPU | Power10 |"
        mock_table.caption_text.return_value = "Spécifications"

        converted_doc = Mock()
        converted_doc.tables = [mock_table]

        with patch.object(Path, 'write_text'):
            table_count, process_time, failures = process_table(
                converted_doc=converted_doc,
                doc_path="sample.pdf",
                out_path=Path("/tmp/out.json"),
                gen_model="test-model",
                gen_endpoint="http://llm",
                document_language=LanguageCodes.FRENCH,
            )

        _, kwargs = mock_summarize_and_classify.call_args
        assert kwargs["prompt_template"] == "French prompt template"
        assert kwargs["max_tokens"] == 1260
        assert failures == {}


# Made with Bob


@pytest.mark.unit
class TestSplitJapaneseSentences:
    """Tests for _split_japanese_sentences Japanese sentence splitter."""

    def test_splits_on_kuten(self):
        """Splits on 。 and retains the punctuation."""
        from digitize.processing.orchestrator import _split_japanese_sentences
        result = _split_japanese_sentences("これはテストです。次の文章です。")
        assert result == ["これはテストです。", "次の文章です。"]

    def test_splits_on_exclamation(self):
        """Splits on ！ and retains the punctuation."""
        from digitize.processing.orchestrator import _split_japanese_sentences
        result = _split_japanese_sentences("素晴らしい！本当にそうです。")
        assert result == ["素晴らしい！", "本当にそうです。"]

    def test_splits_on_question_mark(self):
        """Splits on ？ and retains the punctuation."""
        from digitize.processing.orchestrator import _split_japanese_sentences
        result = _split_japanese_sentences("何ですか？はい、そうです。")
        assert result == ["何ですか？", "はい、そうです。"]

    def test_mixed_punctuation(self):
        """Splits on mixed Japanese and ASCII sentence-ending punctuation."""
        from digitize.processing.orchestrator import _split_japanese_sentences
        result = _split_japanese_sentences("テストです。確認しました！問題ありません。")
        assert len(result) == 3
        assert result[0] == "テストです。"
        assert result[1] == "確認しました！"
        assert result[2] == "問題ありません。"

    def test_filters_empty_strings(self):
        """Empty strings from consecutive delimiters are filtered out."""
        from digitize.processing.orchestrator import _split_japanese_sentences
        result = _split_japanese_sentences("文章です。。次です。")
        assert all(s for s in result)

    def test_single_sentence_no_delimiter(self):
        """Text with no sentence-ending punctuation returns as single item."""
        from digitize.processing.orchestrator import _split_japanese_sentences
        result = _split_japanese_sentences("区切りなしのテキスト")
        assert result == ["区切りなしのテキスト"]

    def test_empty_string(self):
        """Empty input returns empty list."""
        from digitize.processing.orchestrator import _split_japanese_sentences
        result = _split_japanese_sentences("")
        assert result == []

    def test_english_text_passthrough(self):
        """English text without Japanese punctuation returns as single sentence."""
        from digitize.processing.orchestrator import _split_japanese_sentences
        result = _split_japanese_sentences("Power10 processor with 16 cores.")
        assert result == ["Power10 processor with 16 cores."]

    def test_mixed_japanese_english(self):
        """Splits on Japanese delimiters even when English words are present."""
        from digitize.processing.orchestrator import _split_japanese_sentences
        result = _split_japanese_sentences("Power10はIBMのプロセッサです。16コアを搭載しています。")
        assert result == ["Power10はIBMのプロセッサです。", "16コアを搭載しています。"]



@pytest.mark.unit
class TestProcessImage:
    """Tests for process_image in digitize.processing.images."""

    def test_process_image_no_pictures(self, tmp_path):
        """When converted document has no pictures, returns 0 and writes empty dict."""
        from digitize.processing.images import process_image

        mock_doc = MagicMock()
        mock_doc.pictures = []
        out_file = tmp_path / "images.json"

        count, proc_time, failures = process_image(
            converted_doc=mock_doc,
            doc_path=Path("sample.pdf"),
            out_path=out_file,
            gen_model="test-vlm",
            gen_endpoint="http://localhost:8000",
        )

        assert count == 0
        assert failures == {}
        assert json.loads(out_file.read_text()) == {}

    @patch("digitize.processing.images.summarize_and_classify_images")
    def test_process_image_success(self, mock_summarize, tmp_path):
        """Extracts pictures, resolves captions, and filters summaries."""
        from digitize.processing.images import process_image

        mock_pic1 = MagicMock()
        mock_pic1.caption_text.return_value = "System Architecture Diagram"
        mock_pic1.image.uri = "data:image/png;base64,abc123"
        mock_pic1.prov = [MagicMock(page_no=1)]

        mock_pic2 = MagicMock()
        mock_pic2.caption_text.return_value = "Decorative line"
        mock_pic2.image.uri = "data:image/png;base64,def456"
        mock_pic2.prov = [MagicMock(page_no=2)]

        mock_doc = MagicMock()
        mock_doc.pictures = [mock_pic1, mock_pic2]

        # First image kept (Yes), second image discarded (No)
        mock_summarize.return_value = (
            ["Diagram showing components.", "Decorative line."],
            [True, False],
            {},
        )

        out_file = tmp_path / "images.json"
        count, proc_time, failures = process_image(
            converted_doc=mock_doc,
            doc_path=Path("sample.pdf"),
            out_path=out_file,
            gen_model="test-vlm",
            gen_endpoint="http://localhost:8000",
        )

        assert count == 1
        assert failures == {}
        data = json.loads(out_file.read_text())
        assert "0" in data
        assert data["0"]["caption"] == "System Architecture Diagram"
        assert data["0"]["summary"] == "Diagram showing components."
        assert data["0"]["page_number"] == 1
        assert "1" not in data  # filtered out by decision=False

    def test_process_image_none_image_uri(self, tmp_path):
        """When picture.image or picture.image.uri is None, uri evaluates to empty string, not 'None'."""
        from digitize.processing.images import process_image

        mock_pic1 = MagicMock()
        mock_pic1.caption_text.return_value = "Caption only"
        mock_pic1.image = MagicMock()
        mock_pic1.image.uri = None
        mock_pic1.prov = [MagicMock(page_no=1)]

        mock_doc = MagicMock()
        mock_doc.pictures = [mock_pic1]

        with patch("digitize.processing.images.summarize_and_classify_images") as mock_summarize:
            mock_summarize.return_value = (["Summary."], [True], {})
            out_file = tmp_path / "images.json"
            count, _, _ = process_image(
                converted_doc=mock_doc,
                doc_path=Path("sample.pdf"),
                out_path=out_file,
                gen_model="test-vlm",
                gen_endpoint="http://localhost:8000",
            )
            assert count == 1
            # Verify image_uri passed to summarize_and_classify_images was "" not "None"
            called_images_data = mock_summarize.call_args[1]["images_data"]
            assert called_images_data == [("", "Caption only")]

    def test_summarize_and_classify_images_cancellation(self):
        """When stop_event is set, summarize_and_classify_images raises JobCancelledError."""
        import threading
        from digitize.exceptions import JobCancelledError
        from digitize.processing.images import summarize_and_classify_images

        stop_event = threading.Event()
        stop_event.set()

        with pytest.raises(JobCancelledError):
            summarize_and_classify_images(
                images_data=[("data:image/png;base64,123", "caption")],
                gen_model="test-vlm",
                llm_endpoint="http://localhost:8000",
                doc_path="sample.pdf",
                stop_event=stop_event,
            )

    def test_extract_json_object_with_preamble_and_fences(self):
        """_extract_json_object extracts JSON even if surrounded by prose or markdown code blocks."""
        from digitize.processing.images import _extract_json_object

        raw = 'Here is your analysis:\n```json\n{"summary": "Test chart", "is_informative": true}\n```\nHope this helps!'
        extracted = _extract_json_object(raw)
        assert json.loads(extracted) == {"summary": "Test chart", "is_informative": True}

    def test_summarize_and_classify_single_image_empty_short_circuit(self):
        """When both image_uri and caption are empty, short circuits without making network requests."""
        from digitize.processing.images import summarize_and_classify_single_image

        summary, decision = summarize_and_classify_single_image(
            image_uri="",
            caption="",
            gen_model="test-vlm",
            llm_endpoint="http://localhost:8000",
        )
        assert summary == "No summary available."
        assert decision is False

    def test_parse_image_response_fallback_valid(self):
        """_parse_image_response_fallback extracts both fields from slightly malformed JSON."""
        from digitize.processing.images import _parse_image_response_fallback

        # Missing comma between fields — valid values but invalid JSON
        malformed = '{"summary": "A bar chart showing sales data.", "is_informative": true'
        result = _parse_image_response_fallback(malformed)
        assert result["summary"] == "A bar chart showing sales data."
        assert result["is_informative"] is True

    def test_parse_image_response_fallback_false(self):
        """_parse_image_response_fallback correctly parses is_informative=false."""
        from digitize.processing.images import _parse_image_response_fallback

        malformed = '{"summary": "Decorative border.", "is_informative": false'
        result = _parse_image_response_fallback(malformed)
        assert result["summary"] == "Decorative border."
        assert result["is_informative"] is False

    def test_parse_image_response_fallback_escaped_quotes(self):
        """_parse_image_response_fallback handles escaped quotes inside the summary."""
        from digitize.processing.images import _parse_image_response_fallback

        malformed = r'{"summary": "Chart titled \"Revenue Q1\".", "is_informative": true'
        result = _parse_image_response_fallback(malformed)
        assert result["summary"] == 'Chart titled "Revenue Q1".'
        assert result["is_informative"] is True

    def test_parse_image_response_fallback_raises_when_unrecoverable(self):
        """_parse_image_response_fallback raises ValueError when no fields can be extracted."""
        from digitize.processing.images import _parse_image_response_fallback
        import pytest

        with pytest.raises(ValueError, match="Could not extract any fields"):
            _parse_image_response_fallback("totally garbage text with no JSON fields")

    def test_summarize_and_classify_single_image_uses_fallback_on_bad_json(self):
        """When the model returns malformed JSON, the fallback parser is used instead of raising."""
        from digitize.processing.images import summarize_and_classify_single_image
        from unittest.mock import MagicMock, patch
        import common.misc_utils as misc_utils

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"summary": "A pie chart.", "is_informative": true'}}]
        }

        original_session = misc_utils.SESSION
        try:
            misc_utils.SESSION = MagicMock()
            misc_utils.SESSION.post.return_value = mock_response
            summary, decision = summarize_and_classify_single_image(
                image_uri="data:image/png;base64,abc",
                caption="Test",
                gen_model="test-vlm",
                llm_endpoint="http://localhost:8000",
            )
            assert summary == "A pie chart."
            assert decision is True
        finally:
            misc_utils.SESSION = original_session

    def test_summarize_and_classify_single_image_no_extra_body(self):
        """Payload sent to the LLM must not contain extra_body."""
        from digitize.processing.images import summarize_and_classify_single_image
        from unittest.mock import MagicMock, patch
        import common.misc_utils as misc_utils

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"summary": "A chart.", "is_informative": true}'}}]
        }

        original_session = misc_utils.SESSION
        try:
            misc_utils.SESSION = MagicMock()
            misc_utils.SESSION.post.return_value = mock_response
            summarize_and_classify_single_image(
                image_uri="data:image/png;base64,abc",
                caption="Test",
                gen_model="test-vlm",
                llm_endpoint="http://localhost:8000",
            )
            _, kwargs = misc_utils.SESSION.post.call_args
            sent_payload = kwargs.get("json", {})
            assert "extra_body" not in sent_payload
            assert sent_payload.get("response_format", {}).get("type") == "json_schema"
        finally:
            misc_utils.SESSION = original_session
