"""Class for converting PDF documents to text format."""

import time
import logging
from pathlib import Path
from typing import Iterable, Tuple
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.settings import settings
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat


class PDFConverter:
    """
    A class for converting PDF documents to text format using the docling library.
    """

    def __init__(self) -> None:
        """
        Initializes the PDFConverter class.
        """
        self._log = logging.getLogger(self.__class__.__name__)

    def pdf_to_txt(self, input_dir: Path, output_dir: Path) -> None:
        """
        Converts all PDF files in the input directory to text files in the output 
        directory.

        This function uses the `docling` library to convert PDF documents to plain text.
        It disables certain debug visualizations, sets conversion settings, processes
        each PDF document, and saves the results to text (.txt and .md) files in the 
        output directory.

        Args:
            input_dir (Path): The path to the directory containing the PDF files to 
                convert.
            output_dir (Path): The path to the directory where the converted text files 
                will be saved.

        Returns:
            None

        Raises:
            RuntimeError: If the example fails to convert any of the input PDF files.
        """
        logging.basicConfig(level=logging.INFO)
        input_doc_paths = list(input_dir.glob("*.pdf"))

        # Turn off inline debug visualizations:
        settings.debug.visualize_layout = False
        settings.debug.visualize_ocr = False
        settings.debug.visualize_tables = False
        settings.debug.visualize_cells = False

        # Converter Settings
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        start_time = time.time()

        conv_results = doc_converter.convert_all(
            input_doc_paths,
            raises_on_error=False,
            # to let conversion run through all
            # and examine results at the end
        )
        success_count, partial_success_count, failure_count = self._export_documents(
            conv_results, output_dir=output_dir
        )

        end_time = time.time() - start_time

        self._log.info(f"Document conversion complete in {end_time:.2f} seconds.")
        if failure_count > 0:
            raise RuntimeError(
                f"The example failed converting {failure_count} on "
                f"{len(input_doc_paths)}."
            )

    def _export_documents(
        self, conv_results: Iterable[ConversionResult], output_dir: Path
    ) -> Tuple[int, int, int]:
        """
        Exports the converted documents to text (.txt and .md) files.

        This function iterates through the conversion results, extracts successful 
        documents, and saves them to text files in the specified output directory.
        It also logs information about any conversion errors or partial successes.

        Args:
            conv_results (Iterable[ConversionResult]): An iterable of ConversionResult 
                objects representing the results of the document conversions.
            output_dir (Path): The path to the directory where the converted documents 
                should be saved.

        Returns:
            result (Tuple[int, int, int]): A tuple containing the number of successful 
                conversions, partial success conversions, and failed conversions.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0
        failure_count = 0
        partial_success_count = 0

        for conv_res in conv_results:
            if conv_res.status == ConversionStatus.SUCCESS:
                success_count += 1
                doc_filename = conv_res.input.file.stem

                # Export Docling document format to text (.txt and .md):
                with (output_dir / f"{doc_filename}.txt").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(conv_res.document.export_to_text(strict_text=True))
                with (output_dir / f"{doc_filename}.md").open(
                    "w", encoding="utf-8"
                ) as fp:
                    fp.write(conv_res.document.export_to_markdown(strict_text=True))
            elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
                self._log.info(
                    f"Document {conv_res.input.file} was partially converted"
                    f" with the following errors:"
                )
                for item in conv_res.errors:
                    self._log.info(f"\t{item.error_message}")
                partial_success_count += 1
            else:
                self._log.info(f"Document {conv_res.input.file} failed to convert.")
                failure_count += 1

        self._log.info(
            f"Processed {success_count + partial_success_count + failure_count} "
            f"docs, of which {failure_count} failed and "
            f"{partial_success_count} were partially converted."
        )

        return success_count, partial_success_count, failure_count
