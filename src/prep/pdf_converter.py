import time
import typing
import logging
import pathlib

from docling import document_converter
from docling.datamodel import settings
from docling.datamodel import document
from docling.datamodel import base_models
from docling.datamodel import pipeline_options as po


class PDFConverter:

    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)
    

    def pdf_to_txt(self, input_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
        logging.basicConfig(level=logging.INFO)
        input_doc_paths = list(input_dir.glob("*.pdf"))

        # Turn off inline debug visualizations:
        settings.settings.debug.visualize_layout = False
        settings.settings.debug.visualize_ocr = False
        settings.settings.debug.visualize_tables = False
        settings.settings.debug.visualize_cells = False

        # Converter Settings
        pipeline_options = po.PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        
        doc_converter = document_converter.DocumentConverter(
            format_options={
                base_models.InputFormat.PDF: document_converter.PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        start_time = time.time()

        conv_results = doc_converter.convert_all(
            input_doc_paths,
            raises_on_error=False,
            # to let conversion run through all and examine results at the end
        )        
        success_count, partial_success_count, failure_count = self.__export_documents(
            conv_results, output_dir=output_dir
        )

        end_time = time.time() - start_time

        self._log.info(f"Document conversion complete in {end_time:.2f} seconds.")

        if failure_count > 0:
            raise RuntimeError(
                f"The example failed converting {failure_count} on {len(input_doc_paths)}."
            )
    

    def __export_documents(self, conv_results: typing.Iterable[document.ConversionResult], output_dir: pathlib.Path):
        output_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        failure_count = 0
        partial_success_count = 0

        for conv_res in conv_results:
            if conv_res.status == base_models.ConversionStatus.SUCCESS:
                success_count += 1
                doc_filename = conv_res.input.file.stem

                # Export Docling document format to markdown:
                with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
                    fp.write(conv_res.document.export_to_markdown())

                # Export Docling document format to text:
                with (output_dir / f"{doc_filename}.txt").open("w", encoding="utf-8") as fp:
                    fp.write(conv_res.document.export_to_markdown(strict_text=True))

            elif conv_res.status == base_models.ConversionStatus.PARTIAL_SUCCESS:
                self._log.info(
                    f"Document {conv_res.input.file} was partially converted with the following errors:"
                )
                for item in conv_res.errors:
                    self._log.info(f"\t{item.error_message}")
                partial_success_count += 1
            else:
                self._log.info(f"Document {conv_res.input.file} failed to convert.")
                failure_count += 1

        self._log.info(
            f"Processed {success_count + partial_success_count + failure_count} docs, "
            f"of which {failure_count} failed "
            f"and {partial_success_count} were partially converted."
        )
        return success_count, partial_success_count, failure_count
        