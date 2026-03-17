import os
import io
import base64
from pathlib import Path
import logging

import ollama
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat

logger = logging.getLogger(__name__)

STEM_VISION_PROMPT = """You are analyzing a STEM document image. Provide a precise technical description:
- Mathematical equations or formulas: provide the exact LaTeX representation (e.g. $E = mc^2$)
- Charts and graphs: state axis labels, units, scale, data series names, and specific key values or inflection points
- Diagrams and schematics: label all components, describe connections, signal/data flow directions
- Tables: transcribe the structure, headers, and representative values
Be machine-readable and exact. Avoid subjective language."""


class DocumentProcessor:
    def __init__(self, vision_model="qwen2.5vl:7b"):
        self.vision_model = vision_model
        self._vision_model_loaded = False

        # Configure Docling to use CUDA for fast sequential processing on the GPU
        accelerator_options = AcceleratorOptions(num_threads=8, device=AcceleratorDevice.CUDA)
        pipeline_options = PdfPipelineOptions(generate_picture_images=True)
        pipeline_options.accelerator_options = accelerator_options

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def _ensure_vision_loaded(self):
        """
        Lazily warms up the vision model on first figure encounter.
        Avoids consuming ~8GB VRAM on documents that contain no images.
        """
        if not self._vision_model_loaded:
            logger.info(f"First figure detected — loading vision model: {self.vision_model}")
            try:
                ollama.generate(model=self.vision_model, prompt="Hello", keep_alive="30m")
            except Exception as e:
                logger.warning(f"Could not pre-warm vision model: {e}")
            self._vision_model_loaded = True

    def process_images(self, image_data: bytes):
        """
        Sends image data to qwen2.5vl via Ollama for STEM-focused description.
        """
        self._ensure_vision_loaded()
        try:
            encoded_image = base64.b64encode(image_data).decode('utf-8')
            response = ollama.generate(
                model=self.vision_model,
                prompt=STEM_VISION_PROMPT,
                images=[encoded_image]
            )
            # ollama.generate returns a GenerateResponse object, not a dict
            return response.response or '[Image description empty]'
        except Exception as e:
            logger.error(f"Error processing image with vision model: {e}")
            return "[Image description failed]"

    def _item_to_markdown(self, item) -> str:
        """
        Extracts text/markdown from a single non-figure Docling item.
        Tries the item-level export method first (preserves table formatting),
        then falls back to the raw text property.
        Note: item.export_to_markdown() is distinct from doc.export_to_markdown(item)
        — the latter is not valid Docling API.
        """
        # Item-level export works for tables and richly formatted items in Docling 2.x
        if hasattr(item, "export_to_markdown") and callable(item.export_to_markdown):
            try:
                md = item.export_to_markdown()
                if md:
                    return md
            except Exception:
                pass
        # Fallback: raw text content (TextItem, SectionHeaderItem, ListItem, FormulaItem)
        if hasattr(item, "text") and item.text:
            return item.text
        return ""

    def process_pdf(self, file_path: str):
        """
        Parses a PDF using Docling and assembles enriched markdown with vision
        descriptions injected inline at each figure's original document position.

        Inline injection is critical for RAG context quality: a figure describing
        a "Boltzmann probability density plot" must appear in the same chunk as the
        surrounding text about "Boltzmann distribution" so that LightRAG's NER and
        the vector embeddings capture the relationship. Appending figures at the end
        of the document breaks this — vision descriptions would land in isolated chunks
        disconnected from the text that gives them meaning.
        """
        logger.info(f"Starting ingestion of: {file_path}")
        result = self.converter.convert(file_path)
        doc = result.document

        enriched_md = []

        for item, _ in doc.iterate_items():
            label_str = str(item.label).lower() if hasattr(item, "label") else ""

            if any(t in label_str for t in ("picture", "figure", "chart")):
                # Inject vision description inline at the figure's position in the document.
                # This keeps the figure description in the same 1200-token chunk as the
                # surrounding text, preserving the semantic context for both NER and ANN.
                try:
                    if hasattr(item, "get_image"):
                        img = item.get_image(doc)
                        if img is not None:
                            buffered = io.BytesIO()
                            img.save(buffered, format="PNG")
                            description = self.process_images(buffered.getvalue())
                            enriched_md.append(f"\n> [Vision Analysis]: {description}\n")
                except Exception as e:
                    logger.warning(f"Could not process figure: {e}")
            else:
                content = self._item_to_markdown(item)
                if content:
                    enriched_md.append(content)

        return "\n\n".join(enriched_md)


def run_ingestion(input_dir: str, output_dir: str):
    processor = DocumentProcessor()
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".pdf"):
            input_path = os.path.join(input_dir, filename)
            md_content = processor.process_pdf(input_path)

            output_path = os.path.join(output_dir, f"{Path(filename).stem}.md")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.info(f"Processed {filename} -> {output_path}")


if __name__ == "__main__":
    run_ingestion("data", "processed_docs")
