from __future__ import annotations

from src._class_module_support import bind_module_namespace, finalize_split_class
import src.ingestion as _source_module

bind_module_namespace(
    _source_module,
    globals(),
    proxy_functions=_source_module._CLASS_MODULE_PROXY_FUNCTIONS,
)


class _LegacyDocumentProcessor:
    def __init__(self, vision_model="qwen2.5vl:7b"):
        self.vision_model = vision_model
        self._vision_model_loaded = False

        self.converter = _build_docling_converter(
            accelerator="auto",
            num_threads=8,
            generate_picture_images=True,
        )

    def _ensure_vision_loaded(self):
        """
        Lazily warms up the vision model on first figure encounter.
        Avoids consuming ~8GB VRAM on documents that contain no images.
        """
        if not self._vision_model_loaded:
            logger.info(f"First figure detected — loading vision model: {self.vision_model}")
            try:
                _ollama_generate(model=self.vision_model, prompt="Hello", keep_alive="30m")
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
            response = _ollama_generate(
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
        surrounding text about "Boltzmann distribution" so that local retrieval and
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

_LegacyDocumentProcessor.__module__ = _source_module.__name__
finalize_split_class(_source_module, _LegacyDocumentProcessor)

