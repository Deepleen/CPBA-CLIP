import os

import torch

from .simple_tokenizer import SimpleTokenizer


class Prompt_Ensemble:
    """Create class prompts with a BPE-aware context-length budget.

    CLIP's context length (77) includes the start/end tokens.  This project
    additionally replaces ``<|class|>`` with visual prompt tokens and may
    insert deep prompt tokens in ``encode_text``.  The default budget of 73
    therefore refers to BPE content tokens, not to characters and not to the
    total CLIP sequence length.
    """

    def __init__(
        self,
        cla_len,
        tokenizer,
        context_length=77,
        text_token_budget=None,
        deep_prompt_len=1,
        total_d_layer_len=11,
    ):
        self.special_token = "<|class|>"
        self.prompt_templates_abnormal = ["a photo of a damaged <|class|>"]
        self.prompt_templates_normal = ["a photo of a good <|class|>"]

        self.cla_len = int(cla_len)
        self.tokenizer = tokenizer
        self.context_length = int(context_length)
        self.deep_prompt_len = int(deep_prompt_len)
        self.total_d_layer_len = int(total_d_layer_len)
        self._explicit_text_token_budget = text_token_budget
        self._active_context_length = self.context_length

        # Use the same BPE vocabulary as model_CLIP.py.  Resolving the path
        # relative to this file keeps the new module independent of cwd.
        bpe_tokenizer = getattr(tokenizer, "bpe_tokenizer", None)
        if bpe_tokenizer is None:
            bpe_path = os.path.join(
                os.path.dirname(__file__), "bpe_simple_vocab_16e6.txt.gz"
            )
            bpe_tokenizer = SimpleTokenizer(bpe_path)
        self._bpe_tokenizer = bpe_tokenizer

        self._active_text_token_budget = self._compute_content_budget()

    def _compute_content_budget(self, model=None, visual_feature=None):
        """Return the number of BPE content tokens that can safely remain."""
        context_length = int(getattr(model, "context_length", self.context_length))
        visual_tokens = self.cla_len
        if visual_feature is not None and getattr(visual_feature, "ndim", 0) >= 3:
            visual_tokens = int(visual_feature.shape[1])

        deep_prompt_len = int(
            getattr(model, "num_tokens", self.deep_prompt_len)
        )
        total_d_layer = int(
            getattr(model, "total_d_layer", self.total_d_layer_len)
        )
        # Reserve SOT/EOT, the net visual-token increase, and the deep prompt
        # slots that are inserted by the custom text encoder.
        safe_budget = (
            context_length
            - 2
            - max(visual_tokens - 1, 0)
            - (deep_prompt_len if total_d_layer > 0 else 0)
        )
        if self._explicit_text_token_budget is not None:
            budget = min(int(self._explicit_text_token_budget), safe_budget)
        else:
            budget = safe_budget
        return max(1, budget)

    def _bpe_length(self, text):
        return len(self._bpe_tokenizer.encode(text))

    def _fit_prompt_to_budget(self, template, attrs_text, budget):
        """Keep the template and a maximal attribute prefix within ``budget``."""
        if not attrs_text:
            return template

        full_prompt = f"{template}, {attrs_text}"
        if self._bpe_length(full_prompt) <= budget:
            return full_prompt

        # Attribute text follows the class token, so the class token is kept
        # intact while a prefix of the attributes is selected by BPE length.
        low, high = 0, len(attrs_text)
        while low < high:
            middle = (low + high + 1) // 2
            prefix = attrs_text[:middle].rstrip()
            candidate = f"{template}, {prefix}" if prefix else template
            if self._bpe_length(candidate) <= budget:
                low = middle
            else:
                high = middle - 1

        prefix = attrs_text[:low].rstrip(" ,;:")
        if not prefix:
            return template

        # Prefer a readable boundary when it still fits.  The final loop is a
        # defensive check for any non-monotonic BPE merge behavior at a cut.
        boundary = max(prefix.rfind(", "), prefix.rfind(" "))
        if boundary > 0:
            boundary_prefix = prefix[:boundary].rstrip(" ,;:")
            if self._bpe_length(f"{template}, {boundary_prefix}") <= budget:
                prefix = boundary_prefix
        while prefix and self._bpe_length(f"{template}, {prefix}") > budget:
            prefix = prefix[:-1].rstrip(" ,;:")

        return f"{template}, {prefix}" if prefix else template

    def _parse_txt_content(self, txt_content):
        """Parse the expected ``Key: Value`` attribute text format."""
        if txt_content is None:
            return None

        attributes = {}
        for line in txt_content.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if "Object Type" in key:
                attributes["object_type"] = value
            elif "Color" in key:
                attributes["color"] = value
            elif "Shape" in key:
                attributes["shape"] = value
            elif "Texture" in key:
                attributes["texture"] = value
            elif "Material" in key:
                attributes["material"] = value
            elif "Edge Characteristics" in key:
                attributes["edge"] = value

        return attributes if attributes else None

    def _create_enhanced_prompts(
        self, attributes, is_normal=True, max_content_tokens=None
    ):
        """Build prompts and truncate attributes by BPE tokens if necessary."""
        if attributes is None:
            return (
                self.prompt_templates_normal
                if is_normal
                else self.prompt_templates_abnormal
            )

        attribute_order = [
            ("object_type", "Object Type"),
            ("color", "Color"),
            ("shape", "Shape"),
            ("texture", "Texture"),
            ("material", "Material"),
            ("edge", "Edge Characteristics"),
        ]
        all_attrs = [
            f"{label}: {attributes[key]}"
            for key, label in attribute_order
            if attributes.get(key)
        ]
        base_templates = (
            self.prompt_templates_normal
            if is_normal
            else self.prompt_templates_abnormal
        )
        if not all_attrs:
            return list(base_templates)

        budget = (
            self._active_text_token_budget
            if max_content_tokens is None
            else max(1, int(max_content_tokens))
        )
        attrs_text = ", ".join(all_attrs)
        return [
            self._fit_prompt_to_budget(template, attrs_text, budget)
            for template in base_templates
        ]

    def _tokenize(self, texts):
        """Call either the project tokenizer or a compatible custom callable."""
        try:
            return self.tokenizer(
                texts, context_length=self._active_context_length, truncate=True
            )
        except TypeError:
            return self.tokenizer(texts, truncate=True)

    def forward_ensemble(
        self, model, vison_feature, device, prompt_id=0, txt_content=None
    ):
        """Generate text embeddings for base or attribute-conditioned prompts."""
        del prompt_id  # Kept in the signature for compatibility with callers.
        self._active_context_length = int(
            getattr(model, "context_length", self.context_length)
        )
        self._active_text_token_budget = self._compute_content_budget(
            model=model, visual_feature=vison_feature
        )

        if txt_content is not None:
            if isinstance(txt_content, list):
                batch_size = len(txt_content)
                has_any_txt = any(
                    txt is not None and txt.strip() for txt in txt_content
                )

                if not has_any_txt:
                    normal_tokens = self._tokenize(self.prompt_templates_normal).to(device)
                    abnormal_tokens = self._tokenize(self.prompt_templates_abnormal).to(device)
                    normal_embeddings = model.encode_text(normal_tokens, vison_feature)
                    abnormal_embeddings = model.encode_text(abnormal_tokens, vison_feature)
                else:
                    all_normal_embeddings = []
                    all_abnormal_embeddings = []
                    for i in range(batch_size):
                        attrs = (
                            self._parse_txt_content(txt_content[i])
                            if txt_content[i] is not None
                            else None
                        )
                        enhanced_normal = self._create_enhanced_prompts(
                            attrs, is_normal=True
                        )
                        enhanced_abnormal = self._create_enhanced_prompts(
                            attrs, is_normal=False
                        )
                        normal_prompt = [enhanced_normal[0]]
                        abnormal_prompt = [enhanced_abnormal[0]]
                        vf = (
                            vison_feature[i : i + 1]
                            if vison_feature.dim() > 2
                            else vison_feature
                        )

                        try:
                            normal_tokens = self._tokenize(normal_prompt)
                            abnormal_tokens = self._tokenize(abnormal_prompt)
                            normal_emb = model.encode_text(
                                normal_tokens.to(device), vf
                            )
                            abnormal_emb = model.encode_text(
                                abnormal_tokens.to(device), vf
                            )
                        except Exception as exc:
                            # Keep the original fallback behavior, but it is
                            # now only a last resort after token-aware fitting.
                            print(
                                "Warning: Failed to process enhanced prompt; "
                                f"using base template. Error: {exc}"
                            )
                            normal_tokens = self._tokenize(
                                self.prompt_templates_normal
                            )
                            abnormal_tokens = self._tokenize(
                                self.prompt_templates_abnormal
                            )
                            normal_emb = model.encode_text(
                                normal_tokens.to(device), vf
                            )
                            abnormal_emb = model.encode_text(
                                abnormal_tokens.to(device), vf
                            )

                        all_normal_embeddings.append(normal_emb)
                        all_abnormal_embeddings.append(abnormal_emb)

                    normal_embeddings = torch.cat(all_normal_embeddings, dim=0)
                    abnormal_embeddings = torch.cat(all_abnormal_embeddings, dim=0)
            else:
                attributes = self._parse_txt_content(txt_content)
                enhanced_normal = self._create_enhanced_prompts(
                    attributes, is_normal=True
                )
                enhanced_abnormal = self._create_enhanced_prompts(
                    attributes, is_normal=False
                )
                try:
                    normal_tokens = self._tokenize(enhanced_normal).to(device)
                    abnormal_tokens = self._tokenize(enhanced_abnormal).to(device)
                except Exception as exc:
                    print(
                        "Warning: Failed to process enhanced prompt; "
                        f"using base template. Error: {exc}"
                    )
                    normal_tokens = self._tokenize(self.prompt_templates_normal).to(device)
                    abnormal_tokens = self._tokenize(self.prompt_templates_abnormal).to(device)
                normal_embeddings = model.encode_text(normal_tokens, vison_feature)
                abnormal_embeddings = model.encode_text(abnormal_tokens, vison_feature)
        else:
            normal_tokens = self._tokenize(self.prompt_templates_normal).to(device)
            abnormal_tokens = self._tokenize(self.prompt_templates_abnormal).to(device)
            normal_embeddings = model.encode_text(normal_tokens, vison_feature)
            abnormal_embeddings = model.encode_text(abnormal_tokens, vison_feature)

        return torch.cat([normal_embeddings, abnormal_embeddings], dim=1)
