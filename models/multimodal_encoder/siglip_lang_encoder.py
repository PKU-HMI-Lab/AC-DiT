import torch
from transformers import AutoTokenizer, SiglipTextModel


class SiglipLangEncoder:
    available_models = ["google/siglip-so400m-patch14-384"]

    def __init__(
        self,
        device,
        from_pretrained=None,
        *,
        cache_dir=None,
        hf_token=None,
        use_text_preprocessing=True,
        model_kwargs=None,
        torch_dtype=None,
        use_offload_folder=None,
        model_max_length=120,
        local_files_only=False,
    ):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype or torch.bfloat16

        if model_kwargs is None:
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
            }

        assert from_pretrained in self.available_models, f"Model {from_pretrained} not available. Available models: {self.available_models}"

        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained,
            model_max_length=model_max_length,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        self.model = SiglipTextModel.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **model_kwargs,
        ).eval().to(self.device)

        self.model_max_length = model_max_length

    def get_text_embeddings(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = text_tokens_and_mask["input_ids"].to(self.device)
        attention_mask = text_tokens_and_mask["attention_mask"].to(self.device)
        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["last_hidden_state"].detach()
        return text_encoder_embs, attention_mask
    

if __name__ == "__main__":
    encoder = SiglipLangEncoder(
        device="cuda:0",
        from_pretrained="google/siglip-so400m-patch14-384",
    )
    texts = ["Hello world", "This is a test"]
    embeddings, mask = encoder.get_text_embeddings(texts)
    print(embeddings.shape)  # Should print the shape of the embeddings
    print(mask.shape)  # Should print the shape of the attention mask
