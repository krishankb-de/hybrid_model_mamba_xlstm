# BROKEN — always creates a default config (dim=768, 12 layers = ~125M params)
config = HybridConfig()
model = HybridLanguageModel(config)
model.load_state_dict(cleaned_state_dict, strict=False)