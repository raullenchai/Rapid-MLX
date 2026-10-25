# SPDX-License-Identifier: Apache-2.0
"""Auxiliary checkpoint identities not present in the model catalogs.

Update deliberately after checking the upstream asset and its consumers.
Catalog identities remain owned by model_aliases and the audio registry.
"""

AUXILIARY_REVISIONS = {
    "lucasnewman/vocos-mel-24khz": "ee89d00fac94b61e1235e0869f3e8d49711b7484",
    "t5-base": "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1",
    "mlx-community/sam-audio-large-fp16": "110947e651c2f7837c47d8897cfca1cb22ce9633",
    "mlx-community/sam-audio-large": "a3e529408663b65cb418584036db8f7209a80ced",
    "mlx-community/sam-audio-small-fp16": "64f1eb58120088c46f0f2ea78b13d5efd0778a88",
    "mlx-community/sam-audio-small": "4ca84acc5c47f64bb2acee448f835dfac65ed318",
    "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP": "85e5c23b6e864261d86dd9f26cd1cd96621d811e",
    "openai/whisper-large-v3": "06f233fe06e710322aca913c1bc4249a0d71fce1",
    "openai/whisper-large-v3-turbo": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
    "openai/whisper-medium": "abdf7c39ab9d0397620ccaea8974cc764cd0953e",
    "openai/whisper-small": "973afd24965f72e36ca33b3055d56a652f456b4d",
    "openai/whisper-base": "e37978b90ca9030d5170a5c07aadb050351a65bb",
    "openai/whisper-tiny": "169d4a4341b33bc18d8881c4b69c2e104e1cc0af",
    "mlx-community/Qwen3.5-9B-MTP-4bit": "222dfd2c23fc9518d7b817e4f8e0cb0571787489",
    "mlx-community/jina-reranker-v3-4bit-mxfp4": "00ce276e691f7f8041b397131c77f30a3e894d12",
    "mlx-community/mxbai-rerank-large-v2": "76001241414a9e9fbb91e9a0ece8d247164a7ca3",
    "BAAI/bge-reranker-v2-gemma": "1787044f8b6fb740a9de4557c3a12377f84d9e17",
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "google/gemma-4-E2B-it-assistant": "2d874ef7d29f9a30599a1e4b3c1cbc9595f005df",
    "google/gemma-4-E4B-it-assistant": "8d0031ea8c2109e2b1e86bb9368a4539b537f80a",
    "google/gemma-4-12B-it-assistant": "46d4c6f13f0ac0ad827b915669b8df9b81c64c51",
    "google/gemma-4-26B-A4B-it-assistant": "6e5aaaf4c42b98394530b8fda2e95cadd65c151c",
    "google/gemma-4-31B-it-assistant": "627c5ec1458b9086b841a91e0512fd31fd2fbbf1",
    "mlx-community/S3TokenizerV2": "e0c9886f0e1c35ae85b1f27277416fb19fc72bec",
    "ResembleAI/chatterbox": "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
    "dgrauet/ltx-2.3-mlx-q8": "6671a7572a530862d1d60ce393b5d93491e3f76b",
    "mlx-community/gemma-3-12b-it-4bit": "86cc6a8dedbc456dd0e4af01a9d09f396f77e558",
}
