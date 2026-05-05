"""
Run during Docker build (Stage 2) to pre-download HuggingFace models into HF_HOME.
TRANSFORMERS_OFFLINE must NOT be set when this runs.
"""
from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer

print("Downloading microsoft/graphcodebert-base ...")
RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
AutoModel.from_pretrained("microsoft/graphcodebert-base")
print("Done: graphcodebert-base")

print("Downloading claudios/VulBERTa-MLP-D2A ...")
AutoTokenizer.from_pretrained("claudios/VulBERTa-MLP-D2A")
RobertaModel.from_pretrained("claudios/VulBERTa-MLP-D2A")
print("Done: VulBERTa-MLP-D2A")

print("All models cached.")
