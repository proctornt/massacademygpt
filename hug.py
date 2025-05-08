from huggingface_hub import notebook_login, create_repo, HfApi
from sentence_transformers import SentenceTransformer



# Load your model (you already fine-tuned it earlier)
model = SentenceTransformer("fine_tuned_model")

# Push to Hugging Face
model.push_to_hub("ntproctor/mass-academy-faq-embedder")