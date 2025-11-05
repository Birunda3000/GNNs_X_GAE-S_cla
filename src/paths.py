import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

OUTPUT_PATH = os.path.join(DATA_DIR, "output")

# --- Caminhos para os arquivos do Musae-Github ---
class musae_github_paths:
    DATASET_NAME = "musae-github"

    GITHUB_MUSAE_EDGES_PATH = os.path.join(
        DATA_DIR, "datasets", "musae-github", "musae_git_edges.csv"
    )
    GITHUB_MUSAE_TARGET_PATH = os.path.join(
        DATA_DIR, "datasets", "musae-github", "musae_git_target.csv"
    )
    GITHUB_MUSAE_FEATURES_PATH = os.path.join(
        DATA_DIR, "datasets", "musae-github", "musae_git_features.json"
    )



# --- Caminhos para os arquivos do Musae-Facebook ---
class musae_facebook_paths:
    DATASET_NAME = "musae-facebook"

    FACEBOOK_MUSAE_EDGES_PATH = os.path.join(
        DATA_DIR, "datasets", "musae-facebook", "musae_facebook_edges.csv"
    )
    FACEBOOK_MUSAE_TARGET_PATH = os.path.join(
        DATA_DIR, "datasets", "musae-facebook", "musae_facebook_target.csv"
    )
    FACEBOOK_MUSAE_FEATURES_PATH = os.path.join(
        DATA_DIR, "datasets", "musae-facebook", "musae_facebook_features.json"
    )