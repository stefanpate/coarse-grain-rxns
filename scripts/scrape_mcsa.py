import requests
import hydra
from omegaconf import DictConfig

def fetch_json(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

@hydra.main(version_base=None, config_path="../configs", config_name="mcsa_pull")
def main(cfg: DictConfig):
    response = fetch_json(cfg.url)
    print()


if __name__ == '__main__':
    main()