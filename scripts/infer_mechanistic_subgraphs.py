import hydra
from omegaconf import DictConfig
from cgr.inference import ReactantGraph

@hydra.main(version_base=None, config_path='../configs', config_name='infer_mech_subgraphs')
def main(cfg: DictConfig):
    print(cfg)


if __name__ == '__main__':
    main()