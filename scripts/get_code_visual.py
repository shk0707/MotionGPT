import os
import numpy as np
import time
import pytorch_lightning as pl
import torch
from pathlib import Path
from tqdm import tqdm
from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae


def main():

    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1

    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    cfg.TIME = str(time_str)

    model_name = cfg.model.target.split('.')[-2].lower()
    output_dir = Path(
        os.path.join(cfg.FOLDER, model_name, cfg.NAME,
                     "tokens_visual_" + cfg.TIME))
    print(f'Output will be saved to {output_dir}')

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datamodule = build_data(cfg, phase="test")
    print("datasets module {} initialized".format("".join(cfg.DATASET.target)))

    os.makedirs(output_dir, exist_ok=True)

    # create model
    model = build_model(cfg, datamodule)
    print("model {} loaded".format(cfg.model.target))

    # Strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model)

    # loading state dict
    if cfg.TEST.CHECKPOINTS:
        load_pretrained(cfg, model, phase="test")

    if cfg.ACCELERATOR == "gpu":
        model = model.cuda()
    elif cfg.ACCELERATOR == "cpu":
        model = model.cpu()

    model.eval()
    codes = cfg.model.params.codebook_size # [num_ubody, num_lbody]
    with torch.no_grad():
        for i in tqdm(range(codes[0])):

            # Generate motion from token
            # vq_latent = model.vae.quantizer.dequantize(m_token)
            # gen_motion = model.vae.decode(m_token)
            # gen_motion = model.feats2joints(gen_motion).to('cpu').numpy()

            m_token = torch.LongTensor(1, 1, 2).fill_(0).to(model.device)
            m_token[:, :, 0] = i
            gen_motion = model.vae.decode(m_token, no_lz=True)
            gen_motion = model.feats2joints(gen_motion).to('cpu').numpy()

            # Generate translation from token
            # texts = [
            #     f'Generate text: <motion_id_{codes}><motion_id_{i}><motion_id_{codes +1}>'
            # ]
            texts = [
                f'Generate text: <motion_id_som><motion_id_{i}_none><motion_id_eom>'
            ]
            # texts = [f'Use only one word to describe: <motion_id_{codes}><motion_id_{i}><motion_id_{codes +1}>']
            batch = {"text": texts, "length": [0]}

            # out_text = model(batch)['texts']
            # print(out_text)
            # out_text_path = os.path.join(output_dir, f'{i}.txt')
            # Path(out_text_path).parent.mkdir(parents=True, exist_ok=True)
            # with open(out_text_path, 'w') as f:
            #     f.write(out_text[0])

            target_path = os.path.join(output_dir, f'ubody_{i}.npy')
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)

            np.save(target_path, gen_motion)

        for i in tqdm(range(codes[1])):

            # Generate motion from token
            # vq_latent = model.vae.quantizer.dequantize(m_token)
            # gen_motion = model.vae.decode(m_token)
            # gen_motion = model.feats2joints(gen_motion).to('cpu').numpy()

            m_token = torch.LongTensor(1, 1, 2).fill_(0).to(model.device)
            m_token[:, :, 1] = i
            gen_motion = model.vae.decode(m_token, no_uz=True)
            gen_motion = model.feats2joints(gen_motion).to('cpu').numpy()

            # Generate translation from token
            # texts = [
            #     f'Generate text: <motion_id_{codes}><motion_id_{i}><motion_id_{codes +1}>'
            # ]
            texts = [
                f'Generate text: <motion_id_som><motion_id_none_{i}><motion_id_eom>'
            ]
            # texts = [f'Use only one word to describe: <motion_id_{codes}><motion_id_{i}><motion_id_{codes +1}>']
            batch = {"text": texts, "length": [0]}

            # out_text = model(batch)['texts']
            # print(out_text)
            # out_text_path = os.path.join(output_dir, f'{i}.txt')
            # Path(out_text_path).parent.mkdir(parents=True, exist_ok=True)
            # with open(out_text_path, 'w') as f:
            #     f.write(out_text[0])

            target_path = os.path.join(output_dir, f'lbody_{i}.npy')
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)

            np.save(target_path, gen_motion)

    print(
        f'Motion tokenization done, the motion tokens are saved to {output_dir}'
    )


if __name__ == "__main__":
    main()
