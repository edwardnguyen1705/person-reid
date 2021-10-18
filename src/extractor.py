import torch
import numpy as np

from tqdm import tqdm


def feature_extractor(
    model,
    data_loader,
    device,
    description=None,
    flip_inference: bool = False,
):
    r"""Extract feature from dataloader
    Args:
        model (models):
        data_loader (Dataloader):
        device (int): torch.device('cpu') if use_gpu == 0 else torch.device(n_gpu)
    Return:
    """
    model.eval()
    feature, label, camera = [], [], []
    with torch.no_grad():
        model.to(device)
        with tqdm(total=len(data_loader)) as pbar:
            if description is not None:
                pbar.set_description(description)

            for x, y, cam_id in data_loader:
                x_gpu = x.to(device)

                e = model(x_gpu).cpu()

                if flip_inference:
                    f1 = e.cpu()

                    x_gpu = x.flip(dims=[3]).to(device)

                    e = model(x_gpu)

                    e = (f1 + e.cpu()) / 2

                feature.append(e)
                label.extend(y)
                camera.extend(cam_id)
                pbar.update(1)

    feature = torch.cat(feature, dim=0)
    label = np.asarray(label)
    camera = np.asarray(camera)

    return feature, label, camera
