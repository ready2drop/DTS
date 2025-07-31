import os
import pickle
import warnings

from tqdm import tqdm
from collections import OrderedDict

import numpy as np

import torch
from medpy import metric

from engine import Engine
from metric import dice_coeff
from utils import parse_args, get_dataloader

warnings.filterwarnings("ignore")

class Tester(Engine):
    def __init__(
        self,
        model_path: str = None,
        model_name: str = "DTS",
        data_name: str = "btcv",
        data_path: str = None,
        sw_batch_size: int = 4,
        overlap: float = 0.25,
        image_size: int = 256,
        spatial_size: int = 96,
        spacing: tuple = [1.5, 1.5, 2.0],
        classes: str = None,
        epoch: int = None,
        device: str = "cuda:0",
        project_name: str = "DTS",
        wandb_name: str = None,
        include_background: bool = False,
        use_amp: bool = True,
        use_cache: bool = False,
        use_wandb: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            data_name=data_name,
            data_path=data_path,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            image_size=image_size,
            spatial_size=spatial_size,
            classes=classes,
            device=device,
            model_path=model_path,
            project_name=project_name,
            wandb_name=wandb_name,
            include_background=include_background,
            use_amp=use_amp,
            use_cache=use_cache,
            use_wandb=use_wandb,
            mode="test",
        )
        self.epoch = epoch
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.spacing = spacing
        self.model = self.load_model()
        self.log_dir = os.path.dirname(os.path.dirname(model_path))
        self.save_name = os.path.basename(model_path).split(".")[0]
        self.dices = []
        self.hds = []
        self.images = []
        self.outputs = []
        self.labels = []

        self.patient_index = 0

        self.load_checkpoint(model_path)
        self.set_dataloader()

    def load_checkpoint(self, model_path):
        if self.epoch is not None:
            model_path = os.path.join(os.path.dirname(model_path), f"epoch_{self.epoch}.pt")
        state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict['model'])

        print(f"Checkpoint loaded from {model_path}.....")

    def set_dataloader(self):
        self.dataloader = get_dataloader(
            data_name=self.data_name,
            data_path=self.data_path,
            image_size=self.image_size,
            spatial_size=self.spatial_size,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            mode=self.mode
        )

    def test(self):
        self.model.eval()
        dices = []
        all_hds = []

        with torch.cuda.amp.autocast(self.use_amp):
            for batch in tqdm(self.dataloader["val"], total=len(self.dataloader["val"])):
                with torch.no_grad():
                    dice_score = self.validation_step(batch)
                    dices.append(dice_score)

                    current_hds = list(self.hds[-1].values())
                    valid_hds = [hd for hd in current_hds if hd != float('inf')]
                    all_hds.extend(valid_hds)

                self.global_step += 1

        mean_dice = np.mean(dices)
        mean_hd95 = np.mean(all_hds) if all_hds else float('inf')

        print("="*200)
        print(f"Final Results:")
        print(f"Mean Dice: {mean_dice:.4f}")
        print(f"Mean HD95: {mean_hd95:.4f}")
        print("="*200)

        # self.save_results() # This may demand a lot of memory, consider saving results in smaller chunks or only saving necessary data.
        return mean_dice, mean_hd95

    def validation_step(self, batch):
        with torch.cuda.amp.autocast(self.use_amp):
            images, outputs, labels = self.infer(batch)

        _, _, d, w, h = labels.shape
        images = torch.nn.functional.interpolate(images, mode="nearest", size=(d, w, h))
        outputs = torch.nn.functional.interpolate(outputs, mode="nearest", size=(d, w, h))

        dices = OrderedDict({v: 0 for v in self.class_names.values()})
        hds = OrderedDict({v: 0 for v in self.class_names.values()})

        classes = list(self.class_names.values())

        for i in range(self.num_classes):
            output = outputs[:, i]
            label = labels[:, i]

            if output.sum() > 0 and label.sum() == 0:
                dice = 1.0
            else:
                dice = dice_coeff(output, label).to(outputs.device).item()

            dices[classes[i]] = dice

            try:
                output_np = output.squeeze(0).cpu().numpy().astype(bool)
                label_np = label.squeeze(0).cpu().numpy().astype(bool)

                output_sum = output_np.sum()
                label_sum = label_np.sum()

                if output_sum > 0 and label_sum > 0:
                    hd95 = metric.hd95(output_np, label_np, voxelspacing=self.spacing)
                elif output_sum == 0 and label_sum == 0:
                    hd95 = 0.0
                elif output_sum == 0 and label_sum > 0:
                    hd95 = float('inf')
                elif output_sum > 0 and label_sum == 0:
                    hd95 = float('inf')

            except Exception as e:
                print(f"HD95 calculation failed for class {classes[i]}: {e}")
                print(f"  -> output shape: {output.shape}, label shape: {label.shape}")
                hd95 = float('inf')

            hds[classes[i]] = hd95
            print(f"{classes[i]}: dice={dice:.4f}")

        self.dices.append(dices)
        self.hds.append(hds)
        self.images.append(images)
        self.outputs.append(outputs)
        self.labels.append(labels)

        mean_dice = np.mean(list(dices.values()))
        mean_hd95_batch = np.mean([v for v in hds.values() if v != float('inf')])

        print(f"Batch mean dice: {mean_dice:.4f}")
        print(f"Batch mean hd95: {mean_hd95_batch:.4f}")
        print("-" * 200)

        self.patient_index += 1
        return mean_dice

    def save_results(self):
        results = {
            "images": self.images,
            "dices": self.dices,
            "hds": self.hds,
            "labels": self.labels,
            "outputs": self.outputs,
        }

        with open(os.path.join(self.log_dir, f'results.pkl'), 'wb') as file:
            pickle.dump(results, file)

if __name__ == "__main__":
    args = parse_args()
    tester = Tester(**vars(args))
    tester.test()