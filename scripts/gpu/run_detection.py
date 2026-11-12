"""Run a pretrained OpenPCDet KITTI model over every KITTI Tracking LiDAR sequence.

Runs on the GPU machine. Output is deliberately raw so that everything that
doesn't need a GPU (calibration, FOV filtering, tracking, eval) happens locally.

Output layout:
    <out>/<split>/<seq>.txt   one line per box:
        frame class_id score x y z dx dy dz heading
    Boxes are in the LiDAR frame, OpenPCDet convention (center, dims, yaw around z).
    class_id indexes into meta.json["class_names"] (0-based).
    <out>/meta.json           model, versions, GPU, timings.

Resumable: finished sequences are skipped on re-run.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--openpcdet", required=True, type=Path, help="OpenPCDet repo root")
    p.add_argument("--cfg", required=True, help="model cfg, relative to OpenPCDet/tools")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--kitti-root", required=True, type=Path,
                   help="dir containing training/velodyne/<seq>/ and testing/velodyne/<seq>/")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--splits", nargs="+", default=["training", "testing"])
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = args.ckpt.resolve()
    kitti_root = args.kitti_root.resolve()
    out_root = args.out.resolve()

    # OpenPCDet configs use _BASE_CONFIG_ paths relative to tools/.
    os.chdir(args.openpcdet / "tools")
    sys.path.insert(0, str(args.openpcdet))

    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import DatasetTemplate
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils

    class FrameDataset(DatasetTemplate):
        def __init__(self, files, **kw):
            super().__init__(**kw)
            self.files = files

        def __len__(self):
            return len(self.files)

        def __getitem__(self, i):
            f = self.files[i]
            points = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
            return self.prepare_data({"points": points, "frame_id": int(f.stem)})

    cfg_from_yaml_file(args.cfg, cfg)
    logger = common_utils.create_logger()
    class_names = list(cfg.CLASS_NAMES)

    model = None
    meta = {
        "cfg": args.cfg,
        "ckpt": ckpt.name,
        "class_names": class_names,
        "columns": "frame class_id score x y z dx dy dz heading",
        "frame": "lidar (OpenPCDet convention)",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "sequences": {},
    }
    meta_path = out_root / "meta.json"
    if meta_path.exists():
        meta["sequences"] = json.loads(meta_path.read_text()).get("sequences", {})

    for split in args.splits:
        vel_dir = kitti_root / split / "velodyne"
        seqs = sorted(d for d in vel_dir.iterdir() if d.is_dir())
        (out_root / split).mkdir(parents=True, exist_ok=True)

        for seq_dir in seqs:
            out_file = out_root / split / f"{seq_dir.name}.txt"
            if out_file.exists():
                logger.info(f"skip {split}/{seq_dir.name} (done)")
                continue

            files = sorted(seq_dir.glob("*.bin"))
            ds = FrameDataset(files, dataset_cfg=cfg.DATA_CONFIG, class_names=class_names,
                              training=False, root_path=seq_dir, logger=logger)
            if model is None:
                model = build_network(model_cfg=cfg.MODEL, num_class=len(class_names), dataset=ds)
                model.load_params_from_file(filename=str(ckpt), logger=logger, to_cpu=True)
                model.cuda().eval()

            loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=args.workers,
                                                 collate_fn=ds.collate_batch)
            rows = []
            t0 = time.time()
            with torch.no_grad():
                for batch in loader:
                    frame = int(batch["frame_id"][0])
                    load_data_to_gpu(batch)
                    pred, _ = model(batch)
                    boxes = pred[0]["pred_boxes"].cpu().numpy()
                    scores = pred[0]["pred_scores"].cpu().numpy()
                    labels = pred[0]["pred_labels"].cpu().numpy() - 1  # 1-based -> 0-based
                    for b, s, l in zip(boxes, scores, labels):
                        rows.append([frame, l, s, *b[:7]])

            arr = np.array(rows, dtype=np.float64).reshape(-1, 10)
            tmp = out_file.with_suffix(".tmp")
            np.savetxt(tmp, arr, fmt=["%d", "%d", "%.4f"] + ["%.4f"] * 7,
                       header=meta["columns"])
            tmp.rename(out_file)

            dt = time.time() - t0
            meta["sequences"][f"{split}/{seq_dir.name}"] = {
                "frames": len(files), "boxes": len(arr), "seconds": round(dt, 1)}
            meta_path.write_text(json.dumps(meta, indent=2))
            logger.info(f"{split}/{seq_dir.name}: {len(files)} frames, {len(arr)} boxes, "
                        f"{len(files) / dt:.1f} fps")

    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("all done")


if __name__ == "__main__":
    main()
