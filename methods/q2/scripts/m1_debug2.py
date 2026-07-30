"""Debug M1 objectness: FP pretrained vs LSQ vs trained checkpoints."""
import torch
from torch.utils.data import DataLoader
from qat.data.voc import VOCDataset, collate_detection
from qat.models.yolov5 import QYOLOv5
from qat.pretrained import load_yolov5s_pretrained
from yolov5.utils.general import non_max_suppression


def report(tag, model, imgs):
    model.eval()
    with torch.no_grad():
        out = model(imgs)
    dec = out[0]
    print(tag, "obj mean/max", float(dec[..., 4].mean()), float(dec[..., 4].max()), flush=True)
    for conf in (0.001, 0.1, 0.25):
        pred = non_max_suppression(dec, conf, 0.45, max_det=300)
        print(tag, f"ndet@{conf}", [0 if p is None else len(p) for p in pred], flush=True)


def main():
    device = torch.device("cuda")
    ds = VOCDataset("/mnt/hdd2/qat_run/voc_yolo/test.list", img_size=640, augment=False)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_detection)
    imgs, _targets = next(iter(loader))
    imgs = imgs.to(device)

    m = QYOLOv5(nc=20, quant=None, wbits=4, abits=4).to(device)
    load_yolov5s_pretrained(m, "/mnt/hdd2/qat_run/weights/yolov5s.pt")
    report("fp_pretrained", m, imgs)

    m2 = QYOLOv5(nc=20, quant="lsq", wbits=4, abits=4).to(device)
    m2.init_quantizers(640)
    load_yolov5s_pretrained(m2, "/mnt/hdd2/qat_run/weights/yolov5s.pt")
    report("lsq_pt_no_calib", m2, imgs)

    m2.train()
    with torch.no_grad():
        for i, (im, _) in enumerate(loader):
            m2(im.to(device))
            if i >= 5:
                break
    report("lsq_pt_after_fwd", m2, imgs)

    for ep in (0, 1, 5, 29):
        path = f"/mnt/hdd2/qat_run/m1_lsq_baseline/ckpt_ep{ep}.pt"
        m3 = QYOLOv5(nc=20, quant="lsq", wbits=4, abits=4).to(device)
        m3.init_quantizers(640)
        m3.load_state_dict(torch.load(path, map_location=device)["model"])
        report(f"trained_ep{ep}", m3, imgs)

    m3 = QYOLOv5(nc=20, quant="lsq", wbits=4, abits=4).to(device)
    m3.init_quantizers(640)
    m3.load_state_dict(torch.load("/mnt/hdd2/qat_run/m1_lsq_baseline/ckpt_ep29.pt", map_location=device)["model"])
    count = 0
    for n, mod in m3.named_modules():
        if hasattr(mod, "step"):
            s = mod.step.detach().float()
            print(
                f"  {n}: step mean={s.mean():.6g} min={s.min():.6g} max={s.max():.6g} shape={tuple(s.shape)}",
                flush=True,
            )
            count += 1
            if count >= 10:
                break
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
