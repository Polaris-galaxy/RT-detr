# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class RTDETRPredictor(BasePredictor):
    """
    RT-DETR (Real-Time Detection Transformer) Predictor extending the BasePredictor class for making predictions using
    Baidu's RT-DETR model.

    This class leverages the power of Vision Transformers to provide real-time object detection while maintaining
    high accuracy. It supports key features like efficient hybrid encoding and IoU-aware query selection.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.rtdetr import RTDETRPredictor

        args = dict(model='rtdetr-l.pt', source=ASSETS)
        predictor = RTDETRPredictor(overrides=args)
        predictor.predict_cli()
        ```

    Attributes:
        imgsz (int): Image size for inference (must be square and scale-filled).
        args (dict): Argument overrides for the predictor.
    """

    def postprocess(self, preds, img, orig_imgs):
        """
        Postprocess the raw predictions from the model to generate bounding boxes and confidence scores.

        The method filters detections based on confidence and class if specified in `self.args`.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input images.
            orig_imgs (list or torch.Tensor): Original, unprocessed images.

        Returns:
            (list[Results]): A list of Results objects containing the post-processed bounding boxes, confidence scores,
                and class labels.
        """
        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, bbox in enumerate(bboxes):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            score, cls = scores[i].max(-1, keepdim=True)  # (300, 1)
            idx = score.squeeze(-1) > self.args.conf  # (300, )
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, score, cls], dim=-1)[idx]  # filter
            pred = self._filter_wool_priors(pred)
            orig_img = orig_imgs[i]
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow
            pred[..., [1, 3]] *= oh
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    @staticmethod
    def _aspect_ratio_xyxy(xyxy):
        """max(w,h)/min(w,h) for each box, shape (N,)."""
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        w = w.clamp(min=1e-6)
        h = h.clamp(min=1e-6)
        return torch.max(w / h, h / w)

    def _filter_wool_priors(self, pred):
        """Optional size / aspect-ratio prior filter for wool (normalized xyxy in [0,1])."""
        if pred.shape[0] == 0 or not getattr(self.args, 'wool_prior_filter', False):
            return pred
        w = (pred[:, 2] - pred[:, 0]).clamp(min=1e-6)
        h = (pred[:, 3] - pred[:, 1]).clamp(min=1e-6)
        area = w * h
        ar = self._aspect_ratio_xyxy(pred[:, :4])
        min_a = float(self.args.wool_min_area_ratio)
        max_a = float(self.args.wool_max_area_ratio)
        min_ar = float(self.args.wool_min_aspect_ratio)
        max_ar = float(self.args.wool_max_aspect_ratio)
        keep = (area >= min_a) & (area <= max_a) & (ar >= min_ar) & (ar <= max_ar)
        return pred[keep]

    def pre_transform(self, im):
        """
        Pre-transforms the input images before feeding them into the model for inference. The input images are
        letterboxed to ensure a square aspect ratio and scale-filled. The size must be square(640) and scaleFilled.

        Args:
            im (list[np.ndarray] |torch.Tensor): Input images of shape (N,3,h,w) for tensor, [(h,w,3) x N] for list.

        Returns:
            (list): List of pre-transformed images ready for model inference.
        """
        letterbox = LetterBox(self.imgsz, auto=False, scaleFill=True)
        return [letterbox(image=x) for x in im]
