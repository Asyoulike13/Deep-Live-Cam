import importlib
import sys
import types
import unittest

import numpy as np


def _install_import_stubs():
    sys.modules.setdefault(
        "cv2",
        types.SimpleNamespace(
            IMREAD_COLOR=1,
            BORDER_CONSTANT=0,
            NORMAL_CLONE=0,
            INTER_LINEAR=1,
            imread=lambda *_args, **_kwargs: None,
            imwrite=lambda *_args, **_kwargs: None,
            invertAffineTransform=lambda *_args, **_kwargs: None,
            warpAffine=lambda *_args, **_kwargs: None,
            multiply=lambda *_args, **_kwargs: None,
            merge=lambda *_args, **_kwargs: None,
            add=lambda *_args, **_kwargs: None,
            seamlessClone=lambda *_args, **_kwargs: None,
            ellipse=lambda *_args, **_kwargs: None,
            GaussianBlur=lambda *_args, **_kwargs: None,
        ),
    )
    sys.modules.setdefault(
        "insightface",
        types.SimpleNamespace(
            model_zoo=types.SimpleNamespace(get_model=lambda *_args, **_kwargs: None)
        ),
    )
    sys.modules["modules.gpu_processing"] = types.SimpleNamespace(
        gpu_cvt_color=lambda *_args, **_kwargs: None,
        gpu_gaussian_blur=lambda *_args, **_kwargs: None,
        gpu_sharpen=lambda *_args, **_kwargs: None,
        gpu_add_weighted=lambda *_args, **_kwargs: None,
        gpu_resize=lambda *_args, **_kwargs: None,
    )
    sys.modules["modules.cluster_analysis"] = types.SimpleNamespace(
        find_closest_centroid=lambda *_args, **_kwargs: None
    )
    sys.modules["modules.utilities"] = types.SimpleNamespace(
        conditional_download=lambda *_args, **_kwargs: None,
        is_image=lambda *_args, **_kwargs: False,
        is_video=lambda *_args, **_kwargs: False,
    )
    sys.modules["modules.face_analyser"] = types.SimpleNamespace(
        get_one_face=lambda *_args, **_kwargs: None,
        get_many_faces=lambda *_args, **_kwargs: None,
        default_source_face=lambda *_args, **_kwargs: None,
    )
    sys.modules["modules.core"] = types.SimpleNamespace(update_status=lambda *_args, **_kwargs: None)
    sys.modules["modules.typing"] = types.SimpleNamespace(Face=object, Frame=object)
    sys.modules["modules.processors.frame.core"] = types.SimpleNamespace()


class FastPasteBackTests(unittest.TestCase):
    def test_returns_original_frame_when_transform_is_missing(self):
        _install_import_stubs()
        sys.modules.pop("modules.processors.frame.face_swapper", None)
        face_swapper = importlib.import_module("modules.processors.frame.face_swapper")

        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        fake = np.ones((16, 16, 3), dtype=np.uint8)
        aimg = np.zeros((128, 128, 3), dtype=np.uint8)

        result = face_swapper._fast_paste_back(frame, fake, aimg, None)

        self.assertTrue(np.array_equal(result, frame))


if __name__ == "__main__":
    unittest.main()
