"""DECIMER on a folder of crop images, written as predictions.csv. Runs in the decimer environment.

DECIMER preprocesses each image itself; a crop with no drawing left after that gets an empty
SMILES. DECIMER gives one softmax probability per SMILES token; the
confidence written here is their product, the probability of the greedy SMILES. PYSTOW_HOME must
point at the models setup.sh downloaded.
"""

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from crop_runner import base_parser, list_images, run_crops  # noqa: E402

MODELS = ("DECIMER_model", "DECIMER_HandDrawn_model")


def main() -> int:
    parser = base_parser(__doc__)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()

    # Importing DECIMER without its models downloads them, into ~/.data when PYSTOW_HOME is unset.
    home = os.environ.get("PYSTOW_HOME")
    missing = [m for m in MODELS if not home or not Path(home, "DECIMER-V2", m, "saved_model.pb").is_file()]
    if missing:
        parser.error(f"no {', '.join(missing)} under PYSTOW_HOME={home!r}; run setup.sh")

    import tensorflow as tf

    # Listing GPUs initialises CUDA before DECIMER's import sets CUDA_VISIBLE_DEVICES=0, so the
    # GPU SLURM assigned is the one used.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = tf.config.list_physical_devices("GPU")
    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    elif not gpus:
        parser.error("TensorFlow sees no GPU; pass --device cpu to run on CPU")

    from DECIMER import decimer

    if visible is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    print(f"TensorFlow {tf.__version__}, GPUs {[g.name for g in gpus]}, running on {args.device}",
          file=sys.stderr)

    def predict(image: Path):
        # decimer.predict_SMILES(image, confidence=True) of 2.8.0, step by step, so that only its
        # preprocessing is guarded: on a blank crop it raises ValueError (an empty border crop or
        # a zero-size resize), which is an empty prediction. TensorFlow errors still raise.
        try:
            pixels = decimer.pre_process.decode_image(str(image))
        except ValueError as exc:
            print(f"{image.name}: no drawing after DECIMER preprocessing ({exc}); empty prediction",
                  file=sys.stderr)
            return "", None
        tokens, confidences = decimer.DECIMER_V2(tf.constant(pixels))
        smiles = decimer.utils.decoder(decimer.detokenize_output(tokens))
        per_token = decimer.detokenize_output_add_confidence(tokens, confidences)
        return smiles, math.prod(float(c) for _, c in per_token) if per_token else None

    errors = run_crops(predict, args)
    return 1 if errors == len(list_images(args.images)) else 0


if __name__ == "__main__":
    sys.exit(main())
