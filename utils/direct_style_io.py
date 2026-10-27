import datetime
import os
from pathlib import Path
import tempfile


def assert_separate_style_output(output_dir, source_checkpoint):
    if not source_checkpoint:
        raise ValueError("Direct Gaussian style training requires start_checkpoint.")
    source = Path(source_checkpoint).expanduser().resolve()
    source_run = source.parent.parent if source.parent.name == "checkpoints" else source.parent
    output = Path(output_dir).expanduser().resolve()
    if output == source_run or source_run in output.parents:
        raise ValueError(
            "Direct style output must be outside the source experiment directory: "
            f"source={source_run}, output={output}. Set model.model_path to a "
            "separate directory, for example outputs/jumpingjacks-direct-style."
        )
    return output


def create_direct_style_output(output_root, source_checkpoint):
    root = assert_separate_style_output(output_root, source_checkpoint)
    root.mkdir(parents=True, exist_ok=True)
    prefix = "direct-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-"
    return tempfile.mkdtemp(prefix=prefix, dir=os.fspath(root))
