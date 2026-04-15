#!/home/carsten/trinity-vlm/.venv/bin/python

import os
import sys
import traceback

from trinity_vlm.train_ddp import main


def _hard_exit(code: int) -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            _hard_exit(0)
        if isinstance(code, int):
            _hard_exit(code)
        print(code, file=sys.stderr)
        _hard_exit(1)
    except BaseException:
        traceback.print_exc()
        _hard_exit(1)
    else:
        _hard_exit(0)
