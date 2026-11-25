"""Start SGLang with the specsched patch: python -m specsched_sgl_launch <sglang.launch_server arguments>.

SGLang starts its scheduler and detokenizer with multiprocessing's spawn method, which re-imports this module (as
__mp_main__) in each child before running it, so the scheduler-side patch below is applied in every process. The
HTTP endpoints are installed in the main process only, once the port is known.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import specsched_sgl  # noqa: E402

specsched_sgl.apply()

if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(sys.argv[1:])
    specsched_sgl.install_http(server_args.port)
    specsched_sgl.log(f"launching SGLang on port {server_args.port} with the specsched patch")
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
