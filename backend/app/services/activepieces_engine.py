import asyncio
import json
import os

async def run_activepieces_flow(flow_path: str, inputs: dict) -> dict:
    try:
        runner_path = os.path.join("backend/app/third_party/activepieces/flow_runner.js")
        input_json = json.dumps(inputs)

        process = await asyncio.create_subprocess_exec(
            "node", runner_path, flow_path, input_json,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if stdout:
            result = json.loads(stdout.decode())
            return result
        if stderr:
            return {
                "status": "error",
                "message": stderr.decode()
            }

        return {"status": "error", "message": "No output from subprocess."}

    except Exception as e:
        return {
            "status": "error",
            "message": f"Exception in run_activepieces_flow: {e}"
        }
