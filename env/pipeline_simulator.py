from typing import Dict, Tuple, Any


class PipelineSimulator:

    def run_stage(self, config: Dict[str, Any], stage: str) -> Tuple[bool, str]:
        pipeline = config.get("pipeline", {})

        # Safety check
        if not isinstance(pipeline, dict):
            return False, "Invalid pipeline structure"

        stage_config = pipeline.get(stage, {})
        commands = stage_config.get("script", [])

        if not isinstance(commands, list):
            commands = []

        # ---------- BUILD ----------
        if stage == "build":
            for command in commands:
                if "tset" in command:
                    return False, "Build failed: command typo"

            requirements = pipeline.get("requirements", {})
            image = str(stage_config.get("image", ""))

            required_node = requirements.get("node")
            if required_node and required_node not in image:
                return False, f"Build failed: Node version mismatch, expected {required_node}"

            return True, "Build succeeded"

        # ---------- TEST ----------
        if stage == "test":
            if not any("pytest" in cmd for cmd in commands):
                return False, "Test failed: pytest missing"

            return True, "Test succeeded"

        return False, f"Unknown stage: {stage}"

    def run_full_pipeline(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        logs = []

        success, log = self.run_stage(config, "build")
        logs.append(log)
        if not success:
            return False, "\n".join(logs)

        success, log = self.run_stage(config, "test")
        logs.append(log)

        return success, "\n".join(logs)

    def check_fix_correctness(self, config: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        return config == expected