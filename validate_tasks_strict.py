import asyncio
from env.environment import CICDDebuggerEnvironment
from env.tasks import get_tasks_by_difficulty


async def validate():
    env = CICDDebuggerEnvironment()

    # Collect all tasks
    all_tasks = []
    for difficulty in ["easy", "medium", "hard"]:
        tasks = get_tasks_by_difficulty(difficulty)
        all_tasks.extend(tasks)

    failed = False

    for task in all_tasks:
        print(f"\nChecking: {task.task_id}")

        # -------------------------
        # RESET ENVIRONMENT
        # -------------------------
        obs = await env.reset(task_id=task.task_id)

        # -------------------------
        # BEFORE FIX → MUST FAIL
        # -------------------------
        obs, reward, done, info = await env.step({
            "tool": "validate_fix",
            "payload": {}
        })

        if reward == 1.0:
            print("❌ FAIL: passes BEFORE fix")
            print("Info:", info)
            failed = True

        # -------------------------
        # APPLY CORRECT FIX
        # -------------------------
        obs, reward, done, info = await env.step({
            "tool": "edit_config",
            "payload": {"new_config": task.expected_config}
        })

        # -------------------------
        # RUN PIPELINE (REQUIRED BEFORE VALIDATION)
        # -------------------------
        await env.step({
            "tool": "run_pipeline_stage",
            "payload": {"stage": "build"}
        })

        # -------------------------
        # OPTIONAL (STRONGER CHECK)
        # -------------------------
        await env.step({
            "tool": "run_pipeline_stage",
            "payload": {"stage": "test"}
        })

        # -------------------------
        # AFTER FIX → MUST PASS
        # -------------------------
        obs, reward, done, info = await env.step({
            "tool": "validate_fix",
            "payload": {}
        })

        if reward != 0.95:
            print("❌ FAIL: fails AFTER fix")
            print("Info:", info)
            failed = True

    # -------------------------
    # FINAL RESULT
    # -------------------------
    if not failed:
        print("\n✅ ALL TASKS VALID")
    else:
        print("\n⚠️ SOME TASKS FAILED — FIX REQUIRED")


if __name__ == "__main__":
    asyncio.run(validate())