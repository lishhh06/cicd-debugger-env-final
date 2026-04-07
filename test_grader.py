import asyncio
from env.environment import CICDDebuggerEnvironment

async def test():
    env = CICDDebuggerEnvironment()
    obs = await env.reset()

    # 🔥 APPLY CORRECT FIX
    env._state.current_config = env._state.task.expected_config

    obs, reward, done, _ = await env.step({
        "action_type": "submit_solution",
        "payload": {}
    })

    print("Reward:", reward)
    print("Done:", done)

asyncio.run(test())