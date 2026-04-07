import asyncio
from env.environment import CICDDebuggerEnvironment

async def test():
    env = CICDDebuggerEnvironment(max_steps=20, seed=42)
    obs = await env.reset(task_id='easy-command-typo')
    print('✓ Reset OK')
    print(f'  Observation keys: {list(obs.keys())}')
    
    # Test step with valid action
    obs, reward, done, info = await env.step('read_logs: inspect logs')
    print(f'✓ Step OK')
    print(f'  Reward: {reward} (type: {type(reward).__name__})')
    print(f'  Reward in [0,1]: {0.0 <= reward <= 1.0}')
    print(f'  Done: {done}')
    
    # Test validation without pipeline (should fail)
    obs, reward, done, info = await env.step('validate_fix: {}')
    print(f'✓ Validation without pipeline OK')
    print(f'  Score in info: {info.get("score")}')
    print(f'  Score in (0,1): {0.0 < info.get("score", 0) < 1.0}')
    print(f'  is_valid: {info.get("is_valid")}')
    
    # Run through proper flow
    env2 = CICDDebuggerEnvironment(max_steps=20, seed=43)
    obs = await env2.reset(task_id='easy-command-typo')
    
    # Edit config
    config_fix = '''name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
  test:
    runs-on: ubuntu-latest
    needs: build
    steps:
      - uses: actions/checkout@v4
      - run: npm test'''
    
    obs, reward, done, info = await env2.step(f'edit_config: {{"raw": {repr(config_fix)}}}')
    print(f'✓ Edit OK - reward: {reward}')
    
    # Run pipeline stage
    obs, reward, done, info = await env2.step('run_pipeline_stage: {"stage": "build"}')
    print(f'✓ Run build stage OK - reward: {reward}')
    
    obs, reward, done, info = await env2.step('run_pipeline_stage: {"stage": "test"}')
    print(f'✓ Run test stage OK - reward: {reward}')
    
    # Now validate
    obs, reward, done, info = await env2.step('validate_fix: {}')
    print(f'✓ Validation with pipeline OK')
    print(f'  Score: {info.get("score")}')
    print(f'  Score in (0.01, 0.99): {0.01 < info.get("score", 0) < 0.99}')
    print(f'  is_valid: {info.get("is_valid")}')
    
    print('\n✓ All tests passed!')

asyncio.run(test())
