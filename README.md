# MetaWorld KUKA v3

MetaWorld KUKA v3 adapts a focused set of MetaWorld v3 manipulation tasks to a
KUKA IIWA robot with a Robotiq 2F-85 gripper. This standalone repository is
based on the runtime subset of Farama Foundation's MetaWorld v3.1.1 source tree
plus the separate `metaworld_kuka` package, KUKA task definitions, MJCF models,
meshes, tests, and diagnostic scripts. It contains all files required by the
included KUKA environments and does not depend on a sparse or partial clone.

This is an experimental adaptation, not an official Farama Foundation release.
The upstream MetaWorld code remains under its original MIT license. Local KUKA
code and models are kept isolated under `metaworld_kuka` so upstream and local
work are distinguishable.

## KUKA quick start

Python 3.10 through 3.13 is declared in `pyproject.toml`. Install this checkout
and its pinned MuJoCo 3.3.0 dependency in an isolated environment:

```bash
python -m pip install -e '.[testing]'
```

No separate asset download is currently required for the registered KUKA
tasks. Their XML files, KUKA meshes, Robotiq meshes, and referenced upstream
MetaWorld task assets are stored in this repository. Do not exclude
`metaworld/assets` or `metaworld_kuka/assets` when packaging or cloning.

```python
import gymnasium as gym
import metaworld_kuka  # registers Meta-World-KUKA/MT1

env = gym.make("Meta-World-KUKA/MT1", env_name="kuka-reach-v3", seed=42)
observation, info = env.reset(seed=42)
for _ in range(3):
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )
env.close()
```

The entry point currently supports these 15 tasks:

- `kuka-reach-v3`
- `kuka-push-v3`
- `kuka-push-cube-v3`
- `kuka-handle-press-side-v3`
- `kuka-button-press-v3`
- `kuka-button-press-topdown-v3`
- `kuka-peg-unplug-side-v3`
- `kuka-hammer-v3`
- `kuka-drawer-close-v3`
- `kuka-faucet-open-v3`
- `kuka-faucet-close-v3`
- `kuka-window-open-v3`
- `kuka-window-close-v3`
- `kuka-door-open-v3`
- `kuka-pick-place-v3`

On 2026-10-09, all twelve tasks were validated locally with Python 3.10.20,
Gymnasium 1.3.0, MuJoCo 3.3.0, and NumPy 2.2.6: each model loaded, the
environment initialized and reset, and three actions produced finite
observations. This is a smoke test, not evidence of task solvability or trained
policy performance. The three subsequently added tasks are covered by the same
headless registration, model-loading, space-contract, and finite-step tests.

## Public contracts and reward selection

All registered KUKA tasks use a 39-element `float32` observation and a
four-element continuous action. The action contains Cartesian end-effector
motion followed by gripper control. The episode horizon is 200 steps. Task and
goal sampling are controlled by MetaWorld's task payload and seed; callers that
need exact experiment reproduction should retain both.

Select rewards with the `reward_function_version` argument to
`Meta-World-KUKA/MT1`. Ordinary `kuka-push-v3` uses only its Sawyer-inspired
`v2` reward. `kuka-push-cube-v3` is likewise a single Sawyer-inspired v2 task;
its former curricula, contact starts, and experimental rewards have been
removed. Historical checkpoints remain research records but are incompatible
with the new task definition. Registration rejects obsolete reward versions.

```python
env = gym.make(
    "Meta-World-KUKA/MT1",
    env_name="kuka-push-cube-v3",
    reward_function_version="v2",
    seed=42,
)
```

The Hammer and Peg Unplug experiments also expose separately scoped reward
variant parameters through the same registered factory. See the implementation
and tests before changing them; these options are part of existing experiment
contracts.

For headless testing, use a rendering backend supported by the host. In this
checkout the validated interpreter is normally selected explicitly by the
developer; the generic command is:

```bash
MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/metaworld_kuka
```

The plugin setting avoids unrelated globally installed pytest plugins.

## Assets, provenance, and limitations

- `metaworld/assets` contains assets inherited from MetaWorld v3.1.1.
- `metaworld_kuka/assets/kuka_xyz` contains KUKA task MJCF entry points and
  shared fragments.
- `metaworld_kuka/assets/objects/meshes/kuka` contains the KUKA IIWA link meshes
  required by every registered KUKA task.
- `metaworld_kuka/assets/meshes/kuka/robotiq_2f85` contains the Robotiq model,
  meshes, provenance notes, and its BSD-2-Clause license. It was imported from
  `Metaworld-KUKA-IIWA-R800` commit
  `256a721408df1585a6c04e9c5556e857cc760e06`.
- The KUKA IIWA link meshes came from the MIT-licensed public
  `Ghiara/Metaworld-KUKA-IIWA-R800` repository. The direct source and the limits
  of the available authorship record are documented in
  [`docs/asset_provenance.md`](docs/asset_provenance.md).
- This publication contains the MetaWorld v3.1.1 runtime subset needed by the
  KUKA adaptation. It intentionally omits unrelated upstream policies,
  documentation, CI configuration, Docker files, and tests.
- Training output under `runs/` is local-only and ignored. It is not required to
  load the environments.
- Smoke validation does not assess reward quality, policy success rates, visual
  correctness on every renderer, or long-horizon stability.

Additional local engineering notes are in `docs/`, and diagnostic utilities
are in `scripts/`.

## Relationship to continual-learning

This repository owns environment registration, task implementations, reward
semantics, MuJoCo models, and assets. The sibling `continual-learning`
repository consumes the public `Meta-World-KUKA/MT1` entry point through its
`kuka-v3` adapter and owns training configurations, checkpoints, evaluation,
and policy viewers. Keep both repositories at mutually compatible revisions
when reproducing an experiment. Environment cleanup must not rename public
modules, task IDs, reward versions, XML entry points, or observation/action
contracts without a coordinated migration in that consumer.

Repository audit details and deferred cleanup candidates are recorded in
[`docs/REPOSITORY_CLEANUP_REPORT.md`](docs/REPOSITORY_CLEANUP_REPORT.md).

## Original MetaWorld README

[![Python](https://img.shields.io/pypi/pyversions/metaworld.svg)](https://badge.fury.io/py/metaworld)
[![PyPI](https://badge.fury.io/py/metaworld.svg)](https://badge.fury.io/py/metaworld.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2505.11289-b31b1b.svg)](https://arxiv.org/pdf/2505.11289)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
    <a href="https://metaworld.farama.org/" target = "_blank">
    <img src="https://github.com/Farama-Foundation/Metaworld/blob/main/metaworld-text.png" width="500px"/>
</p>

Meta-World is an open source benchmark for developing and evaluating multi-task and meta reinforcement learning algorithms for continuous control robotic manipulation environments, with various benchmarks to evaluate different aspects of reinforcement learning algorithms.

The documentation website is at [metaworld.farama.org](https://metaworld.farama.org), and we have a public discord server (which we also use to coordinate development work) that you can join here: https://discord.gg/bnJ6kubTg6

## Installation

To install Meta-World, use `pip install metaworld`

We support and test for Python 3.10, 3.11, 3.12, and 3.13 on Linux and macOS. We will accept PRs related to Windows, but do not officially support it.

## API

The Meta-World API follows the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) API for environment creation and environment interactions.

To create a benchmark and interact with it:

```python
import gymnasium as gym
import metaworld


env = gym.make("Meta-World/MT1", env_name="reach-v3")

observation, info = env.reset()
for _ in range(500):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

## Available Benchmarks

### Multi-Task Benchmarks
The MT1, MT10, and MT50 benchmarks are the Multi-Task Benchmarks. These benchmarks are used to learn a multi-task policy that can learn 1, 10, or 50 training tasks simultaneously. MT1 benchmarks can be created with any of the 50 tasks available in Meta-World.
In the MT10 and MT50 benchmarks, the observations returned by the benchmark will come with one-hot task IDs appended to the state.

### Meta-Learning Benchmarks
The ML1, ML10, and ML45 benchmarks are 3 meta-reinforcement learning benchmarks available in Meta-World. The ML1 benchmark can be used with any of the 50 tasks available in Meta-World.
The ML1 benchmark tests for few-shot adaptation to goal variations within a single task. The ML10 and ML45 both test few-shot adaptation to new tasks. ML10 comprises 10 train tasks with 5 test tasks, while ML45 comprises of 45 training tasks with 5 test tasks.


## Creating Multi-Task Benchmarks

### MT1
```python
import gymnasium as gym
import metaworld

seed = 42 # for reproducibility

env = gym.make('Meta-World/MT1', env_name='reach-v3', seed=seed) # MT1 with the reach environment

obs, info = env.reset()

a = env.action_space.sample() # randomly sample an action
obs, reward, truncate, terminate, info = env.step(a) # apply the randomly sampled action
```

### MT10
MT10 has two different versions that can be returned by ```gym.make```. The first version is the synchronous version of the benchmark where all environments are contained within the same process.
For users with limited compute resources, the synchronous option needs the least resources.
```python
import gymnasium as gym
import metaworld

seed = 42

envs = gym.make_vec('Meta-World/MT10', vector_strategy='sync', seed=seed) # this returns a Synchronous Vector Environment with 10 environments

obs, info = envs.reset() # reset all 10 environments

a = env.action_space.sample() # sample an action for each environment

obs, reward, truncate, terminate, info = envs.step(a) # step all 10 environments
```
Alternatively, for users with more compute we also provide the asynchronous version of the MT10 benchmark where each environment is isolated in it's own process and must use inter-process messaging via pipes to communicate.

```python
envs = gym.make_vec('Meta-World/MT10', vector_strategy='async', seed=seed) # this returns an Asynchronous Vector Environment with 10 environments
```

### MT50
MT50 also contains two different versions, a synchronous and an asynchronous version, of the environments.
```python
import gymnasium as gym
import metaworld

seed = 42

envs = gym.make_vec('Meta-World/MT50', vector_strategy='sync', seed=seed) # this returns a Synchronous Vector Environment with 50 environments

obs, info = envs.reset() # reset all 50 environments

a = env.action_space.sample() # sample an action for each environment

obs, reward, truncate, terminate, info = envs.step(a) # step all 50 environments
```

```python
envs = gym.make_vec('Meta-World/MT50', vector_strategy='async', seed=seed) # this returns an Asynchronous Vector Environment with 50 environments
```


## Meta-Learning Benchmarks
Each Meta-reinforcement learning benchmark has training and testing environments. These environments must be created separately as follows.

### ML1
```python
import gymnasium as gym
import metaworld

seed = 42


train_envs = gym.make('Meta-World/ML1-train', env_name='reach-V3', seed=seed)
test_envs = gym.make('Meta-World/ML1-test', env_name='reach-V3', seed=seed)

# training procedure use train_envs
# testing procedure use test_envs

```


### ML10
Similar to the Multi-Task benchmarks, the ML10 and ML45 environments can be run in synchronous or asynchronous modes.


```python
import gymnasium as gym
import metaworld

train_envs = gym.make_vec('Meta-World/ML10-train', vector_strategy='sync', seed=seed) # or vector_strategy='async'
test_envs = gym.make_vec('Meta-World/ML10-test', vector_strategy='sync', seed=seed) # or vector_strategy='async'
```


### ML45
```python
import gymnasium as gym
import metaworld

train_envs = gym.make_vec('Meta-World/ML45-train', vector_strategy='sync', seed=seed) # or vector_strategy='async'
test_envs = gym.make_vec('Meta-World/ML45-test', vector_strategy='sync', seed=seed) # or vector_strategy='async'
```


## Custom Benchmarks
Finally, we also provide support for creating custom benchmarks by combining any number of Meta-World environments.

The prefix 'mt' will return environments that are goal observable for Multi-Task reinforcement learning, while the prefix 'ml' will return environments that are partially observable for Meta-reinforcement learning.
Like the included MT and ML benchmarks, these environments can also be run in synchronous or asynchronous mode.
In order to create a custom benchmark, the user must provide a list of environment names with the suffix '-v3'.

```python
import gymnasium as gym
import metaworld

envs = gym.make_vec('Meta-World/custom-mt-envs',vector_strategy='sync', envs_list=['env_name_1-v3', 'env_name_2-v3', 'env_name_3-v3'], seed=seed) # or vector_strategy='async'
envs = gym.make_vec('Meta-World/custom-ml-envs',vector_strategy='sync', envs_list=['env_name_1-v3', 'env_name_2-v3', 'env_name_3-v3'], seed=seed) # or vector_strategy='async'
```

## Development Roadmap

We have a roadmap for future development work for Gymnasium available here: https://github.com/Farama-Foundation/Metaworld/issues/500

## Benchmark Code

Code for producing results found in the paper can be found at: https://github.com/rainx0r/metaworld-algorithms

## Citation

You can cite Meta-World using our related paper (https://openreview.net/forum?id=1de3azE606) as:

```
@inproceedings{
mclean2025metaworld,
title={Meta-World+: An Improved, Standardized, {RL} Benchmark},
author={Reginald McLean and Evangelos Chatzaroulas and Luc McCutcheon and Frank R{\"o}der and Tianhe Yu and Zhanpeng He and K.R. Zentner and Ryan Julian and J K Terry and Isaac Woungang and Nariman Farsad and Pablo Samuel Castro},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2025},
url={https://openreview.net/forum?id=1de3azE606}
}
```
