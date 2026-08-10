# Adding a problem to the viewer

The viewer is deliberately generic. It knows nothing about rockets — it knows
about *parameters* and *trajectories*. Every future week plugs in by adding one
Python file. No JavaScript is ever touched.

## The three-step recipe

Create `src/gnc/problems/your_problem.py`:

```python
from ..registry import Problem, register
from ..types import Param, Series, Trajectory, attitudes_from_thrust


@register
class YourProblem(Problem):
    slug = "scvx-6dof"                 # unique, url-safe
    title = "6-DoF SCvx"
    summary = "One line, shown under the problem picker."
    phase = "Week 3"
    scene_scale = 900.0                # rough scene size in metres, frames the camera

    def params(self):
        return [
            Param("h0", "Initial altitude", 700.0, min=100, max=2000,
                  step=10, unit="m", group="Initial state"),
            Param("mode", "Mode", "fast", kind="choice",
                  choices=["fast", "accurate"], group="Optimisation"),
        ]

    def solve(self, values):
        p = self.merge(values)         # fills defaults, coerces types
        ...
        return Trajectory(...)
```

Restart `run_viewer.py` and it appears in the dropdown with working controls.

## Parameter kinds

| `kind`   | Renders as | Notes                                     |
|----------|-----------|-------------------------------------------|
| `float`  | slider    | uses `min`, `max`, `step`, `unit`         |
| `int`    | slider    | integer-snapped                           |
| `bool`   | checkbox  | `default` must be `True`/`False`          |
| `choice` | dropdown  | requires `choices=[...]`                  |

`group` controls the section heading. `help` renders as small print under the
control — use it to explain a trap rather than restating the label.

## The Trajectory contract

Frames are fixed everywhere: **right-handed, +Y up, pad at the origin**. Three.js
uses the same convention, so nothing is transformed in the browser.

| Field        | Shape     | Meaning                                     |
|--------------|-----------|---------------------------------------------|
| `t_state`    | `(N+1,)`  | state sample times                          |
| `t_control`  | `(N,)`    | control sample times (zero-order hold)      |
| `position`   | `(N+1,3)` | metres                                      |
| `velocity`   | `(N+1,3)` | m/s                                         |
| `thrust`     | `(N,3)`   | commanded acceleration or force vector      |
| `attitude`   | `(N+1,4)` | quaternion `[x, y, z, w]`, body→world       |
| `series`     | list      | named scalars for the telemetry strip       |
| `thrust_max` | scalar    | normalises plume size and path colour       |

A translation-only problem can call `attitudes_from_thrust(thrust)` to point the
vehicle along its thrust vector. A real 6-DoF solver returns its own attitude and
skips that helper.

### Reporting failure

Never raise out of `solve()`. Return a `Trajectory` with `feasible=False`, the
solver's `status`, and a `notes` entry explaining what to relax. The UI turns the
status chip red and surfaces the first note as a toast. `tests/test_problems.py`
asserts this.

### Anything worth saying, say in `notes`

`notes` is a list of strings shown in the flight-data panel. It is the right
place for results a plot cannot convey — `landing_1d.py` uses it to report that
its own objective is degenerate.

## Extras

* `diagnostics` is a free-form dict; numeric entries are auto-formatted into the
  flight-data table. A key named `glideslope_deg` additionally draws the
  approach-corridor cone.
* **EXPORT** downloads the full run as JSON (parameters + trajectory). Those files
  are static, so a solved run can be published without the Python backend.
* Verify with `python tests/test_problems.py` — it checks shapes, quaternion
  normalisation, terminal conditions, and graceful infeasibility for every
  registered problem automatically.
