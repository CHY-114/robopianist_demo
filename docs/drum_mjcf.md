# Building a Drum Kit MJCF Model

This note demonstrates how to build a simple drum kit for MuJoCo in the same style as RoboPianist's procedurally-generated piano.  The implementation lives in [`robopianist/models/drum/drum_mjcf.py`](../robopianist/models/drum/drum_mjcf.py) and follows three main steps:

1. **Author helper routines** – `_add_drum_shell`, `_add_cymbal`, and `_add_hi_hat` create reusable subassemblies for drums, cymbals, and a pedal-driven hi-hat.  Each helper constructs MJCF bodies, geoms, and strike sites that later tasks can bind to for sensing or actuation.
2. **Instantiate the MJCF tree** – `build` sets compiler defaults, adds a stage floor, and then calls the helpers to place the kick, snare, toms, crash, ride, and hi-hat within a parent `drum_kit` body.  Optional extra percussion can be supplied via the `extra_percussion` argument.
3. **(Optional) add actuators** – when `add_actuators=True`, the hi-hat receives a `general` actuator wired to its slide joint, mimicking a foot pedal.  Other controllers (e.g., mallets or sticks) can be added by creating additional actuators on the strike joints exposed by the helpers.

Once the MJCF is generated you can export it to XML, inspect it with MuJoCo's visualizer, or integrate it into a `composer.Entity` the same way RoboPianist wraps the piano MJCF.

## Testing the Drum Kit

To verify that the procedural model remains renderable and simulation-ready, run the repository's automated tests:

```bash
pytest robopianist/models/drum/drum_mjcf_test.py
```

The tests compile the MJCF, advance the physics for several steps, and request a rendered frame, mirroring the checks we use for RoboPianist's piano videos. Installing `dm-control` (and its MuJoCo dependency) is required for the checks to run; otherwise they will be skipped with a descriptive message. If you have `pytest-xdist` installed you can optionally append `-n auto` to parallelize the suite.
