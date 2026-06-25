# Morphos spec file schema

A spec file is a small JSON document that describes one design run. It is the
input to `morphos run <spec_file>` on the command line and to
`morphos.from_json(path)` in Python. Two shapes are supported, and every spec
file carries a `"version"` field for forward compatibility.

Current schema version: **`"1.0"`**.

## Shape A: intent spec (recommended)

An intent spec names a `DesignIntent` subclass and the engineering-unit
parameters its constructor takes. This is the human-facing path and the only
**lossless** one: an intent captures its constructor arguments, so it
round-trips exactly through `morphos.to_json` / `morphos.from_json`.

```json
{
  "version": "1.0",
  "intent": "CantileverIntent",
  "params": {
    "span": 60,
    "height": 30,
    "load": -1.0,
    "volume_fraction": 0.4,
    "max_iter": 200
  }
}
```

| Field     | Type   | Meaning                                                      |
| --------- | ------ | ----------------------------------------------------------- |
| `version` | string | Schema version. `"1.0"` today.                              |
| `intent`  | string | Class name of a concrete `DesignIntent` subclass.           |
| `params`  | object | Keyword arguments for that intent's constructor.            |

`params` keys are exactly the intent constructor's parameters; omitted keys take
the constructor's defaults. Unknown intent names raise a `ValueError` listing the
available intents. The available intents are every concrete subclass of
`morphos.intent.DesignIntent` (for example `CantileverIntent`, `ChannelIntent`,
`ThermalSinkIntent`, `StokesBrinkmanChannelIntent`, `ThermoElasticIntent`,
`HeatExchangerIntent`).

## Shape B: raw spec (structural metadata)

A raw spec describes an already-built `DesignSpec`:

```json
{
  "version": "1.0",
  "spec": {
    "name": "cantilever",
    "initial": { "values": [[0.5, 0.5], [0.5, 0.5]], "spacing": 1.0 },
    "optimizer": { "class": "TopologyOptimizer", "params": { "max_iter": 200 } },
    "oracle": { "class": "ElasticityOracle" },
    "objective": { "class": "MaximizeValue" },
    "constraint": null
  }
}
```

Shape B is **not lossless**. A physics oracle consumes its boundary conditions
(loads, supports, sources) into solver-internal state at construction and cannot
reproduce them, so `DesignSpec.to_dict()` emits only the structural surface (the
initial field, the optimizer parameters, and the class names of the oracle,
objective and constraint). Reconstructing a runnable design from a raw spec dict
therefore raises a clear error directing you to Shape A. To persist a fully
reconstructable design, serialise the originating `DesignIntent` (Shape A).

## Python API

```python
import morphos

intent = morphos.from_json("spec.json")   # -> DesignIntent or DesignSpec
morphos.to_json(intent, "spec.json")      # write either shape back out
```
